"""Group-aware random search for a representative TEST split.

A *group* is one well; groups are never split between train and test.  We
generate many random candidate group subsets whose page count lands near
``round(test_frac * N)``, score each one (lower is better) and keep the best.

Score terms (all non-negative, combined as a weighted sum):

* ``size``            |n_test_pages - target| in pages.
* ``strata_tv``       total-variation distance between the test and train page
                      distributions over strata ``(is_colour, log_type)``.
                      Each page inherits the stratum of its *group* (majority
                      over the group's pages, see :func:`group_strata`) so the
                      strata used for balancing are exactly the units we can
                      move between splits.
* ``smd``             mean over NUMERIC_FEATURES of |standardised mean
                      difference| (pooled SD, NaN-safe), each clipped at
                      SMD_CAP.
* ``empty_stratum``   number of strata containing >= 2 groups but no test
                      group.
* ``dominant_group``  1 + excess share if any single test group contributes
                      more than MAX_GROUP_SHARE of the test pages, else 0.

Everything is evaluated vectorised over a (n_candidates x n_groups) boolean
membership matrix, so 50k candidates score in well under a second.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Weights / constants (lower total score is better)
# --------------------------------------------------------------------------
# 1 point per page away from the target size.  Every other weight is
# expressed relative to "one page off target".
W_SIZE = 1.0
# TV distance is in [0, 1].  With ~10 test pages a single page moved between
# strata changes TV by ~0.1, so W=4 makes "one page in the wrong stratum"
# (~0.4) cheaper than "one page off target" (1.0) — size first, then mix.
W_STRATA_TV = 4.0
# Mean |SMD| over 4 features; ~0.2 is "small", ~0.5 "medium".  W=2 makes a
# medium imbalance (1.0) as bad as one page off target.
W_SMD = 2.0
# Missing a whole multi-group stratum from test is a coverage failure, worth
# more than a couple of pages of size error.
W_EMPTY_STRATUM = 3.0
# A single well dominating the test set makes it unrepresentative.
W_DOMINANT_GROUP = 5.0

MAX_GROUP_SHARE = 0.40
SMD_CAP = 3.0
NUMERIC_FEATURES = ("blank_frac", "texture_diversity", "legend_aspect", "column_height_px")
DEFAULT_N_ITER = 50_000
TARGET_JITTER = 2  # candidate page targets are drawn from target +/- this

WEIGHTS = {
    "size": W_SIZE,
    "strata_tv": W_STRATA_TV,
    "smd": W_SMD,
    "empty_stratum": W_EMPTY_STRATUM,
    "dominant_group": W_DOMINANT_GROUP,
}
TERM_NAMES = tuple(WEIGHTS)

_TOTAL_DECIMALS = 9  # totals are rounded before ranking so float noise can't break ties


# --------------------------------------------------------------------------
# Value coercion helpers (rows may come from CSV and contain strings / None)
# --------------------------------------------------------------------------
def to_float(x) -> float:
    if x is None:
        return math.nan
    if isinstance(x, (bool, np.bool_)):
        return float(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def to_bool(x) -> bool:
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "t", "yes", "y")
    if x is None:
        return False
    try:
        if isinstance(x, float) and math.isnan(x):
            return False
    except TypeError:
        pass
    return bool(x)


def stratum_label(stratum: tuple[bool, str]) -> str:
    is_colour, log_type = stratum
    return f"{'colour' if is_colour else 'BW'}/{log_type}"


# --------------------------------------------------------------------------
# Strata
# --------------------------------------------------------------------------
def _row_index(rows: Sequence[Mapping]) -> dict[str, Mapping]:
    idx: dict[str, Mapping] = {}
    for r in rows:
        stem = str(r["stem"])
        if stem in idx:
            raise ValueError(f"duplicate stem in rows: {stem!r}")
        idx[stem] = r
    return idx


def _validate(rows_by_stem: Mapping[str, Mapping], groups: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """Return stem -> gid; raise if groups and rows don't line up."""
    stem_to_gid: dict[str, str] = {}
    for gid, stems in groups.items():
        if not stems:
            raise ValueError(f"group {gid!r} has no stems")
        for s in stems:
            if s in stem_to_gid:
                raise ValueError(f"stem {s!r} is in two groups: {stem_to_gid[s]!r} and {gid!r}")
            if s not in rows_by_stem:
                raise ValueError(f"stem {s!r} (group {gid!r}) has no row")
            stem_to_gid[s] = gid
    missing = sorted(set(rows_by_stem) - set(stem_to_gid))
    if missing:
        raise ValueError(f"{len(missing)} row stem(s) belong to no group, e.g. {missing[:3]}")
    return stem_to_gid


def group_strata(rows: Sequence[Mapping], groups: Mapping[str, Sequence[str]]) -> dict[str, tuple[bool, str]]:
    """Majority (is_colour, log_type) per group.

    Ties are broken deterministically: highest count first, then the smallest
    key under ordering (is_colour as bool, log_type as str), i.e. BW before
    colour, then alphabetical log_type.
    """
    by_stem = _row_index(rows)
    out: dict[str, tuple[bool, str]] = {}
    for gid in sorted(groups):
        c = Counter()
        for s in groups[gid]:
            r = by_stem[s]
            c[(to_bool(r.get("is_colour")), str(r.get("log_type") or "other"))] += 1
        out[gid] = min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return out


# --------------------------------------------------------------------------
# Pre-computed problem matrices
# --------------------------------------------------------------------------
@dataclass
class _Problem:
    gids: list[str]                 # sorted
    gpages: np.ndarray              # (G,) pages per group
    strata_keys: list[tuple]        # sorted unique strata
    g_stratum: np.ndarray           # (G,) stratum index per group
    g_strata_pages: np.ndarray      # (G, S) pages per stratum (one-hot * pages)
    eligible_strata: np.ndarray     # (S,) bool: stratum has >= 2 groups
    feat_sum: np.ndarray            # (G, F)
    feat_sumsq: np.ndarray          # (G, F)
    feat_cnt: np.ndarray            # (G, F)
    feat_overall_sd: np.ndarray     # (F,)
    target: int
    n_pages: int


def _build(rows, groups, test_frac, strata) -> _Problem:
    by_stem = _row_index(rows)
    _validate(by_stem, groups)
    gids = sorted(groups)
    G = len(gids)
    gpages = np.array([len(groups[g]) for g in gids], dtype=np.int64)
    strata_keys = sorted(set(strata[g] for g in gids))
    s_index = {k: i for i, k in enumerate(strata_keys)}
    g_stratum = np.array([s_index[strata[g]] for g in gids], dtype=np.int64)
    S = len(strata_keys)
    onehot = np.zeros((G, S), dtype=np.float64)
    onehot[np.arange(G), g_stratum] = 1.0
    g_strata_pages = onehot * gpages[:, None]
    eligible = onehot.sum(axis=0) >= 2

    F = len(NUMERIC_FEATURES)
    fsum = np.zeros((G, F))
    fsq = np.zeros((G, F))
    fcnt = np.zeros((G, F))
    allvals: list[list[float]] = [[] for _ in range(F)]
    for gi, g in enumerate(gids):
        for s in groups[g]:
            r = by_stem[s]
            for fi, f in enumerate(NUMERIC_FEATURES):
                v = to_float(r.get(f))
                if not math.isnan(v):
                    fsum[gi, fi] += v
                    fsq[gi, fi] += v * v
                    fcnt[gi, fi] += 1
                    allvals[fi].append(v)
    overall_sd = np.array([np.std(v, ddof=1) if len(v) >= 2 else 0.0 for v in allvals])
    # Centre features to keep sum-of-squares variance numerically stable
    # (column_height_px ~ 1e4 -> squares ~ 1e8).
    means = np.array([np.mean(v) if v else 0.0 for v in allvals])
    fsq = fsq - 2 * means * fsum + fcnt * means**2
    fsum = fsum - fcnt * means

    n_pages = int(gpages.sum())
    return _Problem(
        gids=gids, gpages=gpages, strata_keys=strata_keys, g_stratum=g_stratum,
        g_strata_pages=g_strata_pages, eligible_strata=eligible,
        feat_sum=fsum, feat_sumsq=fsq, feat_cnt=fcnt, feat_overall_sd=overall_sd,
        target=int(round(test_frac * n_pages)), n_pages=n_pages,
    )


def _score_matrix(P: _Problem, M: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorised scoring. M: (C, G) bool membership. Returns arrays of shape (C,...)."""
    Mf = M.astype(np.float64)
    n_test = Mf @ P.gpages.astype(np.float64)
    n_train = P.n_pages - n_test

    # --- size
    size_dev = np.abs(n_test - P.target)

    # --- strata TV
    st_test = Mf @ P.g_strata_pages
    st_tot = P.g_strata_pages.sum(axis=0)
    st_train = st_tot[None, :] - st_test
    with np.errstate(invalid="ignore", divide="ignore"):
        p_test = st_test / n_test[:, None]
        p_train = st_train / n_train[:, None]
        tv = 0.5 * np.abs(p_test - p_train).sum(axis=1)
    tv = np.where((n_test > 0) & (n_train > 0), tv, 1.0)

    # --- SMD (NaN-safe; features pre-centred)
    s1 = Mf @ P.feat_sum
    q1 = Mf @ P.feat_sumsq
    c1 = Mf @ P.feat_cnt
    s2 = P.feat_sum.sum(0)[None, :] - s1
    q2 = P.feat_sumsq.sum(0)[None, :] - q1
    c2 = P.feat_cnt.sum(0)[None, :] - c1
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = s1 / c1
        m2 = s2 / c2
        ss1 = np.maximum(q1 - s1 * m1, 0.0)
        ss2 = np.maximum(q2 - s2 * m2, 0.0)
        dof = c1 + c2 - 2
        pooled = np.sqrt((np.nan_to_num(ss1) + np.nan_to_num(ss2)) / dof)
    # Guard SD ~ 0 (e.g. tiny or constant samples): fall back to overall SD.
    tiny = ~(pooled > 1e-9 * np.maximum(P.feat_overall_sd, 1e-12)[None, :]) | ~np.isfinite(pooled)
    sd = np.where(tiny, P.feat_overall_sd[None, :], pooled)
    diff = m1 - m2
    with np.errstate(invalid="ignore", divide="ignore"):
        smd = np.abs(diff) / sd
    const_feature = (P.feat_overall_sd <= 0)[None, :]
    smd = np.where(const_feature, 0.0, smd)
    # A side with no known values for a feature: can't compare -> max penalty,
    # unless the whole feature is unknown everywhere.
    no_vals = (c1 == 0) | (c2 == 0)
    all_unknown = (P.feat_cnt.sum(0) == 0)[None, :]
    smd = np.where(no_vals & ~all_unknown, SMD_CAP, smd)
    smd = np.where(np.isfinite(smd), smd, SMD_CAP)
    smd = np.minimum(smd, SMD_CAP)
    smd_mean = smd.mean(axis=1) if smd.shape[1] else np.zeros(len(M))

    # --- empty strata
    g_onehot = (P.g_strata_pages > 0).astype(np.float64)
    test_groups_per_stratum = Mf @ g_onehot
    empty = (test_groups_per_stratum == 0) & P.eligible_strata[None, :]
    n_empty = empty.sum(axis=1).astype(np.float64)

    # --- dominant group
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = np.where(M, P.gpages[None, :] / n_test[:, None], 0.0)
    shares = np.nan_to_num(shares)
    max_share = shares.max(axis=1) if shares.shape[1] else np.zeros(len(M))
    dom_idx = shares.argmax(axis=1) if shares.shape[1] else np.zeros(len(M), int)
    excess = max_share - MAX_GROUP_SHARE
    dominant = np.where(excess > 1e-12, 1.0 + excess, 0.0)

    terms = {
        "size": W_SIZE * size_dev,
        "strata_tv": W_STRATA_TV * tv,
        "smd": W_SMD * smd_mean,
        "empty_stratum": W_EMPTY_STRATUM * n_empty,
        "dominant_group": W_DOMINANT_GROUP * dominant,
    }
    total = sum(terms.values())
    return {
        **terms, "total": total,
        "n_test": n_test, "size_dev": size_dev, "tv": tv, "smd_feat": smd, "smd_mean": smd_mean,
        "empty": empty, "n_empty": n_empty, "max_share": max_share, "dom_idx": dom_idx,
        "dominant_raw": dominant,
    }


def _breakdown(P: _Problem, res: dict, i: int) -> dict:
    out = {k: float(res[k][i]) for k in TERM_NAMES}
    out["total"] = float(res["total"][i])
    out["n_test_pages"] = int(res["n_test"][i])
    out["n_pages"] = P.n_pages
    out["target"] = P.target
    out["size_dev"] = float(res["size_dev"][i])
    out["tv"] = float(res["tv"][i])
    out["smd_mean"] = float(res["smd_mean"][i])
    out["smd_per_feature"] = {f: float(res["smd_feat"][i, j]) for j, f in enumerate(NUMERIC_FEATURES)}
    out["n_empty_strata"] = int(res["n_empty"][i])
    out["empty_strata"] = [stratum_label(P.strata_keys[j]) for j in np.flatnonzero(res["empty"][i])]
    out["max_group_share"] = float(res["max_share"][i])
    out["dominant_gid"] = P.gids[int(res["dom_idx"][i])] if out["n_test_pages"] else None
    out["dominant_raw"] = float(res["dominant_raw"][i])
    out["weights"] = dict(WEIGHTS)
    out["max_group_share_limit"] = MAX_GROUP_SHARE
    return out


def score_split(test_groups: Iterable[str], rows, groups, test_frac: float, strata) -> dict:
    """Score one split given its set of test group ids. Lower ``total`` is better."""
    P = _build(rows, groups, test_frac, strata)
    test_groups = set(test_groups)
    unknown = sorted(test_groups - set(P.gids))
    if unknown:
        raise ValueError(f"unknown group id(s): {unknown}")
    M = np.array([[g in test_groups for g in P.gids]], dtype=bool)
    return _breakdown(P, _score_matrix(P, M), 0)


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------
def _resolve_constraints(stems: Iterable[str], stem_to_gid: Mapping[str, str], what: str) -> set[str]:
    stems = [stems] if isinstance(stems, str) else list(stems)
    unknown = sorted(s for s in stems if s not in stem_to_gid)
    if unknown:
        raise ValueError(f"{what}: unknown stem(s): {unknown}")
    return {stem_to_gid[s] for s in stems}


def _generate(P: _Problem, rng: np.random.Generator, n_iter: int,
              forced: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Return (n_iter, G) bool candidate matrix.

    Each candidate = forced groups + free groups taken in a random order until
    the page count reaches a random target in target +/- TARGET_JITTER.  In half
    of the candidates (random), groups that would overshoot the target by more
    than one page are skipped instead of taken, which adds variety (larger
    wells are not always picked when they come early in the permutation).
    """
    G = len(P.gids)
    M = np.zeros((n_iter, G), dtype=bool)
    M[:, forced] = True
    free_idx = np.flatnonzero(free)
    if len(free_idx) == 0:
        return M
    base = int(P.gpages[forced].sum())
    targets = P.target + rng.integers(-TARGET_JITTER, TARGET_JITTER + 1, size=n_iter)
    need = np.maximum(targets - base, 0)
    keys = rng.random((n_iter, len(free_idx)))
    perm = np.argsort(keys, axis=1, kind="stable")
    gp = P.gpages[free_idx][perm].astype(np.int64)            # (n, Ffree)
    skip_mode = rng.random(n_iter) < 0.5

    # Mode A (greedy): take a prefix of the permutation until cumsum >= need.
    cs = np.cumsum(gp, axis=1)
    take_a = (cs - gp) < need[:, None]

    # Mode B (skip overshooters): sequential over permutation positions, but
    # vectorised across candidates.
    take_b = np.zeros_like(take_a)
    acc = np.zeros(n_iter, dtype=np.int64)
    for j in range(gp.shape[1]):
        room = acc < need
        fits = acc + gp[:, j] <= need + 1
        t = room & fits
        take_b[:, j] = t
        acc += np.where(t, gp[:, j], 0)
    # If skipping left a candidate short (only big groups left), fall back to greedy.
    take_b = np.where((acc < need - 1)[:, None], take_a, take_b)

    take = np.where(skip_mode[:, None], take_b, take_a)
    rows_idx = np.repeat(np.arange(n_iter), perm.shape[1]).reshape(perm.shape)
    M[rows_idx[take], free_idx[perm][take]] = True
    return M


def search_split(rows, groups, test_frac: float = 0.2, seed: int = 42, n_iter: int = DEFAULT_N_ITER,
                 force_test: Iterable[str] = (), exclude_test: Iterable[str] = ()) -> dict:
    """Random search over group subsets; returns the best split found.

    Fully deterministic for the same rows, groups, test_frac, seed, n_iter and
    constraints (input ordering of rows / groups does not matter).
    """
    if not 0 < test_frac < 1:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")
    by_stem = _row_index(rows)
    stem_to_gid = _validate(by_stem, groups)
    strata = group_strata(rows, groups)
    P = _build(rows, groups, test_frac, strata)

    forced_g = _resolve_constraints(force_test, stem_to_gid, "force_test")
    excl_g = _resolve_constraints(exclude_test, stem_to_gid, "exclude_test")
    clash = sorted(forced_g & excl_g)
    if clash:
        raise ValueError(f"group(s) {clash} contain both force_test and exclude_test stems")

    forced = np.array([g in forced_g for g in P.gids], dtype=bool)
    excluded = np.array([g in excl_g for g in P.gids], dtype=bool)
    free = ~forced & ~excluded

    rng = np.random.default_rng(seed)
    M = _generate(P, rng, n_iter, forced, free)
    # Deduplicate (np.unique sorts rows -> order-independent of generation).
    M = np.unique(M, axis=0)
    res = _score_matrix(P, M)
    total = np.round(res["total"], _TOTAL_DECIMALS)
    best_val = total.min()
    tied = np.flatnonzero(total == best_val)
    # Tie-break: lexicographically smallest sorted gid tuple. gids are sorted,
    # so index tuples order identically to gid tuples.
    best = min(tied, key=lambda i: tuple(np.flatnonzero(M[i]).tolist()))

    test_gids = [P.gids[j] for j in np.flatnonzero(M[best])]
    test_stems = sorted(s for g in test_gids for s in groups[g])
    train_stems = sorted(s for g in P.gids if g not in set(test_gids) for s in groups[g])
    return {
        "test": test_stems,
        "train": train_stems,
        "test_groups": sorted(test_gids),
        "score": _breakdown(P, res, int(best)),
        "n_iter": int(n_iter),
        "n_unique_candidates": int(len(M)),
        "seed": int(seed),
        "test_frac": float(test_frac),
        "force_test": sorted(forced_g),
        "exclude_test": sorted(excl_g),
        "group_strata": {g: [bool(strata[g][0]), strata[g][1]] for g in P.gids},
    }
