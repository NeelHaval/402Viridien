"""Check regional lithology plausibility; standalone Python 3.10+ / Colab module."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

# Specific classes precede generic classes; first match wins (Dolomitic clay is carbonate).
COARSE_CLASSES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "carbonate": ("limestone", "dolomit", "chalk", "marl", "carbonate"),
    "evaporite": ("halite", "salt", "anhydrite", "gypsum", "sylvite", "evaporite"),
    "organic": ("coal", "lignite", "bitumen"),
    "volcanic": ("tuff", "volcan", "igneous", "basalt", "granite", "extrusive", "intrusive"),
    "sand_silt": ("sand", "silt", "conglomerate", "gravel"),
    "mud": ("shale", "clay", "mud", "argill"),
})
CLASS_NAMES = (*COARSE_CLASSES, "other")
UNKNOWN_NAMES = frozenset(("", "blank", "unknown", "nan", "none"))
REQUIRED_COLUMNS = frozenset(("top_y", "bottom_y", "cluster_id", "lithology", "top_depth", "base_depth"))
API_URL = "https://macrostrat.org/api/units"
REQUEST_TIMEOUT = 30
FEET_PER_METRE = 1 / 0.3048
OCR_TILE_HEIGHT = 1600
OCR_WIDTH = 2400
ASSUMED_HEMISPHERE_FACTOR = 0.8


@dataclass(frozen=True)
class Fingerprint:
    """Thickness-weighted coarse lithology composition and descriptive statistics."""
    proportions: dict[str, float]
    bed_rate: float | None
    entropy: float
    thickness_ft: float
    source: str


@dataclass(frozen=True)
class Location:
    """Decimal coordinates and whether an unsigned hemisphere was assumed positive."""
    lat: float
    lon: float
    assumed_hemisphere: bool


@dataclass(frozen=True)
class Confidence:
    """Heuristic plausibility score; None means insufficient evidence."""
    score: float | None
    presence: float | None
    similarity: float | None
    reasons: tuple[str, ...]


def coarse_class(name: str) -> str:
    """Map a printed or Macrostrat lithology name into the shared vocabulary."""
    lowered = name.lower()
    return next((key for key, words in COARSE_CLASSES.items()
                 if any(word in lowered for word in words)), "other")


def finite_number(value: object) -> float | None:
    """Parse a finite number without accepting booleans or missing values."""
    if isinstance(value, bool):
        return None
    try:
        number = float(str(value))
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def make_fingerprint(weights: Mapping[str, float], source: str,
                     bed_rate: float | None = None) -> Fingerprint:
    """Normalise weights, using zero thickness and all-other for an empty result."""
    total = sum(weights.values())
    proportions = {key: weights.get(key, 0.0) / total if total else float(key == "other")
                   for key in CLASS_NAMES}
    entropy = -sum(p * math.log2(p) for p in proportions.values() if p > 0)
    return Fingerprint(proportions, bed_rate, entropy, total, source)


def bed_weights(row: Mapping[str, Any]) -> tuple[dict[str, float], float, float | None, float | None]:
    """Split a valid bed equally among names and account for unknown thickness."""
    top, base = finite_number(row.get("top_depth")), finite_number(row.get("base_depth"))
    if top is None or base is None or base <= top:
        return {}, 0.0, None, None
    names = str(row.get("lithology", "")).split("|")
    share = (base - top) / len(names)
    weights: dict[str, float] = {}
    skipped = 0.0
    for name in names:
        if name.strip().lower() in UNKNOWN_NAMES:
            skipped += share
        else:
            key = coarse_class(name)
            weights[key] = weights.get(key, 0.0) + share
    return weights, skipped, top, base


def fingerprint_from_rows(rows: list[dict[str, Any]]) -> Fingerprint:
    """Aggregate usable beds and calculate bed rate over their full depth span."""
    weights: dict[str, float] = {}
    skipped, count, invalid = 0.0, 0, 0
    tops: list[float] = []
    bases: list[float] = []
    for row in rows:
        bed, omitted, top, base = bed_weights(row)
        skipped += omitted
        invalid += int(top is None)
        if bed and top is not None and base is not None:
            count += 1
            tops.append(top)
            bases.append(base)
            for key, value in bed.items():
                weights[key] = weights.get(key, 0.0) + value
    logging.info("Skipped %.3f ft of unknown lithology; %d invalid/non-positive beds (thickness uncounted)", skipped, invalid)
    span = max(bases) - min(tops) if tops else 0.0
    return make_fingerprint(weights, "predicted", count * 1000 / span if span else None)


def fingerprint_from_csv(csv_path: str | Path) -> Fingerprint:
    """Read required pipeline CSV; missing or invalid data yields an empty fingerprint."""
    try:
        with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not REQUIRED_COLUMNS.issubset(reader.fieldnames or ()):
                logging.warning("CSV is missing required columns: %s", sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ())))
                return make_fingerprint({}, "predicted")
            return fingerprint_from_rows(list(reader))
    except (OSError, UnicodeError, csv.Error) as error:
        logging.warning("Cannot read predicted CSV: %s", error)
        return make_fingerprint({}, "predicted")


def unit_weights(unit: Mapping[str, Any]) -> dict[str, float]:
    """Use observed lith.name/prop/atts and max_thick or min_thick, in metres."""
    thickness = next((v for key in ("max_thick", "min_thick")
                      if (v := finite_number(unit.get(key))) is not None and v > 0), 0.0)
    liths = unit.get("lith")
    if not isinstance(liths, list) or not thickness:
        return {}
    entries = [(item, lithology_weight(item)) for item in liths
               if isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"].strip()]
    total = sum(weight for _, weight in entries)
    weights: dict[str, float] = {}
    for item, weight in entries:
        key = coarse_class(item["name"])
        weights[key] = weights.get(key, 0.0) + (thickness * FEET_PER_METRE * weight / total if total else 0.0)
    return weights


def lithology_weight(item: Mapping[str, Any]) -> float:
    """Prefer API proportions; recognise dominance only in the observed atts field."""
    prop = finite_number(item.get("prop"))
    if prop is not None and prop >= 0:
        return prop
    attributes = item.get("atts", [])
    if isinstance(attributes, list):
        names = [value.lower() for value in attributes if isinstance(value, str)]
        if "dominant" in names:
            return 2.0
        if "subordinate" in names:
            return 0.5
    return 1.0


def fingerprint_from_response(payload: object) -> Fingerprint | None:
    """Parse only the verified success.data[].lith[] unit response structure."""
    success = payload.get("success") if isinstance(payload, dict) else None
    units = success.get("data") if isinstance(success, dict) else None
    if not isinstance(units, list):
        logging.warning("Macrostrat response lacks success.data list")
        return None
    weights: dict[str, float] = {}
    for unit in units:
        if isinstance(unit, dict):
            for key, value in unit_weights(unit).items():
                weights[key] = weights.get(key, 0.0) + value
    if not sum(weights.values()):
        logging.warning("Macrostrat has no usable lithologies with positive unit thickness")
        return None
    return make_fingerprint(weights, "macrostrat")


def valid_coordinates(lat: float, lon: float) -> bool:
    """Check finite latitude and longitude bounds."""
    return math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180


def fingerprint_from_location(lat: float, lon: float) -> Fingerprint | None:
    """Fetch regional units with response=long; unavailable coverage returns None."""
    if not valid_coordinates(lat, lon):
        logging.warning("Coordinates outside valid geographic bounds")
        return None
    try:
        import requests
        response = requests.get(API_URL, params={"lat": lat, "lng": lon, "response": "long"}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return fingerprint_from_response(response.json())
    except (ImportError, OSError, ValueError) as error:
        logging.warning("Macrostrat request unavailable: %s", error)
        return None
    except Exception as error:
        logging.warning("Macrostrat request failed (%s): %s", type(error).__name__, error)
        return None


def coordinate_tokens(text: str) -> list[str]:
    """Tokenise decimal or DMS coordinates using string operations only."""
    normal = text.lower().replace("degrees", " ").replace("degree", " ").replace("deg", " ")
    for character in ('°', "'", '"', "′", "″", ":", "=", ","):
        normal = normal.replace(character, " ")
    for direction in ("n", "s", "e", "w"):
        normal = normal.replace(direction, f" {direction} ")
    return normal.split()


def parse_coordinate(text: str, limit: float) -> tuple[float, bool] | None:
    """Parse one DMS/decimal coordinate, flagging unsigned coordinates as assumed."""
    tokens = coordinate_tokens(text)
    values = [value for token in tokens if (value := finite_number(token)) is not None]
    directions = [token for token in tokens if token in ("n", "s", "e", "w")]
    if not 1 <= len(values) <= 3 or len(set(directions)) > 1:
        return None
    if directions and directions[0] not in (("n", "s") if limit == 90 else ("e", "w")):
        return None
    if any(value < 0 or value >= 60 for value in values[1:]):
        return None
    magnitude = abs(values[0]) + sum(value / 60 ** index for index, value in enumerate(values[1:], 1))
    negative = values[0] < 0 or any(token in ("s", "w") for token in directions)
    if values[0] < 0 and any(token in ("n", "e") for token in directions):
        return None
    signed = any(token.startswith(("+", "-")) for token in tokens if finite_number(token) is not None)
    return ((-magnitude if negative else magnitude), not directions and not signed) if magnitude <= limit else None


def coordinate_prefix(text: str) -> str:
    """Keep one coordinate value while excluding adjacent OCR header fields."""
    normal = text.lower().replace("degrees", " ").replace("degree", " ").replace("deg", " ")
    normal = normal.strip().lstrip(":=").strip()
    for index, character in enumerate(normal):
        if character in "nsew":
            return normal[:index + 1]
        if character not in "0123456789+-. °'\"′″\t\n\r":
            return normal[:index]
    return normal


def location_from_text(text: str) -> Location | None:
    """Extract labelled coordinates; missing hemispheres assume positive N/E."""
    lowered = text.lower()
    _, label, remainder = lowered.partition("latitude")
    if not label:
        return None
    latitude, label, longitude = remainder.partition("longitude")
    if not label:
        return None
    lat = parse_coordinate(coordinate_prefix(latitude), 90)
    lon = parse_coordinate(coordinate_prefix(longitude), 180)
    if lat is None or lon is None:
        return None
    return Location(lat[0], lon[0], lat[1] or lon[1])


def header_bottom(annotation_path: str | Path) -> int | None:
    """Use the top of the annotated columns/legend as the full-width header limit."""
    try:
        payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
        boxes = payload.get("bounding_boxes", {}) if isinstance(payload, dict) else {}
        tops = [box[1] for key in ("depth_column", "interpreted_lithology_column", "lithology_legend")
                if isinstance((box := boxes.get(key)), list) and len(box) == 4
                and all(finite_number(value) is not None for value in box)] if isinstance(boxes, dict) else []
        return int(min(tops)) if tops and min(tops) > 0 else None
    except (OSError, ValueError, TypeError) as error:
        logging.warning("Cannot read header annotations: %s", error)
        return None


def ocr_strings(result: object) -> list[str]:
    """Read PaddleOCR 2 line records or PaddleOCR 3 rec_texts results."""
    if isinstance(result, dict):
        texts = result.get("rec_texts", [])
        return [value for value in texts if isinstance(value, str)] if isinstance(texts, list) else []
    if not isinstance(result, (list, tuple)):
        return []
    if len(result) == 2 and isinstance(result[0], str) and finite_number(result[1]) is not None:
        return [result[0]]
    return [text for item in result for text in ocr_strings(item)]


def ocr_image_tiles(image: Any, engine: Any, bottom: int) -> str | None:
    """Read overlapping RGB tiles with PaddleOCR 2 or 3."""
    import numpy as np
    texts: list[str] = []
    for top in range(0, min(bottom, image.height), OCR_TILE_HEIGHT - 200):
        tile = image.crop((0, top, image.width, min(top + OCR_TILE_HEIGHT, bottom, image.height))).convert("RGB")
        if tile.width > OCR_WIDTH:
            tile.thumbnail((OCR_WIDTH, OCR_TILE_HEIGHT))
        result = list(engine.predict(np.asarray(tile))) if hasattr(engine, "predict") else engine.ocr(np.asarray(tile))
        texts.extend(ocr_strings(result))
    return "\n".join(texts) or None


def ocr_header(image_path: str | Path, bottom: int) -> str | None:
    """OCR the header lazily, temporarily allowing trusted huge scanned logs."""
    try:
        from PIL import Image
        from paddleocr import PaddleOCR
        engine = PaddleOCR(lang="en")
        previous_limit = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(Path(image_path)) as image:
                return ocr_image_tiles(image, engine, bottom)
        finally:
            Image.MAX_IMAGE_PIXELS = previous_limit
    except Exception as error:
        logging.warning("Header OCR unavailable (%s): %s", type(error).__name__, error)
        return None


def compare_fingerprints(predicted: Fingerprint, regional: Fingerprint | None,
                         location: Location | None) -> Confidence:
    """Average supported-class mass and 1 minus total variation; apply assumption penalty."""
    if predicted.thickness_ft <= 0:
        return Confidence(None, None, None, ("No usable predicted thickness; check CSV and logs.",))
    if location is None:
        return Confidence(None, None, None, ("Location missing or unreadable; supply explicit latitude/longitude.",))
    if regional is None or regional.thickness_ft <= 0:
        return Confidence(None, None, None, ("Macrostrat unavailable, uncovered, or lacking usable thickness/lithology.",))
    presence = sum(p for key, p in predicted.proportions.items() if regional.proportions.get(key, 0) > 0)
    similarity = 1 - sum(abs(predicted.proportions.get(key, 0) - regional.proportions.get(key, 0)) for key in CLASS_NAMES) / 2
    score = (presence + similarity) / 2
    reasons = [f"Supported predicted thickness: {presence:.1%}; distribution similarity: {similarity:.1%}.",
               "Heuristic regional plausibility, not a calibrated probability; regional units are not depth/age matched.",
               "Bed rate and entropy are descriptive; regional units do not resolve individual log beds."]
    if location.assumed_hemisphere:
        score *= ASSUMED_HEMISPHERE_FACTOR
        reasons.append("Unsigned hemisphere assumed N/E; score multiplied by 0.8. Verify coordinates.")
    if predicted.proportions.get("other", 0) or regional.proportions.get("other", 0):
        reasons.append("Unmapped names contribute to 'other', which may inflate agreement.")
    return Confidence(max(0.0, min(1.0, score)), presence, max(0.0, min(1.0, similarity)), tuple(reasons))


def resolve_location(args: argparse.Namespace) -> Location | None:
    """Prefer explicit coordinates, otherwise extract the annotated image header."""
    if args.lat is not None or args.lon is not None:
        if args.lat is not None and args.lon is not None and valid_coordinates(args.lat, args.lon):
            return Location(args.lat, args.lon, False)
        logging.warning("Supply both --lat and --lon with valid bounds")
        return None
    if args.image is None or args.annotation is None:
        return None
    bottom = header_bottom(args.annotation)
    text = ocr_header(args.image, bottom) if bottom else None
    return location_from_text(text) if text else None


def parse_arguments() -> argparse.Namespace:
    """Parse file paths and optional explicit coordinates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def save_report(path: Path, encoded: str) -> None:
    """Save optional JSON output, reporting filesystem errors without crashing."""
    try:
        path.write_text(encoded + "\n", encoding="utf-8")
    except OSError as error:
        logging.warning("Cannot save report: %s", error)


def main() -> None:
    """Run validation and log a JSON report, optionally saving it to a file."""
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    predicted, location = fingerprint_from_csv(args.csv), resolve_location(args)
    regional = fingerprint_from_location(location.lat, location.lon) if location else None
    report = {"confidence": asdict(compare_fingerprints(predicted, regional, location)),
              "location": asdict(location) if location else None,
              "predicted": asdict(predicted), "macrostrat": asdict(regional) if regional else None}
    encoded = json.dumps(report, indent=2, allow_nan=False)
    logging.info("%s", encoded)
    if args.output:
        save_report(args.output, encoded)


if __name__ == "__main__":
    main()
