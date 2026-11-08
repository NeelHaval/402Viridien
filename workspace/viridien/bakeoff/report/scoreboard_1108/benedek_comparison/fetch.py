"""Get a separate source checkout. Never alter the owned live model checkout."""
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
BAKEOFF = HERE.parents[2]
TARGET = BAKEOFF / "out/benedek_source_20261108T1001"
URL = "https://github.com/NeelHaval/viridian402.git"

def main():
    start = time.perf_counter()
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
    command = ["git", "clone", "--filter=blob:none", "--no-checkout", "--no-tags", URL, str(TARGET)]
    with (HERE / "fetch.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
    receipt = {"repository": URL, "checkout": str(TARGET), "command": command,
               "exit_code": result.returncode, "wall_seconds": time.perf_counter()-start,
               "live_model_modified": False, "dataset_opened": False}
    (HERE / "fetch_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    if result.returncode:
        raise SystemExit(result.returncode)
    log = subprocess.run(["git", "-C", str(TARGET), "log", "--all", "-30", "--format=%H %an %s"],
                         capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
    (HERE / "commits.txt").write_text(log.stdout, encoding="utf-8")

if __name__ == "__main__":
    main()
