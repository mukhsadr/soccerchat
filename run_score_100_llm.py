#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(root / "viz" / "soccer" / "score_preds_jsonl.py"),
        "--input",
        str(root / "SoccerChat_valid_xfoul_abs_preds_100.jsonl"),
        "--output",
        str(root / "SoccerChat_valid_xfoul_abs_preds_100_scored_v2.jsonl"),
        "--group",
        "valid_preds_100",
        "--ollama_judge",
    ]
    print("Running:\n", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
