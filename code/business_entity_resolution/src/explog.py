"""Experiment logging: append one row per experiment to experiments.csv."""
from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path

FIELDS = ["timestamp", "git_hash", "description", "blocking_recall", "cands_per_s1", "oof_f05",
          "singleton_acc", "nonsingleton_f05", "cross_country_f05"]


def git_hash() -> str:
    """Short hash of HEAD (or 'nogit')."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "nogit"


def log_experiment(path, description: str, **metrics) -> None:
    """Append a row with the given metrics (missing ones left blank)."""
    path = Path(path)
    new = not path.exists()
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "git_hash": git_hash(), "description": description}
    for k in FIELDS[3:]:
        v = metrics.get(k, "")
        row[k] = f"{v:.5f}" if isinstance(v, float) else v
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
