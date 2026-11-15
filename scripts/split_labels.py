#!/usr/bin/env python3
"""Split the labeled candidates into the files that build_corpus.py reads.

Reads the review file, checks that every expert_label is malicious, benign, or
exclude, and writes:
    malicious_final.csv     candidates labeled malicious (the malicious group)
    benign_generated.csv    candidates labeled benign (hard negatives in the benign group)
    malicious_excluded.csv  candidates excluded from the evaluation

The family column is renamed to level in these files.

Usage:
    python scripts/split_labels.py [--review data/malicious_v2_review.csv] [--out-dir outputs]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ALLOWED = {"malicious", "benign", "exclude"}


def split(review):
    """Return (malicious, benign, excluded) data frames from a labeled review table."""
    rev = review.copy()
    rev["expert_label"] = rev["expert_label"].str.strip().str.lower()
    bad = rev[~rev["expert_label"].isin(ALLOWED)]
    if len(bad):
        print("expert_label must be malicious, benign, or exclude; invalid rows:", file=sys.stderr)
        print(bad[["id", "expert_label"]].to_string(index=False), file=sys.stderr)
        raise SystemExit(1)
    rev = rev.rename(columns={"family": "level", "expert_label": "label"})
    keep = ["id", "level", "sql", "label", "expert_note"]
    malicious = rev[rev["label"] == "malicious"][keep]
    benign = rev[rev["label"] == "benign"][keep]
    excluded = rev[rev["label"] == "exclude"][["id", "level", "sql", "expert_note"]]
    return malicious, benign, excluded


def main():
    parser = argparse.ArgumentParser(description="Split labeled candidates into malicious, benign, and excluded files.")
    parser.add_argument("--review", type=Path, default=REPO / "data" / "malicious_v2_review.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO / "outputs")
    args = parser.parse_args()

    review = pd.read_csv(args.review, dtype=str).fillna("")
    malicious, benign, excluded = split(review)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    malicious.to_csv(args.out_dir / "malicious_final.csv", index=False, encoding="utf-8")
    benign.to_csv(args.out_dir / "benign_generated.csv", index=False, encoding="utf-8")
    excluded.to_csv(args.out_dir / "malicious_excluded.csv", index=False, encoding="utf-8")
    print(f"malicious: {len(malicious)} | benign: {len(benign)} | excluded: {len(excluded)}")
    print(f"written to {args.out_dir}")


if __name__ == "__main__":
    main()
