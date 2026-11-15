#!/usr/bin/env python3
"""Assemble the 213-item evaluation corpus.

Order and sources:
    malicious   malicious_final.csv    (tier = attack family)
    benign      benign_generated.csv   (meta = generated_benign, the hard negatives)
    benign      benign_sql.csv         (meta = human_style or llm_style)
    pi          pi_payloads.csv        (meta = injection category, true label malicious)

The post-hoc position reads the corpus in this order in batches of ten.

Usage:
    python scripts/build_corpus.py [--data-dir data] [--out outputs/corpus.csv]
"""
import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def build(malicious, generated_benign, benign_controls, payloads):
    """Return the corpus with columns item_id, text, true_label, group, tier, meta."""
    mal_rows = pd.DataFrame({
        "item_id": malicious["id"], "text": malicious["sql"], "true_label": "malicious",
        "group": "malicious", "tier": malicious["level"], "meta": "",
    })
    gen_rows = pd.DataFrame({
        "item_id": generated_benign["id"], "text": generated_benign["sql"], "true_label": "benign",
        "group": "benign", "tier": "", "meta": "generated_benign",
    })
    ben_rows = pd.DataFrame({
        "item_id": benign_controls["id"], "text": benign_controls["text"], "true_label": "benign",
        "group": "benign", "tier": "", "meta": benign_controls["style"],
    })
    pi_rows = pd.DataFrame({
        "item_id": payloads["id"], "text": payloads["text"], "true_label": "malicious",
        "group": "pi", "tier": "", "meta": payloads["injection_category"],
    })
    corpus = pd.concat([mal_rows, gen_rows, ben_rows, pi_rows], ignore_index=True)
    corpus["text"] = corpus["text"].astype(str)
    return corpus


def main():
    parser = argparse.ArgumentParser(description="Assemble the evaluation corpus from the labeled sets.")
    parser.add_argument("--data-dir", type=Path, default=REPO / "data")
    parser.add_argument("--out", type=Path, default=REPO / "outputs" / "corpus.csv")
    args = parser.parse_args()

    d = args.data_dir
    corpus = build(pd.read_csv(d / "malicious_final.csv"), pd.read_csv(d / "benign_generated.csv"),
                   pd.read_csv(d / "benign_sql.csv"), pd.read_csv(d / "pi_payloads.csv"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(args.out, index=False, encoding="utf-8")
    print(f"corpus: {len(corpus)} items")
    print(corpus.groupby(["group", "true_label"]).size().to_string())
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
