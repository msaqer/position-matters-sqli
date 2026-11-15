#!/usr/bin/env python3
"""Rebuild the paper's tables and Fig. 2 from the run log.

Writes, to --out-dir:
    table1_detection.csv        Table I: detection rate and false positive rate with
                                Wilson 95% confidence intervals, and F1
    table2_mcnemar.csv          Table II: exact two-sided McNemar tests between positions
    table3_latency_cost.csv     Table III: latency per call and per query, tokens, and cost
    fig2_family_detection.csv   Fig. 2: detection rate by attack family
    payload_detection.csv       Section VI-C: prompt-injection payloads flagged

An item's label at a position is the majority verdict over its repeats. The script
uses only the Python standard library; scripts/verify_paper_numbers.py compares
these quantities with the values printed in the paper.

Usage:
    python scripts/analyze.py [--log results/logs/runs_v2.jsonl] [--out-dir outputs/tables]
"""
import argparse
import collections
import csv
import json
import math
from decimal import ROUND_HALF_UP, Decimal
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

POSITIONS = ["review_time", "runtime_gatekeeper", "post_hoc"]
DETECTORS = ["frontier", "small"]
FAMILIES = ["auth_bypass", "blind_boolean", "encoding_obf", "error_based",
            "file_os", "stacked", "time_based", "union_exfil"]
# USD per 1,000 tokens: the launch list price of the frontier model
# ($2.50 / $15 per million input / output tokens), as in config.yaml.
PRICE_IN_PER_1K, PRICE_OUT_PER_1K = 0.0025, 0.015
Z95 = 1.959963984540054


def wilson(k, n, z=Z95):
    """Wilson score interval for k successes in n trials."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def mcnemar_exact(b, c):
    """Exact two-sided McNemar test on the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def format_p(p, digits):
    """Format a p-value as printed in the paper (round half up; '<0.0001' below that)."""
    if p < 0.0001:
        return "<0.0001"
    return str(Decimal(repr(p)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def load_runs(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def majority_flags(runs):
    """(detector, position, item_id) -> True if at least half of the repeats said malicious."""
    votes = collections.defaultdict(list)
    for r in runs:
        votes[(r["detector"], r["position"], r["item_id"])].append(r["parsed_label"] == "malicious")
    return {key: 2 * sum(v) >= len(v) for key, v in votes.items()}


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Rebuild Tables I to III and Fig. 2 from the run log.")
    parser.add_argument("--log", type=Path, default=REPO / "results" / "logs" / "runs_v2.jsonl")
    parser.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "tables")
    args = parser.parse_args()

    runs = load_runs(args.log)
    flagged = majority_flags(runs)
    group = {r["item_id"]: r["group"] for r in runs}
    family = {r["item_id"]: r["tier"] for r in runs}
    items = {g: sorted(i for i in group if group[i] == g) for g in ("malicious", "benign", "pi")}
    mal, ben = items["malicious"], items["benign"]

    def n_flagged(det, pos, ids):
        return sum(flagged[(det, pos, i)] for i in ids)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Table I
    rows = []
    for det in DETECTORS:
        for pos in POSITIONS:
            tp, fp = n_flagged(det, pos, mal), n_flagged(det, pos, ben)
            d_lo, d_hi = wilson(tp, len(mal))
            f_lo, f_hi = wilson(fp, len(ben))
            f1 = 2 * tp / (2 * tp + fp + (len(mal) - tp))
            rows.append([det, pos, tp, len(mal), round(100 * tp / len(mal), 1), round(100 * d_lo), round(100 * d_hi),
                         fp, len(ben), round(100 * fp / len(ben), 1), round(100 * f_lo), round(100 * f_hi),
                         round(f1, 2)])
    write_csv(args.out_dir / "table1_detection.csv",
              ["detector", "position", "detected", "n_malicious", "detection_pct", "detection_ci_low_pct",
               "detection_ci_high_pct", "false_positives", "n_benign", "fpr_pct", "fpr_ci_low_pct",
               "fpr_ci_high_pct", "f1"], rows)
    table1 = rows

    # Table II
    rows = []
    for det in DETECTORS:
        for first, second in combinations(POSITIONS, 2):
            for grp, ids, digits in (("malicious", mal, 2), ("benign", ben, 3)):
                only_first = sum(flagged[(det, first, i)] and not flagged[(det, second, i)] for i in ids)
                only_second = sum(flagged[(det, second, i)] and not flagged[(det, first, i)] for i in ids)
                p = mcnemar_exact(only_first, only_second)
                rows.append([det, first, second, grp, len(ids), only_first, only_second,
                             f"{p:.6g}", format_p(p, digits)])
    write_csv(args.out_dir / "table2_mcnemar.csv",
              ["detector", "first", "second", "group", "n_items", "only_first", "only_second",
               "p_value", "p_as_printed"], rows)
    table2 = rows

    # Table III
    rows = []
    for det in DETECTORS:
        for pos in POSITIONS:
            recs = [r for r in runs if r["detector"] == det and r["position"] == pos]
            calls = [r for r in recs if r["latency_ms"] is not None]
            latency = sum(r["latency_ms"] for r in calls)
            tokens_in = sum(r["tokens_in"] or 0 for r in recs)
            tokens_out = sum(r["tokens_out"] or 0 for r in recs)
            cost = (tokens_in * PRICE_IN_PER_1K + tokens_out * PRICE_OUT_PER_1K) / 1000 if det == "frontier" else 0.0
            rows.append([det, pos, len(calls), len(recs), round(latency / len(calls)), round(latency / len(recs)),
                         tokens_in, tokens_out, f"{cost:.2f}", f"{1000 * cost / len(recs):.2f}"])
    write_csv(args.out_dir / "table3_latency_cost.csv",
              ["detector", "position", "calls", "judged_queries", "latency_ms_per_call", "latency_ms_per_query",
               "tokens_in", "tokens_out", "cost_usd", "cost_usd_per_1k_queries"], rows)
    table3 = rows

    # Fig. 2
    rows = []
    for fam in FAMILIES:
        ids = [i for i in mal if family[i] == fam]
        rows.append([fam, len(ids)] + [round(100 * n_flagged(d, p, ids) / len(ids)) for d in DETECTORS for p in POSITIONS])
    write_csv(args.out_dir / "fig2_family_detection.csv",
              ["family", "n"] + [f"{d}_{p}" for d in DETECTORS for p in POSITIONS], rows)
    fig2 = rows

    # Prompt-injection payloads
    rows = [[d, p, n_flagged(d, p, items["pi"]), len(items["pi"])] for d in DETECTORS for p in POSITIONS]
    write_csv(args.out_dir / "payload_detection.csv", ["detector", "position", "flagged", "n_payloads"], rows)
    payloads = rows

    print("Table I  detection % [95% CI] | FPR % [95% CI] | F1")
    for r in table1:
        print(f"  {r[0]:8s} {r[1]:18s} {r[4]:5.1f} [{r[5]}, {r[6]}] | {r[9]:4.1f} [{r[10]}, {r[11]}] | {r[12]:.2f}")
    print("Table II  only first / only second / p (malicious, then benign)")
    for r in table2:
        comparison = f"{r[1]} vs {r[2]}"
        print(f"  {r[0]:8s} {comparison:40s} {r[3]:9s} {r[5]:3d} {r[6]:3d}  p = {r[8]}")
    print("Table III  ms per call / ms per query / tokens in / tokens out / cost $ / $ per 1k")
    for r in table3:
        print(f"  {r[0]:8s} {r[1]:18s} {r[4]:5d} {r[5]:5d} {r[6]:7d} {r[7]:7d} {r[8]:>5s} {r[9]:>5s}")
    print("Fig. 2  detection % by family (frontier rev/run/post, small rev/run/post)")
    for r in fig2:
        print(f"  {r[0]:14s} ({r[1]:2d}) " + " ".join(f"{v:3d}" for v in r[2:]))
    print("Payloads flagged (of 12): " + ", ".join(f"{d}/{p} {k}" for d, p, k, _ in payloads))
    print(f"\nwritten to {args.out_dir}")


if __name__ == "__main__":
    main()
