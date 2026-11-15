#!/usr/bin/env python3
"""Check every number reported in the camera-ready paper against the run log.

Paper: "Position Matters: Evaluating Large Language Models as SQL Injection
Detectors Across Deployment Positions" (IEEE BigData 2026, BigD1194).

The script recomputes each reported quantity from results/logs/runs_v2.jsonl
(and, for the dataset counts, from the corpus and label files) and compares it
with the value printed in the paper. It uses only the Python standard library.
No GPU and no API key are needed.

Usage (from the repository root; the defaults point at the released files):
    python scripts/verify_paper_numbers.py
    python scripts/verify_paper_numbers.py --log results/logs/runs_v2.jsonl \
        --corpus data/corpus.csv --review data/malicious_v2_review.csv

Exit status 0 means every check passed.
"""
import argparse
import collections
import csv
import json
import math
import sys
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

POSITIONS = ["review_time", "runtime_gatekeeper", "post_hoc"]
DETECTORS = ["frontier", "small"]
# USD per 1,000 tokens: the launch list price of the frontier model
# ($2.50 / $15 per million input / output tokens), as fixed in config.yaml.
PRICE_IN_PER_1K, PRICE_OUT_PER_1K = 0.0025, 0.015

CHECKS = []          # (passed, description, reported, computed)


def check(description, reported, computed, tol=0.0):
    """Record one comparison between a reported value and a recomputed one."""
    if isinstance(reported, (int, float)) and isinstance(computed, (int, float)):
        passed = abs(reported - computed) <= tol + 1e-12
    else:
        passed = reported == computed
    CHECKS.append((passed, description, reported, computed))


def wilson(k, n, z=1.959963984540054):
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
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pct(k, n, digits=1):
    return round(100 * k / n, digits)


def fmt_p(p, digits):
    """Format a p-value as the paper prints it (round half up, e.g. 0.0625 -> 0.063)."""
    if p < 0.0001:
        return "<0.0001"
    return str(Decimal(repr(p)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(REPO / "results" / "logs" / "runs_v2.jsonl"))
    ap.add_argument("--corpus", default=str(REPO / "data" / "corpus.csv"))
    ap.add_argument("--review", default=str(REPO / "data" / "malicious_v2_review.csv"))
    args = ap.parse_args()

    with open(args.log, encoding="utf-8") as fh:
        runs = [json.loads(line) for line in fh if line.strip()]
    with open(args.corpus, encoding="utf-8-sig", newline="") as fh:
        corpus = list(csv.DictReader(fh))
    with open(args.review, encoding="utf-8-sig", newline="") as fh:
        review = list(csv.DictReader(fh))

    # ------------------------------------------------------------------ log shape
    check("logged verdicts", 3834, len(runs))
    check("2 detectors x 213 items x 3 positions x 3 repeats", 3834, 2 * 213 * 3 * 3)
    check("unparseable verdicts", 0, sum(r["parsed_label"] not in ("malicious", "benign") for r in runs))
    group = {r["item_id"]: r["group"] for r in runs}
    family = {r["item_id"]: r["tier"] for r in runs}
    counts = collections.Counter(group.values())
    check("items in the evaluation set", 213, len(group))
    check("malicious queries", 135, counts["malicious"])
    check("benign controls", 66, counts["benign"])
    check("prompt-injection payloads", 12, counts["pi"])

    votes = collections.defaultdict(dict)
    for r in runs:
        votes[(r["detector"], r["position"], r["item_id"])][r["repeat"]] = r["parsed_label"] == "malicious"
    check("every (detector, position, item) has three repeats", True,
          all(sorted(v) == [0, 1, 2] for v in votes.values()) and len(votes) == 2 * 3 * 213)
    flagged = {k: sum(v.values()) >= 2 for k, v in votes.items()}      # majority verdict

    MAL = sorted(i for i in group if group[i] == "malicious")
    BEN = sorted(i for i in group if group[i] == "benign")
    PI = sorted(i for i in group if group[i] == "pi")

    def n_flagged(det, pos, items):
        return sum(flagged[(det, pos, i)] for i in items)

    # ------------------------------------------------------------------ Table I
    table1 = {  # (detector, position): (detection %, CI, FPR %, CI, F1)
        ("frontier", "review_time"): (81.5, (74, 87), 6.1, (2, 15), 0.88),
        ("frontier", "runtime_gatekeeper"): (84.4, (77, 90), 12.1, (6, 22), 0.89),
        ("frontier", "post_hoc"): (80.7, (73, 87), 22.7, (14, 34), 0.84),
        ("small", "review_time"): (51.9, (43, 60), 9.1, (4, 18), 0.66),
        ("small", "runtime_gatekeeper"): (78.5, (71, 85), 22.7, (14, 34), 0.83),
        ("small", "post_hoc"): (51.9, (43, 60), 16.7, (10, 27), 0.65),
    }
    tp, fp = {}, {}
    for key, (det_rate, det_ci, fpr, fpr_ci, f1) in table1.items():
        d, p = key
        tp[key], fp[key] = n_flagged(d, p, MAL), n_flagged(d, p, BEN)
        lo, hi = wilson(tp[key], 135)
        check(f"Table I detection % {d} {p}", det_rate, pct(tp[key], 135))
        check(f"Table I detection CI {d} {p}", det_ci, (round(100 * lo), round(100 * hi)))
        lo, hi = wilson(fp[key], 66)
        check(f"Table I FPR % {d} {p}", fpr, pct(fp[key], 66))
        check(f"Table I FPR CI {d} {p}", fpr_ci, (round(100 * lo), round(100 * hi)))
        fn = 135 - tp[key]
        check(f"Table I F1 {d} {p}", f1, round(2 * tp[key] / (2 * tp[key] + fp[key] + fn), 2))

    # ------------------------------------------------------------------ Table II
    table2 = {  # (detector, first, second): (mal 1st, mal 2nd, mal p, ben 1st, ben 2nd, ben p)
        ("frontier", "review_time", "runtime_gatekeeper"): (4, 8, "0.39", 0, 4, "0.125"),
        ("frontier", "review_time", "post_hoc"): (14, 13, "1.00", 0, 11, "0.001"),
        ("frontier", "runtime_gatekeeper", "post_hoc"): (13, 8, "0.38", 0, 7, "0.016"),
        ("small", "review_time", "runtime_gatekeeper"): (0, 36, "<0.0001", 1, 10, "0.012"),
        ("small", "review_time", "post_hoc"): (20, 20, "1.00", 0, 5, "0.063"),
        ("small", "runtime_gatekeeper", "post_hoc"): (43, 7, "<0.0001", 5, 1, "0.219"),
    }
    pvals = {}
    for (d, a, b), (m1, m2, mp, b1, b2, bp) in table2.items():
        for side, items, r1, r2, rp, nd in (("malicious", MAL, m1, m2, mp, 2), ("benign", BEN, b1, b2, bp, 3)):
            first = sum(flagged[(d, a, i)] and not flagged[(d, b, i)] for i in items)
            second = sum(flagged[(d, b, i)] and not flagged[(d, a, i)] for i in items)
            p = mcnemar_exact(first, second)
            pvals[(d, a, b, side)] = p
            shown = fmt_p(p, nd)
            check(f"Table II {side} discordant counts {d} {a} vs {b}", (r1, r2), (first, second))
            check(f"Table II {side} p {d} {a} vs {b}", rp, shown)
    check("frontier detection: all p > 0.38", True,
          all(p > 0.38 for (d, a, b, s), p in pvals.items() if d == "frontier" and s == "malicious"))
    check("frontier benign flags nested across positions (no first-only items)", True,
          all(table2[k][3] == 0 for k in table2 if k[0] == "frontier"))
    significant = [p for p in pvals.values() if p < 0.05]
    check("every significant comparison below the Bonferroni threshold 0.05/3", True,
          all(p < 0.05 / 3 for p in significant))
    check("Bonferroni threshold", 0.0167, round(0.05 / 3, 4))
    check("number of significant comparisons (2 frontier benign, 2 small malicious, 1 small benign)", 5, len(significant))

    # ------------------------------------------------------------------ benign subgroups
    meta = {row["item_id"]: row["meta"] for row in corpus}
    hand = [i for i in BEN if meta[i] in ("human_style", "llm_style")]
    hard = [i for i in BEN if i not in hand]
    check("hand-authored benign controls", 40, len(hand))
    check("hand-authored controls per style", (20, 20),
          (sum(meta[i] == "human_style" for i in hand), sum(meta[i] == "llm_style" for i in hand)))
    check("generated hard negatives", 26, len(hard))
    check("false positives on hand-authored controls (any detector, any position)", 0,
          sum(n_flagged(d, p, hand) for d in DETECTORS for p in POSITIONS))
    check("Wilson upper bound for 0 of 40 (%)", 8.8, round(100 * wilson(0, 40)[1], 1))
    check("frontier flags on 26 hard negatives (review, runtime, post-hoc)", (4, 8, 15),
          tuple(n_flagged("frontier", p, hard) for p in POSITIONS))
    check("small flags on 26 hard negatives (review, runtime, post-hoc)", (6, 15, 11),
          tuple(n_flagged("small", p, hard) for p in POSITIONS))

    # ------------------------------------------------------------------ attack families
    fam_sizes = collections.Counter(family[i] for i in MAL)
    check("attack families", 8, len(fam_sizes))
    check("family sizes", {"auth_bypass": 18, "blind_boolean": 14, "encoding_obf": 9, "error_based": 11,
                           "file_os": 16, "stacked": 23, "time_based": 23, "union_exfil": 21}, dict(fam_sizes))
    check("smallest and largest family", (9, 23), (min(fam_sizes.values()), max(fam_sizes.values())))

    def fam_rate(d, p, f):
        items = [i for i in MAL if family[i] == f]
        return round(100 * n_flagged(d, p, items) / len(items))

    heatmap = {  # Fig. 2 as printed: frontier rev/run/post, then small rev/run/post
        "auth_bypass": (61, 78, 61, 50, 94, 56),
        "blind_boolean": (14, 50, 79, 0, 14, 36),
        "encoding_obf": (100, 100, 100, 78, 100, 89),
        "error_based": (100, 100, 100, 64, 91, 64),
        "file_os": (100, 100, 100, 75, 100, 75),
        "stacked": (100, 100, 100, 48, 91, 39),
        "time_based": (100, 100, 100, 100, 100, 52),
        "union_exfil": (71, 52, 24, 5, 38, 33),
    }
    for f, row in heatmap.items():
        got = tuple(fam_rate(d, p, f) for d in DETECTORS for p in POSITIONS)
        check(f"Fig. 2 row {f}", row, got)
    check("frontier at 100% in five families at every position", 5,
          sum(all(fam_rate("frontier", p, f) == 100 for p in POSITIONS) for f in fam_sizes))
    check("frontier blind Boolean (rev, run, post)", (14, 50, 79),
          tuple(fam_rate("frontier", p, "blind_boolean") for p in POSITIONS))
    check("frontier UNION (rev, run, post)", (71, 52, 24),
          tuple(fam_rate("frontier", p, "union_exfil") for p in POSITIONS))
    ab = [fam_rate("frontier", p, "auth_bypass") for p in POSITIONS]
    check("frontier authentication bypass range", (61, 78), (min(ab), max(ab)))
    bb = [fam_rate("small", p, "blind_boolean") for p in POSITIONS]
    un = [fam_rate("small", p, "union_exfil") for p in POSITIONS]
    check("small blind Boolean range", (0, 36), (min(bb), max(bb)))
    check("small UNION range", (5, 38), (min(un), max(un)))

    # ------------------------------------------------------------------ payloads
    check("frontier payloads flagged (rev, run, post)", (11, 10, 10),
          tuple(n_flagged("frontier", p, PI) for p in POSITIONS))
    check("small payloads flagged (rev, run, post)", (6, 8, 6),
          tuple(n_flagged("small", p, PI) for p in POSITIONS))
    pi_meta = collections.Counter(meta[i] for i in PI)
    check("payload categories, two each", [2] * 6, sorted(pi_meta.values()))

    # ------------------------------------------------------------------ repeat consistency
    def consistency(d, p):
        return pct(sum(len(set(votes[(d, p, i)].values())) == 1 for i in group), 213)

    check("small: identical verdicts over repeats, all positions (%)", [100.0] * 3,
          [consistency("small", p) for p in POSITIONS])
    fc = [consistency("frontier", p) for p in POSITIONS]
    check("frontier: identical verdicts over repeats, min and max (%)", (90.6, 92.5), (min(fc), max(fc)))

    # ------------------------------------------------------------------ flag rates (Discussion)
    def flag_rate(d, p):
        return pct(n_flagged(d, p, list(group)), 213)

    check("small flag rate on all 213 items (rev, run, post)", (38.5, 60.6, 40.8),
          tuple(flag_rate("small", p) for p in POSITIONS))
    fr = [flag_rate("frontier", p) for p in POSITIONS]
    check("frontier flag rate range", (58.7, 62.9), (min(fr), max(fr)))
    j = tuple(round(100 * (tp[("small", p)] / 135 - fp[("small", p)] / 66), 1) for p in POSITIONS)
    check("small detection minus FPR (rev, run, post), points", (42.8, 55.8, 35.2), j)
    check("runtime gap, frontier minus small detection, under six points", True,
          0 < 100 * (tp[("frontier", "runtime_gatekeeper")] - tp[("small", "runtime_gatekeeper")]) / 135 < 6)

    # ------------------------------------------------------------------ post-hoc batches
    order = [row["item_id"] for row in corpus]
    for d in DETECTORS:
        for rep in (0, 1, 2):
            seq = [r["item_id"] for r in runs if r["detector"] == d and r["position"] == "post_hoc" and r["repeat"] == rep]
            check(f"post-hoc pass follows corpus order ({d}, repeat {rep})", True, seq == order)
    batches = collections.defaultdict(list)
    for r in runs:
        if r["detector"] == "frontier" and r["position"] == "post_hoc" and r["repeat"] == 0:
            batches[r["batch_id"]].append(r["item_id"])
    check("batches per pass", 22, len(batches))
    check("batch sizes: 21 of ten and 1 of three", {10: 21, 3: 1},
          dict(collections.Counter(len(v) for v in batches.values())))
    kinds = collections.Counter("+".join(sorted({group[i] for i in items})) for items in batches.values())
    check("batches holding only malicious queries", 13, kinds["malicious"])
    check("batches holding only benign controls", 6, kinds["benign"])
    check("batches holding only payloads", 1, kinds["pi"])
    check("single-group batches", 20, kinds["malicious"] + kinds["benign"] + kinds["pi"])
    fams_per_mal_batch = [len({family[i] for i in items}) for items in batches.values()
                          if {group[i] for i in items} == {"malicious"}]
    check("malicious-only batches span at most two families", True, max(fams_per_mal_batch) <= 2)
    for d in DETECTORS:
        for rep in (0, 1, 2):
            b2 = collections.defaultdict(list)
            for r in runs:
                if r["detector"] == d and r["position"] == "post_hoc" and r["repeat"] == rep:
                    b2[r["batch_id"].split("-")[-1]].append(r["item_id"])
            ref = {k.split("-")[-1]: v for k, v in batches.items()}
            check(f"same batch composition in every pass ({d}, repeat {rep})", True, dict(b2) == ref)

    # frontier: benign items flagged post-hoc but not at review-time, and their batch mates
    extra = [i for i in BEN if flagged[("frontier", "post_hoc", i)] and not flagged[("frontier", "review_time", i)]]
    check("frontier additional post-hoc false positives over review-time", 11, len(extra))
    item_batch = {i: b for b, items in batches.items() for i in items}
    mixed = [b for b, items in batches.items() if {group[i] for i in items} == {"malicious", "benign"}]
    best = None
    for b in mixed:
        items = batches[b]
        mal_fams = collections.Counter(family[i] for i in items if group[i] == "malicious")
        ben_here = [i for i in items if group[i] == "benign"]
        if mal_fams == {"file_os": 5} and len(ben_here) == 5:
            best = (b, ben_here)
    check("a batch of five file-access attacks and five hard negatives exists", True, best is not None)
    if best:
        check("all five of those hard negatives are among the 11 additional false positives", 5,
              sum(i in extra for i in best[1]))
        check("those five are hard negatives (generated candidates)", 5, sum(i in hard for i in best[1]))

    # ------------------------------------------------------------------ Table III
    table3 = {  # (detector, position): (ms per call, ms per query, tokens in, tokens out, cost, cost per 1k)
        ("frontier", "review_time"): (1829, 1829, 69555, 64293, 1.14, 1.78),
        ("frontier", "runtime_gatekeeper"): (1051, 1051, 54219, 23700, 0.49, 0.77),
        ("frontier", "post_hoc"): (3768, 389, 24315, 18264, 0.33, 0.52),
        ("small", "review_time"): (3040, 3040, 87708, 43458, 0.0, 0.0),
        ("small", "runtime_gatekeeper"): (240, 240, 69816, 1665, 0.0, 0.0),
        ("small", "post_hoc"): (750, 77, 26799, 576, 0.0, 0.0),
    }
    total_cost = 0.0
    for (d, p), (ms_call, ms_query, tin, tout, cost, per_1k) in table3.items():
        rows = [r for r in runs if r["detector"] == d and r["position"] == p]
        calls = [r for r in rows if r["latency_ms"] is not None]
        latency = sum(r["latency_ms"] for r in calls)
        got_in = sum(r["tokens_in"] or 0 for r in rows)
        got_out = sum(r["tokens_out"] or 0 for r in rows)
        got_cost = (got_in * PRICE_IN_PER_1K + got_out * PRICE_OUT_PER_1K) / 1000 if d == "frontier" else 0.0
        total_cost += got_cost
        check(f"Table III judged queries per row {d} {p}", 639, len(rows))
        check(f"Table III calls {d} {p}", 66 if p == "post_hoc" else 639, len(calls))
        check(f"Table III latency per call (ms) {d} {p}", ms_call, round(latency / len(calls)))
        check(f"Table III latency per query (ms) {d} {p}", ms_query, round(latency / 639))
        check(f"Table III tokens in {d} {p}", tin, got_in)
        check(f"Table III tokens out {d} {p}", tout, got_out)
        check(f"Table III cost {d} {p}", cost, round(got_cost, 2))
        check(f"Table III cost per 1,000 queries {d} {p}", per_1k, round(1000 * got_cost / 639, 2))
    check("full frontier run cost", 1.96, round(total_cost, 2))
    rt = [r["tokens_out"] for r in runs if r["detector"] == "frontier" and r["position"] == "runtime_gatekeeper"]
    check("frontier runtime: mean billed output tokens per one-word verdict", 37, round(sum(rt) / len(rt)))
    check("frontier runtime calls billed more than six output tokens (%)", 87, round(100 * sum(x > 6 for x in rt) / len(rt)))
    check("frontier runtime answers are one word", True,
          all(r["raw_output"].strip().upper() in ("SAFE", "UNSAFE") for r in runs
              if r["detector"] == "frontier" and r["position"] == "runtime_gatekeeper"))
    days = {datetime.fromtimestamp(r["ts"], tz=timezone.utc).date().isoformat() for r in runs if r["detector"] == "frontier"}
    check("frontier run dates (UTC)", {"2026-08-08"}, days)

    # ------------------------------------------------------------------ labeling
    check("generated candidates", 200, len(review))
    labels = collections.Counter(r["expert_label"].strip().lower() for r in review)
    check("label outcome malicious / benign / excluded", (135, 26, 39),
          (labels["malicious"], labels["benign"], sum(v for k, v in labels.items() if k.startswith("exclu"))))
    changed = sum(r["auto_label"].strip().lower() != r["expert_label"].strip().lower() for r in review)
    check("pre-labels changed by the annotator", 68, changed)

    # ------------------------------------------------------------------ example query (Section VII)
    example = "SELECT * FROM orders WHERE ASCII(SUBSTRING(status, 1, 1)) = 97;"
    hits = [row for row in corpus if row["text"] == example]
    check("example query is one corpus item, verbatim", 1, len(hits))
    if hits:
        ex_id = hits[0]["item_id"]
        check("example query is a malicious blind Boolean item", ("malicious", "blind_boolean"),
              (group[ex_id], family[ex_id]))
        check("example query: judgments that flagged it (of 18)", 0,
              sum(v for (d, p, i), reps in votes.items() if i == ex_id for v in reps.values()))

    # ------------------------------------------------------------------ report
    width = max(len(c[1]) for c in CHECKS)
    failed = 0
    for passed, description, reported, computed in CHECKS:
        if not passed:
            failed += 1
        print(f"{'PASS' if passed else 'FAIL'}  {description.ljust(width)}  paper: {reported}  log: {computed}")
    print(f"\n{len(CHECKS) - failed} of {len(CHECKS)} checks passed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
