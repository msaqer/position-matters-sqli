#!/usr/bin/env python3
"""Run both LLM detectors over the corpus at the three deployment positions.

Detectors
    frontier   gpt-5.6-terra through the OpenAI API (reads OPENAI_API_KEY)
    small      Qwen2.5-7B-Instruct at a pinned revision, 4-bit on GPU 0, greedy decoding

Positions
    review_time         code review of one query: brief reasoning, then a VERDICT line
    runtime_gatekeeper  one word, SAFE or UNSAFE, for one query
    post_hoc            a numbered batch of ten queries in corpus order; the answer lists
                        the line numbers of the malicious ones

Every judgment is appended to a JSON Lines log, one record per (item, position,
detector, repeat). An existing log is resumed: records already written are
skipped. For post-hoc calls, latency and tokens are recorded on the first item of
each batch, and every item in the batch carries the same batch_id. A running
estimate of the API cost stops the per-item positions at the budget cap.

Usage:
    python scripts/run_detection.py [--corpus data/corpus.csv] [--log outputs/runs.jsonl]
                                    [--detectors frontier small]
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]

REVIEW_PROMPT = (
    "You are a security code reviewer inspecting a SQL statement for SQL injection risk. "
    "Consider tautologies, comment tricks, UNION reads, stacked queries, encoding or "
    "function obfuscation, and time-based blind patterns. Reason briefly, then end with a "
    "single final line exactly in this form: VERDICT: MALICIOUS  or  VERDICT: BENIGN.")

RUNTIME_PROMPT = (
    "You are an inline SQL firewall that must decide fast. For the single query below, "
    "reply with exactly one word and nothing else: UNSAFE if it is or contains a SQL "
    "injection attack, or SAFE if it is a legitimate query.")

POSTHOC_PROMPT = (
    "You are auditing a batch of SQL statements from a query log. Some may be SQL "
    "injection attacks. Return only the line numbers of the malicious ones as a "
    "comma-separated list, for example: 1,4,5. If none are malicious, return: none")


def msgs_review(query):
    return [{"role": "user", "content": REVIEW_PROMPT + "\n\nSQL:\n" + query}]


def msgs_runtime(query):
    return [{"role": "user", "content": RUNTIME_PROMPT + "\n\nQuery: " + query}]


def msgs_posthoc(texts):
    listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return [{"role": "user", "content": POSTHOC_PROMPT + "\n\n" + listing}]


def parse_verdict(text):
    """Map a review-time or runtime answer to malicious, benign, or unparseable.
    A VERDICT line wins; otherwise "unsafe"/"malicious" are checked before
    "safe"/"benign", so the "safe" inside "unsafe" is not misread."""
    low = text.lower()
    m = re.search(r"verdict\s*[:\-]\s*(malicious|benign|unsafe|safe)", low)
    tok = m.group(1) if m else None
    if tok is None:
        if "unsafe" in low or "malicious" in low:
            tok = "malicious"
        elif re.search(r"\bsafe\b", low) or "benign" in low:
            tok = "benign"
    if tok in ("malicious", "unsafe"):
        return "malicious"
    if tok in ("benign", "safe"):
        return "benign"
    return "unparseable"


def parse_indices(text, n):
    """Read the line numbers flagged in a post-hoc answer for a batch of n queries."""
    if "none" in text.lower():
        return set()
    return {int(x) for x in re.findall(r"\d+", text) if 1 <= int(x) <= n}


class FrontierDetector:
    """OpenAI chat completions. As in the reported runs, each request sets only the
    model and the messages: no temperature, no other sampling parameter, and no
    output limit."""

    def __init__(self, model_id):
        from openai import OpenAI
        self.client = OpenAI()          # reads OPENAI_API_KEY from the environment
        self.model_id = model_id

    def __call__(self, messages, max_new_tokens):
        # max_new_tokens applies to the local model only; it is not sent to the API.
        t0 = time.time()
        resp = self.client.chat.completions.create(model=self.model_id, messages=messages)
        latency_ms = (time.time() - t0) * 1000
        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", None) if usage else None
        tokens_out = getattr(usage, "completion_tokens", None) if usage else None
        return text, latency_ms, tokens_in, tokens_out


class LocalDetector:
    """A Hugging Face chat model in 4-bit on GPU 0 with greedy decoding."""

    def __init__(self, model_id, revision):
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.sha = HfApi().model_info(model_id, revision=revision).sha
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=self.sha)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=self.sha, quantization_config=bnb,
                                                          device_map={"": 0}, dtype=torch.float16)
        self.model.eval()

    def __call__(self, messages, max_new_tokens):
        with self.torch.no_grad():
            enc = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                     return_tensors="pt", return_dict=True)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            n_in = enc["input_ids"].shape[1]
            t0 = time.time()
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id)
            latency_ms = (time.time() - t0) * 1000
        generated = out[0][n_in:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text, latency_ms, int(n_in), int(generated.shape[0])


def logged_keys(log_path):
    """Return the (item_id, position, detector, repeat) keys already in the log."""
    done = set()
    if log_path.exists():
        with log_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["item_id"], r["position"], r["detector"], r["repeat"]))
    return done


def run(items, detectors, log_path, repeats, batch_size, max_new_tokens,
        price_in_per_1k, price_out_per_1k, usd_cap):
    """Judge every item at every position. `detectors` maps a detector name to a
    callable (messages, max_new_tokens) -> (text, latency_ms, tokens_in, tokens_out)."""
    from tqdm.auto import tqdm

    log_path.parent.mkdir(parents=True, exist_ok=True)
    done = logged_keys(log_path)
    spend = 0.0
    stop = False

    def write(rec):
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def record(item, position, det, rep, parsed, raw, latency_ms, tokens_in, tokens_out, batch_id):
        return {"item_id": item["item_id"], "group": item["group"],
                "true_label": item["true_label"], "tier": item["tier"],
                "position": position, "detector": det, "repeat": rep,
                "parsed_label": parsed, "raw_output": raw,
                "latency_ms": latency_ms, "tokens_in": tokens_in, "tokens_out": tokens_out,
                "batch_id": batch_id, "ts": time.time()}

    # Review-time and runtime gatekeeper: one query per call.
    for det, call in detectors.items():
        for position, build in (("review_time", msgs_review), ("runtime_gatekeeper", msgs_runtime)):
            for rep in range(repeats):
                for item in tqdm(items, desc=f"{det}/{position}/r{rep}"):
                    key = (item["item_id"], position, det, rep)
                    if key in done:
                        continue
                    if stop:
                        break
                    text, latency_ms, tin, tout = call(build(item["text"]), max_new_tokens[position])
                    if det == "frontier" and tin is not None:
                        spend += (tin / 1000.0) * price_in_per_1k + (tout / 1000.0) * price_out_per_1k
                    write(record(item, position, det, rep, parse_verdict(text), text,
                                 round(latency_ms, 1), tin, tout, None))
                    done.add(key)
                    if det == "frontier" and spend >= usd_cap:
                        print(f"Budget cap reached at ${spend:.2f}; stopping.")
                        stop = True
                        break

    # Post-hoc audit: numbered batches of batch_size queries in corpus order.
    if not stop:
        for det, call in detectors.items():
            for rep in range(repeats):
                for b0 in range(0, len(items), batch_size):
                    batch = items[b0:b0 + batch_size]
                    if all((it["item_id"], "post_hoc", det, rep) in done for it in batch):
                        continue
                    text, latency_ms, tin, tout = call(msgs_posthoc([it["text"] for it in batch]),
                                                       max_new_tokens["post_hoc"])
                    if det == "frontier" and tin is not None:
                        spend += (tin / 1000.0) * price_in_per_1k + (tout / 1000.0) * price_out_per_1k
                    flagged = parse_indices(text, len(batch))
                    batch_id = f"{det}-r{rep}-b{b0 // batch_size}"
                    for i, it in enumerate(batch):
                        first = i == 0
                        write(record(it, "post_hoc", det, rep,
                                     "malicious" if (i + 1) in flagged else "benign",
                                     text if first else "",
                                     round(latency_ms, 1) if first else None,
                                     tin if first else 0, tout if first else 0, batch_id))
                        done.add((it["item_id"], "post_hoc", det, rep))

    print(f"done. estimated frontier API spend: ${spend:.2f}")
    return spend


def main():
    parser = argparse.ArgumentParser(description="Run the LLM detectors at the three deployment positions.")
    parser.add_argument("--config", type=Path, default=REPO / "config.yaml")
    parser.add_argument("--corpus", type=Path, default=REPO / "data" / "corpus.csv")
    parser.add_argument("--log", type=Path, default=REPO / "outputs" / "runs.jsonl")
    parser.add_argument("--detectors", nargs="+", choices=["frontier", "small"], default=["frontier", "small"])
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    det_cfg, budget = cfg["detection"], cfg["budget"]
    items = pd.read_csv(args.corpus, dtype=str, keep_default_na=False).to_dict("records")

    detectors = {}
    if "frontier" in args.detectors:
        detectors["frontier"] = FrontierDetector(cfg["models"]["detector_frontier"]["model_id"])
    if "small" in args.detectors:
        small = cfg["models"]["detector_small"]
        detectors["small"] = LocalDetector(small["model_id"], str(small["revision"]))
        print("small detector commit:", detectors["small"].sha)

    run(items, detectors, args.log,
        repeats=int(det_cfg["repeats"]),
        batch_size=int(det_cfg["posthoc_batch_size"]),
        max_new_tokens={k: int(v) for k, v in det_cfg["local_max_new_tokens"].items()},
        price_in_per_1k=float(budget["price_per_1k_tokens"]["input"]),
        price_out_per_1k=float(budget["price_per_1k_tokens"]["output"]),
        usd_cap=float(budget["usd_cap"]))


if __name__ == "__main__":
    main()
