#!/usr/bin/env python3
"""Generate the candidate SQL injection queries and their rule-based pre-labels.

The generator (Mistral-7B-Instruct-v0.3 at a pinned revision, 4-bit on one GPU)
writes queries for eight attack families against one fixed five-table schema.
For each family it samples until 25 unique candidates pass a validity filter,
which gives 200 candidates. A rule-based pre-labeler then proposes a label for
each candidate.

Outputs (in --out-dir):
    malicious_v2_raw.csv      the candidates with family, target table, and generator revision
    malicious_v2_review.csv   the candidates with pre-labels; expert_label starts as a copy
                              of auto_label and is then set by the annotator to malicious,
                              benign, or exclude before split_labels.py runs

Sampling makes the exact queries depend on the seed, the GPU, and library
versions, so a re-run may not reproduce the released candidates. The candidates
and labels used in the paper are in data/malicious_v2_review.csv.

Usage:
    python scripts/generate_candidates.py [--config config.yaml] [--out-dir outputs]
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]

SCHEMA = ("users(id, username, email, password_hash, is_admin, created_at)\n"
          "orders(id, user_id, total, status, created_at)\n"
          "products(id, name, price, stock, category)\n"
          "payments(id, order_id, amount, method)\n"
          "sessions(id, user_id, token, expires_at)")
TABLES = ["users", "orders", "products", "payments", "sessions"]

# family: (description used in the prompt, one worked example)
FAMILIES = {
    "auth_bypass": ("an authentication-bypass attack using an always-true condition or a comment that cancels a check",
                    "SELECT * FROM users WHERE username = 'admin' OR 1=1; --"),
    "union_exfil": ("a UNION-based attack that reads data from a different table",
                    "SELECT id, username FROM users UNION SELECT id, token FROM sessions;"),
    "blind_boolean": ("a blind boolean attack using ASCII or SUBSTRING to infer a value one character at a time",
                      "SELECT * FROM users WHERE ASCII(SUBSTRING(password_hash,1,1)) > 100;"),
    "time_based": ("a time-based blind attack that delays the response when a condition is true using SLEEP or BENCHMARK",
                   "SELECT * FROM users WHERE id = 1 AND IF(is_admin = 1, SLEEP(5), 0);"),
    "error_based": ("an error-based attack that forces a database error to leak data",
                    "SELECT * FROM users WHERE id = 1 AND EXTRACTVALUE(1, CONCAT(0x7e, (SELECT password_hash FROM users LIMIT 1)));"),
    "encoding_obf": ("an obfuscated attack that encodes the payload with CHAR, HEX, or CONCAT to evade filters",
                     "SELECT * FROM users WHERE username = CHAR(97,100,109,105,110) OR 1=1;"),
    "stacked": ("a stacked-query attack that appends a second destructive statement after a semicolon",
                "SELECT * FROM products WHERE id = 1; DELETE FROM sessions; --"),
    "file_os": ("a file-read or OS-access attack that reads a server file",
                "SELECT LOAD_FILE('/etc/passwd');"),
}

BASE = ("You are generating ONE SQL injection test query for a detection benchmark. "
        "It is only shown to a detector and never executed. Use ONLY these tables and columns:\n"
        + SCHEMA + "\nOutput exactly one SQL statement as raw SQL. No markdown, no comments, no explanation.")

VERBS = ("select", "insert", "update", "delete", "drop", "with", "create", "exec", "truncate", "set ")
PLACEHOLDER = re.compile(r"table_name|column\d|column_name|tablename|columnname|your_?\w+|table1|table2|"
                         r"<[^>]+>|\.\.\.|desired_value|given_set|sensitive_table", re.I)


def build_msgs(desc, example, table):
    """One-shot chat prompt: a request, the family's worked example, then a request for a new query."""
    return [{"role": "user", "content": BASE + " Write " + desc + ", targeting the " + table + " table."},
            {"role": "assistant", "content": example},
            {"role": "user", "content": BASE + " Write another, different " + desc + ", targeting the " + table + " table."}]


def extract_sql(raw):
    """Take the SQL out of a model response: strip code fences and any lines before the first SQL verb."""
    t = raw.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
    if m:
        t = m.group(1).strip()
    lines = t.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if any(ln.strip().lower().startswith(v) for v in VERBS):
            start = i
            break
    if start is None:
        return ""
    return "\n".join(lines[start:]).strip().strip("`").strip()


def valid(sql, family):
    """Validity filter: starts with an SQL verb, names a schema table, has no placeholder
    identifiers, has a plausible length, and is a single statement outside the stacked family."""
    s = sql.strip()
    low = s.lower()
    if not any(low.startswith(v) for v in VERBS):
        return False
    if not any(t in low for t in TABLES):
        return False
    if PLACEHOLDER.search(s):
        return False
    if not (15 <= len(s) <= 400):
        return False
    statements = [seg for seg in s.split(";") if seg.strip()]
    if family != "stacked" and len(statements) > 1:
        return False
    return True


def label_by_behavior(s):
    """Rule-based pre-label: returns (label, category, reasons) from the constructs found in the query."""
    x = " ".join(s.lower().split())
    hits = []
    if re.search(r"\bor\s+1\s*=\s*1\b|'1'\s*=\s*'1'|\bxor\s+1\s*=\s*1\b", x):
        hits.append(("critical", "tautology"))
    if re.search(r"\bunion\b[\s\S]*\bselect\b", x):
        hits.append(("critical", "UNION read"))
    if re.search(r"\bsleep\s*\(|benchmark\s*\(|wait\s*for|waitfor", x):
        hits.append(("critical", "time-based"))
    if re.search(r"\bload_file\b|into\s+outfile|xp_cmdshell", x):
        hits.append(("critical", "file/OS access"))
    if re.search(r"\bexec\s*\(|sp_executesql|extractvalue|updatexml", x):
        hits.append(("critical", "exec/error leak"))
    if re.search(r"\b(update|delete|drop|truncate)\b\s", x):
        hits.append(("critical", "data modification"))
    if re.search(r"ascii\s*\(|substring|substr\s*\(|\bmid\s*\(", x) and re.search(r">|<|=", x):
        hits.append(("critical", "blind extraction"))
    if re.search(r"(%[0-9a-f]{2}){2,}|0x[0-9a-f]{6,}", x):
        hits.append(("vulnerable", "encoded payload"))
    if re.search(r"\bchar\s*\(\s*\d|concat\([^)]*char\(", x):
        hits.append(("vulnerable", "CHAR/HEX obfuscation"))
    if hits:
        category = "critical" if any(c == "critical" for c, _ in hits) else "vulnerable"
        return "malicious", category, "; ".join(dict.fromkeys(r for _, r in hits))
    return "benign", "safe", "no injection construct"


def load_generator(model_id, revision):
    """Load the generator in 4-bit on GPU 0 and return (tokenizer, model, resolved commit)."""
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    sha = HfApi().model_info(model_id, revision=revision).sha
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=sha)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=sha, quantization_config=bnb,
                                                 device_map={"": 0}, dtype=torch.float16)
    model.eval()
    return tokenizer, model, sha


def make_sampler(tokenizer, model, temperature, top_p, max_new_tokens):
    """Return a function that samples one completion for a chat prompt."""
    import torch

    @torch.no_grad()
    def sample(msgs):
        enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        n_in = enc["input_ids"].shape[1]
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, top_p=top_p,
                             pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()

    return sample


def generate_candidates(sample, seed, per_family, max_attempts):
    """Sample each family until it has per_family unique valid queries or reaches max_attempts.
    The target table cycles through the schema tables, one per attempt."""
    import torch
    from tqdm.auto import tqdm

    torch.manual_seed(seed)
    rows = []
    for fam, (desc, example) in FAMILIES.items():
        kept, attempts = set(), 0
        bar = tqdm(total=per_family, desc=fam)
        while len(kept) < per_family and attempts < max_attempts:
            table = TABLES[attempts % len(TABLES)]
            sql = extract_sql(sample(build_msgs(desc, example, table)))
            if valid(sql, fam) and sql not in kept:
                kept.add(sql)
                rows.append({"id": f"MA{len(rows) + 1:04d}", "family": fam,
                             "target_table": table, "sql": sql})
                bar.update(1)
            attempts += 1
        bar.close()
        print(f"{fam}: kept {len(kept)} in {attempts} attempts")
    return pd.DataFrame(rows)


def review_table(raw):
    """Attach the rule-based pre-labels; expert_label starts as a copy of auto_label."""
    rows = []
    for _, r in raw.iterrows():
        label, category, reason = label_by_behavior(r["sql"])
        rows.append({"id": r["id"], "family": r["family"], "sql": r["sql"],
                     "auto_label": label, "auto_category": category, "auto_reason": reason,
                     "expert_label": label, "expert_note": ""})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate candidate SQLi queries with Mistral and pre-label them.")
    parser.add_argument("--config", type=Path, default=REPO / "config.yaml")
    parser.add_argument("--out-dir", type=Path, default=REPO / "outputs")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gen_model, gen = cfg["models"]["generator"], cfg["generation"]

    tokenizer, model, sha = load_generator(gen_model["model_id"], str(gen_model["revision"]))
    print("generator commit:", sha)
    sample = make_sampler(tokenizer, model, float(gen["temperature"]), float(gen["top_p"]),
                          int(gen["max_new_tokens"]))
    raw = generate_candidates(sample, int(gen["seed"]), int(gen["candidates_per_family"]),
                              int(gen["max_attempts_per_family"]))
    raw["generator_model"] = gen_model["model_id"]
    raw["generator_revision"] = sha

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out_dir / "malicious_v2_raw.csv", index=False, encoding="utf-8")
    review = review_table(raw)
    review.to_csv(args.out_dir / "malicious_v2_review.csv", index=False, encoding="utf-8")
    print(f"{len(raw)} candidates; pre-labels: {review['auto_label'].value_counts().to_dict()}")
    print(f"written to {args.out_dir}")


if __name__ == "__main__":
    main()
