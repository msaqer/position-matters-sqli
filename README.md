# Position Matters: LLMs as SQL Injection Detectors Across Deployment Positions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24265893.svg)](https://doi.org/10.5281/zenodo.24265893)

Code, data, and run logs for the paper:

M. Abusaqer, "Position Matters: Evaluating Large Language Models as SQL Injection Detectors Across Deployment Positions," 2026 IEEE International Conference on Big Data (BigData), Phoenix, AZ, USA, December 2026 (to appear).

The study asks whether a prompted large language model can serve as an SQL injection detector on LLM-generated queries, and whether the answer depends on where in the pipeline the detector sits. Two detectors, a frontier API model (gpt-5.6-terra) and a locally hosted 7B model (Qwen2.5-7B-Instruct), judge a 213-item corpus at three deployment positions:

- **Review-time**: offline code review of one query.
- **Runtime gatekeeper**: an inline, one-word verdict.
- **Post-hoc**: a batched audit of the query log.

Each item was judged three times at each position, for 3,834 logged verdicts.

## Contents

| Path | Contents |
|---|---|
| `data/corpus.csv` | The 213-item evaluation set: 135 malicious queries, 66 benign controls, and 12 prompt-injection payloads |
| `data/malicious_v2_review.csv` | The 200 generated candidates with rule-based pre-labels and the annotator's final labels |
| `data/malicious_final.csv`, `data/benign_generated.csv`, `data/malicious_excluded.csv` | The candidates split by final label (135 malicious, 26 benign, 39 excluded) |
| `data/benign_sql.csv`, `data/pi_payloads.csv` | The 40 hand-authored benign controls and the 12 payloads |
| `results/logs/runs_v2.jsonl` | All 3,834 verdicts with the raw model output, latency, and token counts |
| `results/tables/` | Tables I to III, Fig. 2, and the payload counts, rebuilt from the log |
| `scripts/` | The code for each stage of the study |
| `config.yaml` | Model revisions, decoding settings, and prices used for the reported runs |

[data/README.md](data/README.md) describes every column and the labeling policy.

## Reproduce the reported numbers

These two scripts need only Python 3.8 or later, with no GPU, API key, or third-party packages.

```
python scripts/verify_paper_numbers.py
python scripts/analyze.py --out-dir results/tables
```

`verify_paper_numbers.py` recomputes every number reported in the paper from the run log, the corpus, and the review file, then compares each one with the printed value. It ends with `182 of 182 checks passed.` `analyze.py` writes the tables as CSV files.

## Re-run the experiments

The experiment scripts need:

- Python 3.12 (the version used for the study).
- An NVIDIA GPU. The study used one RTX A2000 with 12 GB, with both local models loaded in 4-bit.
- An OpenAI API key in the `OPENAI_API_KEY` environment variable, for the frontier detector.
- Access to the two Hugging Face models.

Install PyTorch with CUDA support for your platform first (https://pytorch.org/get-started/locally/), then:

```
pip install -r requirements.txt
```

The scripts write to `outputs/` unless told otherwise, so the released files in `data/` and `results/` stay unchanged.

| Step | Command | Writes |
|---|---|---|
| 1. Hand-authored controls | `python scripts/build_controls.py` | `outputs/benign_sql.csv`, `outputs/pi_payloads.csv` |
| 2. Candidate generation (GPU) | `python scripts/generate_candidates.py` | `outputs/malicious_v2_raw.csv`, `outputs/malicious_v2_review.csv` |
| 3. Labeling | Set `expert_label` in `outputs/malicious_v2_review.csv` to `malicious`, `benign`, or `exclude` | |
| 4. Split by label | `python scripts/split_labels.py --review outputs/malicious_v2_review.csv` | `outputs/malicious_final.csv`, `outputs/benign_generated.csv`, `outputs/malicious_excluded.csv` |
| 5. Corpus | `python scripts/build_corpus.py --data-dir outputs` | `outputs/corpus.csv` |
| 6. Detection (GPU and API) | `python scripts/run_detection.py --corpus outputs/corpus.csv` | `outputs/runs.jsonl` |
| 7. Tables | `python scripts/analyze.py --log outputs/runs.jsonl` | `outputs/tables/` |

To rerun only the detectors on the released corpus, skip steps 1 to 5 and run `python scripts/run_detection.py`. To try the local model alone, add `--detectors small`. The detection run resumes from an existing log, so it can be stopped and restarted. At the launch list price the frontier runs cost about $1.96. `verify_paper_numbers.py` checks the released log against the paper; a new run will differ, because the frontier model is not deterministic.

## Notes on the reported runs

- **Frontier settings:** the frontier detector ran at the API's default settings. Each request set only the model and the messages, with no temperature, other sampling parameter, or output limit. Its verdicts are therefore not deterministic, which is why every item was judged three times and its label at a position is the majority verdict.
- **Frontier snapshot:** the provider exposes no dated snapshot of gpt-5.6-terra. All frontier runs were made on 8 August 2026.
- **Billed tokens:** frontier output tokens are billed tokens. They include reasoning tokens that the API does not return as text.
- **Prices:** costs use the launch list price of $2.50 per million input tokens and $15 per million output tokens.
- **Local detector:** it decodes greedily. It returned identical verdicts across the three repeats for every item.
- **Post-hoc batches:** they follow corpus order, so most of them hold a single group.
- **Latency:** it was measured on sequential single calls. Throughput under concurrent load was not tested.
- **Generation:** candidates were sampled (temperature 0.8, top-p 0.95, seed 42), so a rerun of step 2 may produce different candidates. The released files are the ones used in the paper.
- **Labels:** one annotator judged the query text. No query was executed against a database.

## Intended use

The corpus contains SQL injection attack strings and prompt-injection payloads for research on detection. They are meant to be shown to detectors, not executed against systems you do not control.

## Citation

```bibtex
@inproceedings{abusaqer2026position,
  author    = {Muhammad Abusaqer},
  title     = {Position Matters: Evaluating Large Language Models as {SQL} Injection Detectors Across Deployment Positions},
  booktitle = {2026 IEEE International Conference on Big Data (BigData)},
  address   = {Phoenix, AZ, USA},
  month     = dec,
  year      = {2026},
  note      = {To appear}
}
```

The code, data, and run logs are archived on Zenodo. Version 1.0.0, the release that matches the paper, has the DOI [10.5281/zenodo.24265894](https://doi.org/10.5281/zenodo.24265894). The DOI [10.5281/zenodo.24265893](https://doi.org/10.5281/zenodo.24265893) covers all versions and always resolves to the latest one.

GitHub's "Cite this repository" link reads the same information from `CITATION.cff`.

## License

- **Code:** MIT License (see `LICENSE`).
- **Data, run logs, and tables** in `data/` and `results/`: Creative Commons Attribution 4.0 International (see `LICENSE-DATA`).
