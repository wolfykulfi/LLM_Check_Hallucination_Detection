# Sentence-level hallucination detection in RAG with small LMs

Major project built on top of LLM-Check (NeurIPS 2024). The original LLM-Check
files at the repository root (`common_utils.py`, `utils_*.py`,
`run_detection_combined.py`, `README.md`) are **kept unchanged** as the
reference implementation. Everything new lives in `rag_halluc/`.

**Goal:** a small (0.5B to 3B) reader model and a small NLI checker that say
*which sentence* of a RAG answer the retrieved passages do not support, route
only uncertain sentences to the costly check, and repair the flagged ones,
evaluated on RAGTruth and RAGBench's five domains.

## Layout

```
rag_halluc/
  models.py               reader registry (Qwen2.5, Llama-3.2, Gemma-2), eager-attention loading,
                          chat-template encoding with exact answer token span
  data/ragtruth.py        download + join RAGTruth (response.jsonl x source_info.jsonl)
  eval/metrics.py         AUROC (label 1 = hallucinated), direction, balanced acc, TPR@5%FPR
  repro/phase0_llmcheck.py  Phase 0: original LLM-Check scores with a small reader
configs/                  one YAML per experiment
notebooks/                Kaggle / Colab notebooks (free T4)
docs/                     explanations for the report and viva
tests/                    CPU-only tests with a tiny random model (no downloads)
results/                  CSV result tables written by code (never typed by hand)
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q                       # CPU, a few seconds, no internet needed
python -m rag_halluc.repro.phase0_llmcheck --config configs/phase0_qwen25_3b.yaml   # needs a GPU
```

On Kaggle or Colab, open `notebooks/phase0_reproduce_llmcheck.ipynb` and run all
cells (Accelerator: GPU T4, Internet: on). RAGTruth is downloaded automatically
from its GitHub repo into `data/ragtruth/`.

## Progress

| Phase | What | Status |
|---|---|---|
| 0 | Setup; reproduce LLM-Check logit + attention scores on 200 RAGTruth Summary answers with a small reader | code done, run on Kaggle |
| 1 | Data pipeline: all tasks and generators, source-grouped validation split, sentence labels and token spans | next |
| 2 | Per-sentence internal features incl. context-aware attention and lookback ratio; AUROC vs reader size (Review 1) | |
| 3 | NLI checkers (MiniCheck, DeBERTa, HHEM) and baselines (Cheung, LettuceDetect, observer probe) | |
| 4 | Fusion probe, cascade, ablations, bootstrap CIs (Review 2) | |
| 5 | Repair loop and RAGBench transfer | |
| 6 | Gradio demo, final evaluation, report | |

Phase 0 explanations of every score: [docs/phase0_scores_explained.md](docs/phase0_scores_explained.md).

## Reproducibility rules

* One config file per run; it is copied next to the results.
* Fixed seed (`seed` in the config).
* Every results table is written to CSV by code.
* Validation split by `source_id`, never by row (Phase 1), and thresholds are
  never tuned on test.
