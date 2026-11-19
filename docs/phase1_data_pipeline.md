# Phase 1: the data pipeline

```bash
python -m spacy download en_core_web_sm
python -m rag_halluc.data.build_ragtruth                              # CPU, about 3 minutes
python -m rag_halluc.data.build_ragtruth --check_reader qwen2.5-1.5b   # + token check with a real tokenizer
```

## What goes in, what comes out

RAGTruth gives, for each of 2,965 source documents, one answer from each of
six generators (GPT-4, GPT-3.5, Mistral-7B, Llama-2 7B / 13B / 70B), across
three tasks: QA (MS MARCO passages), Data2txt (Yelp business records) and
Summary (news articles). Annotators marked hallucinated **character spans** in
each answer and typed them as *evident* or *subtle*, *conflict* (contradicts
the context) or *baseless info* (adds facts that are not there).

`build_ragtruth.py` turns this into two tables:

| file | one row per | key columns |
|---|---|---|
| `data/processed/ragtruth_responses.parquet` | answer (17,790) | id, source_id, task_type, generator, split, prompt, context, question, response, spans_json, label |
| `data/processed/ragtruth_sentences.parquet` | answer sentence (about 132k) | id, sentence_idx, sent_start, sent_end, sentence, label, span_types, is_evident / is_subtle / is_conflict / is_baseless, implicit_true_only, split |

`context` is the evidence text exactly as it appears inside the prompt, so its
position in the tokenized prompt can be found (Phase 2 needs it for the
context-aware attention features).

## The four design decisions (and why)

1. **Validation split by source, not by row** (`assign_splits`). Each source
   has six answers. If we split rows, the same article would appear in both
   train and validation, and the probe could learn the article instead of the
   hallucination signal. We hold out 10% of the *train sources* (252 of 2,515,
   seed 0). The official test split (450 sources) is never touched until the
   final runs.
2. **Sentences as claims** (`claims/split.py`). We cut at line breaks first
   (bullet lists, headings), then let spaCy find sentence boundaries, then
   merge bare list markers like `1.` into the next sentence. Each sentence
   keeps its exact `[start, end)` character offsets.
3. **Sentence label = overlap with any labelled span.** A span that crosses a
   sentence boundary labels both sentences. The span types are kept for error
   analysis ("do we miss subtle conflicts more than evident baseless info?").
   `implicit_true_only` marks sentences whose only spans the annotators flagged
   as true-but-unsupported, for an ablation.
4. **Token mapping by character offsets** (`models.encode_with_sentences`).
   We render the reader's chat template, tokenize with
   `return_offsets_mapping=True`, and take every token whose characters overlap
   the sentence (or the passages). This is tokenizer-agnostic, so the same code
   works for Qwen, Llama and Gemma.

## Numbers (written by code to `results/phase1/`)

`ragtruth_stats.csv`, official splits:

| task | split | answers | hallucinated answers | sentences | hallucinated sentences |
|---|---|---:|---:|---:|---:|
| Data2txt | train | 5,298 | 3,675 (69%) | 48,933 | 6,931 (14%) |
| Data2txt | test | 900 | 579 (64%) | 8,235 | 936 (11%) |
| QA | train | 5,034 | 1,564 (31%) | 36,857 | 3,768 (10%) |
| QA | test | 900 | 160 (18%) | 5,785 | 369 (6%) |
| Summary | train | 4,758 | 1,482 (31%) | 26,954 | 1,783 (7%) |
| Summary | test | 900 | 204 (23%) | 4,991 | 258 (5%) |

* Answer counts per task (QA 5,934; Data2txt 6,198; Summary 5,658; 17,790
  total, 2,700 test) match the RAGTruth paper's statistics. Compare the
  hallucinated-answer counts with the paper's table before Review 1. The
  released data was updated in 2024 with more annotated spans, so small
  differences from the paper are expected.
* Sentence-level positives are rare (5% to 14%). That is why we report AUPRC
  and F1 next to AUROC from Phase 2 on: AUROC alone looks good on imbalanced
  data.

`ragtruth_splits.csv`: train 13,578 answers / 2,263 sources, val 1,512 / 252,
test 2,700 / 450.

## Checks (the "done when" of Phase 1)

* Every hallucinated answer has at least one hallucinated sentence: 0 misses.
* No source in both train and val: 0.
* Token mapping: on 500 random real RAGTruth sentences, decoding the mapped
  tokens gives back exactly the sentence text, and the passages are located in
  the prompt every time (checked with the byte-level test tokenizer; run
  `--check_reader` on Kaggle to repeat it with Qwen / Llama tokenizers).
* Unit tests: `tests/test_phase1.py` (split offsets, overlap labels, spans
  crossing sentences, source-grouped split, context lookup per task, decode
  check on 20 random sentences).

## Not yet done

* RAGBench loader (`data/ragbench.py`): moved to Phase 5, where the transfer
  test uses it, because its schema can only be checked with Hugging Face
  access (Kaggle).
* Inputs longer than 2,048 tokens are handled in Phase 2 by truncating the
  passages, not the answer.
