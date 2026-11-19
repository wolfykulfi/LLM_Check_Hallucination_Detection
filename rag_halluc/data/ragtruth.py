"""RAGTruth: download and join.

RAGTruth (Niu et al., ACL 2024) ships two JSONL files in its GitHub repo
(github.com/ParticleMedia/RAGTruth, folder dataset/):

  response.jsonl     one row per generated answer (~17.8k rows):
                     id, source_id, model (one of 6 generators), split
                     (train/test), response, labels (list of hallucinated
                     character spans with a label_type), quality, ...
  source_info.jsonl  one row per source (~3k rows): source_id, task_type
                     (QA / Data2txt / Summary), source_info (passages), prompt

Every source has an answer from each of the six generators, so the two files
are joined on source_id. Phase 0 only needs the joined rows and a
response-level label (1 if any span is labelled).

Phase 1 adds:
  * context_and_question(): the passages text exactly as it appears in the
    prompt, and the question (QA only)
  * assign_splits(): a validation split of 10% of train *sources*
  * sentence_rows(): one row per answer sentence with its hallucination label
  * build_tables(): the response table and the sentence table
  * label_stats(): per-task counts, to compare with the RAGTruth paper
"""

from __future__ import annotations

import json
import os
import random
import urllib.request
from typing import Dict, List, Optional, Tuple

RAGTRUTH_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/{name}"
FILES = ("response.jsonl", "source_info.jsonl")
DEFAULT_DIR = os.path.join("data", "ragtruth")

TASKS = ("QA", "Data2txt", "Summary")
GENERATORS = (
    "gpt-4-0613",
    "gpt-3.5-turbo-0613",
    "mistral-7B-instruct",
    "llama-2-7b-chat",
    "llama-2-13b-chat",
    "llama-2-70b-chat",
)


def download_ragtruth(data_dir: str = DEFAULT_DIR, overwrite: bool = False) -> str:
    """Download the two RAGTruth JSONL files into data_dir (about 37 MB)."""
    os.makedirs(data_dir, exist_ok=True)
    for name in FILES:
        path = os.path.join(data_dir, name)
        if os.path.exists(path) and not overwrite:
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(RAGTRUTH_URL.format(name=name), path)
    return data_dir


def _read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_ragtruth(
    data_dir: str = DEFAULT_DIR,
    tasks: Optional[List[str]] = None,
    generators: Optional[List[str]] = None,
    split: Optional[str] = None,
    download: bool = True,
) -> List[Dict]:
    """Return joined RAGTruth rows, optionally filtered.

    Each returned dict has:
        id, source_id, task_type, generator, split, prompt, source_info,
        response, labels (list of span dicts), quality,
        label (int: 1 if the response contains any labelled hallucination)

    The order of rows follows response.jsonl, so "the first N rows" is
    deterministic and matches what the original LLM-Check loader saw.
    """
    if download:
        download_ragtruth(data_dir)
    sources = {s["source_id"]: s for s in _read_jsonl(os.path.join(data_dir, "source_info.jsonl"))}

    rows = []
    for r in _read_jsonl(os.path.join(data_dir, "response.jsonl")):
        src = sources[r["source_id"]]
        row = {
            "id": r["id"],
            "source_id": r["source_id"],
            "task_type": src["task_type"],
            "generator": r["model"],
            "split": r["split"],
            "prompt": src["prompt"],
            "source_info": src["source_info"],
            "response": r["response"],
            "labels": r["labels"],
            "quality": r.get("quality", "good"),
            "label": int(len(r["labels"]) > 0),
        }
        if tasks and row["task_type"] not in tasks:
            continue
        if generators and row["generator"] not in generators:
            continue
        if split and row["split"] != split:
            continue
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Phase 1: context, splits, sentence labels
# --------------------------------------------------------------------------


def context_and_question(row: Dict) -> Tuple[str, str]:
    """Return (context, question) for one joined row.

    context is the evidence text exactly as it appears inside the prompt, so
    it can be located there and mapped to tokens:
      Summary   the news article (source_info is a string)
      QA        source_info["passages"]; question = source_info["question"]
      Data2txt  the structured JSON record, which the prompt embeds as str(dict)
    """
    si = row["source_info"]
    if row["task_type"] == "Summary":
        return si, ""
    if row["task_type"] == "QA":
        return si["passages"], si["question"]
    return str(si), ""


def assign_splits(rows: List[Dict], val_frac: float = 0.1, seed: int = 0) -> List[Dict]:
    """Add row["split3"] in {"train", "val", "test"}.

    The validation set is 10% of the *source documents* of the official train
    split, not 10% of the rows. Each source has answers from six generators;
    a row-level split would put the same passage on both sides and leak.
    The official test split is kept as is.
    """
    train_sources = sorted({r["source_id"] for r in rows if r["split"] == "train"}, key=int)
    rng = random.Random(seed)
    val_sources = set(rng.sample(train_sources, round(val_frac * len(train_sources))))
    for r in rows:
        if r["split"] == "test":
            r["split3"] = "test"
        else:
            r["split3"] = "val" if r["source_id"] in val_sources else "train"
    return rows


def _span_flags(label_types: List[str]) -> Dict[str, int]:
    """RAGTruth label_type is e.g. "Evident Conflict" or "Subtle Baseless Info"."""
    lt = " ".join(label_types).lower()
    return {
        "is_evident": int("evident" in lt),
        "is_subtle": int("subtle" in lt),
        "is_conflict": int("conflict" in lt),
        "is_baseless": int("baseless" in lt),
    }


def sentence_rows(row: Dict, nlp=None) -> List[Dict]:
    """Split one answer into sentences and label each one.

    A sentence is labelled 1 if its character range [start, end) overlaps any
    labelled hallucination span. We also keep which kinds of span it overlaps
    (evident / subtle, conflict / baseless) for error analysis, and whether
    all overlapping spans are marked implicit_true (correct but not in the
    context), so that variant can be excluded in an ablation.
    """
    from rag_halluc.claims.split import split_sentences

    sents = split_sentences(row["response"], nlp)
    out = []
    for k, sent in enumerate(sents):
        hits = [sp for sp in row["labels"] if sp["start"] < sent.end and sp["end"] > sent.start]
        types = sorted({sp.get("label_type", "") for sp in hits})
        out.append(
            {
                "id": row["id"],
                "source_id": row["source_id"],
                "task_type": row["task_type"],
                "generator": row["generator"],
                "split": row.get("split3", row["split"]),
                "sentence_idx": k,
                "n_sentences": len(sents),
                "sent_start": sent.start,
                "sent_end": sent.end,
                "sentence": sent.text,
                "label": int(bool(hits)),
                "n_spans": len(hits),
                "span_types": "|".join(types),
                **_span_flags(types),
                "implicit_true_only": int(bool(hits) and all(sp.get("implicit_true", False) for sp in hits)),
            }
        )
    return out


def build_tables(rows: List[Dict], nlp=None):
    """Return (responses_df, sentences_df) as pandas DataFrames.

    responses_df: one row per answer with prompt, context, question, response
                  and labels (the context is stored once here, not per sentence).
    sentences_df: one row per answer sentence with its label.
    """
    import pandas as pd

    resp, sents = [], []
    for r in rows:
        context, question = context_and_question(r)
        resp.append(
            {
                "id": r["id"],
                "source_id": r["source_id"],
                "task_type": r["task_type"],
                "generator": r["generator"],
                "split": r.get("split3", r["split"]),
                "quality": r["quality"],
                "prompt": r["prompt"],
                "context": context,
                "question": question,
                "response": r["response"],
                "spans_json": json.dumps(r["labels"]),
                "label": r["label"],
            }
        )
        sents.extend(sentence_rows(r, nlp))
    return pd.DataFrame(resp), pd.DataFrame(sents)


def label_stats(responses_df, sentences_df):
    """Per task and official split: answers, hallucinated answers, sentences,
    hallucinated sentences. Response counts are comparable to the RAGTruth
    paper's statistics table; sentence counts are new to this project."""
    import pandas as pd

    r = responses_df.assign(official=responses_df["split"].replace({"val": "train"}))
    s = sentences_df.assign(official=sentences_df["split"].replace({"val": "train"}))
    a = r.groupby(["task_type", "official"]).agg(responses=("id", "size"), halluc_responses=("label", "sum"))
    b = s.groupby(["task_type", "official"]).agg(sentences=("id", "size"), halluc_sentences=("label", "sum"))
    out = a.join(b).reset_index()
    out["halluc_response_rate"] = (out["halluc_responses"] / out["responses"]).round(3)
    out["halluc_sentence_rate"] = (out["halluc_sentences"] / out["sentences"]).round(3)
    return pd.DataFrame(out)
