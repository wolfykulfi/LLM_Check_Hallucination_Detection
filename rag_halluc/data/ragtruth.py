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
response-level label (1 if any span is labelled). Phase 1 extends this module
with the source-grouped validation split and sentence labels.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Dict, List, Optional

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
