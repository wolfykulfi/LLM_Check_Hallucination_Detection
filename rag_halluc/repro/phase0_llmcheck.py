"""Phase 0: run the *unchanged* LLM-Check scores on RAGTruth with a small reader.

What it does
------------
1. Loads N RAGTruth responses (default: 200 Summary answers written by
   llama-2-7b-chat, train split -- the same slice the original
   utils_ragtruth.py used).
2. For each answer, renders [prompt, answer] with the reader's chat template
   and runs ONE teacher-forced forward pass (no generation).
3. Computes LLM-Check's scores on the answer tokens by calling the original
   functions in common_utils.py, untouched:
     logit  -> perplexity, window_logit_entropy (w=1), logit_entropy (top-50)
     attns  -> get_attn_eig_prod for every layer 1..L-1
     hidden -> get_svd_eval for every layer 1..L (optional, slower)
4. Labels each answer 1 if RAGTruth marks any hallucinated span in it.
5. Writes per-answer scores and a per-score AUROC table to CSV, and a plot of
   attention-score AUROC against layer.

Run (from the repo root):
    python -m rag_halluc.repro.phase0_llmcheck --config configs/phase0_llama32_3b.yaml

Differences from the original run_detection_combined.py (all deliberate):
  * small 2024-25 readers via tokenizer.apply_chat_template instead of fastchat
  * attn_implementation="eager" + float16 (works on a free T4)
  * the response span stops at the last answer token, so end-of-turn template
    tokens are not scored
  * scores are computed layer by layer on the GPU instead of first copying
    every attention map to the CPU in float32 (which needs ~2x more RAM)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

# The original LLM-Check code lives at the repository root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import common_utils as llmcheck  # noqa: E402  (original, unmodified LLM-Check code)

from rag_halluc.data.ragtruth import load_ragtruth  # noqa: E402
from rag_halluc.eval.metrics import score_report  # noqa: E402
from rag_halluc.models import encode_conversation, forward_with_internals, load_reader  # noqa: E402

DEFAULTS = {
    "reader": "llama3.2-3b",
    "task": "Summary",
    "generator": "llama-2-7b-chat",
    "split": "train",
    "n_samples": 200,
    "scores": ["logit", "attns"],
    "max_tokens": 2048,
    "data_dir": "data/ragtruth",
    "out_dir": "results/phase0",
    "seed": 0,
    "device": None,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def llmcheck_scores(logits, hidden, attns, input_ids, span, which):
    """Compute the LLM-Check scores for one answer using the original functions.

    The original functions take *lists over samples*, so every argument is
    wrapped in a one-element list and the single result is unpacked with [0].
    """
    out = {}
    if "logit" in which:
        lg = logits[0].float().cpu()  # (seq, vocab)
        ids = input_ids.cpu()  # (1, seq)
        out["perplexity"] = llmcheck.perplexity([lg], [ids], [span])[0]
        out["window_entropy"] = llmcheck.window_logit_entropy([lg], [span], w=1)[0]
        out["logit_entropy"] = float(llmcheck.logit_entropy([lg], [span], top_k=50)[0])

    if "attns" in which:
        # attns[l] is (1, heads, seq, seq). Keep it on the GPU in float32.
        # In fp16 a very small attention weight can round to exactly 0, and
        # log(0) = -inf would wreck the mean, so we floor weights at 1e-8
        # (the original code ran in bf16, which has a wider range).
        per_layer = [a[0] for a in attns]
        for layer in range(1, len(per_layer)):  # the repo skips layer 0
            one = [None] * len(per_layer)
            one[layer] = per_layer[layer].float().clamp_min(1e-8)
            out[f"attn_L{layer}"] = llmcheck.get_attn_eig_prod([one], layer, [span], True)[0]

    if "hidden" in which:
        # hidden[0] is the embedding output, hidden[l] the output of block l.
        for layer in range(1, len(hidden)):
            one = [None] * len(hidden)
            one[layer] = hidden[layer][0].float().cpu()  # (seq, d)
            out[f"hidden_L{layer}"] = llmcheck.get_svd_eval([one], layer, [span], True)[0]
    return out


def run(cfg: dict) -> pd.DataFrame:
    set_seed(cfg["seed"])
    rows = load_ragtruth(cfg["data_dir"], tasks=[cfg["task"]], generators=[cfg["generator"]], split=cfg["split"])
    rows = rows[: cfg["n_samples"]]
    print(f"{len(rows)} answers, {sum(r['label'] for r in rows)} hallucinated")

    model, tok = load_reader(cfg["reader"], device=cfg["device"])
    print(f"Reader {cfg['reader']} on {model.device}, dtype {model.dtype}")

    records = []
    for i, r in enumerate(rows):
        enc = encode_conversation(tok, r["prompt"], r["response"], reader_name=cfg["reader"])
        n_tok = enc.input_ids.shape[1]
        if n_tok > cfg["max_tokens"]:
            print(f"  skip id={r['id']}: {n_tok} tokens > max_tokens")
            continue

        t0 = time.perf_counter()
        logits, hidden, attns = forward_with_internals(model, enc.input_ids)
        scores = llmcheck_scores(logits, hidden, attns, enc.input_ids, list(enc.response_span), cfg["scores"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        del logits, hidden, attns

        records.append(
            {"id": r["id"], "label": r["label"], "n_tokens": n_tok,
             "n_answer_tokens": enc.response_span[1] - enc.response_span[0], "seconds": secs, **scores}
        )
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)} done, last {secs:.2f}s")
    return pd.DataFrame(records)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    meta = {"id", "label", "n_tokens", "n_answer_tokens", "seconds"}
    table = [{"score": c, **score_report(df["label"], df[c])} for c in df.columns if c not in meta]
    return pd.DataFrame(table)


def plot_layers(table: pd.DataFrame, prefix: str, path: str, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    sub = table[table["score"].str.startswith(prefix)].copy()
    if sub.empty:
        return
    sub["layer"] = sub["score"].str.replace(prefix, "").astype(int)
    sub = sub.sort_values("layer")
    plt.figure(figsize=(6, 3.5))
    plt.plot(sub["layer"], sub["auroc"], marker="o", label="AUROC (raw, 1 = hallucinated)")
    plt.plot(sub["layer"], sub["auroc_best"], marker=".", linestyle="--", label="AUROC (best direction)")
    plt.axhline(0.5, color="grey", linewidth=0.8)
    plt.xlabel("layer")
    plt.ylabel("AUROC")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="YAML file; any key in DEFAULTS")
    ap.add_argument("--reader")
    ap.add_argument("--n_samples", type=int)
    ap.add_argument("--task")
    ap.add_argument("--generator")
    ap.add_argument("--scores", nargs="+", choices=["logit", "attns", "hidden"])
    args = ap.parse_args(argv)

    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f) or {})
    for k, v in vars(args).items():
        if k != "config" and v is not None:
            cfg[k] = v

    tag = f"{cfg['reader']}_{cfg['task']}_{cfg['generator']}_{cfg['n_samples']}".replace("/", "-")
    out_dir = os.path.join(cfg["out_dir"], tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    df = run(cfg)
    df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
    table = summarise(df)
    table.to_csv(os.path.join(out_dir, "auroc.csv"), index=False)
    plot_layers(table, "attn_L", os.path.join(out_dir, "attn_auroc_by_layer.png"), f"Attention score, {cfg['reader']}")
    plot_layers(table, "hidden_L", os.path.join(out_dir, "hidden_auroc_by_layer.png"), f"Hidden score, {cfg['reader']}")

    print(f"\nMean seconds per answer: {df['seconds'].mean():.3f}")
    best_attn = table[table["score"].str.startswith("attn_L")].sort_values("auroc_best").tail(3)
    print(table[~table["score"].str.contains("_L")].to_string(index=False))
    print("\nTop-3 attention layers:\n" + best_attn.to_string(index=False))
    print(f"\nWrote {out_dir}/")


if __name__ == "__main__":
    main()
