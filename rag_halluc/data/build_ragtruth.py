"""Phase 1: build the RAGTruth response and sentence tables.

Run from the repo root (CPU only, about 2 minutes):
    python -m rag_halluc.data.build_ragtruth
    python -m rag_halluc.data.build_ragtruth --check_reader qwen2.5-1.5b   # + token-mapping check

Writes:
    data/processed/ragtruth_responses.parquet   one row per answer (prompt, context, question, response, label, split)
    data/processed/ragtruth_sentences.parquet   one row per answer sentence (char span, label, span types, split)
    results/phase1/ragtruth_stats.csv           per task x official split: answers / sentences, hallucinated counts
    results/phase1/ragtruth_splits.csv          answers, sources and hallucinated answers per train / val / test
    results/phase1/token_check_<reader>.csv     (with --check_reader) decoded tokens vs sentence text

Sanity checks printed at the end (the "done when" of Phase 1):
  * every hallucinated answer has at least one hallucinated sentence
  * no source_id appears in both train and val
  * response counts per task match the RAGTruth paper (QA 5,934; Data2txt 6,198; Summary 5,658)
"""

from __future__ import annotations

import argparse
import os
import random

import pandas as pd

from rag_halluc.claims.split import get_nlp
from rag_halluc.data.ragtruth import assign_splits, build_tables, label_stats, load_ragtruth


def token_check(responses: pd.DataFrame, sentences: pd.DataFrame, reader: str, n: int, seed: int) -> pd.DataFrame:
    """Decode the tokens mapped to n random sentences and compare with the sentence text.

    A token can straddle a sentence boundary (e.g. ".\\n" or " The"), so the
    decoded text may carry an extra space or punctuation mark at either end;
    we count a match when the sentence text is contained in the decoded text
    and the decoded text is at most a few characters longer.
    """
    from transformers import AutoTokenizer

    from rag_halluc.models import encode_with_sentences, resolve_reader

    tok = AutoTokenizer.from_pretrained(resolve_reader(reader))
    rng = random.Random(seed)
    picks = rng.sample(range(len(sentences)), n)
    by_id = responses.set_index("id")
    out = []
    for i in picks:
        s = sentences.iloc[i]
        r = by_id.loc[s["id"]]
        enc, spans, ctx = encode_with_sentences(
            tok, r["prompt"], r["response"], [(s["sent_start"], s["sent_end"])], context=r["context"], reader_name=reader
        )
        decoded = tok.decode(enc.input_ids[0, spans[0][0]: spans[0][1]])
        ok = s["sentence"] in decoded and len(decoded.strip()) - len(s["sentence"]) <= 3
        out.append({"id": s["id"], "sentence_idx": s["sentence_idx"], "sentence": s["sentence"],
                    "decoded": decoded, "match": ok, "context_found": ctx is not None})
    return pd.DataFrame(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build RAGTruth sentence tables (Phase 1).")
    ap.add_argument("--data_dir", default="data/ragtruth")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--results_dir", default="results/phase1")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check_reader", help="reader name for the token-mapping check (needs HF access)")
    ap.add_argument("--n_check", type=int, default=20)
    args = ap.parse_args(argv)

    rows = assign_splits(load_ragtruth(args.data_dir), val_frac=args.val_frac, seed=args.seed)
    responses, sentences = build_tables(rows, get_nlp())

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    responses.to_parquet(os.path.join(args.out_dir, "ragtruth_responses.parquet"), index=False)
    sentences.to_parquet(os.path.join(args.out_dir, "ragtruth_sentences.parquet"), index=False)

    stats = label_stats(responses, sentences)
    stats.to_csv(os.path.join(args.results_dir, "ragtruth_stats.csv"), index=False)
    splits = responses.groupby("split").agg(
        responses=("id", "size"), sources=("source_id", "nunique"), halluc_responses=("label", "sum")
    ).reset_index()
    splits.to_csv(os.path.join(args.results_dir, "ragtruth_splits.csv"), index=False)

    print(stats.to_string(index=False))
    print()
    print(splits.to_string(index=False))

    # ---- sanity checks ----
    sent_any = sentences.groupby("id")["label"].max()
    resp_lab = responses.set_index("id")["label"]
    missed = int(((resp_lab == 1) & (sent_any.reindex(resp_lab.index).fillna(0) == 0)).sum())
    leak = set(responses.loc[responses.split == "train", "source_id"]) & set(
        responses.loc[responses.split == "val", "source_id"])
    print(f"\nHallucinated answers with no hallucinated sentence: {missed}")
    print(f"Sources in both train and val: {len(leak)}")
    print(f"Answers with no sentences: {int((~resp_lab.index.isin(sentences['id'])).sum())}")

    if args.check_reader:
        chk = token_check(responses, sentences, args.check_reader, args.n_check, args.seed)
        path = os.path.join(args.results_dir, f"token_check_{args.check_reader}.csv")
        chk.to_csv(path, index=False)
        print(f"Token mapping: {chk['match'].sum()}/{len(chk)} sentences decode correctly; "
              f"passages located in {chk['context_found'].sum()}/{len(chk)}  -> {path}")


if __name__ == "__main__":
    main()
