"""Phase 1 tests: sentence split, labels, validation split, token mapping.

CPU only, no internet (spaCy's en_core_web_sm must be installed:
python -m spacy download en_core_web_sm).
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tiny_reader import make_tokenizer  # noqa: E402

from rag_halluc.claims.split import split_sentences  # noqa: E402
from rag_halluc.data.ragtruth import assign_splits, build_tables, context_and_question, sentence_rows  # noqa: E402
from rag_halluc.models import encode_with_sentences  # noqa: E402

spacy = pytest.importorskip("spacy")

ARTICLE = "The cat sat on the mat. It was a sunny day in Paris. The dog slept."
RESPONSE = "The cat sat on the mat in Paris. It was rainy.\n\n1. The dog slept.\n2. The bird sang."


def _row(response=RESPONSE, labels=None, **kw):
    r = {"id": "0", "source_id": "1", "task_type": "Summary", "generator": "gpt-4-0613", "split": "train",
         "quality": "good", "source_info": ARTICLE,
         "prompt": "Summarize the following news within 50 words:\n" + ARTICLE + "\noutput:",
         "response": response, "labels": labels or [], "label": int(bool(labels))}
    r.update(kw)
    return r


def test_split_offsets_are_exact():
    sents = split_sentences(RESPONSE)
    assert [s.text for s in sents] == ["The cat sat on the mat in Paris.", "It was rainy.",
                                       "1. The dog slept.", "2. The bird sang."]
    assert all(RESPONSE[s.start:s.end] == s.text for s in sents)


def test_sentence_label_by_overlap():
    start = RESPONSE.index("rainy")
    labels = [{"start": start, "end": start + 5, "text": "rainy", "label_type": "Evident Conflict"}]
    rows = sentence_rows(_row(labels=labels))
    assert [r["label"] for r in rows] == [0, 1, 0, 0]
    assert rows[1]["is_evident"] == 1 and rows[1]["is_conflict"] == 1 and rows[1]["is_baseless"] == 0


def test_span_crossing_two_sentences_labels_both():
    start = RESPONSE.index("Paris")
    end = RESPONSE.index("rainy") + 5
    rows = sentence_rows(_row(labels=[{"start": start, "end": end, "label_type": "Subtle Baseless Info"}]))
    assert [r["label"] for r in rows] == [1, 1, 0, 0]
    assert rows[0]["is_subtle"] == 1 and rows[0]["is_baseless"] == 1


def test_validation_split_is_by_source():
    rows = [_row(id=str(i), source_id=str(i // 6), split="train" if i < 120 else "test") for i in range(150)]
    assign_splits(rows, val_frac=0.1, seed=0)
    by_src = {}
    for r in rows:
        by_src.setdefault(r["source_id"], set()).add(r["split3"])
    assert all(len(v) == 1 for v in by_src.values())  # a source never straddles splits
    n_val_sources = sum(1 for v in by_src.values() if v == {"val"})
    assert n_val_sources == 2  # 10% of 20 train sources
    assert all(r["split3"] == "test" for r in rows if r["split"] == "test")


def test_context_found_in_prompt_for_each_task():
    qa = _row(task_type="QA", source_info={"question": "q?", "passages": "passage 1: abc"},
              prompt="Briefly answer the following question:\nq?\npassages:\npassage 1: abc\noutput:")
    d2t = _row(task_type="Data2txt", source_info={"name": "X", "stars": 4.0})
    d2t["prompt"] = "Structured data:\n" + str(d2t["source_info"]) + "\nOverview:"
    for r in (_row(), qa, d2t):
        ctx, _ = context_and_question(r)
        assert ctx.strip() in r["prompt"]


def test_build_tables_shapes():
    resp, sents = build_tables([_row()])
    assert len(resp) == 1 and len(sents) == 4
    assert resp.loc[0, "context"] == ARTICLE


def test_token_mapping_decodes_to_sentences():
    """The Phase 1 unit test from the build guide: decoded tokens == sentence text."""
    tok = make_tokenizer()
    rng = random.Random(0)
    words = "the cat sat on a mat in Paris . it was sunny and the dog slept".split()
    for _ in range(20):
        sents = [" ".join(rng.choices(words, k=rng.randint(3, 8))).capitalize() + "." for _ in range(3)]
        response = "  " + " ".join(sents)
        spans = [(s.start, s.end) for s in split_sentences(response)]
        enc, tok_spans, ctx = encode_with_sentences(tok, _row()["prompt"], response, spans, context=ARTICLE)
        for (cs, ce), (ts, te) in zip(spans, tok_spans):
            assert tok.decode(enc.input_ids[0, ts:te]).strip() == response[cs:ce]
        c0, c1 = ctx
        assert tok.decode(enc.input_ids[0, c0:c1]).strip() == ARTICLE
        assert c1 <= enc.response_span[0]  # passages come before the answer
