"""Phase 0 tests. Run from the repo root:  python -m pytest tests -q

They use a tiny random model and tokenizer (tests/tiny_reader.py), so they run
on CPU in a few seconds and need no internet.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tiny_reader import make_model, make_tokenizer  # noqa: E402

from rag_halluc.data.ragtruth import load_ragtruth  # noqa: E402
from rag_halluc.eval.metrics import score_report  # noqa: E402
from rag_halluc.models import encode_conversation, forward_with_internals  # noqa: E402
from rag_halluc.repro import phase0_llmcheck as p0  # noqa: E402

PROMPT = "Summarize the following news within 50 words: The cat sat on the mat. It was a sunny day in Paris."
RESPONSE = "The cat sat on the mat in Paris. It was sunny."


@pytest.fixture(scope="module")
def tok():
    return make_tokenizer()


@pytest.fixture(scope="module")
def model(tok):
    return make_model(len(tok))


def test_response_span_decodes_to_response(tok):
    enc = encode_conversation(tok, PROMPT, "  " + RESPONSE + "\n")
    s, e = enc.response_span
    assert tok.decode(enc.input_ids[0, s:e]).strip() == RESPONSE
    # the token right after the answer is the end-of-turn marker, not answer text
    assert tok.convert_ids_to_tokens(int(enc.input_ids[0, e])) == "<|im_end|>"


def test_response_found_even_if_prompt_contains_it(tok):
    prompt = "Repeat this: " + RESPONSE
    enc = encode_conversation(tok, prompt, RESPONSE)
    # rfind picks the occurrence inside the assistant turn, i.e. the later one
    assert enc.response_char_span[0] > enc.text.find(RESPONSE)


def test_llmcheck_scores_are_finite(tok, model):
    enc = encode_conversation(tok, PROMPT, RESPONSE)
    logits, hidden, attns = forward_with_internals(model, enc.input_ids)
    out = p0.llmcheck_scores(logits, hidden, attns, enc.input_ids, list(enc.response_span), ["logit", "attns", "hidden"])
    n_layers = model.config.num_hidden_layers
    expected = {"perplexity", "window_entropy", "logit_entropy"}
    expected |= {f"attn_L{l}" for l in range(1, n_layers)}  # repo skips attention layer 0
    expected |= {f"hidden_L{l}" for l in range(1, n_layers + 1)}
    assert set(out) == expected
    assert all(np.isfinite(v) for v in out.values())
    assert out["perplexity"] > 1.0  # a random model is surprised by everything


def test_attention_logdet_equals_sum_of_log_diagonal(model, tok):
    """Viva point: for a lower-triangular matrix, det = product of the diagonal,
    so log det = sum of log diagonal entries. LLM-Check uses the *mean*."""
    import torch

    enc = encode_conversation(tok, PROMPT, RESPONSE)
    _, _, attns = forward_with_internals(model, enc.input_ids)
    s, e = enc.response_span
    a = attns[1][0, 0, s:e, s:e].double()
    assert torch.allclose(torch.triu(a, diagonal=1), torch.zeros_like(a))  # causal => lower-triangular
    logdet = torch.linalg.slogdet(a).logabsdet
    assert torch.isclose(logdet, torch.log(torch.diagonal(a)).sum(), atol=1e-6)


def test_score_report_direction():
    labels = np.array([0, 0, 0, 1, 1, 1])
    good = score_report(labels, [0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    flipped = score_report(labels, [0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert good["auroc"] == 1.0 and good["direction"] == "+"
    assert flipped["auroc"] == 0.0 and flipped["direction"] == "-" and flipped["auroc_best"] == 1.0
    assert flipped["tpr_at_5fpr"] == 1.0


def _write_fixture(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    sources = [
        {"source_id": "s1", "task_type": "Summary", "source": "CNN/DM", "source_info": "The cat sat on the mat.",
         "prompt": PROMPT},
        {"source_id": "s2", "task_type": "QA", "source": "MARCO",
         "source_info": {"question": "q", "passages": "p"}, "prompt": "Briefly answer the following question: q"},
    ]
    responses = []
    for i, (sid, gen, split, labels) in enumerate([
        ("s1", "llama-2-7b-chat", "train", []),
        ("s1", "llama-2-7b-chat", "train", [{"start": 0, "end": 7, "text": "The cat", "label_type": "Evident Conflict"}]),
        ("s1", "gpt-4-0613", "train", []),
        ("s2", "llama-2-7b-chat", "test", []),
        ("s1", "llama-2-7b-chat", "train", [{"start": 8, "end": 11, "text": "sat", "label_type": "Subtle Conflict"}]),
        ("s1", "llama-2-7b-chat", "train", []),
    ]):
        responses.append({"id": str(i), "source_id": sid, "model": gen, "temperature": 0.7, "labels": labels,
                          "split": split, "quality": "good", "response": RESPONSE})
    for name, rows in [("source_info.jsonl", sources), ("response.jsonl", responses)]:
        with open(os.path.join(data_dir, name), "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")


def test_ragtruth_join_and_filters(tmp_path):
    d = str(tmp_path / "rt")
    _write_fixture(d)
    rows = load_ragtruth(d, tasks=["Summary"], generators=["llama-2-7b-chat"], split="train", download=False)
    assert [r["id"] for r in rows] == ["0", "1", "4", "5"]
    assert [r["label"] for r in rows] == [0, 1, 1, 0]
    assert rows[0]["prompt"] == PROMPT and rows[0]["task_type"] == "Summary"


def test_phase0_end_to_end(tmp_path, tok, model, monkeypatch):
    d = str(tmp_path / "rt")
    _write_fixture(d)
    monkeypatch.setattr(p0, "load_reader", lambda name, device=None: (model, tok))
    cfg = dict(p0.DEFAULTS, reader="tiny", data_dir=d, n_samples=4, device="cpu", out_dir=str(tmp_path / "out"))
    df = p0.run(cfg)
    assert len(df) == 4 and df["label"].tolist() == [0, 1, 1, 0]
    table = p0.summarise(df)
    assert {"perplexity", "attn_L1", "attn_L2"} <= set(table["score"])
    assert table["auroc"].between(0, 1).all()


def test_phase0_cli_writes_csvs(tmp_path, tok, model, monkeypatch):
    d = str(tmp_path / "rt")
    _write_fixture(d)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(f"reader: tiny\ndata_dir: {d}\nout_dir: {tmp_path / 'out'}\nn_samples: 4\ndevice: cpu\n")
    monkeypatch.setattr(p0, "load_reader", lambda name, device=None: (model, tok))
    p0.main(["--config", str(cfg_path)])
    out = tmp_path / "out" / "tiny_Summary_llama-2-7b-chat_4"
    assert (out / "scores.csv").exists() and (out / "auroc.csv").exists() and (out / "config.yaml").exists()
