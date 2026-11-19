"""Reader-model registry, loading and prompt encoding.

The "reader" is a small instruct LM that *reads* (question + passages + answer)
in one forward pass with teacher forcing. It never generates; we only look at
its logits, attention maps and hidden states while it reads the answer.

Why these choices (viva notes):
  * attn_implementation="eager": the fast SDPA / flash kernels never build the
    full attention matrix, so they cannot return it. Eager attention does, which
    is what output_attentions=True needs.
  * float16: free Kaggle / Colab T4 GPUs have no fast bfloat16, so fp16 is the
    right half-precision type there. On CPU we fall back to float32.
  * tokenizer.apply_chat_template: every modern instruct model ships its own
    chat format. Using the tokenizer's template replaces the old `fastchat`
    dependency of the original LLM-Check code and works for Qwen, Llama-3.2 and
    Gemma alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

# --------------------------------------------------------------------------
# Registry: short name -> Hugging Face id. Short names are used in configs and
# result file names so tables stay readable.
# --------------------------------------------------------------------------
READERS: Dict[str, str] = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",  # gated: accept licence on HF
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",  # gated: accept licence on HF
    "gemma2-2b": "google/gemma-2-2b-it",  # gated: accept licence on HF
}

# Gemma-2 has no system role in its chat template; we fold the system text
# into the user turn for any model listed here.
NO_SYSTEM_ROLE = {"gemma2-2b"}


def resolve_reader(name: str) -> str:
    """Map a short reader name (or a full HF id / local path) to a loadable id."""
    return READERS.get(name.lower(), name)


def _dtype_kwarg(dtype: torch.dtype) -> dict:
    """transformers>=4.56 renamed `torch_dtype` to `dtype`; support both."""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    if (major, minor) >= (4, 56):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def load_reader(name: str, device: Optional[str] = None, dtype: Optional[torch.dtype] = None):
    """Load a reader model and tokenizer ready for attention / hidden-state extraction.

    Args:
        name: short registry name (see READERS) or any HF model id / local path.
        device: "cuda", "cpu", ... Defaults to cuda when available.
        dtype: defaults to float16 on GPU and float32 on CPU.

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

    model_id = resolve_reader(name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="eager",  # required for output_attentions
        **_dtype_kwarg(dtype),
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


@dataclass
class EncodedExample:
    """One (prompt, response) pair encoded for a single forward pass.

    input_ids: shape (1, seq_len)
    response_span: [start, end) token indices of the response *text* inside
        input_ids. It excludes the chat-template tokens that follow the answer
        (e.g. <|im_end|> or <|eot_id|>), so every score is computed on answer
        tokens only.
    text: the full templated string (useful for debugging / unit tests).
    response_char_span: [start, end) character offsets of the response in `text`.
    offsets: per-token (char_start, char_end) in `text`, used to map any
        character range (a sentence, the passages) to tokens.
    """

    input_ids: torch.Tensor
    response_span: Tuple[int, int]
    text: str
    response_char_span: Tuple[int, int]
    offsets: List[Tuple[int, int]]

    def char_to_token_span(self, c_start: int, c_end: int) -> Tuple[int, int]:
        """Token range [s, e) covering the characters [c_start, c_end) of `text`."""
        return char_span_to_token_span(self.offsets, c_start, c_end)


def char_span_to_token_span(offsets, c_start: int, c_end: int) -> Tuple[int, int]:
    """Smallest token range [s, e) whose tokens overlap characters [c_start, c_end).

    We test for overlap rather than strict containment because a token may
    straddle a boundary (e.g. a leading space merged with the first word).
    """
    hit = [i for i, (a, b) in enumerate(offsets) if b > a and a < c_end and b > c_start]
    if not hit:
        raise ValueError(f"No tokens overlap characters [{c_start}, {c_end}).")
    return min(hit), max(hit) + 1


def encode_conversation(
    tokenizer,
    prompt: str,
    response: str,
    system_prompt: Optional[str] = None,
    reader_name: str = "",
) -> EncodedExample:
    """Render [system?, user=prompt, assistant=response] with the model's chat
    template and find exactly which tokens belong to the response.

    How the response span is found: we render the template to a string, locate
    the response text in it (searching from the end, because the prompt may
    quote the same words), then tokenize with return_offsets_mapping=True and
    keep every token whose character range lies inside the response.
    The same offset trick is reused in Phase 1 to map each *sentence* to tokens.
    """
    response = response.strip()
    messages = []
    user_text = prompt.strip()
    if system_prompt:
        if reader_name.lower() in NO_SYSTEM_ROLE:
            user_text = system_prompt.strip() + "\n\n" + user_text
        else:
            messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": response})

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    c_start = text.rfind(response)
    if c_start < 0:
        raise ValueError("Response text not found in rendered chat template.")
    c_end = c_start + len(response)

    # The template already contains BOS / special tokens, so do not add more.
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
    offsets = [tuple(o) for o in enc["offset_mapping"][0].tolist()]
    t_start, t_end = char_span_to_token_span(offsets, c_start, c_end)

    return EncodedExample(
        input_ids=enc["input_ids"],
        response_span=(t_start, t_end),
        text=text,
        response_char_span=(c_start, c_end),
        offsets=offsets,
    )


@torch.no_grad()
def forward_with_internals(model, input_ids: torch.Tensor):
    """One teacher-forced forward pass returning logits, hidden states and attentions.

    Returns the raw HF tuples:
        logits: (1, seq, vocab)
        hidden_states: tuple of (n_layers + 1) tensors (1, seq, d) -- index 0 is the embeddings
        attentions: tuple of n_layers tensors (1, heads, seq, seq)
    """
    out = model(
        input_ids=input_ids.to(model.device),
        use_cache=False,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
    )
    return out.logits, out.hidden_states, out.attentions


def encode_with_sentences(
    tokenizer,
    prompt: str,
    response: str,
    sentence_char_spans: List[Tuple[int, int]],
    context: Optional[str] = None,
    system_prompt: Optional[str] = None,
    reader_name: str = "",
):
    """Encode one RAG example and map every answer sentence (and the passages) to tokens.

    Args:
        sentence_char_spans: [start, end) offsets of each sentence in the *raw*
            `response` string (as returned by rag_halluc.claims.split).
        context: the passages text; it must appear verbatim inside `prompt`.

    Returns:
        enc: EncodedExample for the whole conversation
        sent_spans: list of token ranges [s, e), one per sentence
        context_span: token range [c0, c1) of the passages, or None
    """
    enc = encode_conversation(tokenizer, prompt, response, system_prompt, reader_name)
    # encode_conversation strips the response; convert raw offsets to `enc.text` offsets.
    lead = len(response) - len(response.lstrip())
    shift = enc.response_char_span[0] - lead
    sent_spans = [enc.char_to_token_span(s + shift, e + shift) for s, e in sentence_char_spans]

    context_span = None
    if context:
        ctx = context.strip()
        c0 = enc.text.find(ctx)
        if c0 >= 0 and c0 < enc.response_char_span[0]:
            context_span = enc.char_to_token_span(c0, c0 + len(ctx))
    return enc, sent_spans, context_span
