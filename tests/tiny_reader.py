"""A tiny, randomly initialised reader + tokenizer built entirely offline.

Tests use it so they run on CPU in seconds with no Hugging Face download.
The tokenizer has a ChatML-style template (like Qwen2.5) so that
encode_conversation is exercised exactly as with a real model.
"""

from __future__ import annotations

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

CHATML = (
    "{% for m in messages %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

CORPUS = [
    "Summarize the following news within 50 words: The cat sat on the mat. It was a sunny day in Paris.",
    "Briefly answer the following question: Who wrote the report? passage 1: The report was written by Ana.",
    "The Anne Frank House has revealed new research about the Bergen-Belsen camp in 1945.",
    "0123456789 ,.;:!?'\"()[]{}-_/ abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ",
]


def make_tokenizer() -> PreTrainedTokenizerFast:
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator(CORPUS * 5, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    fast.chat_template = CHATML
    return fast


def make_model(vocab_size: int) -> LlamaForCausalLM:
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
    )
    cfg._attn_implementation = "eager"
    model = LlamaForCausalLM(cfg).eval()
    model.requires_grad_(False)
    return model
