"""Split an answer into sentences ("claims"), keeping character offsets.

Sentences are the default claim unit because RAGTruth's hallucination labels
are character spans: a sentence's [start, end) range can be compared with the
labelled spans directly (see rag_halluc/data/ragtruth.py).

How the split works:
  1. Cut the text at line breaks first. Data2txt and QA answers often contain
     bullet lists or headings with no full stop, and spaCy's parser tends to
     glue such lines together.
  2. Run spaCy's sentence segmenter on each non-empty line.
  3. Merge a bare list marker ("1.", "*", "-") into the sentence after it;
     spaCy otherwise returns "1." as a sentence of its own.
  4. Trim surrounding whitespace and return (start, end, text) where
     text == answer[start:end] exactly.

An optional LM-based decomposer into atomic claims is a later extension.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List, NamedTuple


class Sentence(NamedTuple):
    start: int  # character offset in the original answer (inclusive)
    end: int  # character offset (exclusive)
    text: str  # == answer[start:end]


@lru_cache(maxsize=1)
def get_nlp(model: str = "en_core_web_sm"):
    """Load spaCy once. Only the components needed for sentence boundaries run."""
    import spacy

    return spacy.load(model, disable=["ner", "lemmatizer", "attribute_ruler"])


def _trim(text: str, start: int, end: int):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


LIST_MARKER = re.compile(r"^(\d{1,3}[.)]|[*+\-\u2022])$")


def split_sentences(text: str, nlp=None) -> List[Sentence]:
    """Return the sentences of `text` with exact character offsets."""
    nlp = nlp or get_nlp()
    out: List[Sentence] = []
    pending_start = None  # start of a bare list marker waiting for its sentence
    for line in re.finditer(r"[^\n]+", text):
        l_start = line.start()
        if not line.group().strip():
            continue
        for sent in nlp(line.group()).sents:
            s, e = _trim(text, l_start + sent.start_char, l_start + sent.end_char)
            if e <= s:
                continue
            if LIST_MARKER.match(text[s:e]):
                pending_start = s if pending_start is None else pending_start
                continue
            if pending_start is not None:
                s, pending_start = pending_start, None
            out.append(Sentence(s, e, text[s:e]))
    if pending_start is not None:  # a trailing marker with nothing after it
        out.append(Sentence(pending_start, len(text.rstrip()), text[pending_start:len(text.rstrip())]))
    return out
