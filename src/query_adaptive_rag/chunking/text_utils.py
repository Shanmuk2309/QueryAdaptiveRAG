"""Token counting and sentence segmentation with character offsets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class OffsetTokenizer(Protocol):
    def __call__(self, text: str, **kwargs): ...
    def decode(self, token_ids, **kwargs) -> str: ...


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int


def count_tokens(text: str, tokenizer: OffsetTokenizer) -> int:
    """Count tokens using a Hugging Face fast tokenizer."""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def segment_sentences(text: str) -> list[Sentence]:
    """Split on sentence-ending punctuation while retaining exact source offsets."""
    sentences = []
    cursor = 0
    for match in re.finditer(r"[^.!?]+(?:[.!?]+[\"'”’)]*)?|[.!?]+", text, flags=re.DOTALL):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            sentences.append(Sentence(text[start:end], start, end))
        cursor = match.end()
    if not sentences and text.strip():
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())
        sentences.append(Sentence(text[start:end], start, end))
    return sentences
