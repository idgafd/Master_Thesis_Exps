"""Character vocabulary for CTC-based ASR (English)."""

import json
import re
from pathlib import Path
from typing import List


_KEEP_RE = re.compile(r"[^a-z ']")


def normalize_text(text: str) -> str:
    """Lowercase + strip non-English chars (keep a-z, space, apostrophe)."""
    text = text.lower()
    text = _KEEP_RE.sub("", text)
    text = " ".join(text.split())
    return text


class CharVocab:
    """Character-level vocabulary with CTC blank at index 0.

    Default English vocab: blank + space + apostrophe + a-z = 29 tokens.
    """

    BLANK = "<blank>"

    def __init__(self, token_to_id: dict):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> List[int]:
        return [self.token_to_id[c] for c in text if c in self.token_to_id]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id_to_token[i] for i in ids if i in self.id_to_token and i != 0)

    @classmethod
    def build_english(cls) -> "CharVocab":
        """Build the canonical English character vocab."""
        token_to_id = {cls.BLANK: 0, " ": 1, "'": 2}
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
            token_to_id[c] = i + 3
        return cls(token_to_id)

    @classmethod
    def from_texts(cls, texts: List[str]) -> "CharVocab":
        chars = sorted(set("".join(texts)))
        token_to_id = {cls.BLANK: 0}
        for c in chars:
            token_to_id[c] = len(token_to_id)
        return cls(token_to_id)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharVocab":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))
