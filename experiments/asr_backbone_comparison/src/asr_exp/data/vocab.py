"""Character vocabulary for CTC-based ASR."""

import json
import re
from pathlib import Path
from typing import List


# Characters kept after normalization (space, apostrophe, Ukrainian Cyrillic)
_KEEP_RE = re.compile(r"[^а-яіїєґ ']", re.UNICODE)


def normalize_text(text: str) -> str:
    """Lowercase + strip non-Ukrainian chars (keep space and apostrophe)."""
    text = text.lower()
    text = _KEEP_RE.sub("", text)
    text = " ".join(text.split())   # collapse whitespace
    return text


class CharVocab:
    """Character-level vocabulary with CTC blank at index 0.

    JSON format: {"<blank>": 0, " ": 1, "'": 2, "а": 3, ...}
    """

    BLANK = "<blank>"

    def __init__(self, token_to_id: dict):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> List[int]:
        """Encode normalized text to list of token ids (unknown chars dropped)."""
        return [self.token_to_id[c] for c in text if c in self.token_to_id]

    def decode(self, ids: List[int]) -> str:
        """Decode token ids to string, skipping blank (0) and unknowns."""
        return "".join(self.id_to_token[i] for i in ids if i in self.id_to_token and i != 0)

    @classmethod
    def from_texts(cls, texts: List[str]) -> "CharVocab":
        """Build vocab from a list of already-normalized texts."""
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
