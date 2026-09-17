"""Small, typed Python interface; all text processing stays in Go."""

import os
from collections.abc import Sequence

from ._native import NativeHandle
from ._native import normalize as _normalize


def _text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("text must be a string")
    return value


def normalize(text: str) -> str:
    """Apply the model's NFC and whitespace normalization in Go."""
    return _normalize(_text(text))


class Tokenizer:
    """Load a Go JSON model once; reuse it for single or batch encoding.

    Concurrent encode/decode calls are supported. Close only after concurrent
    work has finished. Prefer a context manager for deterministic native cleanup.
    """

    def __init__(self, model_path: str | os.PathLike[str]) -> None:
        path = os.fspath(model_path)
        if not isinstance(path, str) or "\0" in path:
            raise ValueError("model_path must be a string path without NUL characters")
        self._native = NativeHandle(path)

    @property
    def vocab_size(self) -> int:
        return self._native.vocab_size

    @property
    def closed(self) -> bool:
        return self._native.closed

    def encode(self, text: str) -> list[int]:
        self._native.check_open()
        return self._native.encode([_text(text).encode("utf-8")])[0]

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Encode a batch in one native call, preserving input order."""
        self._native.check_open()
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not a single string")
        return self._native.encode([_text(s).encode("utf-8") for s in texts])

    def decode(self, ids: Sequence[int]) -> str:
        self._native.check_open()
        values = list(ids)
        if any(type(value) is not int for value in values):
            raise TypeError("token IDs must be integers")
        return self._native.decode(values)

    def close(self) -> None:
        """Release the native model; repeated close calls are harmless."""
        self._native.close()

    def __enter__(self) -> "Tokenizer":
        self._native.check_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
