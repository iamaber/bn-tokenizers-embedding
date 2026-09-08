"""Small, typed Python interface; all text processing stays in Go."""

import os
import weakref
from collections.abc import Sequence

from ._native import call, library, release


def _text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("text must be a string")
    # Python permits isolated surrogates; the Go engine accepts Unicode scalar text.
    value.encode("utf-8")
    return value


def normalize(text: str) -> str:
    """Apply the model's NFC and whitespace normalization in Go."""
    return call(library(), {"op": "normalize", "text": _text(text)})["text"]


class Tokenizer:
    """Load a Go JSON model once; reuse it for single or batch encoding.

    Concurrent encode/decode calls are supported. Close only after concurrent
    work has finished. Prefer a context manager for deterministic native cleanup.
    """

    def __init__(self, model_path: str | os.PathLike[str]) -> None:
        path = os.fspath(model_path)
        if not isinstance(path, str) or "\0" in path:
            raise ValueError("model_path must be a string path without NUL characters")
        self._lib = library()
        response = call(self._lib, {"op": "load", "path": path})
        self._handle = response["handle"]
        self._vocab_size = response["vocabulary"]
        self._finalizer = weakref.finalize(self, release, self._lib, self._handle)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError("tokenizer is closed")

    def encode(self, text: str) -> list[int]:
        self._check_open()
        return call(self._lib, {"op": "encode", "handle": self._handle, "text": _text(text)}).get(
            "ids", []
        )

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Encode a batch in one native call, preserving input order."""
        self._check_open()
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not a single string")
        return call(
            self._lib,
            {"op": "encode_batch", "handle": self._handle, "texts": [_text(s) for s in texts]},
        ).get("batch", [])

    def decode(self, ids: Sequence[int]) -> str:
        self._check_open()
        values = list(ids)
        if any(type(value) is not int for value in values):
            raise TypeError("token IDs must be integers")
        return call(self._lib, {"op": "decode", "handle": self._handle, "ids": values})["text"]

    def close(self) -> None:
        """Release the native model; repeated close calls are harmless."""
        self._finalizer()

    def __enter__(self) -> "Tokenizer":
        self._check_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
