"""Owned-string C ABI; the shared library loads on first use."""

import ctypes
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import TypedDict, cast


class Response(TypedDict, total=False):
    handle: int
    vocabulary: int
    ids: list[int]
    batch: list[list[int]]
    text: str
    error: str


@lru_cache(maxsize=1)
def library() -> ctypes.CDLL:
    suffix = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
    path = Path(__file__).with_name("_go" + suffix)
    lib = ctypes.CDLL(str(path))
    lib.bntok_call.argtypes = [ctypes.c_char_p]
    # Preserve the allocated address until bntok_free; c_char_p would copy and lose it.
    lib.bntok_call.restype = ctypes.c_void_p
    lib.bntok_free.argtypes = [ctypes.c_void_p]
    lib.bntok_free.restype = None
    return lib


def call(lib: ctypes.CDLL, request: dict[str, object]) -> Response:
    payload = json.dumps(request, ensure_ascii=True, allow_nan=False).encode("ascii")
    pointer = lib.bntok_call(payload)
    if not pointer:
        raise MemoryError("Go tokenizer returned a null response")
    try:
        response = cast(Response, json.loads(ctypes.string_at(pointer)))
    finally:
        lib.bntok_free(pointer)
    if error := response.get("error"):
        raise ValueError(error)
    return response


def release(lib: ctypes.CDLL, handle: int) -> None:
    call(lib, {"op": "close", "handle": handle})
