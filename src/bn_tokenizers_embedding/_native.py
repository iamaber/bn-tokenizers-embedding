"""Owned-buffer C interfaces; the shared library loads on first use."""

import ctypes
import json
import struct
import sys
from functools import lru_cache
from pathlib import Path
from typing import TypedDict, cast


class Response(TypedDict, total=False):
    handle: int
    vocabulary: int
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
    lib.bntok_encode.argtypes = [
        ctypes.c_ulonglong,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.bntok_encode.restype = ctypes.c_void_p
    return lib


def call(lib: ctypes.CDLL, request: dict[str, object]) -> Response:
    payload = json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")
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


def encode(lib: ctypes.CDLL, handle: int, texts: list[bytes]) -> list[list[int]]:
    """Length-prefixed UTF-8 input and uint32 output; buffers are owned by each side."""
    chunks = []
    size = 0
    for data in texts:
        size += 4 + len(data)
        if size > 2**31 - 1:
            raise ValueError("encode input exceeds 2 GiB")
        chunks.extend((struct.pack("<I", len(data)), data))
    payload = b"".join(chunks)
    length = ctypes.c_size_t()
    pointer = lib.bntok_encode(handle, payload, len(payload), ctypes.byref(length))
    if not pointer:
        raise MemoryError("Go tokenizer returned a null response")
    try:
        response = ctypes.string_at(pointer, length.value)
    finally:
        lib.bntok_free(pointer)
    if not response or response[0] not in (0, 1):
        raise RuntimeError("invalid native encode response")
    if response[0]:
        raise ValueError(response[1:].decode("utf-8"))
    result = []
    offset = 1
    while offset < len(response):
        if len(response) - offset < 4:
            raise RuntimeError("truncated native encode response")
        count = struct.unpack_from("<I", response, offset)[0]
        offset += 4
        if count > (len(response) - offset) // 4:
            raise RuntimeError("truncated native token IDs")
        result.append(list(struct.unpack_from(f"<{count}I", response, offset)))
        offset += 4 * count
    if len(result) != len(texts):
        raise RuntimeError("native encode response count mismatch")
    return result
