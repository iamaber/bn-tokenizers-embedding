"""Local E5 encoder inputs and explicit vocabulary transfer for research runs."""

import json
import shutil
import time
from pathlib import Path

import torch
from prepare import sha256
from torch import nn
from transformers import AutoModel, AutoTokenizer


def transfer_embeddings(model, original_tokenizer, path: Path) -> dict:
    """Decomposition initialization; not an implementation of FOCUS or ZeTT."""
    started = time.perf_counter()
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact["version"] != 1 or artifact.get("space_fusion"):
        raise ValueError("first transfer experiment requires an unfused version-1 tokenizer")
    if model.config.model_type != "bert":
        raise ValueError("transfer currently supports the pinned E5-small BertModel only")
    old = model.get_input_embeddings().weight.detach()
    size = 260 + len(artifact["pieces"])
    table = nn.Embedding(size, old.shape[1], padding_idx=0, dtype=old.dtype)
    surfaces = []
    invalid_bytes = 0
    for byte in range(256):
        try:
            surfaces.append(bytes([byte]).decode("utf-8"))
        except UnicodeDecodeError:
            surfaces.append("")
            invalid_bytes += 1
    surfaces.extend(piece["text"] for piece in artifact["pieces"])
    fallback = 0
    with torch.no_grad():
        for new_id, old_id in enumerate(
            (
                original_tokenizer.pad_token_id,
                original_tokenizer.unk_token_id,
                original_tokenizer.bos_token_id,
                original_tokenizer.eos_token_id,
            )
        ):
            if old_id is None:
                raise ValueError("original tokenizer lacks a required control ID")
            table.weight[new_id].copy_(old[old_id])
        for start in range(0, len(surfaces), 512):
            batch = original_tokenizer(surfaces[start : start + 512], add_special_tokens=False)[
                "input_ids"
            ]
            for offset, ids in enumerate(batch):
                if not ids:
                    ids = [original_tokenizer.unk_token_id]
                    fallback += 1
                table.weight[start + offset + 4].copy_(old[ids].mean(0))
        table.weight[0].zero_()
    model.set_input_embeddings(table)
    model.config.vocab_size = size
    model.config.pad_token_id = 0
    model.config.bos_token_id = 2
    model.config.eos_token_id = 3
    return {
        "method": "original-subword mean initialization",
        "not_focus_or_zett": True,
        "seconds": time.perf_counter() - started,
        "vocabulary": size,
        "invalid_single_utf8_bytes": invalid_bytes,
        "empty_decompositions_using_unk": fallback,
        "tokenizer_sha256": sha256(path),
        "limitations": "piece-in-isolation heuristic; measure recovery, not assumed compatibility",
    }


class Encoder:
    """Own a pretrained body and either its original or trained Go token inputs."""

    def __init__(self, path: Path, custom: Path | None = None, max_length: int | None = None):
        saved = path / "encoder.json"
        metadata = json.loads(saved.read_text()) if saved.exists() else {}
        if max_length is None:
            max_length = metadata.get("maximum_tokens", 128)
        if max_length < 3:
            raise ValueError("maximum length must leave control and content positions")
        self.model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        if max_length > self.model.config.max_position_embeddings:
            raise ValueError("maximum length exceeds encoder position capacity")
        self.max_length = max_length
        self.custom_path = custom
        self.initialization = {
            "method": "retain original tokenizer and input embeddings",
            "seconds": 0.0,
        }
        if metadata.get("custom_tokenizer"):
            if custom is not None:
                raise ValueError("cannot reinitialize an already adapted custom checkpoint")
            filename = metadata["custom_tokenizer"]
            if Path(filename).name != filename:
                raise ValueError("checkpoint tokenizer must be a local filename")
            self.custom_path = path / filename
            if sha256(self.custom_path) != metadata["tokenizer_sha256"]:
                raise ValueError("checkpoint tokenizer checksum mismatch")
            self.initialization = {"method": "load adapted custom embedding table", "seconds": 0.0}
        if self.custom_path is not None:
            from bn_tokenizers_embedding import Tokenizer

            self.tokenizer = Tokenizer(self.custom_path)
            if not metadata.get("custom_tokenizer"):
                original = AutoTokenizer.from_pretrained(
                    path, local_files_only=True, trust_remote_code=False
                )
                self.initialization = transfer_embeddings(self.model, original, self.custom_path)
            if self.tokenizer.vocab_size != self.model.config.vocab_size:
                raise ValueError("tokenizer and encoder embedding vocabulary differ")
            self.pad_id = 0
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                path, local_files_only=True, trust_remote_code=False
            )
            self.pad_id = self.tokenizer.pad_token_id
            # The pinned checkpoint's Bert config says PAD=0 while its XLM-R
            # tokenizer uses PAD=1/BOS=0. Align padding for training so BOS is
            # trainable in both the original and replacement vocabularies.
            self.model.get_input_embeddings().padding_idx = self.pad_id
            self.model.config.pad_token_id = self.pad_id

    def inputs(self, texts: list[str]) -> tuple[dict[str, torch.Tensor], dict]:
        if not texts:
            raise ValueError("empty encoder batch")
        if self.custom_path is None:
            full = self.tokenizer(texts, padding=False, truncation=False)["input_ids"]
            batch = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            encoded = self.tokenizer.encode_batch(texts)
            full = [[2, *ids, 3] for ids in encoded]
            limited = [[2, *ids[: self.max_length - 2], 3] for ids in encoded]
            width = max(map(len, limited))
            batch = {
                "input_ids": torch.tensor([ids + [0] * (width - len(ids)) for ids in limited]),
                "attention_mask": torch.tensor(
                    [[1] * len(ids) + [0] * (width - len(ids)) for ids in limited]
                ),
            }
        return dict(batch), {
            "untruncated_tokens": sum(map(len, full)),
            "emitted_tokens": int(batch["attention_mask"].sum()),
            "padded_token_positions": batch["input_ids"].numel(),
            "truncated_texts": sum(len(ids) > self.max_length for ids in full),
        }

    def pool(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        return nn.functional.normalize(pooled, p=2, dim=1)

    def encode(self, texts: list[str], batch_size: int = 16) -> tuple[torch.Tensor, dict]:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        self.model.eval()
        outputs = []
        counts = {
            "untruncated_tokens": 0,
            "emitted_tokens": 0,
            "padded_token_positions": 0,
            "truncated_texts": 0,
        }
        started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                inputs, statistics = self.inputs(texts[start : start + batch_size])
                outputs.append(self.pool(inputs))
                for key, value in statistics.items():
                    counts[key] += value
        if not outputs:
            raise ValueError("no encoder texts")
        return torch.cat(outputs), dict(counts, seconds=time.perf_counter() - started)

    def save(self, output: Path) -> None:
        self.model.save_pretrained(output)
        if self.custom_path is None:
            self.tokenizer.save_pretrained(output)
            metadata = {"custom_tokenizer": None, "maximum_tokens": self.max_length}
        else:
            destination = output / "go-tokenizer.json"
            shutil.copyfile(self.custom_path, destination)
            metadata = {
                "custom_tokenizer": destination.name,
                "maximum_tokens": self.max_length,
                "tokenizer_sha256": sha256(destination),
            }
        (output / "encoder.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def close(self) -> None:
        if self.custom_path is not None:
            self.tokenizer.close()
