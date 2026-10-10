"""FOCUS-inspired overlap/sparse-context initialization; not a FOCUS reproduction."""

import argparse
import json
import time
from pathlib import Path

import torch
from encoder import Encoder
from prepare import read_rows, sha256


def sparsemax(scores: torch.Tensor) -> torch.Tensor:
    """Project rows onto the probability simplex with sparse support."""
    if scores.ndim != 2 or scores.shape[1] == 0 or not torch.isfinite(scores).all():
        raise ValueError("require finite nonempty score rows")
    values = scores - scores.max(dim=1, keepdim=True).values
    ordered = values.sort(dim=1, descending=True).values
    cumulative = ordered.cumsum(dim=1)
    ranks = torch.arange(1, scores.shape[1] + 1, device=scores.device)
    support = 1 + ranks * ordered > cumulative
    count = support.sum(dim=1, keepdim=True)
    threshold = (cumulative.gather(1, count - 1) - 1) / count
    return (values - threshold).clamp_min(0)


def build(base: Path, tokenizer: Path, alignment: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("initialization exists; preserve versions")
    manifest = json.loads((alignment / "manifest.json").read_text())
    if any(sha256(alignment / name) != digest for name, digest in manifest["files"].items()):
        raise ValueError("alignment snapshot changed")
    pairs = read_rows(alignment / "pairs.jsonl")
    texts = [row[field] for row in pairs for field in ("query", "document")]
    torch.set_num_threads(4)
    torch.manual_seed(17)
    started = time.perf_counter()
    original = Encoder(base)
    student = Encoder(base, tokenizer)
    try:
        old_vocabulary = original.tokenizer.get_vocab()
        pieces = json.loads(tokenizer.read_text())["pieces"]
        anchors = [(260 + i, old_vocabulary[piece["text"]]) for i, piece in enumerate(pieces)
                   if piece["text"] in old_vocabulary]
        if not anchors:
            raise ValueError("no exact overlap anchors")
        size = student.model.config.vocab_size
        dimensions, window = 512, 2
        contexts = torch.zeros(size, dimensions)
        flat = contexts.flatten()
        occurrences = torch.zeros(size, dtype=torch.int64)
        # Multiplicative hash is stable across processes; no Python randomized hash.
        for start in range(0, len(texts), 256):
            encoded = student.tokenizer.encode_batch(texts[start:start + 256])
            centers, neighbours = [], []
            for ids in encoded:
                occurrences += torch.bincount(torch.tensor(ids, dtype=torch.int64), minlength=size)
                for offset in range(1, window + 1):
                    centers.extend(ids[:-offset]); neighbours.extend(ids[offset:])
                    centers.extend(ids[offset:]); neighbours.extend(ids[:-offset])
            if centers:
                indices = torch.tensor(centers) * dimensions + (
                    (torch.tensor(neighbours) * 2654435761) & 0xffffffff
                ) % dimensions
                flat.index_add_(0, indices, torch.ones(len(indices)))
        row_mass = contexts.sum(1, keepdim=True)
        column_mass = contexts.sum(0, keepdim=True)
        total = contexts.sum()
        if total <= 0:
            raise ValueError("no usable token contexts")
        ppmi = ((contexts * total) / (row_mass * column_mass).clamp_min(1)).clamp_min(1).log()
        vectors = torch.nn.functional.normalize(ppmi, dim=1)
        active = vectors.norm(dim=1) > 0
        valid_anchors = [(new, old) for new, old in anchors if active[new]]
        if not valid_anchors:
            raise ValueError("no observed overlap anchors")
        new_ids = torch.tensor([new for new, _ in valid_anchors])
        old_ids = torch.tensor([old for _, old in valid_anchors])
        anchor_vectors = vectors[new_ids]
        old_weights = original.model.get_input_embeddings().weight.detach()
        anchor_weights = old_weights[old_ids]
        initialized = student.model.get_input_embeddings().weight.detach().clone()
        overlap = {new for new, _ in anchors}
        candidates = [i for i in range(260, size) if i not in overlap and active[i]]
        support_counts = []
        with torch.no_grad():
            for new, old in anchors:
                initialized[new] = old_weights[old]
            for start in range(0, len(candidates), 128):
                ids = candidates[start:start + 128]
                weights = sparsemax(vectors[ids] @ anchor_vectors.T)
                initialized[ids] = weights @ anchor_weights
                support_counts.extend((weights > 0).sum(1).tolist())
        if not torch.isfinite(initialized).all() or torch.count_nonzero(initialized[0]):
            raise ValueError("invalid initialized embedding table")
        seconds = time.perf_counter() - started
        output.mkdir(parents=True)
        torch.save(initialized, output / "embeddings.pt")
        report = {
            "method": "FOCUS-inspired exact-overlap and hashed-PPMI sparsemax transfer",
            "focus_reproduction": False, "build_seconds": seconds,
            "tokenizer_sha256": sha256(tokenizer),
            "alignment_manifest_sha256": sha256(alignment / "manifest.json"),
            "base_files": {p.name: sha256(p) for p in base.iterdir() if p.is_file()},
            "embedding_sha256": sha256(output / "embeddings.pt"), "shape": list(initialized.shape),
            "exact_overlap_anchors": len(anchors), "observed_context_anchors": len(valid_anchors),
            "context_initialized_pieces": len(candidates),
            "decomposition_fallback_pieces": size - 260 - len(anchors) - len(candidates),
            "controls_and_bytes": "retain existing semantic control remap and byte decomposition",
            "anchor_rule": "exact raw piece string equals original token; boundary markers not stripped",
            "contexts": {"texts": len(texts), "dimensions": dimensions, "symmetric_window": window,
                         "hash": "((token_id * 2654435761) & 0xffffffff) % 512",
                         "transform": "positive PMI then L2 normalization", "observed_tokens": int((occurrences > 0).sum())},
            "sparse_support": {"mean": sum(support_counts) / len(support_counts) if support_counts else 0,
                               "maximum": max(support_counts, default=0)},
            "implementation_sha256": sha256(Path(__file__)),
            "limitations": ["hashed co-occurrence replaces FOCUS fastText space",
                            "native pieces lack original boundary semantics",
                            "initialization quality must be measured after matched adaptation"],
        }
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        original.close(); student.close()


def apply(encoder: Encoder, artifact: Path, base: Path) -> dict:
    report = json.loads((artifact / "manifest.json").read_text())
    if encoder.custom_path is None or sha256(encoder.custom_path) != report["tokenizer_sha256"]:
        raise ValueError("initialization tokenizer mismatch")
    if report["base_files"] != {p.name: sha256(p) for p in base.iterdir() if p.is_file()}:
        raise ValueError("initialization base mismatch")
    if sha256(artifact / "embeddings.pt") != report["embedding_sha256"]:
        raise ValueError("initialization checksum mismatch")
    weights = torch.load(artifact / "embeddings.pt", map_location="cpu", weights_only=True)
    table = encoder.model.get_input_embeddings().weight
    if weights.shape != table.shape or not torch.isfinite(weights).all() or torch.count_nonzero(weights[0]):
        raise ValueError("invalid initialization tensor")
    with torch.no_grad():
        table.copy_(weights)
    encoder.initialization = {**report, "artifact_manifest_sha256": sha256(artifact / "manifest.json")}
    return encoder.initialization


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.tokenizer, args.alignment, args.output), indent=2))
