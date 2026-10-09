"""Verify expanded corpus provenance and every reserved canonical surface."""

import argparse
import json
from collections import Counter
from pathlib import Path

from prepare import canonical, sha256


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def audit(directory: Path, original: Path, material: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    for filename, digest in manifest["files"].items():
        path = Path(filename)
        if path.is_absolute() or ".." in path.parts or sha256(directory / path) != digest:
            raise ValueError("expanded corpus file checksum mismatch")
    for path in [material, *(original / f"{s}.jsonl" for s in ("validation", "test"))]:
        if manifest["inputs"].get(str(path)) != sha256(path):
            raise ValueError("reserved input differs from expansion provenance")
    reserved = {
        canonical(text)
        for row in rows(directory / "original-pairs.jsonl")
        if row["split"] != "train"
        for text in row["forms"].values()
    }
    for split in ("validation", "test"):
        reserved.update(canonical(row["text"]) for row in rows(original / f"{split}.jsonl"))
    reserved.update(
        canonical(row[field]) for row in rows(material)
        for field in ("bn", "bl", "en", "q_bn", "q_bl", "q_en")
    )
    groups = {}
    counts = Counter()
    for split in ("train", "validation", "test"):
        for row in rows(directory / "prepared" / f"{split}.jsonl"):
            if row["source"] == "benchmark-reserve":
                raise ValueError("reserve-only benchmark row entered corpus output")
            if split == "train" and canonical(row["text"]) in reserved:
                raise ValueError("reserved canonical surface entered expanded train")
            previous = groups.setdefault(row["group"], split)
            if previous != split:
                raise ValueError("expanded group crosses splits")
            counts[split] += 1
    return {
        "status": "passed", "manifest_sha256": sha256(directory / "manifest.json"),
        "reserved_canonical_keys": len(reserved), "expanded_rows_by_split": dict(counts),
        "checks": ["all original source val/test forms excluded from train",
                   "all project val/test forms excluded from train",
                   "all authored benchmark forms excluded from train",
                   "reserve-only sources absent from every split", "no group crosses splits"],
        "scope": "integrity only; no frozen test evaluation",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--original", type=Path, default=Path("data/processed"))
    parser.add_argument("--material", type=Path,
                        default=Path("experiments/retrieval/domain_material.jsonl"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report exists; use a new version")
    result = audit(args.expanded, args.original, args.material)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
