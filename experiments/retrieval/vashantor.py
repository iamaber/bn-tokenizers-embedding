"""Import original Vashantor forms through the existing Go grouping/filter engine."""

import argparse
import csv
import hashlib
import io
import json
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

from prepare import canonical, normalize, sha256, write_rows

ARCHIVE_SHA256 = "5abb437220ad6fede5344b974f1abe48f4be8d5a9c250b807c0fea6fe4899919"
ARCHIVE_BYTES = 1073445
ATTRIBUTION = (
    "Vashantor v2, DOI 10.17632/bj5jgk878b.2; CC BY 4.0; author-supplied human "
    "translations and source utterances; NFC/whitespace normalization and grouped filtering"
)


def original_rows(archive: Path) -> list[dict]:
    if archive.stat().st_size != ARCHIVE_BYTES or sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Vashantor archive checksum mismatch")
    rows = []
    with zipfile.ZipFile(archive) as zipped:
        for name in sorted(zipped.namelist()):
            if not name.endswith(".csv"):
                continue
            parent = Path(name).parent.name.lower()
            if parent not in ("train", "validation", "test"):
                raise ValueError("unknown original source split")
            with zipped.open(name) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
                for line_number, fields in enumerate(reader, start=2):
                    row = {k.strip(): normalize(v) for k, v in fields.items()}
                    regional_bn = [k for k in row if k.endswith("_bangla_speech")]
                    regional_bl = [k for k in row if k.endswith("_banglish_speech")]
                    if len(regional_bn) != 1 or len(regional_bl) != 1 or len(row) != 6:
                        raise ValueError("unexpected Vashantor column schema")
                    forms = {
                        "bangla": row["bangla_speech"],
                        "banglish": row["banglish_speech"],
                        "english": row["english_speech"],
                        "regional_bangla": row[regional_bn[0]],
                        "regional_banglish": row[regional_bl[0]],
                    }
                    if any(not value for value in forms.values()):
                        raise ValueError("missing aligned Vashantor form")
                    core_hash = hashlib.sha256(canonical(forms["bangla"]).encode()).hexdigest()
                    rows.append(
                        {
                            "group": f"vashantor:{core_hash}",
                            "split": parent,
                            "region": row["region_name"],
                            "raw_id": f"{name}:{line_number}",
                            "forms": forms,
                        }
                    )
    if len(rows) != 12500:
        raise ValueError("unexpected Vashantor v2 row count")
    return rows


def expand(archive: Path, prepared: Path, material: Path, output: Path, bntok: Path) -> dict:
    if output.exists():
        raise ValueError("expansion output exists; preserve corpus snapshots")
    rows = original_rows(archive)
    inputs = [prepared / f"{split}.jsonl" for split in ("train", "validation", "test")]
    if any(not p.exists() for p in inputs):
        raise ValueError("all existing splits required for integrity exclusion")
    output.mkdir(parents=True)
    combined = output / "combined-records.jsonl"
    with combined.open("w", encoding="utf-8") as stream:
        for path in inputs:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    stream.write(line)
        for row in rows:
            for field, text in row["forms"].items():
                bucket = (
                    "banglish"
                    if field == "regional_banglish"
                    else ("bangla" if field == "regional_bangla" else field)
                )
                record = {
                    "text": text,
                    "bucket": bucket,
                    "source": "vashantor",
                    "group": row["group"],
                    "split": row["split"],
                    "attribution": ATTRIBUTION,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        # Authored benchmark forms are reserve-only blockers, not corpus additions.
        for line in material.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for field in ("bn", "bl", "en", "q_bn", "q_bl", "q_en"):
                record = {
                    "text": row[field],
                    "bucket": "english",
                    "source": "benchmark-reserve",
                    "group": f"agent-domain:{row['id']}",
                    "split": "test",
                    "attribution": "agent-authored evaluation blocker; not training data",
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_rows(output / "original-pairs.jsonl", rows)
    result = subprocess.run(
        [
            str(bntok.resolve()),
            "prepare-records",
            "-input",
            str(combined),
            "-out",
            str(output / "prepared"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    stats = json.loads(result.stdout)
    accepted = Counter()
    final_counts = Counter()
    final_characters = Counter()
    reserve_removed = 0
    for split in ("train", "validation", "test"):
        path = output / "prepared" / f"{split}.jsonl"
        temporary = path.with_suffix(".filtered")
        with (
            path.open(encoding="utf-8") as stream,
            temporary.open("w", encoding="utf-8") as filtered,
        ):
            for line in stream:
                row = json.loads(line)
                if row["source"] == "benchmark-reserve":
                    reserve_removed += 1
                    continue
                filtered.write(line)
                key = f"{split}/{row['bucket']}"
                final_counts[key] += 1
                final_characters[key] += len(row["text"])
                if row["source"] == "vashantor":
                    accepted[key] += 1
        temporary.replace(path)
    stats["split_bucket_records"] = dict(final_counts)
    stats["split_bucket_characters"] = dict(final_characters)
    stats["reserve_records_suppressed"] = reserve_removed
    (output / "prepared/stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    manifest = {
        "source": "original Vashantor v2",
        "archive_sha256": sha256(archive),
        "original_rows": len(rows),
        "original_splits": dict(Counter(r["split"] for r in rows)),
        "accepted_vashantor_records": dict(accepted),
        "reserve_records_suppressed": reserve_removed,
        "prepare_stats": stats,
        "grouping": "Go paired/core groups, all canonical forms and long-text SimHash",
        "split_priority": "test > validation > hash assignment; rejected text carries holdout",
        "frozen_project_test_read_for_integrity_only": True,
        "frozen_project_test_scored": False,
        "benchmark_forms": "reserve-only records cannot enter train",
        "quality": "existing Go length/corruption/PII heuristics; not complete privacy detection",
        "inputs": {str(p): sha256(p) for p in [archive, material, *inputs]},
        "files": {
            "original-pairs.jsonl": sha256(output / "original-pairs.jsonl"),
            **{
                f"prepared/{s}.jsonl": sha256(output / "prepared" / f"{s}.jsonl")
                for s in ("train", "validation", "test")
            },
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("data/raw/bangladual/Vashantor_CSV_Format.zip")
    )
    parser.add_argument("--prepared", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--material", type=Path, default=Path("experiments/retrieval/domain_material.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/vashantor-expanded-v1"))
    parser.add_argument("--bntok", type=Path, default=Path("bin/bntok"))
    args = parser.parse_args()
    print(
        json.dumps(
            expand(args.archive, args.prepared, args.material, args.output, args.bntok), indent=2
        )
    )
