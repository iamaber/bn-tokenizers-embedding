"""Longer matched adaptation with balanced English replay and reviewed hard negatives."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from context_transfer import apply
from encoder import Encoder
from prepare import canonical, read_rows, sha256
from torch.nn import functional as F


def balanced_batch(rows: list[dict], rng: random.Random, size: int) -> list[dict]:
    if size < 2 or size % 2:
        raise ValueError("balanced batches require a positive even size")
    english = [row for row in rows if row["query_language"] == "english"]
    banglish = [row for row in rows if row["query_language"] == "banglish"]
    if min(len(english), len(banglish)) < size // 2:
        raise ValueError("not enough pairs in each language")
    return rng.sample(english, size // 2) + rng.sample(banglish, size // 2)


def verified_rows(path: Path, digest: str) -> list[dict]:
    if sha256(path / "manifest.json") != digest:
        raise ValueError("training manifest differs from protocol")
    manifest = json.loads((path / "manifest.json").read_text())
    if sha256(path / "pairs.jsonl") != manifest["files"]["pairs.jsonl"]:
        raise ValueError("training pairs changed")
    rows = read_rows(path / "pairs.jsonl")
    if len({row["group"] for row in rows}) != len(rows):
        raise ValueError("duplicate training groups")
    return rows


def train(protocol: Path, base: Path, arm: str, seed: int, budget: str, output: Path) -> dict:
    if output.exists():
        raise ValueError("checkpoint exists; preserve versions")
    specification = json.loads(protocol.read_text())
    if arm not in specification["arms"] or seed not in specification["training_seeds"]:
        raise ValueError("run is outside frozen protocol")
    if budget not in ("equal-data", "equal-compute") or base.name != specification["base_revision"]:
        raise ValueError("unknown budget or base revision")
    training = specification["alignment"]
    pairs = verified_rows(Path(training["directory"]), training["manifest_sha256"])
    negatives = specification["hard_negatives"]
    hard = verified_rows(Path(negatives["directory"]), negatives["manifest_sha256"])
    positive_lookup = {row["group"]: row for row in pairs}
    if any(row["pair_correct"] is not True or row["negative_grade"] != 0
           or row["group"] == row["negative_group"]
           or row["group"] not in positive_lookup
           or any(row[field] != positive_lookup[row["group"]][field]
                  for field in ("query", "document", "query_language")) for row in hard):
        raise ValueError("hard negatives must be explicitly reviewed nonanswers")
    if any(row["query_language"] not in ("english", "banglish") for row in pairs):
        raise ValueError("unexpected training query language")
    surfaces = [canonical(row[field]) for row in pairs for field in ("query", "document")]
    if len(set(surfaces)) != len(surfaces):
        raise ValueError("duplicate training surfaces")
    settings = specification["arms"][arm]
    tokenizer = Path(settings["tokenizer"]) if settings["tokenizer"] else None
    if tokenizer and sha256(tokenizer) != settings["tokenizer_sha256"]:
        raise ValueError("tokenizer differs from protocol")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    started = time.perf_counter()
    student = Encoder(base, tokenizer, specification["maximum_tokens"])
    teacher = Encoder(base, max_length=specification["maximum_tokens"])
    teacher.model.eval().requires_grad_(False)
    preprocessing_cost = 0.
    if settings["initialization"]:
        artifact = Path(settings["initialization"])
        if sha256(artifact / "manifest.json") != settings["initialization_manifest_sha256"]:
            raise ValueError("initialization differs from protocol")
        initializer = apply(student, artifact, base)
        if initializer["alignment_manifest_sha256"] != training["manifest_sha256"]:
            raise ValueError("context initialization uses different training data")
        preprocessing_cost = initializer["build_seconds"]
    optimizer = torch.optim.AdamW(student.model.parameters(), lr=2e-5, weight_decay=.01)
    torch.manual_seed(seed)
    student.model.train()
    elapsed = lambda: time.perf_counter() - started + preprocessing_cost
    limit = specification["equal_compute_seconds"] if budget == "equal-compute" else None
    if limit and elapsed() >= limit:
        raise ValueError("initialization exhausted compute budget")
    representation_deadline = None if limit is None else elapsed() + (limit - elapsed()) * .35
    history, stages = [], {}
    sequence = hashlib.sha256()
    batch_size = specification["batch_size"]
    english_retention = specification["english_retention_weight"]
    try:
        for stage, requested in (("representation", specification["representation_steps"]),
                                 ("contrastive", specification["contrastive_steps"])):
            rng = random.Random(f"{seed}:{stage}")
            losses, language_counts, hard_examples = [], {"english": 0, "banglish": 0}, 0
            maximum = requested if limit is None else 100000
            for step in range(maximum):
                deadline = representation_deadline if stage == "representation" else limit
                if deadline is not None and elapsed() >= deadline:
                    break
                use_hard = stage == "contrastive" and step % 3 == 2
                batch = balanced_batch(hard if use_hard else pairs, rng, batch_size)
                queries = ["query: " + row["query"] for row in batch]
                documents = ["passage: " + row["document"] for row in batch]
                for row in batch:
                    language_counts[row["query_language"]] += 1
                    sequence.update(f"{stage}:{row['group']}:{int(use_hard)}\n".encode())
                optimizer.zero_grad(set_to_none=True)
                if stage == "representation":
                    inputs, _ = student.inputs(queries + documents)
                    reference, _ = teacher.inputs(queries + documents)
                    with torch.no_grad():
                        targets = teacher.pool(reference)
                    loss = (1 - (student.pool(inputs) * targets).sum(1)).mean()
                else:
                    query_inputs, _ = student.inputs(queries)
                    doc_inputs, _ = student.inputs(documents)
                    q_vectors, d_vectors = student.pool(query_inputs), student.pool(doc_inputs)
                    logits = q_vectors @ d_vectors.T / .05
                    targets = torch.arange(batch_size)
                    loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
                    if use_hard:
                        negative_inputs, _ = student.inputs(["passage: " + row["negative"] for row in batch])
                        negative_vectors = student.pool(negative_inputs)
                        pair_logits = torch.stack([(q_vectors * d_vectors).sum(1),
                                                   (q_vectors * negative_vectors).sum(1)], dim=1) / .05
                        loss = loss + F.cross_entropy(pair_logits, torch.zeros(batch_size, dtype=torch.long))
                        hard_examples += batch_size
                    english_inputs, _ = teacher.inputs(queries[:batch_size // 2])
                    with torch.no_grad():
                        retained = teacher.pool(english_inputs)
                    retention_loss = (1 - (q_vectors[:batch_size // 2] * retained).sum(1)).mean()
                    loss = loss + english_retention * retention_loss
                if not torch.isfinite(loss):
                    raise ValueError("non-finite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                if step % 32 == 0:
                    entry = {"stage": stage, "steps": step + 1, "loss": losses[-1], "charged_seconds": elapsed()}
                    history.append(entry)
                    print(json.dumps(entry), flush=True)
            if not losses:
                raise ValueError("budget allowed no steps in stage")
            stages[stage] = {"steps": len(losses), "mean_loss": sum(losses) / len(losses),
                             "first_loss": losses[0], "last_loss": losses[-1],
                             "query_languages": language_counts, "verified_negative_examples": hard_examples}
        training_seconds = elapsed()
        output.mkdir(parents=True)
        student.save(output)
        result = {"arm": arm, "seed": seed, "budget": budget,
                  "protocol_sha256": sha256(protocol), "stages": stages, "history": history,
                  "charged_training_seconds": training_seconds, "preprocessing_charged_seconds": preprocessing_cost,
                  "equal_compute_limit_seconds": limit,
                  "overrun_seconds": max(0, training_seconds - limit) if limit else 0,
                  "tokenizer_initialization": student.initialization,
                  "training_sequence_sha256": sequence.hexdigest(),
                  "base_files": {p.name: sha256(p) for p in base.iterdir() if p.is_file()},
                  "checkpoint_files": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
                  "implementations": {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("encoder.py"), Path(__file__).with_name("context_transfer.py"))},
                  "objective": "cosine teacher adaptation; symmetric in-batch CE plus verified-negative CE and English teacher cosine retention",
                  "english_sampling": "eight English and eight Banglish queries per batch; no reversal",
                  "limitations": ["single-agent negative judgments", "bulk in-batch semantic negatives remain unreviewed",
                                  "balanced replay and retention do not guarantee English quality",
                                  "cached context artifact build cost conservatively charged in full per run"]}
        (output / "training-report.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        student.close()
        teacher.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", choices=("equal-data", "equal-compute"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args.protocol, args.base, args.arm, args.seed, args.budget, args.output)
