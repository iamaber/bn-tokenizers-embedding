"""Matched local representation adaptation and contrastive encoder training."""

import argparse
import hashlib
import json
import math
import platform
import random
import resource
import sys
import time
from pathlib import Path

import torch
from encoder import Encoder
from prepare import canonical, read_rows, sha256
from torch.nn import functional as F


def train(
    base: Path,
    alignment: Path,
    output: Path,
    custom: Path | None = None,
    seed: int = 17,
    adapt_steps: int = 64,
    contrastive_steps: int = 128,
    batch_size: int = 16,
    max_length: int = 128,
    max_seconds: float | None = None,
) -> dict:
    if output.exists():
        raise ValueError("checkpoint exists; use a new run directory")
    if min(adapt_steps, contrastive_steps) < 1 or batch_size < 2:
        raise ValueError("require positive stage steps and at least two pairs per batch")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("compute budget must be finite and positive")
    started = time.perf_counter()
    manifest = json.loads((alignment / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if Path(name).name != name or sha256(alignment / name) != digest:
            raise ValueError("alignment input checksum mismatch")
    pairs = read_rows(alignment / "pairs.jsonl")
    groups = {r["group"] for r in pairs}
    if len(groups) != len(pairs) or len(pairs) < batch_size:
        raise ValueError("require distinct training groups and enough pairs for a full batch")
    surfaces = [canonical(r[field]) for r in pairs for field in ("query", "document")]
    if len(set(surfaces)) != len(surfaces):
        raise ValueError("duplicate training surfaces risk in-batch false negatives")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    student = Encoder(base, custom, max_length)
    teacher = Encoder(base, max_length=max_length)
    teacher.model.eval().requires_grad_(False)
    total_parameters = sum(p.numel() for p in student.model.parameters())
    embedding_parameters = student.model.get_input_embeddings().weight.numel()
    optimizer = torch.optim.AdamW(student.model.parameters(), lr=2e-5, weight_decay=0.01)
    # Reset after tokenizer transfer so its construction cannot change dropout RNG.
    torch.manual_seed(seed)
    student.model.train()
    initialization_seconds = time.perf_counter() - started
    if max_seconds is not None and initialization_seconds >= max_seconds:
        student.close()
        teacher.close()
        raise ValueError("initialization exhausted the total compute budget")
    remaining = None if max_seconds is None else max_seconds - initialization_seconds
    adapt_deadline = None if remaining is None else time.perf_counter() + remaining * 0.35
    total_deadline = None if max_seconds is None else started + max_seconds
    history = []
    stage_results = {}
    sequence_digest = hashlib.sha256()
    for stage, steps in (("representation", adapt_steps), ("contrastive", contrastive_steps)):
        rng = random.Random(f"{seed}:{stage}")
        stage_started = time.perf_counter()
        losses = []
        tokens = {
            "untruncated_tokens": 0,
            "emitted_tokens": 0,
            "padded_token_positions": 0,
            "truncated_texts": 0,
        }
        deadline = adapt_deadline if stage == "representation" else total_deadline
        for step in range(steps):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            batch = rng.sample(pairs, batch_size)
            queries, documents = [], []
            for row in batch:
                reverse = bool(rng.getrandbits(1))
                query, document = (
                    (row["document"], row["query"]) if reverse else (row["query"], row["document"])
                )
                queries.append("query: " + query)
                documents.append("passage: " + document)
                sequence_digest.update(f"{stage}:{row['group']}:{int(reverse)}\n".encode())
            optimizer.zero_grad(set_to_none=True)
            if stage == "representation":
                texts = queries + documents
                inputs, counts = student.inputs(texts)
                reference_inputs, _ = teacher.inputs(texts)
                with torch.no_grad():
                    targets = teacher.pool(reference_inputs)
                loss = F.mse_loss(student.pool(inputs), targets)
            else:
                query_inputs, query_counts = student.inputs(queries)
                document_inputs, document_counts = student.inputs(documents)
                counts = {k: query_counts[k] + document_counts[k] for k in tokens}
                logits = student.pool(query_inputs) @ student.pool(document_inputs).T / 0.05
                targets = torch.arange(batch_size)
                loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            for key, value in counts.items():
                tokens[key] += value
            if step % 16 == 0 or step == steps - 1:
                entry = {
                    "stage": stage,
                    "step": step + 1,
                    "loss": losses[-1],
                    "elapsed_seconds": time.perf_counter() - started,
                }
                history.append(entry)
                print(json.dumps(entry), flush=True)
        if not losses:
            student.close()
            teacher.close()
            raise ValueError(f"compute budget allowed no {stage} training steps")
        stage_results[stage] = {
            "steps": len(losses),
            "requested_steps": steps,
            "examples": len(losses) * batch_size,
            "mean_loss": sum(losses) / len(losses),
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "seconds": time.perf_counter() - stage_started,
            **tokens,
        }
    training_seconds = time.perf_counter() - started
    output.mkdir(parents=True)
    student.save(output)
    report = {
        "seed": seed,
        "budget_kind": "equal-data" if max_seconds is None else "equal-compute",
        "total_training_budget_seconds": max_seconds,
        "total_training_seconds": training_seconds,
        "budget_overrun_seconds": 0
        if max_seconds is None
        else max(0, training_seconds - max_seconds),
        "budget_granularity": "checked between optimizer steps; report last-step overrun",
        "initialization_and_loading_seconds": initialization_seconds,
        "tokenizer_initialization": student.initialization,
        "stages": stage_results,
        "history": history,
        "batch_size": batch_size,
        "maximum_tokens": max_length,
        "threads": 4,
        "device": "cpu",
        "objective": (
            "teacher pooled-vector MSE then symmetric in-batch contrastive CE, temperature .05"
        ),
        "deterministic_algorithms": True,
        "optimizer": {"name": "AdamW", "learning_rate": 2e-5, "weight_decay": 0.01},
        "parameters": {
            "total": total_parameters,
            "embedding": embedding_parameters,
            "body": total_parameters - embedding_parameters,
        },
        "training_sequence_sha256": sequence_digest.hexdigest(),
        "alignment_manifest_sha256": sha256(alignment / "manifest.json"),
        "base_model_files": {p.name: sha256(p) for p in sorted(base.iterdir()) if p.is_file()},
        "criteria_sha256": sha256(Path(__file__).with_name("criteria.json")),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024 * 1024 if sys.platform == "darwin" else 1024),
        "memory_scope": "training process including teacher, optimizer and loaded tokenizers",
        "checkpoint_files": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
        "implementations": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("encoder.py", "train_encoder.py")
        },
        "limitations": [
            "small adaptation budget",
            "decomposition initialization is heuristic",
            "original and replacement embedding table capacities differ",
            "distinct source groups do not guarantee semantic negative validity",
        ],
    }
    (output / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    student.close()
    teacher.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, default=Path("data/processed/alignment-v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--adapt-steps", type=int, default=64)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()
    report = train(
        args.base,
        args.alignment,
        args.output,
        args.tokenizer,
        args.seed,
        args.adapt_steps,
        args.steps,
        args.batch_size,
        args.max_length,
        args.max_seconds,
    )
    print(
        json.dumps({k: report[k] for k in ("seed", "stages", "total_training_seconds")}, indent=2)
    )
