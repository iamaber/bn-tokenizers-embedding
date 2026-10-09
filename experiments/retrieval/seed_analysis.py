"""Paired uncertainty over both training seeds and held-out intent groups."""

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from compare import score_rankings
from prepare import read_rows, sha256, verify_manifest

METRICS = ("recall_at_10", "ndcg_at_10", "mrr_at_10")
DIRECTIONS = {f"{q}-{d}" for q in ("bl", "bn", "en") for d in ("bl", "bn", "en")}


def selection_scores(directory: Path, report: dict) -> dict:
    scores = {}
    for direction, detail in report["results"].items():
        if direction not in DIRECTIONS:
            continue  # Sentence-pilot reporting does not select domain models.
        dataset = Path(detail["dataset"])
        verify_manifest(dataset)
        if any(row.get("split") != "development" for row in read_rows(dataset / "queries.jsonl")):
            raise ValueError("model selection requires development-only domain queries")
        if sha256(dataset / "manifest.json") != detail["dataset_manifest_sha256"]:
            raise ValueError("evaluation dataset differs")
        ranking = directory / direction / "rankings.jsonl"
        if sha256(ranking) != detail["ranking_sha256"]:
            raise ValueError("evaluation rankings differ")
        scores[direction] = score_rankings(dataset, ranking)
    if scores.keys() != DIRECTIONS:
        raise ValueError("selection requires all nine domain directions")
    return scores


def combine_directions(cohort, directions):
    return {
        s: [dict(row, query_id=f"{d}:{row['query_id']}") for d in directions for row in rows[d]]
        for s, rows in cohort.items()
    }


def paired_seeds(
    reference: dict[int, list[dict]],
    candidate: dict[int, list[dict]],
    seed: int = 17,
    repeats: int = 2000,
) -> dict:
    if not reference or reference.keys() != candidate.keys() or repeats < 40:
        raise ValueError("matching nonempty seed sets and at least 40 draws required")
    differences = {}
    common_queries = None
    for training_seed in sorted(reference):
        left = {row["query_id"]: row for row in reference[training_seed]}
        right = {row["query_id"]: row for row in candidate[training_seed]}
        if (
            not left
            or left.keys() != right.keys()
            or len(left) != len(reference[training_seed])
            or len(right) != len(candidate[training_seed])
        ):
            raise ValueError("paired seeds require identical unique query IDs")
        identity = {qid: row["group"] for qid, row in left.items()}
        if common_queries is not None and identity != common_queries:
            raise ValueError("query/group identity differs across training seeds")
        common_queries = identity
        grouped = defaultdict(list)
        for qid, original in left.items():
            other = right[qid]
            if original["group"] != other["group"]:
                raise ValueError("paired groups differ")
            if any(
                not math.isfinite(row["metrics"][metric])
                for row in (original, other)
                for metric in METRICS
            ):
                raise ValueError("non-finite metric")
            grouped[original["group"]].append(
                {
                    metric: other["metrics"][metric] - original["metrics"][metric]
                    for metric in METRICS
                }
            )
        differences[training_seed] = {
            group: {metric: statistics.mean(row[metric] for row in rows) for metric in METRICS}
            for group, rows in grouped.items()
        }
    groups, seeds = sorted(next(iter(differences.values()))), sorted(differences)
    rng = random.Random(seed)
    draws = {metric: [] for metric in METRICS}
    for _ in range(repeats):
        sampled_groups = rng.choices(groups, k=len(groups))
        sampled_seeds = rng.choices(seeds, k=len(seeds))
        for metric in METRICS:
            draws[metric].append(
                statistics.mean(
                    differences[s][g][metric] for s in sampled_seeds for g in sampled_groups
                )
            )
    seed_means = {
        str(s): {
            metric: statistics.mean(row[metric] for row in differences[s].values())
            for metric in METRICS
        }
        for s in seeds
    }
    return {
        "groups": len(groups),
        "queries": len(common_queries),
        "training_seeds": seeds,
        "bootstrap_repeats": repeats,
        "per_seed_difference": seed_means,
        "metrics": {
            metric: {
                "mean": statistics.mean(row[metric] for row in seed_means.values()),
                "bootstrap_95_percentile": [
                    sorted(draws[metric])[int(repeats * fraction)] for fraction in (0.025, 0.975)
                ],
            }
            for metric in METRICS
        },
        "effect": "candidate minus reference; crossed paired seed/group bootstrap",
        "uncertainty_scope": "empirical three-seed and intent sampling; excludes label uncertainty",
    }


def analyze(suite: Path, criteria: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("decision exists; preserve the frozen selection")
    specification = json.loads(suite.read_text())
    rules = json.loads(criteria.read_text())
    seeds = rules["required_training_seeds"]
    runs, reports = {}, {}
    for row in specification["runs"]:
        directory = Path(row["directory"])
        if sha256(directory / "report.json") != row["report_sha256"]:
            raise ValueError("evaluation report checksum mismatch")
        report = json.loads((directory / "report.json").read_text())
        runs[row["name"]] = selection_scores(directory, report)
        reports[row["name"]] = report
    directions = sorted(runs["unchanged"])
    decisions = {}
    original_adaptation_effect = {}
    for budget in ("equal-data", "equal-compute"):
        controls = {s: runs[f"{budget}/original-seed{s}"] for s in seeds}
        original_adaptation_effect[budget] = {
            direction: paired_seeds(
                {s: runs["unchanged"][direction] for s in seeds},
                {s: controls[s][direction] for s in seeds},
            )
            for direction in directions
        }
        for arm in ("unigram-32000", "bpe-32000"):
            candidates = {s: runs[f"{budget}/{arm}-seed{s}"] for s in seeds}
            comparisons = {
                direction: paired_seeds(
                    {s: controls[s][direction] for s in seeds},
                    {s: candidates[s][direction] for s in seeds},
                )
                for direction in directions
            }
            english_directions = [d for d in directions if d.startswith("en-")]

            english_comparison = paired_seeds(
                combine_directions(controls, english_directions),
                combine_directions(candidates, english_directions),
            )
            ratios = []
            for training_seed in seeds:
                base = reports[f"{budget}/original-seed{training_seed}"]
                other = reports[f"{budget}/{arm}-seed{training_seed}"]
                if (
                    base["runtime"]["single_query"]["workload_sha256"],
                    base["runtime"]["document_workload_sha256"],
                ) != (
                    other["runtime"]["single_query"]["workload_sha256"],
                    other["runtime"]["document_workload_sha256"],
                ):
                    raise ValueError("latency workload differs")
                ratios.append(
                    {
                        "seed": training_seed,
                        "query_latency_ratio": other["runtime"]["single_query"]["median_seconds"]
                        / base["runtime"]["single_query"]["median_seconds"],
                        "peak_memory_ratio": other["peak_process_rss_bytes"]
                        / base["peak_process_rss_bytes"],
                    }
                )
            primary = comparisons["bl-bn"]["metrics"][rules["primary_metric"]]
            english_effect = english_comparison["metrics"][rules["primary_metric"]]
            gates = {
                "minimum_gain": primary["mean"] >= rules["minimum_absolute_gain"],
                "positive_paired_interval": primary["bootstrap_95_percentile"][0]
                > rules["paired_interval_lower_bound_must_exceed"],
                "english_retention": english_effect["mean"] >= -rules["maximum_english_regression"],
                "query_latency": all(
                    r["query_latency_ratio"] <= rules["maximum_query_latency_ratio"] for r in ratios
                ),
                "memory": all(
                    r["peak_memory_ratio"] <= rules["maximum_peak_memory_ratio"] for r in ratios
                ),
            }
            decisions[f"{budget}/{arm}"] = {
                "passes_all_gates": all(gates.values()),
                "gates": gates,
                "paired_directions": comparisons,
                "english_retention": english_comparison,
                "runtime_ratios": ratios,
            }
    qualified = [
        arm
        for arm in ("unigram-32000", "bpe-32000")
        if all(decisions[f"{b}/{arm}"]["passes_all_gates"] for b in ("equal-data", "equal-compute"))
    ]
    chosen = (
        max(
            qualified,
            key=lambda arm: decisions[f"equal-data/{arm}"]["paired_directions"]["bl-bn"]["metrics"][
                rules["primary_metric"]
            ]["mean"],
        )
        if qualified
        else "unchanged"
    )
    result = {
        "selection": chosen,
        "selection_frozen_before_reserved_test": True,
        "selection_rule": (
            "custom arm must pass declared gates in both budgets; otherwise retain unchanged E5"
        ),
        "criteria_sha256": sha256(criteria),
        "evaluation_suite": str(suite),
        "evaluation_suite_sha256": sha256(suite),
        "evaluation_report_hashes": {r["name"]: r["report_sha256"] for r in specification["runs"]},
        "decisions": decisions,
        "original_adaptation_minus_unchanged": original_adaptation_effect,
        "provenance": "agent-authored/judged controlled domain proxy",
        "limits": (
            "42 development intents; only three seeds; no claim of natural-traffic generalization"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument(
        "--criteria", type=Path, default=Path("experiments/retrieval/criteria.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.suite, args.criteria, args.output)
    print(
        json.dumps(
            {
                "selection": result["selection"],
                "gates": {name: details["gates"] for name, details in result["decisions"].items()},
            },
            indent=2,
        )
    )
