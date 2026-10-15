"""Evaluate frozen original-tokenizer controls, select on development, then confirm."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from compare import score_rankings
from prepare import read_rows, sha256, verify_manifest
from run_extended_matrix import verify
from seed_analysis import combine_directions, paired_seeds


def save_frozen(path: Path, result: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError(f"frozen artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def gates(primary: dict, unchanged: dict, english: dict, ratios: list[dict], rules: dict) -> dict:
    return {
        "gain_over_short": primary["mean"] >= rules["minimum_gain_over_short_original"],
        "positive_paired_interval": primary["bootstrap_95_percentile"][0] > rules["paired_interval_lower_bound_must_exceed"],
        "gain_over_unchanged": unchanged["mean"] > rules["minimum_gain_over_unchanged"],
        "english_retention": english["mean"] >= -rules["maximum_english_regression_vs_unchanged"],
        "query_latency": all(r["latency"] <= rules["maximum_query_latency_ratio_vs_unchanged"] for r in ratios),
        "memory": all(r["memory"] <= rules["maximum_peak_memory_ratio_vs_unchanged"] for r in ratios),
    }


def compare(scores: dict, reports: dict, rules: dict) -> dict:
    results = {}
    seeds = rules["required_training_seeds"]
    for budget in rules["required_budgets"]:
        long = {s: scores[f"long/{budget}/original-seed{s}"] for s in seeds}
        short = {s: scores[f"short/{budget}/original-seed{s}"] for s in seeds}
        base = {s: scores["unchanged"] for s in seeds}
        directions = {}
        for direction in sorted(scores["unchanged"]):
            directions[direction] = {
                "long_minus_short": paired_seeds({s: short[s][direction] for s in seeds}, {s: long[s][direction] for s in seeds}),
                "long_minus_unchanged": paired_seeds({s: base[s][direction] for s in seeds}, {s: long[s][direction] for s in seeds}),
                "short_minus_unchanged": paired_seeds({s: base[s][direction] for s in seeds}, {s: short[s][direction] for s in seeds}),
            }
        english = paired_seeds(combine_directions(base, rules["english_directions"]),
                               combine_directions(long, rules["english_directions"]))
        ratios = []
        reference = reports["unchanged"]
        for seed in seeds:
            other = reports[f"long/{budget}/original-seed{seed}"]
            for field in ("query_workload_sha256", "document_workload_sha256"):
                if other["runtime"][field] != reference["runtime"][field]:
                    raise ValueError("runtime workloads differ")
            if other["runtime"]["single_query"]["workload_sha256"] != reference["runtime"]["single_query"]["workload_sha256"]:
                raise ValueError("single-query workloads differ")
            ratios.append({"seed": seed,
                           "latency": other["runtime"]["single_query"]["median_seconds"] / reference["runtime"]["single_query"]["median_seconds"],
                           "memory": other["peak_process_rss_bytes"] / reference["peak_process_rss_bytes"]})
        primary = directions[rules["primary_direction"]]
        metric = rules["primary_metric"]
        passed = gates(primary["long_minus_short"]["metrics"][metric],
                       primary["long_minus_unchanged"]["metrics"][metric], english["metrics"][metric], ratios, rules)
        results[budget] = {"directions": directions, "english_minus_unchanged": english,
                           "runtime_ratios": ratios, "gates": passed, "passes_all": all(passed.values())}
    return results


def evaluate_split(models: dict, benchmark: Path, split: str, root: Path) -> tuple[dict, dict]:
    reports, scores = {}, {}
    datasets = sorted((benchmark / split).iterdir())
    for name, model in models.items():
        directory = root / split / name
        if not (directory / "report.json").exists():
            if directory.exists():
                raise ValueError(f"incomplete evaluation needs inspection: {directory}")
            subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate_encoder.py")),
                            "--model", str(model), "--domains", str(benchmark / split),
                            "--output", str(directory), "--benchmark-repeats", "5"], check=True)
        report = json.loads((directory / "report.json").read_text())
        if report["model_files"] != {p.name: sha256(p) for p in model.iterdir() if p.is_file()}:
            raise ValueError("evaluated model changed")
        if report["results"].keys() != {p.name for p in datasets}:
            raise ValueError("evaluation direction inventory changed")
        scores[name] = {}
        for dataset in datasets:
            verify_manifest(dataset)
            if any(r["split"] != split for r in read_rows(dataset / "queries.jsonl")):
                raise ValueError("query split differs")
            detail = report["results"][dataset.name]
            if detail["dataset_manifest_sha256"] != sha256(dataset / "manifest.json"):
                raise ValueError("evaluated dataset changed")
            for filename, field in (("rankings.jsonl", "ranking_sha256"), ("scores.jsonl", "scores_sha256")):
                if sha256(directory / dataset.name / filename) != detail[field]:
                    raise ValueError("evaluation outputs changed")
            scores[name][dataset.name] = score_rankings(dataset, directory / dataset.name / "rankings.jsonl")
        for filename, digest in report["implementation_sha256"].items():
            if sha256(Path(__file__).with_name(filename)) != digest:
                raise ValueError("evaluation implementation changed")
        reports[name] = report
    return reports, scores


def verify_short(directory: Path, base: Path, budget: str, seed: int) -> None:
    report = json.loads((directory / "training-report.json").read_text())
    metadata = json.loads((directory / "encoder.json").read_text())
    if ((report["seed"], report["budget_kind"]) != (seed, budget)
            or metadata["custom_tokenizer"] is not None
            or report["tokenizer_initialization"]["method"] != "retain original tokenizer and input embeddings"):
        raise ValueError("short control identity differs")
    if report["base_model_files"] != {p.name: sha256(p) for p in base.iterdir() if p.is_file()}:
        raise ValueError("short control uses a different base snapshot")
    files = {p.name for p in directory.iterdir() if p.is_file()} - {"training-report.json"}
    if files != report["checkpoint_files"].keys():
        raise ValueError("short control checkpoint inventory changed")
    for filename, digest in report["checkpoint_files"].items():
        if Path(filename).name != filename or sha256(directory / filename) != digest:
            raise ValueError("short control checkpoint changed")


def run(benchmark: Path, inventory: Path, base: Path, criteria: Path, root: Path,
        protocol: Path) -> dict:
    rules = json.loads(criteria.read_text())
    matrix = json.loads(inventory.read_text())
    if matrix["protocol_sha256"] != sha256(protocol):
        raise ValueError("training inventory uses a different protocol")
    models = {"unchanged": base}
    expected = {f"{b}/original-seed{s}" for b in rules["required_budgets"] for s in rules["required_training_seeds"]}
    if {r["name"] for r in matrix["runs"]} != expected or len(matrix["runs"]) != len(expected):
        raise ValueError("training inventory is incomplete")
    for row in matrix["runs"]:
        directory = Path(row["directory"])
        if sha256(directory / "training-report.json") != row["training_report_sha256"]:
            raise ValueError("training report changed")
        budget, seed_name = row["name"].split("/")
        seed = int(seed_name.removeprefix("original-seed"))
        verify(directory, protocol, base, "original", seed, budget)
        models[f"long/{row['name']}"] = directory
        short = Path("models/encoders") / row["name"]
        verify_short(short, base, budget, seed)
        models[f"short/{row['name']}"] = short
    benchmark_manifest = json.loads((benchmark / "manifest.json").read_text())
    for filename, digest in benchmark_manifest["files"].items():
        if sha256(benchmark / filename) != digest:
            raise ValueError("benchmark manifest binding differs")
        verify_manifest((benchmark / filename).parent)
    frozen = {"benchmark_manifest_sha256": sha256(benchmark / "manifest.json"),
              "training_inventory_sha256": sha256(inventory), "criteria_sha256": sha256(criteria),
              "training_protocol_sha256": sha256(protocol),
              "models": {name: {p.name: sha256(p) for p in model.iterdir() if p.is_file()} for name, model in models.items()},
              "implementations": {name: sha256(Path(__file__).with_name(name)) for name in
                                  ("original_comparison.py", "evaluate_encoder.py", "encoder.py", "prepare.py",
                                   "compare.py", "metrics.py", "seed_analysis.py", "run_extended_matrix.py")}}
    save_frozen(root / "frozen-inputs.json", frozen)
    reports, scores = evaluate_split(models, benchmark, "development", root)
    development = compare(scores, reports, rules)
    selection = "long-original" if all(r["passes_all"] for r in development.values()) else "unchanged"
    decision = {"selection": selection, "selection_frozen_before_reserved_test": True,
                "frozen_inputs_sha256": sha256(root / "frozen-inputs.json"), "development": development,
                "development_report_sha256": {name: sha256(root / "development" / name / "report.json") for name in models},
                "uncertainty": rules["uncertainty"]}
    save_frozen(root / "selection.json", decision)
    # This is the only path into reserved-test evaluation, after the durable decision.
    reports, scores = evaluate_split(models, benchmark, "test", root)
    confirmation = compare(scores, reports, rules)
    result = {"selection": selection, "selection_sha256": sha256(root / "selection.json"),
              "selection_changed_using_test": False, "test": confirmation,
              "test_report_sha256": {name: sha256(root / "test" / name / "report.json") for name in models},
              "long_original_confirmed": selection == "long-original" and all(r["passes_all"] for r in confirmation.values()),
              "limits": [rules["causal_limit"], rules["uncertainty"],
                         "source-backed agent-authored benchmark; no human review or natural-traffic claim",
                         "test results cannot select a different seed, budget or model"]}
    save_frozen(root / "confirmation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark", "inventory", "base", "criteria", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--protocol", type=Path,
                        default=Path(__file__).with_name("round2_training_protocol.json"))
    args = parser.parse_args()
    run(args.benchmark, args.inventory, args.base, args.criteria, args.output, args.protocol)
