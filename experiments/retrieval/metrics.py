"""Retrieval metrics and intent-group bootstrap, independent of model libraries."""

import math
import random
import statistics
from collections import defaultdict


def query_metrics(ranking: list[str], relevant: dict[str, int]) -> dict[str, float]:
    k = 10
    if not relevant or any(type(grade) is not int or grade < 0 for grade in relevant.values()):
        raise ValueError("require nonnegative integer relevance judgments")
    positive = {doc for doc, grade in relevant.items() if grade > 0}
    if not positive or len(ranking) != len(set(ranking)):
        raise ValueError("require positive judgments and unique ranked documents")
    top = ranking[:k]
    gains = [(2 ** relevant.get(doc, 0) - 1) / math.log2(rank + 2) for rank, doc in enumerate(top)]
    ideal = sum(
        (2**grade - 1) / math.log2(rank + 2)
        for rank, grade in enumerate(sorted(relevant.values(), reverse=True)[:k])
    )
    return {
        "recall_at_10": len(positive.intersection(top)) / len(positive),
        "ndcg_at_10": sum(gains) / ideal,
        "mrr_at_10": next((1 / (i + 1) for i, doc in enumerate(top) if doc in positive), 0.0),
    }


def summarize(rows: list[dict], seed: int = 20260921, repeats: int = 2000) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row["metrics"])
    if not groups:
        raise ValueError("no queries")
    metrics = ("recall_at_10", "ndcg_at_10", "mrr_at_10")
    means = [
        {m: statistics.mean(r[m] for r in group) for m in metrics}
        for _, group in sorted(groups.items())
    ]
    rng = random.Random(seed)
    samples = {m: [] for m in metrics}
    for _ in range(repeats):
        draw = rng.choices(means, k=len(means))
        for metric in metrics:
            samples[metric].append(statistics.mean(r[metric] for r in draw))
    return {
        "queries": len(rows),
        "groups": len(groups),
        "weighting": "equal group weight",
        "bootstrap_repeats": repeats,
        "metrics": {
            m: {
                "mean": statistics.mean(r[m] for r in means),
                "bootstrap_95_percentile": [
                    sorted(samples[m])[int(0.025 * repeats)],
                    sorted(samples[m])[int(0.975 * repeats)],
                ],
            }
            for m in metrics
        },
    }
