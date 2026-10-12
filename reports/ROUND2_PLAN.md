# Retrieval follow-up — second experiment

The user requested all five follow-ups after the completed first experiment.
The first experiment's datasets, criteria, checkpoints and selected unchanged E5
remain frozen. This file tracks the new work; pending work is not a result.

| Work | Required outcome | Status |
|---|---|---|
| Expanded judgments | Union of all 19 existing encoders' top-10 candidates; explicitly judge every new pair, preserve prior labels, recompute comparisons and coverage | Complete: 6,711 new agent judgments added to 2,496 preserved labels; all 19 systems have complete top-10 coverage. Graded and exact-match rescoring preserve frozen selection. [Results](encoder-expanded-pool-analysis-v1.json). |
| Harder benchmark | Larger source-grounded six-domain benchmark, confusing neighbours and Banglish spelling variants; source-disjoint development/test groups and training-overlap audit | Source audit and pinned downloads complete; material/labels/split pending. |
| Original-tokenizer training | Longer training, reviewed hard negatives and balanced English replay; execute matched runs and score fresh development/test after frozen selection | All 128 mined candidates reviewed; 79 verified negatives retained, 18 known positive errors removed. Five-arm protocol frozen; original/context smoke and reload pass. Long runs and fresh evaluation pending. [Review](HARD_NEGATIVE_REVIEW.md). |
| Stronger transfer | Implement and measure a compatible transfer method, with mean-initialization control and matched training budgets | Both sparse context tables rebuilt on the same revised 7,982-pair alignment; real apply, hard-negative training, inference and reload verified. Matched long training and quality comparison pending. |
| Deployment workloads | Execute fresh-process cold starts, larger real-document indexes, long inputs and concurrent requests; report throughput, p50/p95/p99 and memory | Executed for unchanged and three seed-17 controls: 18 workloads/model, three cold loads/model; concurrent/serial rankings identical. [Report](DEPLOYMENT_WORKLOADS.md). |

Agent-authored queries and agent judgments are authorized by the user. They must
carry that provenance; independent human review and natural user traffic are not
claimed. Unknown relevance outside the pooled candidates remains unknown.

New selection uses fresh development material and the original practical gates
(+0.03 Banglish→Bengali nDCG, positive paired interval, ≤0.01 English loss,
≤1.25× latency/memory). The new protocol must bind data and training settings
before adapted scoring. Reserved test is scored only after the new choice freezes.
