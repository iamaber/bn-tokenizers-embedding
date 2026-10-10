# Larger serving workloads — 9 October 2026

The four measured encoders are unchanged E5 and the first experiment's equal-data
original/Unigram/BPE seed-17 checkpoints. These measurements extend the earlier
small-index timing; they do not revise its frozen quality decision.

Each fresh process indexed 10,000 distinct audited natural training forms, then
served against 1,000- and 10,000-document exact-search indexes. The same workload
includes Bengali, Banglish and English, short requests and synthetic 128-/512-word
requests, with concurrency 1, 4 and 8. All concurrent top-10 rankings matched the
serial reference. Each of the 18 settings has 36 requests, service latency and
queued burst response distributions; each model also has three fresh-process
model/index loads and first responses.

At 10,000 documents, short inputs, concurrency 4:

| Encoder | Requests/s | Service p95 ms | Burst response p95 ms | Model/index first response ms | Cold-process peak MiB |
|---|---:|---:|---:|---:|---:|
| Unchanged E5 | 391.7 | 10.94 | 91.17 | 714.8 | 873.4 |
| Adapted original | 388.9 | 10.82 | 91.52 | 723.4 | 873.2 |
| Adapted Unigram 32k | 367.3 | 12.57 | 97.60 | 125.5 | 458.8 |
| Adapted BPE 32k | 382.3 | 11.79 | 93.77 | 127.9 | 466.0 |

The custom embedding tables reduce loading time and process memory, while this
workload shows similar short-query throughput. Longer inputs take substantially
more time; increasing concurrency improves throughput but raises request latency.
These are measured tradeoffs, not evidence of retrieval quality improvement.

The first-response column starts after Python/module imports and includes model
and saved-index loading plus the first query/search. OS file caches can be warm.
Service latency includes the tokenizer mutex but excludes the executor queue;
burst response latency includes that queue for 36 simultaneous submissions.
The p99 is the maximum of 36 samples, so it is not a stable production tail
estimate. Long requests repeat natural sentences and can truncate at 512 tokens;
they are stress tests, not natural traffic. This is CPU exact search, not an ANN,
network, distributed-service or machine-reboot benchmark. One checkpoint per arm
does not measure training-seed variation in serving cost.

Full sample distributions, token/truncation counts, source/model/workload hashes,
index cost, cold runs and peak memory are preserved in
[the four machine-readable reports](deployment-v2/). Text/index artifacts stay
in ignored local data directories.

Reproduction:

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/retrieval/bin/python \
  experiments/retrieval/deployment.py workload \
  --alignment data/processed/alignment-expanded-v1 \
  --output data/processed/deployment-workload-new
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/retrieval/bin/python \
  experiments/retrieval/deployment.py benchmark \
  --model models/encoders/equal-data/original-seed17 \
  --workload data/processed/deployment-workload-new \
  --output data/processed/deployment-new/original
```
