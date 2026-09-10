# Installed Python API throughput

Measured locally on Apple M5 arm64, macOS 26.6.2, Python 3.14.6, using the installed 0.1.0 wheel built from its source archive. Both models have 32,000 vocabulary entries. Raw measurements, model hashes, workload hashes and environment details are in [Unigram JSON](python-api-unigram-32000.json) and [BPE JSON](python-api-bpe-32000.json).

| Model | Input | Single calls, MB/s | Batch calls, MB/s | Batch speedup |
|---|---|---:|---:|---:|
| Unigram | Short | 9.28 | 14.35 | 1.55× |
| Unigram | Long | 15.49 | 16.29 | 1.05× |
| BPE | Short | 12.75 | 23.76 | 1.86× |
| BPE | Long | 24.09 | 25.89 | 1.07× |

Each workload contains 64 texts cycling through four fixed Bangla, English, Banglish and mixed-script sentences. Long inputs repeat each sentence 16 times. A single-call operation encodes all 64 texts individually; a batch operation encodes the same texts in one call. Each timing sample runs 100 operations, with five repeats and alternating measurement order. Values use the median elapsed time; MB means one million UTF-8 input bytes. Samples are retained in JSON.

Timing includes Python validation, JSON conversion, the native call, Go tokenization and returned Python IDs. Model loading is timed separately, and warmup, ID equivalence and normalized round-trip checks run outside the timed region. Both models passed these correctness checks. Batch calls reduce overhead most noticeably for short inputs on this fixture.

These are synthetic local microbenchmarks, not corpus throughput, retrieval scores or a cross-machine performance guarantee. They do not isolate native binding overhead. The earlier Go benchmarks use different inputs, so their numbers cannot establish a direct Go/Python ratio. CI runs only a tiny model smoke fixture and does not enforce a throughput threshold.

The same macOS wheel passed seven native API tests on each of Python 3.10, 3.11, 3.12, 3.13 and 3.14; logs are saved as `python-310-tests.txt` through `python-314-tests.txt`. Hosted Linux/macOS CI execution remains pending until the workflow is pushed.
