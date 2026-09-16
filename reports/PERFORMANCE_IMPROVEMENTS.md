# Performance improvements and space-aware vocabulary experiment

Measured 16 September 2026 on Apple M5, macOS arm64, Python 3.12.13. **The existing Python API is now about twice as fast on our validation corpus, with identical token IDs.** A separate research prototype reduces sequence lengths by 32.4% compared with our current 32k BPE. We have narrowed the gap to SentencePiece, but have not beaten it overall.

## Implemented: faster encoding with unchanged models

Five full validation passes per configuration, median throughput, batches of 64, 14,779 records and 3,013,257 normalized UTF-8 bytes. Higher MB/s is better. Both baseline and optimized wheels were measured during this session with the same comparison harness; this is not a comparison against an older day's timing.

| Model / call style | Before, MB/s | After, MB/s | Speedup |
|---|---:|---:|---:|
| Unigram 32k / individual | 8.77 | 18.27 | 2.08× |
| Unigram 32k / batch | 10.50 | 22.18 | 2.11× |
| BPE 32k / individual | 14.80 | 30.90 | 2.09× |
| BPE 32k / batch | 19.10 | 36.70 | 1.92× |

The optimized BPE's batch throughput is still below BNLP's 40.95 MB/s and the freshly trained SentencePiece model's 44.71 MB/s in the after run. The tested Hugging Face byte-level BPE measured 5.66 MB/s. These comparisons retain the corpus, algorithm, vocabulary and threading qualifications in [the framework comparison](FRAMEWORK_COMPARISON.md). Control timings vary between runs; they are local measurements, not cross-machine guarantees.

Changes in the production code:

1. **Binary encoding interface:** text crosses the Python/Go boundary as length-prefixed UTF-8; token IDs return as packed 32-bit integers. Encoding no longer serializes and parses JSON in both directions. Loading, decoding, normalization and lifecycle operations retain JSON, now using UTF-8 rather than ASCII escapes.
2. **Fewer temporary allocations:** BPE appends directly to the output buffer. Unigram reuses one per-call scratch buffer across words instead of allocating several arrays for each word. Long words grow the scratch buffer as needed.
3. **Cheaper BPE lookups:** merge-pair keys use one 64-bit integer, with model validation ensuring IDs fit the representation.
4. **Immutable vocabulary cache:** loading a model precomputes the actual segmentation of its vocabulary strings. This is bounded by the vocabulary, does not retain caller text, and does not assume that a learned piece necessarily encodes as one ID. Returned lists do not alias cached storage. Direct Go `New` construction remains uncached; `Load` prepares the cache.
5. **Normalization fast path:** already normalized whitespace avoids splitting and rebuilding the whole string. NFC and invalid-UTF-8 behavior are preserved.

These changes are shipped together in the tested local wheel. The total speedup is measured; individual contributions have not been isolated in a complete ablation.

## Native Go measurements

The Go-only benchmark encodes the same full validation corpus, excluding Python and model loading. Baseline source was extracted from commit `2471dc6e061b9009aac8b2c409321c6fe2b8f8c7`; both final measurements ran without profiling enabled, using the same Go toolchain and benchmark.

| BPE 32k metric | Before | After | Change |
|---|---:|---:|---:|
| Median throughput | 30.20 MB/s | 48.46 MB/s | 1.60× faster |
| Allocations per corpus pass | 316,555 | 31,251 | 90.1% fewer |
| Allocated bytes per corpus pass | ~58.25 MB | ~21.22 MB | 63.6% fewer |

Allocated bytes are cumulative allocation volume, not resident memory. The CPU profile pointed to BPE merge lookups and allocation/runtime work; it was used diagnostically, not as the final throughput measurement. The Go-only result is not directly equivalent to competitors' Python API timings.

## Startup and memory tradeoff

Each sample constructs a tokenizer in a fresh Python process. Five samples per model/build; constructor timing includes lazy native-library startup. RSS is **whole-process peak**, not isolated model memory or a leak estimate. Filesystem caches were not flushed.

| Model | Before load | After load | Before peak RSS | After peak RSS |
|---|---:|---:|---:|---:|
| BPE 32k | 25.24 ms | 39.90 ms | 50.31 MiB | 59.22 MiB |
| Unigram 32k | 15.37 ms | 36.04 ms | 42.09 MiB | 50.86 MiB |

The cache moves work to initialization and retains additional data. Reuse a loaded tokenizer for repeated requests; these improvements target that usage. The measurements compare complete builds, so they do not isolate cache costs from every other change.

## Experimental: fuse spaces with adjacent tokens

This experiment starts with our existing **16k BPE** and learns space-plus-next-token pairs from the **training split only**, using unweighted occurrence counts. Each new ID expands exactly to the original space ID and original token ID. This avoids overloading a literal Unicode marker such as `▁`.

The resulting vocabulary has **29,341 IDs**: 16,000 base IDs plus 13,341 learned pairs, below the 32k budget. This is a Python post-tokenization research prototype and a separate experimental artifact, **not a model supported by the production Go/Python loader**. Its runtime throughput has not been benchmarked.

| Configuration | Vocabulary | Total validation IDs | Tokens/word |
|---|---:|---:|---:|
| Existing BPE 16k, experiment's base | 16,000 | 601,516 | 2.501 |
| Existing BPE 32k | 32,000 | 555,829 | 2.311 |
| Experimental space-fused BPE | 29,341 | 375,823 | 1.563 |
| SentencePiece Unigram baseline | 32,000 | 350,939 | 1.459 |

The experiment produces **37.5% fewer IDs than its own 16k base**, and **32.4% fewer than our existing 32k BPE**. It still emits **7.1% more IDs than SentencePiece** overall. Only one validation record exceeds 510 tokens, compared with nine for our existing 32k BPE. The 510 threshold assumes two reserved control positions in a 512-position encoder.

| Bucket | Experimental tokens/word | SentencePiece tokens/word |
|---|---:|---:|
| Bangla | 1.637 | 1.480 |
| Banglish | 1.525 | 1.433 |
| Code-mixed | 1.586 | 1.594 |
| English | 1.231 | 1.406 |

The experimental layer recovers the exact base IDs and normalized text for every validation record. The small code-mixed bucket has only 156 records. Training weights, algorithms and vocabulary sizes differ, so these are configuration comparisons, not an isolated language-quality result. New IDs change the embedding-table contract: using this vocabulary in an encoder requires integration and training.

## Correctness and scope

- Production Unigram and BPE IDs are byte-for-byte identical to the baseline on **14,785 sequences**: the entire validation split plus six edge cases. The saved digest covers the ordered integer-ID sequences, not merely decoded text or token counts.
- Both production models retain **100% normalized round-trip preservation** on validation and all **16/16** framework-comparison edge cases. Sequence lengths and unknown-ID counts are unchanged by the speed optimization.
- Go race tests and vet pass. Native protocol tests cover malformed/truncated input, invalid handles, empty batches and embedded NUL. Cache tests check competing segmentations and output ownership.
- Short fuzz runs passed for normalization equivalence and malformed binary requests. Seven installed-wheel Python tests pass on Python 3.12 and 3.14; four benchmark/experiment tests pass. The newly optimized build has not been exercised on Linux or the other Python versions yet.
- The test split remains unused. No additional natural code-mixed corpus was collected, no full controlled training-recipe sweep was run, and no embedding/retrieval accuracy improvement is claimed. Those require further experiments and labeled evaluation data.

## Reproduce and inspect

The original wheel remains in `dist/`; the optimized wheel and source archive are in `dist/optimized/`. Both are local development builds with package version 0.1.0. Use separate environments or force reinstall when switching. No package was published.

```sh
# Install the optimized local build in your comparison environment.
python -m pip install --force-reinstall --no-deps dist/optimized/*.whl
export TOKENIZERS_PARALLELISM=false
export NLTK_DATA="$PWD/data/raw/nltk"
python benchmarks/compare_frameworks.py --cpu-model "Apple M5" \
  --output reports/performance-after.json
python benchmarks/load_cost.py --model models/bpe-32000.json
python benchmarks/space_fusion.py
python -m unittest discover -s benchmarks -p 'test_*.py' -v

# Go-only corpus benchmark, with the desired source checkout and toolchain.
BNTOK_MODEL="$PWD/models/bpe-32000.json" \
BNTOK_CORPUS="$PWD/data/processed/validation.jsonl" \
go test ./tokenizer -run '^$' -bench '^BenchmarkValidation$' -benchtime=3x -count=3
```

Replace the CPU label with your actual model. Benchmark-only dependencies and external models are documented in [FRAMEWORK_COMPARISON.md](FRAMEWORK_COMPARISON.md). The space-fusion artifact is saved under ignored `models/comparison/space-fusion.json`; its hash and corpus hashes are recorded.

Evidence: [before](performance-before.json), [after](performance-after.json), [ID equivalence](performance-id-equivalence.json), [Go before](profile-bpe-before.txt), [Go after](profile-bpe-after.txt), [space-fusion results](space-fusion.json), [BPE load before](load-bpe-before.json), [BPE load after](load-bpe-after.json), [Unigram load before](load-unigram-before.json), [Unigram load after](load-unigram-after.json), [build provenance](performance-builds.json).

The next concrete steps are to profile the remaining encoding cost, integrate and benchmark the space-fusion format if its tradeoffs are acceptable, and run controlled tokenizer/encoder experiments before making retrieval-accuracy claims.

## Subsequent code cleanup

A simplify/deslop pass removed duplicate UTF-8 conversion, unused response fields, redundant state and unnecessary training-library imports in benchmark helpers. It retained native buffer validation and ownership checks. The cleaned wheel is in `dist/simplified/`; the timings and build hashes above remain the historical optimization measurements, not a new performance claim for this cleanup.
