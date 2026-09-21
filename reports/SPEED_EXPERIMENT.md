# Training-word cache and native space fusion

Measured 19 September 2026, Apple M5 arm64, macOS 27.0, Python 3.12.13. The selected fused model is **12.0% faster than the tested BNLP tokenizer in batches of 64**. Individual calls are still slower. No embedding/retrieval accuracy improvement has been measured.

## Final comparison

Median of nine complete validation passes per configuration and call style, shuffled timing order, 14,779 records / 3,013,257 normalized UTF-8 bytes. Same single-thread comparison settings as [the framework comparison](FRAMEWORK_COMPARISON.md). Shared normalization, model loading and quality checks are outside timing; Python ID-list production is inside. BNLP's batch row loops over its single-text API. Original validation/test boundaries remain unchanged; the test split was not evaluated.

| Model | Individual MB/s | Batch MB/s | Validation tokens | Normalized round trips |
|---|---:|---:|---:|---:|
| Original BPE 32k, current runtime | 31.83 | 39.02 | 555,829 | 100% |
| BPE 32k + 64k training words | 35.58 | 43.42 | 555,829 | 100% |
| Fused BPE 29,341 + 64k training words | 36.77 | 46.09 | 375,823 | 100% |
| BNLP pretrained 50k | 41.28 | 41.16 | 447,589 | 87.95% |
| Direct SentencePiece / BNLP model | 42.41 | 41.69 | 447,589 | 87.95% |
| SentencePiece Unigram 32k | 48.10 | 47.44 | 350,939 | 100% |

Caching alone improves batch speed by 11.3% against our original BPE in the final run and by 5.5% against BNLP, preserving all original IDs. Fusion plus caching improves batch speed by 18.1% against original BPE and reduces output IDs by 32.4%. It still produces 7.1% more IDs than SentencePiece Unigram. The 50 MB/s engineering target has not been reached.

These are configuration-level results on one warm validation corpus and machine. Training recipes and vocabularies differ. Round trips measure preservation of normalized text, not semantic accuracy. BNLP's normalization contract differs, so its failures here do not establish a library bug. The 156-record code-mixed bucket is too small for strong generalization claims. New-source performance, tail latency, end-to-end encoder speed and multi-seed retrieval accuracy remain unmeasured.

## What changed

- Optional `cache_words` stores training words in the JSON model. `Load` computes their actual existing segmentation once; encoding appends those IDs on a hit. The vocabulary and original IDs do not change. `New` remains uncached. Cache size is fixed by the artifact; runtime user text never grows it.
- Version-2 BPE adds `space_fusion`, an ordered list of base token IDs. New IDs start at `260 + len(pieces)`. Each represents space followed by one listed base ID, including byte fallback. Encoding performs fusion in Go while emitting each word; decoding reverses it. Duplicates, reserved IDs, space itself, recursive/out-of-range IDs and fusion in version 1 or Unigram are rejected.
- The existing space-fusion research artifact is converted to a self-contained version-2 model after checking its base-model and training hashes. The original artifact itself is still not directly loadable. Original models remain untouched under `models/`.
- Go `Expand` restores base IDs for verification and evaluation. CLI fertility and byte-fallback accounting expand fused IDs; sequence lengths and vocabulary utilization measure emitted IDs.

Training-word selection counts only records explicitly labeled `train`, excludes existing vocabulary strings, sorts by descending frequency with lexical tie breaks, and limits words to 1024 UTF-8 bytes. Fusion uses the earlier training-only frequency experiment. No validation words are inserted into the cache. Validation was used to compare cache sizes.

## Cache-size tradeoff

The seven-repeat ablation measured fused-model batch throughput of 38.72, 43.59, 46.37 and 46.73 MB/s for 0, 16k, 64k and 128k additional training words respectively. The 128k cache adds little throughput for substantially more startup and memory, so 64k is the selected candidate.

Five fresh-process samples each; RSS is whole-process peak, not isolated model size. Filesystem caches were not flushed. Constructor timings include lazy native-library startup.

| Model | Constructor median | Peak process RSS median |
|---|---:|---:|
| BPE 32k, no extra words | 40.48 ms | 57.19 MiB |
| BPE 32k + 64k words | 104.09 ms | 72.86 MiB |
| Fused BPE, no extra words | 17.73 ms | 38.61 MiB |
| Fused BPE + 64k words | 77.11 ms | 58.97 MiB |
| Fused BPE + 128k words | 141.89 ms | 78.39 MiB |

Reuse loaded instances. Fused models use a 16k base plus 13,341 fusion IDs, which helps explain their lower startup/memory than cached 32k BPE; the experiment does not isolate individual implementation costs.

## Verification and artifacts

All eight generated models passed full-validation plus 16 stress-case verification (14,795 sequences each): unchanged IDs for ordinary BPE, exactly recoverable base IDs for fusion, and normalized round trips. All 11 installed-wheel Python tests, four benchmark tests, Go race tests, vet, Ruff lint and formatting passed. CLI tests cover a fused leading byte of an emoji so fallback counts cannot silently disappear.

The local wheel and source archive are in `dist/speed/`, version 0.1.0. The wheel was built from the source archive and tested on this macOS arm64 host; no Linux validation or publication was performed. This build requires macOS 27 per its wheel tag. Model artifacts are ignored by Git under `models/speed/`.

- [Fresh HEAD baseline, seven repeats](speed-baseline.json), starting from `da0dc00`.
- [All cache sizes and fusion ablation, seven repeats](speed-ablation.json).
- [Final installed-wheel comparison, nine repeats](speed-final.json), including native-library and harness hashes.
- [Model/training provenance](speed-models.json).
- [ID equivalence and fresh-process load samples](speed-verification.json).

The baseline and ablation preceded minor harness changes (extra model selection and native hash recording); each records its own harness hash. Dependency versions are in comparison reports. Historical September 14–16 reports retain their original values and environments.

## Reproduce and use

After preparing the corpus, existing base models and the earlier `models/comparison/space-fusion.json` artifact:

```sh
uv build --out-dir dist/speed
# Install the new wheel in an environment with comparison-requirements.txt.
python -m pip install --force-reinstall --no-deps dist/speed/*.whl
python benchmarks/prepare_speed_models.py
export TOKENIZERS_PARALLELISM=false
export NLTK_DATA="$PWD/data/raw/nltk"
python benchmarks/compare_frameworks.py --repeats 9 --cpu-model 'your CPU' \
  --model models/speed/bpe-cache-64000.json \
  --model models/speed/fused-cache-64000.json \
  --output reports/speed-final.json
python benchmarks/verify_speed_models.py
```

```python
from bn_tokenizers_embedding import Tokenizer

# Preserves original 32k BPE IDs:
with Tokenizer("models/speed/bpe-cache-64000.json") as tokenizer:
    ids = tokenizer.encode_batch(["আমি ভালো আছি", "ami valo asi"])

# New vocabulary; requires encoder adaptation/training:
with Tokenizer("models/speed/fused-cache-64000.json") as tokenizer:
    ids = tokenizer.encode_batch(["আমি ভালো আছি", "ami valo asi"])
```

## Next accuracy experiment

There is no trained embedding encoder or labeled retrieval evaluation in this repository. The next milestone is a matched encoder experiment with original BPE, fused BPE, BNLP and SentencePiece tokenizers, using verified Bangla–Banglish positives and hard negatives. Evaluate Recall@10 and nDCG@10 by language bucket, disclose parameter/compute differences, and repeat seeds. Keep the frozen test split unused until model selection finishes. Semantic improvements cannot be inferred from these tokenization results.
