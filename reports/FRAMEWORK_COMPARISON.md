# Tokenizer framework comparison

Measured on 14 September 2026. **Our package does not win across the board.** On this workload, our BPE is faster than the tested Hugging Face byte-level BPE configuration, but BNLP and SentencePiece are faster than our package. Our models preserve normalized text more reliably than BNLP's published model. A freshly trained SentencePiece model matches our validation text preservation while producing substantially shorter sequences.

These are **encoding speed and tokenization quality measurements**, not embedding, search, classification, or linguistic segmentation accuracy. There are no human segmentation labels or matched trained encoders in this experiment, so reporting an NLP accuracy score would be unsupported.

## What was compared

| Configuration | Package version | Vocabulary | Model origin |
|---|---|---:|---|
| Our Unigram | bn-tokenizers-embedding 0.1.0 | 32,000 | Existing repository model |
| Our BPE | bn-tokenizers-embedding 0.1.0 | 32,000 | Existing repository model |
| BNLP pretrained | bnlp-toolkit 4.4.1 | 50,000 | Official `bn_spm.model` |
| SentencePiece / BNLP model | sentencepiece 0.2.2 | 50,000 | Exactly the same file as the BNLP row |
| SentencePiece Unigram | sentencepiece 0.2.2 | 32,000 | Trained for this comparison on our training split |
| Hugging Face byte-level BPE | tokenizers 0.23.2 | 32,000 | Trained for this comparison on our training split |

BNLP's `SentencepieceTokenizer` delegates to SentencePiece. Its basic and NLTK word tokenizers solve a different problem and are excluded. The direct SentencePiece/BNLP row checks the wrapper with the same model; it is not a separate independently trained competitor. [BNLP documentation](https://github.com/sagorbrur/bnlp/blob/main/docs/README.md), [SentencePiece](https://github.com/google/sentencepiece), [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers).

## Encoding speed

Higher MB/s is better. Each value is calculated from the median of five full validation passes. MB means one million UTF-8 input bytes.

| Configuration | Individual calls, MB/s | Batches of 64, MB/s |
|---|---:|---:|
| Our Unigram 32k | 9.13 | 10.72 |
| Our BPE 32k | 14.77 | 19.13 |
| BNLP pretrained 50k | 41.08 | 40.76* |
| SentencePiece / BNLP model 50k | 42.21 | 41.95 |
| SentencePiece Unigram 32k | **47.62** | **47.72** |
| Hugging Face byte-level BPE 32k | 5.74 | 5.77 |

\* BNLP's tested wrapper has no native batch method; this entry loops over its `text2id` method. It is labeled as a Python loop in the raw results.

For batches, our BPE is **3.32× faster** than this Hugging Face byte-level BPE configuration. BNLP is **2.13× faster** than our BPE, and the new SentencePiece Unigram model is **2.49× faster**. These are configuration-level comparisons: differing vocabularies, segmentation rules and output lengths affect work performed. They do not establish that Go beats Rust, C++, or Python, or that all Hugging Face tokenizers have the performance of this one configuration.

## Text preservation and coverage

“Exact round trip” means `decode(encode(text))` equals the shared NFC/whitespace-normalized input. It does not mean linguistic accuracy or preservation of the original whitespace and Unicode byte spelling.

| Configuration | Exact round trips | Texts with unknown IDs | Sequence tokens/word | Texts over 510 tokens | Edge cases preserved |
|---|---:|---:|---:|---:|---:|
| Our Unigram 32k | 14,779/14,779 (100%) | 0 | 2.366 | 0.061% | 16/16 |
| Our BPE 32k | 14,779/14,779 (100%) | 0 | 2.311 | 0.061% | 16/16 |
| BNLP pretrained 50k | 12,998/14,779 (87.95%) | 647 | 1.861 | 0.007% | 8/16 |
| SentencePiece / BNLP model 50k | 12,998/14,779 (87.95%) | 647 | 1.861 | 0.007% | 8/16 |
| SentencePiece Unigram 32k | 14,779/14,779 (100%) | 0 | **1.459** | 0% | 15/16 |
| Hugging Face byte-level BPE 32k | 14,779/14,779 (100%) | 0 | 3.205 | 1.164% | 16/16 |

BNLP emitted 1,089 unknown IDs, or 0.243% of its output tokens. Its round-trip failures also include normalization changes; they are not all unknown-character failures. Its published normalization policy differs from our preservation contract, so these failures do not by themselves establish a BNLP bug. The raw edge-case outputs show emoji and unsupported scripts becoming unknown text, and compatibility characters and joiners changing.

The external SentencePiece model uses identity normalization and byte fallback. Its one edge-case failure is the literal `▁` character, which its whitespace representation turns into a space. This is an authored stress test, not an estimate of how often users encounter the issue. Our two models and Hugging Face byte-level BPE preserved all 16 authored cases, including empty text, emoji, NUL, mixed scripts, combining marks and token-like literals. Inputs, expected outputs and decoded outputs are saved in the JSON.

Sequence tokens/word includes every emitted ID, including our standalone space tokens. SentencePiece and byte-level BPE can represent a space together with adjacent text. The new SentencePiece model emits **38.3% fewer IDs** overall than our Unigram model. This is relevant to encoder sequence budgets, but it does not prove better linguistic segmentation or retrieval. An unknown ID can also collapse otherwise lengthy input, so lower token counts from a lossy model need caution. The 510-token threshold assumes a 512-position encoder reserving two control positions; it is not an intrinsic tokenizer limit. JSON p95 lengths use the nearest-rank convention.

## Results by language bucket

Each cell is **sequence tokens/word · exact round-trip percentage**. Bucket names are corpus labels; “code-mixed” is the existing script-based proxy.

| Configuration | Bangla (4,925) | Banglish (6,160) | Code-mixed (156) | English (3,538) |
|---|---:|---:|---:|---:|
| Our Unigram 32k | 2.447 · 100% | 2.326 · 100% | 2.379 · 100% | 2.002 · 100% |
| Our BPE 32k | 2.387 · 100% | 2.276 · 100% | 2.319 · 100% | 1.962 · 100% |
| BNLP pretrained 50k | 1.381 · 92.53% | 2.714 · 77.68% | 2.203 · 77.56% | 1.838 · 99.92% |
| SentencePiece / BNLP model 50k | 1.381 · 92.53% | 2.714 · 77.68% | 2.203 · 77.56% | 1.838 · 99.92% |
| SentencePiece Unigram 32k | 1.480 · 100% | 1.433 · 100% | 1.594 · 100% | 1.406 · 100% |
| Hugging Face byte-level BPE 32k | 4.566 · 100% | 1.346 · 100% | 2.491 · 100% | 1.295 · 100% |

The tested Hugging Face configuration is compact for English and Banglish but fragments Bangla more heavily. This result concerns the selected byte-level BPE training recipe. It is not a result for Hugging Face Unigram, WordPiece, or pretrained multilingual tokenizers.

## Method and limitations

- **Host:** Apple M5 arm64, 10 logical CPUs, macOS 26.6.2, Python 3.12.13. Our installed native wheel was used; no subprocess tokenization. All configurations ran in the same isolated environment.
- **Evaluation:** all 14,779 frozen validation records, totaling 3,013,257 normalized UTF-8 bytes. No validation sampling or length filtering. The test split remains unused. Text and timing order use seed 20260914; configuration/mode order is shuffled each repeat to reduce timing-order bias.
- **Timed work:** end-to-end public encoding calls returning Python integer-ID lists, including each API's internal normalization, pretokenization and conversion overhead. Hugging Face also constructs its normal Encoding results before IDs are extracted. Shared input normalization happens before timing; any normalization repeated inside an API remains timed. Imports, model loading, downloads, training, quality checks and decoding are outside timers. Results are discarded after each call; no encoder inference runs.
- **Concurrency:** sequential single calls; batches of 64 with SentencePiece `num_threads=1` and `TOKENIZERS_PARALLELISM=false`. This is not a maximum-throughput multicore comparison. All candidates were warmed up; single and batch IDs were checked for equality outside timing.
- **Training:** external models were offered all 563,476 training records, once each, after NFC/whitespace normalization. SentencePiece logged that it skipped one record containing a reserved character and loaded 563,475. Our existing models use weighted bucket/sentence/word contributions. Thus this is the same source split, **not an identical training objective or fully controlled algorithm ablation**. BNLP's published model is trained on Bengali Wikipedia; any overlap with our Wikipedia-derived evaluation records is unknown.
- **External settings:** SentencePiece Unigram uses 32k target vocabulary, full character coverage, byte fallback, identity normalization, maximum piece length 12, no configured sentence sampling (`input_sentence_size=0`) and one training thread. Hugging Face uses 32k byte-level BPE, the full 256-byte alphabet, minimum frequency 2, no prefix space and no special tokens. Its default byte-level pretokenization differs from our whitespace-word boundaries. Full settings and hashes are recorded.
- **Uncertainty:** five timing repeats on one small, warm corpus and one machine; no hardware sweep, cold-cache benchmark or confidence interval. The 156-record code-mixed bucket is small. The 16 edge cases are authored diagnostics, not a human-labeled accuracy benchmark. Native training recipes can vary with library versions; saved model hashes identify the exact artifacts used. Training durations in metadata are single setup runs, not a controlled training-speed comparison. JSON `model_bytes` values are serialized artifact sizes in different formats, not runtime memory measurements.

## Reproduce

Run from the repository root after building our wheel and preparing the corpus/models as described in the README. Benchmark dependencies are separate from the package's runtime dependencies.

```sh
python3.12 -m venv .venv/comparison
.venv/comparison/bin/python -m pip install -r benchmarks/comparison-requirements.txt
.venv/comparison/bin/python -m pip install dist/*.whl
export TOKENIZERS_PARALLELISM=false
export NLTK_DATA="$PWD/data/raw/nltk"
.venv/comparison/bin/python -m nltk.downloader -d "$NLTK_DATA" punkt punkt_tab
mkdir -p models/comparison
curl --fail --location \
  https://raw.githubusercontent.com/sagorbrur/bnlp/master/model/bn_spm.model \
  -o models/comparison/bnlp.model
.venv/comparison/bin/python benchmarks/train_comparison.py
.venv/comparison/bin/python benchmarks/compare_frameworks.py --cpu-model "Apple M5"
.venv/comparison/bin/python -m unittest discover -s benchmarks -p 'test_*.py' -v
```

Replace `--cpu-model` with your actual hardware model. The BNLP download URL can change upstream: compare its SHA-256 against the saved results before treating a rerun as identical. Models and training text remain in ignored `models/comparison/`; no external library was added to our package's runtime requirements.

Evidence: [raw measurements and model hashes](framework-comparison.json), [external training settings and hashes](comparison-training.json), [installed dependency versions](comparison-environment.json), [benchmark harness](../benchmarks/compare_frameworks.py), [training script](../benchmarks/train_comparison.py).

## What this means for the project

Our measured strengths are preservation of mixed Unicode text and better speed than this byte-level BPE baseline. **SentencePiece is the stronger speed/sequence-length baseline in this experiment.** The next engineering work should profile our Go tokenizer and Python binding separately, investigate standalone-space and vocabulary-training costs, and retain the preservation tests while optimizing. Claims of better embedding or search accuracy still require matched encoder training and labeled retrieval evaluation.
