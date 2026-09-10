# Bangla–English–Banglish tokenizers: Python API, Go engine

Python package backed by native Go Unigram/BPE tokenization, with Go training tools, a collected checksum-locked corpus, grouped held-out splits, and reproducible intrinsic evaluation. The initial Unigram/BPE models use 16k, 32k and 48k vocabularies. These are tokenizer candidates for embedding experiments, **not a trained embedding model or a demonstrated retrieval improvement**.

Read [the research audit](RESEARCH_AUDIT.md), [corpus documentation](data/CORPUS.md), and [measured results](reports/RESULTS.md). The original [research plan](RESEARCH.md) remains the downstream experimental roadmap.

## Python package

The built wheel in `dist/` contains the native Go runtime and has no third-party Python runtime dependencies. It has been installed and tested locally on macOS 26 arm64. Nothing has been published to PyPI. Models are separate artifacts under `models/`; the wheel does not bundle training data or automatically download a model.

```sh
python -m pip install dist/bn_tokenizers_embedding-0.1.0-py3-none-macosx_26_0_arm64.whl
```

```python
from bn_tokenizers_embedding import Tokenizer, normalize

with Tokenizer("models/unigram-32000.json") as tokenizer:
    ids = tokenizer.encode("আমি ভালো আছি। ami valo achi 🙂")
    print(tokenizer.decode(ids))
    print(tokenizer.encode_batch(["meetingটা কখন?", "When is the meeting?"]))
    print(tokenizer.vocab_size)

print(normalize("  আমি\tভালো আছি। "))
```

Use `models/bpe-32000.json` for BPE with the same API. Model loading happens once per `Tokenizer`; single and batch calls execute inside Go through `ctypes`, without a subprocess. A context manager or `close()` releases the native handle; finalization also releases forgotten handles. Instances support concurrent calls; finish active work before closing. Create a separate instance in each spawned worker process rather than trying to pickle a native handle.

Building from source requires Go 1.24+, a C compiler for the CGo bridge, and Python 3.10+. Installed wheels require neither Go nor a compiler. The core Go tokenizer/CLI itself does not require CGo. Source builds produce a platform-specific `py3-none` wheel, not a CPython-version-specific extension; other OS/architecture combinations need their own builds and validation.

```sh
# Standard source installation (Go must be on PATH).
python -m pip install .

# This workspace's local Go toolchain.
BNTOK_GO="$PWD/.tools/go/bin/go" python -m pip install .

# Build both an sdist and a wheel; uv builds the wheel from the source archive.
BNTOK_GO="$PWD/.tools/go/bin/go" uv build

# Tests exercise the installed native library, not a mock.
python -m unittest discover -s tests -v
```

The Python API deliberately stays small: `Tokenizer`, `normalize`, `encode`, `encode_batch`, `decode`, `vocab_size`, and lifecycle methods. Corpus preparation and training remain available through the Go CLI below. Native inference requires only `tokenizer/` and the Unicode dependency; Parquet processing is not linked into the wheel.

## CI and Python API benchmarks

[CI](.github/workflows/ci.yml) builds source archives and wheels on Linux and macOS for Python 3.10–3.14, then tests each installed wheel in a clean environment. It also runs formatting, Go race tests and vet with Go 1.24 and stable, and builds the CLI without CGo. Tests and benchmark smoke checks use tiny fixtures and require no corpus downloads. Uploaded wheels are CI artifacts; Linux release wheels still need portable platform packaging and validation before publishing.

Locally, the same macOS arm64 wheel passed all seven tests on each of Python 3.10–3.14. The hosted Linux/macOS matrix has not run yet; it starts when these changes are pushed.

Measure the installed Python API with a trained model:

```sh
python benchmarks/python_api.py --model models/unigram-32000.json > reports/python-api-unigram-32000.json
python benchmarks/python_api.py --model models/bpe-32000.json > reports/python-api-bpe-32000.json
```

The benchmark alternates single-call and batch timing order across repeats and verifies matching IDs and normalized round trips outside timing. See [Python throughput results](reports/PYTHON_API.md) for methodology and limitations. Omitting `--model` runs a tiny smoke fixture, not a representative performance benchmark.

## Go tools

Requires Go 1.24 or later. A checksum-verified Go 1.27.1 toolchain is already available locally at `.tools/go/bin/go` in this workspace. The Go library and CLI run without Python, CGo, a GPU, or network access. Corpus preparation additionally uses the Go Parquet reader.

```sh
go test -race ./...
go build -o bin/bntok ./cmd/bntok

# Raw data and trained models are already present in this workspace.
bin/bntok encode -model models/unigram-32000.json \
  -text 'আমি আজ ভালো আছি। ami aaj valo asi. meetingটা কখন?'

# Reproduce the downloads and grouped splits.
bin/bntok fetch
bin/bntok prepare
bin/bntok audit

# Train the recommended first candidate and its controlled baseline.
bin/bntok train -vocab 32000 -out models/unigram-32000.json
bin/bntok train -algorithm bpe -vocab 32000 -out models/bpe-32000.json
bin/bntok evaluate -model models/unigram-32000.json
bin/bntok evaluate -model models/bpe-32000.json

# Full vocabulary sweep; each model uses the same training snapshot.
make matrix

# Separate alternative-mixture ablation.
bin/bntok train -balanced -out models/unigram-32000-balanced.json
bin/bntok evaluate -model models/unigram-32000-balanced.json
```

For the bundled workspace toolchain, substitute `.tools/go/bin/go` for `go`, or use `make GO=.tools/go/bin/go build`. In restricted environments, `GOCACHE=/tmp/bn-go-build GOMODCACHE=/tmp/bn-go-mod` uses the caches populated during this task. Downloaded files and trained models are ignored by Git but live in this repository's directories. Keep `data/sources.lock.json`, `go.sum`, source code, and reports in version control.

## Go API

```go
package main

import (
    "fmt"
    "log"
    "github.com/iamaber/bn-tokenizers-embedding/tokenizer"
)

func main() {
    tok, err := tokenizer.Load("models/unigram-32000.json")
    if err != nil { log.Fatal(err) }
    ids := tok.Encode("বইগুলোতে meetingটা ami valo achi 🙂")
    text, err := tok.Decode(ids)
    if err != nil { log.Fatal(err) }
    fmt.Println(ids, text)
}
```

An initialized tokenizer can be shared across goroutines. `Encode` is deterministic; `Decode(Encode(text))` equals `tokenizer.Normalize(text)`. `Sample(text, alpha, rng)` provides optional Unigram posterior segmentation sampling using a caller-owned seeded `math/rand.Rand`; it is intended for encoder training, not production inference.

## Model and normalization contract

- NFC plus Unicode whitespace collapse/trim; invalid UTF-8 becomes U+FFFD. Case, Bengali marks, ZWJ/ZWNJ, punctuation, digits, emoji and natural spelling are retained. No stemming or automatic transliteration.
- IDs `0..3` reserve PAD/UNK/BOS/EOS positions for future encoder integration. `Encode` does not emit or insert them, and `Decode` rejects them; strip encoder control IDs explicitly before decoding. IDs `4..259` represent raw bytes. Learned pieces start at `260`.
- Ordinary text such as `<unk>`, `▁`, and `<0x20>` is literal, never interpreted as a control token. Unseen code points fall back to their UTF-8 bytes. Decode rejects invalid IDs and byte sequences that are not valid UTF-8.
- Pretokenization uses whitespace words and preserves mixed-script words such as `meetingটা`. Spaces are standalone byte tokens. Fertility excludes spaces; sequence-length, latency, and truncation measurements include them. Evaluation reserves two encoder control positions by default; configure `-max-tokens` and `-special-tokens` for the actual encoder.
- The custom Unigram trainer uses substring seeding, forward–backward EM, expected-count pruning and Viterbi decoding. It is **not SentencePiece-compatible** and does not reproduce SentencePiece's seed selection, pruning or whitespace representation. BPE uses ranked learned merges over Unicode characters with the same corpus and boundary constraints.
- Models are versioned JSON. Every training run writes a `.meta.json` sidecar with parameters, weights, corpus/model hashes, Go version and training duration. No model vocabulary should be substituted into an existing encoder without adapting its embedding table and retraining.

## Corpus and evaluation scope

There are **563,476 training records / 65.97 million normalized characters**, plus 14,779 validation and 16,712 frozen test records. Sources are BanglaTLit, Bengali Wikipedia and attributed Tatoeba sentences. Paired forms, article paragraphs, canonical duplicates, and approximate long-text neighbors stay together. Raw source checksums, saved split overlap, and official BanglaTLit holdout exclusion are checked independently.

The training mixture is 40% Bangla, 25% English, 25% Banglish, 10% script-mixed by weighted sentence contribution. Each sentence's weight is divided among its whitespace words; fitting ignores words longer than 64 runes. `-balanced` uses 25% each. The small code-mixed bucket is explicitly a script proxy and needs more natural data. Automatic privacy/near-duplicate filters are incomplete; inspect [CORPUS.md](data/CORPUS.md) before redistributing data or treating these splits as a publishable benchmark.

Reports include fertility, sequence percentiles, byte fallback, vocabulary utilization, sequence-limit rates, round trips, and local throughput; a byte baseline counts UTF-8 bytes on the same text. They do not measure morphological correctness or semantic retrieval. External tokenizer comparisons, matched embedding encoders, retrieval evaluation, annotation and uncertainty across training seeds remain future research work.
