# Fresh source-backed retrieval benchmark

The benchmark has 90 distinct intents, 15 per domain: business, banking, technology, science, communication guidance and general QA. Sixty intents are development and 30 are reserved test. Each intent has Bengali, Banglish and English passages and queries, plus a manually authored Banglish spelling variant. This expands the first experiment's 60-intent proxy; it does not reach the earlier proposed 180-intent target.

Pinned SBA, CFPB, NIST, NASA, GSA and USGS snapshots ground the material. Canonical source pages/publications stay wholly within one split. GSA files are pinned to a repository commit. The small source collection creates topic shifts between splits. Communication guidance is a proxy for conversation; the material does not represent naturally collected Bengali conversations or search traffic. Passages are short extracts or agent summaries, not the proposed long-document benchmark. Banking/legal policy facts refer to the source's U.S. snapshot; historical estimates are identified as historical.

Two agents reviewed all 90 intents against every same-split candidate, including cross-domain candidates, and checked parallel forms and variants. Translation and variant issues were corrected and rechecked before freezing. The 4,500 same-split intent pairs have explicit reviewed grades; those judgments project across the nine language directions and the meaning-preserving Banglish variants. This projection assumes the reviewed translations preserve meaning and does not constitute human language validation.

The [material, source provenance and review](../experiments/retrieval/source_benchmark_v1/) and [dataset manifest](source-benchmark-v2.json) bind the frozen inputs. The [overlap audit](source-benchmark-overlap-v1.json) found zero canonical or near-copy matches across original train/validation/test, expanded training, actual alignment pairs and reviewed negatives. Eight-word shingle containment detects longer verbatim reuse; it cannot rule out semantic overlap or base-model pretraining exposure. Original project test text was inspected only for integrity, never scored.

Metrics give equal weight to six domains. Banglish spelling variants remain inside their domain block. Paired uncertainty jointly resamples training seeds and six domain blocks; variants are not treated as independent evidence. Annotation uncertainty is excluded.

Fresh adapted quality evaluation is pending. The [original-tokenizer criteria](../experiments/retrieval/round2_original_criteria.json) are frozen before scoring: gain over short adaptation, a positive paired interval, gain over unchanged E5, English retention, latency and memory gates in both budgets. Development selects the candidate; a durable selection artifact must exist before reserved-test scoring.

Reproduce from the repository root, using the pinned local raw snapshots and corpora:

```sh
.venv/retrieval/bin/python experiments/retrieval/source_benchmark.py audit \
  --material experiments/retrieval/source_benchmark_v1 \
  --corpus data/processed/train.jsonl \
  --corpus data/processed/validation.jsonl \
  --corpus data/processed/test.jsonl \
  --corpus data/processed/vashantor-expanded-v2/prepared/train.jsonl \
  --corpus data/processed/alignment-round2-v1/pairs.jsonl \
  --corpus data/processed/hard-negatives-reviewed-v1/pairs.jsonl \
  --output reports/source-benchmark-overlap-repro.json
.venv/retrieval/bin/python experiments/retrieval/source_benchmark.py build \
  --material experiments/retrieval/source_benchmark_v1 \
  --overlap reports/source-benchmark-overlap-repro.json \
  --output data/processed/source-benchmark-repro
```

Longer original-tokenizer runs can resume from verified completed checkpoints:

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/retrieval/bin/python \
  experiments/retrieval/run_original_training.py \
  --protocol experiments/retrieval/round2_training_protocol.json \
  --base models/hf-cache/models--intfloat--multilingual-e5-small/snapshots/614241f622f53c4eeff9890bdc4f31cfecc418b3 \
  --root models/encoders-round2 \
  --output reports/round2-original-training-v1.json
```

After all six original-tokenizer runs complete, execute the fresh comparison:

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/retrieval/bin/python \
  experiments/retrieval/original_comparison.py \
  --benchmark data/processed/source-benchmark-v2 \
  --inventory reports/round2-original-training-v1.json \
  --base models/hf-cache/models--intfloat--multilingual-e5-small/snapshots/614241f622f53c4eeff9890bdc4f31cfecc418b3 \
  --criteria experiments/retrieval/round2_original_criteria.json \
  --output data/processed/round2-original-comparison-v1
```

The runner verifies trained checkpoint identity and checksums, records frozen inputs, evaluates all 13 models on development, saves the development selection, and only then evaluates reserved test. The 13 models are unchanged E5 plus six short and six longer original-tokenizer runs. Short and longer protocols have different exposure/time budgets; the comparison measures their joint intervention rather than attributing any gain to one training change.

Dataset v2 adds the explicit reviewed-label status consumed by the evaluator. Texts, judgments, splits and overlap evidence are byte-identical to v1.
