# First retrieval pilot — 21 September 2026

The repository now has a reproducible development pilot and two executed baselines. **These scores evaluate finding a Bengali sentence corresponding to a Banglish input, not natural search relevance or our tokenizer's embedding accuracy.**

| Baseline | Recall@10 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|
| BM25, no transliteration | 0.0167 | 0.0127 | 0.0115 |
| Multilingual E5-small, original tokenizer | 0.7567 | 0.6559 | 0.6235 |

E5 retrieved the source counterpart in the top ten for 227 of 300 queries; 73 were missed. Its group-bootstrap 95% percentile interval for Recall@10 is 0.7067–0.8067, and for nDCG@10 is 0.6114–0.7013. These intervals do not account for uncertain relevance assumptions or pretrained-model contamination. No claim of improvement over E5 is made.

## Scope

- 300 existing validation pairs from distinct connected corpus groups; 292 Banglish and eight mixed-script queries.
- 2,000 Bengali BanglaTLit candidate sentences, including each counterpart.
- Exact original pair recovery, source attribution, fixed seed 20260921 and SHA-256 provenance.
- No group/canonical overlap with the tokenizer training corpus; frozen test files were not opened.
- One unreviewed source-pair positive per query; other documents are unjudged, not verified negatives.
- No human annotations or new natural query variants have been completed.

BM25 found no matching term for 275 queries. Its pure-Banglish Recall@10 was zero; its five hits came from mixed-script inputs. This is expected evidence of the lexical script barrier, not a strong semantic baseline to beat. E5's pure-Banglish Recall@10 was 0.7534. Eight mixed-script queries are insufficient to establish subgroup performance. Bengali-query and English-query evaluation are absent.

E5 used pinned revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, original preprocessing/tokenizer, masked mean pooling and cosine ranking. No fine-tuning was performed. CPU execution used four threads, batches of 16 and a 512-token limit; no texts were truncated. The single local pass took about 6.78 seconds to embed the 2,000 documents and 0.80 seconds to embed the 300 queries, excluding loading and a separate length audit. These are execution observations, not repeat-based latency benchmarks.

The runtime was Python 3.12.13 with torch 2.14.0, transformers 5.17.0 and tokenizers 0.23.2. Model-file hashes, input/output hashes, platform, bootstrap intervals and bucket-level results are saved in [the raw report](retrieval-pilot-v1.json).

## Available review work

Local `data/processed/retrieval-pilot-v1/review-pool.jsonl` contains 3,152 query–candidate rows pooled from both systems' top ten and the source counterpart, shuffled with model names/ranks hidden and all judgment fields blank. `authoring.jsonl` has 300 source-pair review and query-authoring forms. Data is not committed or redistributed.

The next action is a 30-group reviewer calibration, followed by pair verification, alternative-positive judgments and natural query collection. The [protocol and reproduction instructions](../experiments/retrieval/README.md) define grading, limitations, and the matched-encoder experiments that follow. This pilot makes evaluation executable; it does not resolve annotation, real search coverage or scientific novelty.
