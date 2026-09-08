# Initial tokenizer screening

These figures are intrinsic tokenizer measurements on the frozen validation split, not embedding retrieval scores. The 14,779 validation records are grouped with their paired/transliteration variants before splitting; the test split was not used for selection. Each candidate uses the same 563,476-record training split, NFC plus whitespace normalization, weighted sentence mixture, 12-rune maximum learned piece length, all observed characters, and byte fallback. The Unigram and BPE models are custom Go implementations and are not SentencePiece-compatible.

| Candidate | Vocabulary | Sequence tokens/word | Fertility (spaces excluded) | Byte fallback | Over 512-token budget |
|---|---:|---:|---:|---:|---:|
| Unigram | 16,000 | 2.463 | 1.524 | 0.00035 | 0.00095 |
| BPE | 16,000 | 2.501 | 1.563 | 0.00034 | 0.00108 |
| Unigram | 32,000 | 2.366 | 1.428 | 0.00037 | 0.00061 |
| BPE | 32,000 | 2.311 | 1.373 | 0.00039 | 0.00061 |
| Unigram | 48,000 | 2.364 | 1.426 | 0.00037 | 0.00061 |
| BPE | 48,000 | 2.240 | 1.302 | 0.00041 | 0.00054 |
| Unigram, balanced 25/25/25/25 ablation | 32,000 | 2.399 | 1.461 | 0.00036 | 0.00088 |

The default 32k Unigram model encoded the validation set at 23.3 MB/s across five benchmark runs on an Apple M5 arm64 host, with 52,928 bytes and 466 allocations per benchmark operation. The 32k BPE model measured 42.6 MB/s, with 29,120 bytes and 154 allocations per operation. A repeated-word BPE scaling benchmark measured approximately 76–79 MB/s at 1,000, 10,000, and 100,000 characters, demonstrating linear behavior after batched occurrence replacement. These are local preprocessing measurements; they are not a general Go-versus-Python claim and do not include encoder inference.

Round-trip validation passed for every record and every candidate (`Decode(Encode(text)) == Normalize(text)`). The default 32k Unigram model used 13,633 of 32,000 IDs on validation (42.6% overall; per-bucket utilization is in the JSON report). The code-mixed bucket has only 156 validation examples, so its metrics have high uncertainty.

The measurements do not evaluate morphology-boundary F1, semantic similarity, cross-script retrieval, or encoder quality. Lower fertility is not a reason to select BPE. Final selection requires matched encoders, real natural Banglish and code-mixed relevance judgments, multiple seeds, confidence intervals, memory/compute measurements, and an unseen-source/time-held-out test collection as described in [RESEARCH.md](../RESEARCH.md) and [RESEARCH_AUDIT.md](../RESEARCH_AUDIT.md).

Raw JSON measurements are saved as `*-validation.json`; corpus counts and split hashes are in `corpus-stats.json` and `corpus-audit.json`; Go and Python verification logs are in `tests-final.txt`, `benchmark-*.txt`, and `python-tests.txt`.
