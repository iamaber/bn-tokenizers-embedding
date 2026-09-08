# Research verification — 2026-09-08

The plan is a sound experimental starting point. The references support testing Unigram; they do not establish an optimal vocabulary, corpus mixture, or retrieval winner. The existing qualifications about architecture, perplexity, alignment data, vocabulary capacity, and leakage are appropriate.

| Claim or resource | Verification and consequence |
|---|---|
| Bengali/Hindi Unigram evidence | [Shahriar and Barbosa, 2024](https://aclanthology.org/2024.lrec-main.764.pdf) reports downstream NLU gains. The character input includes a CNN; it is not a tokenizer-only comparison. This is evidence to test Unigram, not evidence of Banglish retrieval improvement. |
| Unigram versus BPE | [Bostrom and Durrett, 2020](https://aclanthology.org/2020.findings-emnlp.414/) studies English/Japanese pretraining and morphology. The research correctly limits extrapolation. |
| Segmentation sampling | [Kudo, 2018](https://aclanthology.org/P18-1007/) supports subword regularization in translation. The Go API supports posterior sampling as an optional encoder-training experiment. |
| Multilingual imbalance | [Rust et al., 2021](https://aclanthology.org/2021.acl-long.243/) motivates language-specific tokenization measurements. Lower fertility alone is not a semantic quality result. |
| Alignment | [Multilingual E5](https://arxiv.org/abs/2402.05672) supports contrastive training; [Pires et al., 2019](https://aclanthology.org/P19-1493/) shows transfer can emerge without explicit parallel training. The plan correctly avoids claiming parallel data is universally necessary. |
| BanglaTLit | [Paper](https://aclanthology.org/2024.findings-emnlp.859/) reports 42.7k paired and 245.7k pretraining samples. [Repository](https://github.com/farhanishmam/BanglaTLit) explicitly identifies paired data as a subset. See the observed file discrepancy below. |
| BanglaDual | [Mendeley v1](https://data.mendeley.com/datasets/pxw2bb4wyt/1) is dated July 3, 2026, reports NFKC preprocessing, 85.34% Wikipedia origin, and CC BY 4.0. Its aggregate license and supplied random splits do not resolve upstream overlap or synthetic provenance. Excluded from this snapshot to avoid mixing it with its upstream sources. |
| BengaliBPE novelty | [November 2025 preprint](https://arxiv.org/abs/2511.05324) already proposes grapheme initialization and morphology-aware merges, evaluated on news classification. No novelty or retrieval claim is made here. |
| Bengali ASR | [Samin, 2025](https://aclanthology.org/2025.banglalp-1.1/) studies a different task and vocabulary scale; it supports retaining BPE as a comparison. |
| Existing-model normalization | [BanglaBERT repository](https://github.com/csebuetnlp/banglabert) requires its own normalization pipeline. Custom NFC must not replace preprocessing required by reference encoders. |
| Retrieval benchmark | [MIRACL](https://github.com/project-miracl/miracl) includes Bengali Wikipedia retrieval. It does not supply the full proposed natural Banglish/cross-language benchmark. No MIRACL query or relevance data was used for fitting. Wikipedia training still means source-domain independence from MIRACL must not be claimed. |
| Other transliteration leads | [Banglakit repository](https://github.com/banglakit/transliteration-data) exists, but was not needed or included. The [BdNC link](https://corpus.bangla.gov.bd/docs/romanized-corpus) remains inaccessible in this audit; contents and licensing remain unverified. |

## Observed data differs from the summary

At BanglaTLit commit `f780fcdd5176ddfe6364ffe7948acb6a1c2ea58e`, `data/BanglaTLit_train.csv` has **245,727 rows**, including **42,864 nonempty Bengali fields** and **202,863 empty Bengali fields**. The validation and test CSVs have 1,500 and 2,500 rows respectively. These are file observations, not corrected paper statistics or independent unique-pair counts. Do not interpret every CSV row as a labeled transliteration pair.

The downloaded pretraining text is retained for provenance, but is not ingested as a second independent source: the training CSV already contains the pretraining Romanized strings. Pair/document groups and canonical duplicates are joined before assigning splits, with official test membership taking precedence over validation and training. Long-text SimHash neighbors are also grouped. The saved split audit detects zero canonical-text or connected-group overlap. This does not prove the absence of every semantic paraphrase or transliteration variant; manual and stronger multilingual near-duplicate review remain necessary for publishable retrieval claims.

## Concrete engineering decisions

- Pure Go training and inference. `golang.org/x/text` supplies NFC; Parquet dependencies are used only by corpus preparation, not the tokenizer package.
- A real Unigram likelihood model: substring seed inventory, log-space forward–backward EM, expected-count pruning and maximum-likelihood Viterbi inference. Pruning is **not** SentencePiece's loss-based pruning; initialization and whitespace handling also differ. The [SentencePiece reference implementation](https://github.com/google/sentencepiece) is a separate baseline, not an interoperability claim.
- BPE shares the same normalized weighted word inventory, maximum 12-rune piece length, full observed character inventory, special/byte IDs, and whitespace boundaries. A byte-oriented baseline is also counted on the identical normalized validation text.
- Vocabulary sizes include 4 reserved IDs and 256 fallback byte IDs. Retaining all observed characters is an explicit 100% character-coverage choice, not grapheme-aware tokenization.
- The 40/25/25/10 mixture is implemented as equal sentence mass within each bucket and inverse word-count mass within each sentence. This weights sentence contributions explicitly; it is not a character-proportional mixture. Words over 64 runes are excluded from fitting and remain encodable through shorter pieces/bytes.
- Whitespace is normalized to single spaces. Spaces have standalone byte IDs, so sequence lengths include an extra token at each word boundary. Fertility excludes these space tokens; full sequence counts and truncation include them. This is an explicit departure from SentencePiece's attached whitespace marker.
- Code-mixed labels are an observable **Bengali-plus-Latin script proxy** in BanglaTLit. Latin-only English/Banglish code-switching cannot be identified by this rule and remains in its source-provided bucket.
- Go is an implementation choice, not proof of superior performance. Reports measure this implementation locally; no Python/C++ speedup claim is supported.

## What remains experimental

The collected corpus is sufficient to fit and screen the initial vocabulary matrix, with 563,476 training records and 65,967,860 normalized characters. It is not established as sufficient for training a high-quality embedding encoder or reaching tokenizer data saturation. Natural code-mixed coverage, conversational English, target-domain documents, dialect coverage, and independently held-out sources need expansion. The finite intrinsic sample cannot select the retrieval winner.

Matched encoder pretraining, alignment-data ablations, existing embedding-model baselines, morphology annotations, retrieval judgments, source/time-held-out natural Banglish evaluation, and multi-seed uncertainty are still separate research work. No encoder was trained or pretrained tokenizer replaced in this implementation task.
