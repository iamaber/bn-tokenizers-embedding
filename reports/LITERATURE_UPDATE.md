# Literature update — 9 October 2026

This is an automated, multi-source literature review of the central question in [RESEARCH.md](../RESEARCH.md):

> At comparable training cost, does a dedicated joint Bangla–English–Banglish tokenizer improve embedding retrieval beyond what better alignment/contrastive data alone gives an existing multilingual encoder?

**None of the sources reviewed answers this question.** This targeted search does not establish that no published study exists. None of the verified sources measures sentence-embedding or retrieval quality as a function of the tokenizer for Bengali or Banglish. Every link from tokenizer to retrieval below is inferred from LLM perplexity, classification, intrinsic metrics, or bitext mining on other languages. The project now has a [controlled encoder comparison](ENCODER_COMPARISON.md) that tests both 32k tokenizers, plus a [reviewed sentence pilot](RETRIEVAL_REVIEW.md). Neither custom tokenizer passed the declared domain quality and English-retention criteria; unchanged E5 remains selected. This is project evidence from small agent-reviewed proxies, not a result established by the cited literature. See the [source audit](LITERATURE_SOURCE_CHECK.md) and [implementation plan](IMPROVEMENT_PLAN.md).

## Method

Five search angles were used: tokenizer effects on downstream quality, Banglish and code-mixed benchmarks, tokenizer swapping and vocabulary transfer, Indic baselines on Romanized input, and byte-level or morphology-aware alternatives. The original automated review reported 23 sources, 111 extracted claims and votes on 25 claims. Its source inventory and individual reviewer decisions are not available in this repository, so those counts and votes are not independently reproducible. Confidence labels below are editorial judgments; the linked source audit takes precedence over the original votes. Several sources are single arXiv preprints; treat them as leads, not settled results.

## Findings

### 1. Tokenizer choice matters, but intrinsic metrics do not predict how much — high confidence (3-0, 3-0)

Ali et al. trained 24 mono- and multilingual decoder LLMs at 2.6B parameters. They found tokenizer choice "can significantly impact the model's downstream performance and training costs", while fertility and parity were "not always predictive" of downstream performance. Multilingual tokenizers needed about 3× the vocabulary of English-only ones. Scope: decoder LLMs on five European languages, with no Indic languages or embedding models. [Findings of NAACL 2024](https://preview.aclanthology.org/dashboard/2024.findings-naacl.247)

**Implication:** the fertility, fallback and utilization numbers in [RESULTS.md](RESULTS.md) cannot select a tokenizer. Selection needs a downstream retrieval evaluation.

### 2. Language-specific Indic tokenizers fragment words far less — medium confidence (3-0, 3-0)

At 32k ULM, 33.02% of words are split by monolingual tokenizers against 57.28% for multilingual ones. For Bengali at 256k, a cluster tokenizer splits 27.1% of words against 32.4% for a joint tokenizer. The authors run no extrinsic evaluation and list it as future work. [arXiv 2506.17789](https://arxiv.org/html/2506.17789v2)

**Implication:** this supports a dedicated tokenizer in principle, but the retrieval benefit is unproven.

### 3. BPE and Unigram are close on intrinsic fertility at 32k in one study — medium confidence

Thakur et al. report Bengali token-to-word ratios of 4.37 (BPE) vs 4.51 (Unigram) at 32K, about 3% apart. The gap widens to 3.01 vs 3.82 at 256K, where SentencePiece Unigram became numerically unstable. Their fertility-driven data mixture (AdaptMix) cut average fertility by about 6% and gave the lowest perplexity: 56.16 for Bengali, against 101–158 for the other mixtures. Caveats: the perplexities are per-token and not normalized across tokenizers, there are no retrieval metrics, and the setting is a 128K LLM vocabulary. [arXiv 2508.06533](https://arxiv.org/pdf/2508.06533)

**Implication:** retain both algorithms as experimental factors. The study did not test 16k or 48k and establishes no retrieval ranking. Run the broader vocabulary sweep after the controlled 32k encoder comparison.

### 4. Vocabulary design helps cross-lingual alignment, mostly when training from scratch — medium confidence (3-0, 3-0)

Trained from scratch on 13 low-resource languages, Parallel Tokenizers lowered average bitext-mining xsim error from 83.56 (joint tokenizer) to 74.08. With continual pretraining from mBERT, task scores were about the same (NusaX 78.3 vs 78.6), but cross-lingual representation similarity was still stronger, with smaller gains than from scratch. The study covers no Bengali or Romanized scripts, and its tokenizer design differs from a generic joint tokenizer. [arXiv 2510.06128](https://arxiv.org/html/2510.06128v1)

**Implication:** training from scratch and alignment-sensitive tasks are hypotheses worth testing later. Shared semantic token indices in this paper differ from our generic joint tokenizer; its results do not establish the best next experiment for this project.

### 5. Swapping a tokenizer needs strong embedding initialization — high confidence (3-0 ×4)

- **ZeTT:** a hypernetwork predicts embeddings for any tokenizer and generalizes to new tokenizers for XLM-R. Earlier heuristics "often perform at chance level" without further training. [Minixhofer et al., NeurIPS 2024](https://arxiv.org/abs/2405.07883)
- **FOCUS:** replacing the vocabulary, with FOCUS initialization plus language-adaptive pretraining, beats both extending the vocabulary and keeping the original, on average. Keeping the original won only for high-resource German. Extending the vocabulary generally underperforms because XLM-R's SentencePiece can already spell any unseen word out of characters. [Dobler & de Melo, EMNLP 2023](https://arxiv.org/abs/2305.14481)

Neither paper evaluates sentence embeddings, retrieval, or Bengali/Banglish.

### 6. A strong data-only Banglish baseline already exists — medium confidence (3-0 ×3)

BanglaTLit released a 42.7k-pair back-transliteration benchmark and BanglaTLit-PT, a 245.7k-sample Romanized pretraining corpus. TB-XLM-R was further pretrained on it with its original tokenizer and gained about 4 F1 on TB-Sentiment and about 4.5 F1 on TB-Emotion, reaching state of the art. These results are classification only. [Findings of EMNLP 2024](https://preview.aclanthology.org/new-sigs/2024.findings-emnlp.859/)

**Implication:** an existing encoder with its original tokenizer, trained on Banglish data, is the baseline our tokenizer has to beat.

### 7. Larger aligned Banglish resources now exist — medium confidence (3-0, 3-0, 2-1, 3-0)

- **BanglaDual:** 1,122,189 Banglish→Bangla pairs and 119,727 Banglish→English pairs, under deposit-level CC BY 4.0 (July 2026). The 1,122,998-row merged file is an outer join, not complete triples. Metadata reports 85.34% Wikipedia and references BanglaTLit; upstream provenance and cross-resource overlap require inspection before splitting. It is a dataset deposit with no peer-reviewed paper. [Mendeley Data](https://data.mendeley.com/datasets/pxw2bb4wyt/1)
- **IndoTranslit:** 975,190 Bengali pairs, produced by rule-based romanization of Bengali Wikipedia. That risks overlap with our Wikipedia training data, and the text is not natural Banglish. [arXiv 2511.22769](https://arxiv.org/abs/2511.22769v1)

### 8. FIRE CMIR is a natural code-mixed retrieval set, but small — medium confidence (3-0)

The 2024 track has 107,900 Roman-transliterated Bengali–English Facebook documents, 20 training queries and 30 test queries on migrant and COVID topics. The CMIR-2025 edition retains native Bengali and Roman scripts, including mixed-script test queries, with the same counts. Thirty test queries give high variance; specify the edition and verify access before depending on it. [2024 paper](https://arxiv.org/html/2411.04752v3), [2025 organizer overview](https://ceur-ws.org/Vol-4173/T3-1.pdf)

## Rejected claims

| Claim | Vote | Source |
|---|---|---|
| A 16k–48k joint vocabulary will badly over-fragment Bengali compared with 256k | 0-3 | [2506.17789](https://arxiv.org/html/2506.17789v2) |
| Diacritic-separating pretokenization raises fertility yet improves perplexity in otherwise identical LMs | 0-3 | [2508.06533](https://arxiv.org/pdf/2508.06533) |
| multilingual-e5-large-instruct is the best public multilingual embedder overall | 0-3 | [2502.13595](https://arxiv.org/pdf/2502.13595) |
| Under 1B tokens of continued training closes the gap after a tokenizer swap | 0-3 | [2405.07883](https://arxiv.org/abs/2405.07883) |

The original review incorrectly rejected a valid paper-specific MMTEB result: MTEB(Indic) contains 23 tasks and multilingual-e5-large-instruct ranked first among that paper's evaluated models. This is a historical comparison, not a current universal ranking. ZeTT reports substantial recovery after 800M continued-training tokens in decoder experiments, but not complete recovery on every task or a Bengali retrieval budget. Keep the existing pinned E5-small as a reproducible first control and estimate post-swap adaptation cost ourselves.

## Recommended next steps

1. **Extend the retrieval evaluation.** Add Bengali- and English-query directions to the [pilot](RETRIEVAL_PILOT.md) using audited complete cases and human-authored queries, with edition-specific FIRE CMIR as an external check. BanglaTLit's aligned benchmark is a subset of its pretraining corpus: remove held-out groups from both adaptation and contrastive training. Audit BanglaDual's upstream overlap before using it. Complete the human review pool before treating any result as final.
2. **Data-only baseline.** Contrastively fine-tune multilingual-e5 or BGE-M3 on the cross-script pairs, keeping their original tokenizers.
3. **Matched tokenizer swap.** Put our tokenizer into the same encoder using ZeTT or FOCUS initialization. Give it the same continued-training and contrastive budget, and compare with confidence intervals, especially in the Banglish↔Bangla directions.
4. **Secondary ablations.** After the controlled 32k Unigram/BPE comparison, test 16k/48k and language weighting. Training an encoder from scratch remains a later hypothesis, contingent on data and compute feasibility.

## Coverage gaps and open questions

- **Not covered:** verified comparisons of MuRIL, IndicBERT, BanglaBERT, Sarvam, Krutrim or IndicTrans2 tokenizers on Romanized input; byte-level or morphology-aware Bengali tokenization for retrieval; and the Dakshina, Aksharantar and IndicXNLI datasets.
- **Open questions:**
  - Does a tokenizer advantage appear only when training from scratch?
  - Are BanglaDual plus BanglaTLit and FIRE CMIR enough for statistically reliable evaluation, or do we need a held-out, human-written Banglish cross-script test set?

## Executed project follow-up — 9 October 2026

At the user’s request, the implementation agent authored and judged business, banking, technology, science, conversation and QA material. This replaces the initial human-review dependency for the requested implementation; no independent human review is claimed. All five implementation outcomes are recorded in the [ledger](IMPROVEMENT_PLAN.md).

The 60-intent domain proxy has all candidate intent pairs judged and 42/18 development/test holdouts. Both training budgets and all three seeds were executed, with inference costs and paired intervals. Custom vocabulary initialization used mean original-subword embeddings followed by shared teacher distillation, so this is a heuristic feasibility experiment rather than a FOCUS/ZeTT reproduction. Natural expansion used audited original Vashantor training forms while preserving all holdouts. The completed original sentence review and 73-case audit are documented separately. Future independent review and natural-search validation remain necessary before broad generalization claims.
