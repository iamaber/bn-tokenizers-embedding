# Research Plan: A Bangla–English–Banglish Tokenizer for Text Embeddings

> Implementation audit, 2026-09-08: the core recommendation remains a hypothesis supported by the cited task-specific evidence. See [RESEARCH_AUDIT.md](RESEARCH_AUDIT.md) for the source verification and observed download discrepancies, and [README.md](README.md) for the Go implementation, corpus snapshot, and measured intrinsic results. The implementation uses custom Unigram EM with expected-count pruning, not an exact SentencePiece reproduction. No embedding retrieval superiority has been established.

## Executive summary

This document investigates how to build and evaluate a tokenizer for a text embedding model that must handle:

- **Bangla** in Bengali script;
- **English** in Latin script;
- **Banglish**, used here to mean Romanized Bangla; code-mixed Bangla–English text is tracked separately because it can occur in either script.

The recommended starting point is a **joint Unigram subword tokenizer**, trained on a carefully balanced Bangla–English–Banglish corpus. It should be compared fairly with BPE at multiple vocabulary sizes. Only after establishing strong baselines should grapheme-aware or morphology-aware constraints be introduced.

The central conclusion is:

> A tokenizer should not be declared better merely because it produces fewer tokens or apparently cleaner morphemes. The winning tokenizer is the one that produces the best downstream embedding retrieval and semantic-similarity results at an acceptable compute, memory, latency, and truncation cost.

A tokenizer alone does not establish semantic equivalence across Bangla, English, and Banglish. Paired contrastive training is the proposed practical approach to improving alignment, but parallel supervision is not a universal prerequisite: multilingual pretraining can produce cross-language transfer without it. Transfer alone does not guarantee useful sentence retrieval. [Multilingual BERT study](https://aclanthology.org/P19-1493/)

The central research question is:

> At comparable training cost, does a joint tokenizer improve Bangla–English–Banglish retrieval beyond the gains from better alignment data alone?

This is an experimental plan, not a report of demonstrated gains. First evaluate an existing multilingual embedding model with its original tokenizer, then fine-tune that baseline on the proposed alignment data. Use separate, matched encoder experiments to isolate tokenizer effects.

## 1. Problem definition

The goal is not simply to compress text. The goal is to provide input units from which an encoder can learn high-quality semantic representations across three related but distinct input conditions:

```text
Bengali script:  আমি আজ ভালো আছি।
Romanized Bangla: ami aj bhalo achi
Spelling variant: ami aaj valo asi
English:           I am doing well today.
Code-mixed:        ajke meetingটা কখন?
```

A successful system should:

1. represent Bangla morphology efficiently;
2. tolerate nonstandard Banglish spelling;
3. retain strong English performance;
4. handle mixed scripts and mixed languages;
5. avoid excessive sequence expansion and truncation;
6. align semantically equivalent text across scripts;
7. generalize to rare words, inflections, names, and new transliterations;
8. remain fast enough for the intended deployment environment.

## 2. Important distinction: tokenizer quality versus embedding quality

A tokenizer can be evaluated intrinsically, but an intrinsically attractive tokenizer is not guaranteed to produce the best embedding model.

Examples:

- A whole-word tokenizer can produce very few tokens but fail badly on unseen inflections.
- A character tokenizer has nearly complete coverage but creates long sequences.
- A tokenizer may align well with morphemes but lose to another tokenizer on retrieval.
- A tokenizer may work well for Bangla while allocating too little vocabulary capacity to English or Banglish.

Therefore, evaluation must have two levels:

1. **Intrinsic tokenizer evaluation**: fragmentation, coverage, morphology, sequence length, fallback usage, and speed.
2. **Extrinsic model evaluation**: retrieval, semantic similarity, clustering, robustness, latency, and memory after training matched encoders.

The second level makes the final decision.

## 3. Research findings

### 3.1 Unigram is a strong first candidate

Shahriar and Barbosa's 2024 study, *Improving Bengali and Hindi Large Language Models*, reports that WordPiece often breaks Bengali and Hindi words into linguistically meaningless pieces and fails to separate roots and affixes. Their Bengali and Hindi experiments found that Unigram and character-level tokenization produced better pretrained models than their WordPiece-based comparisons.

This supports testing Unigram for Bangla. However, their study concerns pretraining and NLU tasks, not Bangla–English–Banglish retrieval. Its character model uses a CNN to construct word representations, so this comparison changes the input architecture as well as tokenization. It is not a plain character-tokenized Transformer baseline. The study also compares masked-token perplexity across tokenizations; these scores use different prediction units and masking contexts and should not be interpreted as directly comparable text-level likelihoods. Its downstream task results provide more relevant evidence for this proposal. [Paper, Sections 3.2 and 5.2](https://aclanthology.org/2024.lrec-main.764.pdf)

Bostrom and Durrett's 2020 study, *Byte Pair Encoding is Suboptimal for Language Model Pretraining*, found that Unigram tokenization recovered units that aligned more closely with morphology and matched or outperformed BPE in controlled English and Japanese experiments. This gives another reason to test Unigram, although it does not establish that Unigram will necessarily win for Banglish or cross-script embeddings.

### 3.2 Subword regularization may improve robustness

Unigram tokenization supports sampling from multiple plausible segmentations during training. Kudo's work on subword regularization uses segmentation ambiguity as training noise rather than committing to one segmentation for every occurrence.

This may be useful for:

- unseen Bangla inflections;
- noisy Banglish spellings;
- rare and compound words;
- reducing overdependence on one fixed segmentation.

The cited evidence concerns machine translation; these embedding benefits remain hypotheses. Training may use sampled segmentations while inference should remain deterministic. [Subword regularization](https://aclanthology.org/P18-1007/)

### 3.3 Multilingual tokenizers can disadvantage lower-resource languages

Multilingual vocabularies are generally learned from corpus frequency. Without careful sampling, high-resource English data can consume a disproportionate amount of vocabulary capacity. Bengali words may consequently be fragmented more heavily than English words.

Tokenizer **fertility**, commonly measured as the average number of subword tokens per word, is useful for identifying this imbalance. Higher fertility increases sequence lengths, compute requirements, and the chance that relevant text is truncated. However, fertility should not be optimized in isolation.

Rust et al.'s *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models* studies the relationship between multilingual tokenizer quality and downstream performance and motivates reporting tokenization quality separately by language.

### 3.4 Banglish is not just Bengali written with another alphabet

Romanized Bangla has no universally followed spelling convention. The same word may appear as:

```text
ভালো
bhalo
valo
```

Variation can reflect pronunciation, region, personal habit, keyboard behavior, or influence from English orthography. Banglish also includes genuine English lexical insertions and forms in which Bengali morphology attaches to an English word:

```text
meetingটা
filesগুলো
updateটা
```

A joint tokenizer can cover these strings, but tokenization alone will not tell an embedding model that `ভালো`, `bhalo`, and `valo` are related. Paired or contrastively related examples provide a direct training signal; shared multilingual pretraining can also contribute alignment without explicit pairs. Measure the incremental benefit of paired training instead of assuming it is the only possible mechanism.

BanglaTLit is a particularly relevant resource. Its repository describes:

- **BanglaTLit**: 42,705 Romanized-Bangla/Bangla back-transliteration pairs;
- **BanglaTLit-PT**: 245,727 Romanized Bangla samples for continued pretraining.

The paired collection is a **subset of BanglaTLit-PT**, not an independent additional corpus. For strict held-out evaluation, remove validation/test examples, their paired forms, and near-duplicate variants from the larger pretraining corpus before either tokenizer or encoder fitting. Keeping the paired dataset's official splits intact is insufficient if the same examples re-enter through BanglaTLit-PT. Record the exclusions and audit overlap across all sources. [Official dataset description](https://github.com/farhanishmam/BanglaTLit)

### 3.5 Contrastive training is a strong embedding baseline

Multilingual E5 demonstrates a successful recipe of large-scale contrastive pretraining followed by supervised fine-tuning. It supports this training approach, but does not isolate tokenizer effects or establish contrastive learning as the only way to obtain useful embeddings. [Technical report](https://arxiv.org/abs/2402.05672)

For this project, useful positive pairs include:

```text
Bangla sentence        ↔ naturally written Banglish equivalent
Bangla sentence        ↔ English translation
Banglish query         ↔ relevant Bangla passage
Code-mixed query       ↔ relevant document
Spelling variant       ↔ canonical or alternate spelling
Paraphrased query      ↔ semantically equivalent query
```

Useful hard negatives should be topically similar but semantically wrong:

```text
Query:    পাসপোর্ট নবায়নের ফি কত?
Positive: Document explaining passport-renewal fees
Negative: Document explaining passport-application tracking
```

The model must also preserve meaning-changing morphology and syntax. For example, positive and negated statements must not collapse into the same embedding:

```text
আমি যাব
আমি যাব না
```

### 3.6 Related work and limits on novelty

The November 2025 **BengaliBPE** preprint already investigates grapheme-level initialization and morphology-aware merge rules, with tokenization and news-classification evaluation. These techniques should be treated as prior work rather than claimed as new on their own. The preprint does not settle the proposed cross-script retrieval question. [BengaliBPE](https://arxiv.org/abs/2511.05324)

A 2025 Bengali speech-recognition study also evaluates BPE against character and Unigram approaches and reports BPE improvements over its character baseline. Its speech task and vocabulary scale differ from embedding retrieval; it reinforces the need to evaluate choices on the target task. [Bengali ASR study](https://aclanthology.org/2025.banglalp-1.1/)

The potential contribution here is a controlled retrieval study, an independently held-out natural Banglish benchmark, or a reproducible quality–cost improvement. This targeted literature review does not establish an exhaustive novelty claim.

## 4. Linguistic design considerations

### 4.1 Bangla morphology

Bangla uses inflection, derivation, compounding, classifiers, case markers, tense/aspect/person marking, and other productive patterns. A simplified illustrative analysis is:

```text
বইগুলোতে → বই + গুলো + তে
             book + plural/classifier + locative
```

The tokenizer should ideally allow useful sharing between related forms without mechanically stripping every suffix. Hard-coded stemming is risky because:

- suffix-like character sequences can be part of a lexical root;
- morpheme boundaries can involve spelling changes;
- some segmentations are context-dependent or ambiguous;
- aggressive stemming can erase differences needed for semantic retrieval.

Unigram tokenization is statistically learned and is not itself a morphological analyzer. Morphological alignment must be measured rather than assumed.

### 4.2 Bengali grapheme clusters

Bengali Unicode text includes combining marks, vowel signs, virama/hasanta, and conjunct consonants. Splitting inside a written grapheme cluster may produce orthographically awkward units.

A grapheme-aware candidate can discourage such splits, but two concepts must remain separate:

- **grapheme boundaries** describe written character clusters;
- **morpheme boundaries** describe meaningful linguistic units.

Protecting grapheme boundaries improves orthographic coherence but does not solve morphology.

### 4.3 Ambiguity in Latin-script text

A Latin-script string can be:

- English;
- Romanized Bangla;
- a name;
- an acronym;
- a mixture of the above.

For example, `bar` could be an English word or part of a Romanized Bengali expression. The system should therefore avoid an irreversible rule that transliterates all Latin-script input before tokenization.

A safer strategy is to retain original input and optionally introduce transliterated or normalized variants during data augmentation and contrastive training.

### 4.4 Code-mixed word boundaries

Mixed forms can combine English stems with Bengali suffixes. The pre-tokenizer must not assume that a script transition always marks a complete semantic word. Such examples must be represented in both tokenizer-training data and downstream embedding-training data.

## 5. Corpus design

Tokenizer performance will depend heavily on the corpus. A sophisticated algorithm cannot compensate for missing Banglish, conversational, or domain-specific text.

### 5.1 Recommended corpus buckets

| Bucket | Examples |
|---|---|
| Bangla | News, books, Wikipedia, educational material, forums, conversations, technical and target-domain documents |
| English | General English plus text from actual deployment domains |
| Banglish | Naturally written Romanized Bangla, spelling variants, regional forms, informal queries and conversations |
| Code-mixed | Bengali-script/English and Banglish/English messages, queries, comments and conversations |
| Parallel/aligned | Bangla–Banglish, Bangla–English, paraphrases and query–document relevance pairs |

### 5.2 Initial sampling experiment

A possible first mixture is:

```text
40% Bangla
25% English
25% Banglish
10% code-mixed
```

These percentages are experimental starting values, not research-established optima. Compare them against at least one more balanced distribution.

Define sampling by a documented unit such as normalized characters or sentences. Raw byte counts are unsuitable for comparing Bengali and Latin scripts because UTF-8 represents them differently.

### 5.3 Data quality requirements

- Deduplicate exact and near-duplicate material.
- Split by source before training when possible.
- Keep paired transliterations and near-duplicates in the same data split.
- Preserve genuine Banglish spelling variation.
- Prevent synthetic transliteration from overwhelming naturally written Banglish.
- Track domain, language/script composition, and data provenance.
- Respect licenses.
- Remove secrets and sensitive personal information.
- Reserve unseen sources and domains for final testing.
- Keep benchmark validation/test examples out of tokenizer and model training, including copies embedded in larger pretraining sources such as BanglaTLit-PT.
- Report any known benchmark exposure in existing pretrained baselines; use a new source- or time-held-out collection to assess generalization beyond that exposure.

### 5.4 Data sources to investigate

- [BanglaTLit](https://github.com/farhanishmam/BanglaTLit): paired and unpaired Romanized Bangla resources.
- [BanglaDual](https://data.mendeley.com/datasets/pxw2bb4wyt/1): a July 2026 multi-source transliteration/translation collection. Its authors report NFKC preprocessing and predominantly Wikipedia-derived data; audit original-source overlap and natural versus synthetic provenance before using its supplied splits. Applying NFC later cannot recover distinctions already removed by earlier normalization.
- [banglakit/transliteration-data](https://github.com/banglakit/transliteration-data): transliteration resources and rules.
- [BdNC Romanized Corpus](https://corpus.bangla.gov.bd/docs/romanized-corpus): an unverified lead; the linked page could not be retrieved during this review. Do not assume availability, contents, or licensing until verified.
- BanglaBERT's pretraining sources and normalization pipeline, subject to their licenses and intended use.
- Application-specific search queries and documents, after privacy review.

Every source must be independently audited for license, quality, overlap, annotation method, and permissible use before inclusion.

## 6. Normalization strategy

Normalization must be deterministic, versioned, tested, and identical during tokenizer training, encoder training, and production inference.

### 6.1 Recommended starting policy

- Apply Unicode NFC normalization.
- Normalize clearly accidental whitespace consistently.
- Preserve meaningful Bengali code points and combining marks.
- Handle zero-width characters with explicit, tested rules rather than deleting all of them.
- Preserve English case initially; compare lowercasing only as an ablation.
- Preserve natural Banglish spelling variation.
- Retain numbers, punctuation, URLs, emoji, and symbols unless the product explicitly excludes them.
- Normalize duplicated noise only when it is clearly accidental and does not carry pragmatic meaning.

### 6.2 Avoid

- Removing Bengali vowel signs or combining marks.
- Blindly deleting virama/hasanta or zero-width characters.
- Stemming all Bangla before tokenization.
- Lowercasing all text without an experiment.
- Replacing every Banglish input with Bengali script.
- Collapsing all Romanized spelling variants into one spelling.
- Using different normalization in training and inference.

BanglaBERT explicitly requires the normalization pipeline used during its pretraining. This illustrates an important rule: when benchmarking an existing model, follow that model's own normalization contract rather than forcing the new pipeline onto it.

## 7. Tokenizer candidates

### 7.1 Baseline matrix

Train at least these controlled candidates:

| Algorithm | Vocabulary sizes |
|---|---|
| Unigram | 16k, 32k, 48k |
| BPE | 16k, 32k, 48k |

Keep all other variables matched:

- corpus snapshot;
- language sampling;
- normalization;
- pre-tokenization;
- special tokens;
- character coverage;
- fallback behavior where available;
- training-data volume;
- evaluation set.

A **32k joint Unigram tokenizer** is the recommended first candidate, but it is a hypothesis rather than the presumed winner.

### 7.2 Character and byte baselines

Include at least one character- or byte-oriented baseline. Such models offer robust coverage and can reveal whether learned subwords are actually helping. Their disadvantages are often longer sequences, more encoder compute, and increased truncation.

Byte fallback is useful for rare or unseen characters. It prevents unknown-token failure, but it does not guarantee that the resulting rare-text representation will be semantically strong.

### 7.3 Existing reference tokenizers

Compare intrinsic behavior against tokenizers from:

- BanglaBERT;
- XLM-R;
- the selected multilingual embedding baseline;
- any strong Bengali-specific encoder used as a baseline.

These are practical references, not controlled algorithm comparisons, because they differ in vocabulary size, corpus, objectives, and normalization.

### 7.4 Morphology-aware ablations

After identifying strong Unigram/BPE baselines, test the following separately:

1. **Soft boundary preferences** that favor plausible stem–suffix boundaries.
2. **Grapheme-aware restrictions** that discourage splitting inside Bengali grapheme clusters.
3. **Vocabulary seeding** with carefully validated frequent stems, affixes, and mixed-script suffix patterns.
4. **Subword regularization** during encoder training.
5. **Alternative corpus sampling** that gives more vocabulary capacity to Banglish and code-mixed text.

Avoid hard-coding every suffix boundary. Each intervention must be justified by improvements on held-out embedding tasks rather than visual appeal alone.

## 8. Intrinsic tokenizer evaluation

All metrics must be reported separately for:

- Bangla;
- English;
- Banglish;
- mixed Bengali script and English;
- mixed Latin-script Banglish and English;
- frequent and rare words;
- formal and informal domains.

### 8.1 Core metrics

| Metric | Purpose |
|---|---|
| Tokens per whitespace-delimited word | Basic fragmentation/fertility measurement |
| Tokens per Unicode grapheme cluster | Script-aware fragmentation measurement |
| Tokens per sentence/document | Direct estimate of model input cost |
| Truncation rate | Fraction of examples losing content at the target context length |
| Unknown-token rate | Detects unsupported text |
| Byte/fallback rate | Measures dependence on fallback mechanisms |
| Token-length distribution | Reveals excessive tiny fragments or overlong memorized tokens |
| Vocabulary utilization | Detects unused or over-specialized vocabulary entries |
| Tokenization throughput | Measures preprocessing cost |
| Round-trip/decode correctness | Compare decoded text with the declared normalized input; NFC and whitespace normalization do not preserve arbitrary original bytes |

Report distributions, medians, high percentiles, and frequency-stratified results—not only global averages.

### 8.2 Morphological fidelity

Construct a manually reviewed set containing:

- noun inflections and case markers;
- plural/classifier forms;
- verb tense/aspect/person forms;
- derivational forms;
- compounds;
- borrowed English words with Bengali affixes;
- proper names;
- dialectal forms;
- ambiguous cases.

Possible metrics:

- morpheme-boundary precision;
- morpheme-boundary recall;
- morpheme-boundary F1;
- over-segmentation rate;
- under-segmentation rate;
- root/stem preservation rate.

Because morphological analyses can be ambiguous, annotation guidelines should permit multiple valid analyses or flag disputed examples.

### 8.3 Robustness sets

Evaluate consistent behavior across:

```text
ভালো / bhalo / valo
আছি / achi / asi
করছি / korchi / kortesi
```

Also test:

- spelling errors;
- repeated characters;
- punctuation variation;
- mixed casing;
- emoji;
- URLs and handles;
- Bengali and Arabic numerals;
- regional transliteration variants;
- English words carrying Bengali suffixes.

The objective is not necessarily identical tokenization. It is robust model performance despite different tokenizations.

## 9. Extrinsic embedding evaluation

### 9.1 Retrieval tasks

The final model should be tested on:

1. Bangla query → Bangla document.
2. English query → English document.
3. Banglish query → Bangla document.
4. Bangla query → English document.
5. English query → Bangla document.
6. Code-mixed query → Bangla or English document.
7. Spelling variants → the same relevant documents.
8. Rare inflections and entities.
9. Long documents where tokenizer efficiency affects truncation.

Example:

```text
Query: passport renew korte ki ki lage?
Relevant document: পাসপোর্ট নবায়নের প্রয়োজনীয় কাগজপত্র ...
```

### 9.2 Metrics

Use task-appropriate metrics such as:

- nDCG@10;
- Recall@1, Recall@5, Recall@10, and Recall@100;
- MRR;
- MAP where relevant;
- Spearman correlation for semantic textual similarity;
- clustering scores for related text groups;
- robustness delta between standard and noisy variants.

Also report:

- tokenization and encoder latency;
- examples per second;
- GPU/CPU memory use;
- average and percentile sequence lengths;
- truncation rates;
- index size where model dimension or quantization differs.

### 9.3 Benchmarks

MIRACL includes Bengali retrieval over Wikipedia passages. It does not by itself cover natural Banglish, application domains, or Bangla–English cross-language retrieval. Supplement it with independently authored Banglish and code-mixed queries, cross-language relevance judgments, and target-domain documents. Automatically transliterated queries alone will not capture real spelling variation or natural code-mixing. [Official MIRACL repository](https://github.com/project-miracl/miracl)

A custom test collection should include human relevance judgments and be split by source or time to reduce leakage.

### 9.4 Fair experimental controls

For a tokenizer comparison, train matched encoders with the same:

- architecture;
- hidden dimension;
- initialization policy;
- corpus;
- training objective;
- optimizer and schedule;
- batch construction;
- number of seeds;
- downstream fine-tuning data.

Tokenizers change sequence lengths, so fairness should be reported in at least two ways:

1. comparable raw-text exposure;
2. comparable measured compute budget, with processed-token counts reported separately.

Equal optimizer steps or processed-token counts alone do not guarantee equal compute. Record sequence-length distributions, padding, batch sizes, accelerator type, training time, and an explicit compute estimate or measurement.

At fixed hidden size, vocabulary changes also change the embedding table and any vocabulary prediction head. Report total parameters, embedding parameters, peak memory, and training/inference cost for every candidate. Compare Unigram and BPE at the same vocabulary size to isolate algorithm effects; treat vocabulary-size sweeps as a separate capacity–cost tradeoff. Character/CNN models belong in a separately labeled architecture comparison.

Use a predeclared set of multiple training seeds; three is a practical starting choice, not a universal sufficiency guarantee. Report means and seed variability, and paired bootstrap confidence intervals for retrieval-score differences over queries. Keep related spelling/transliteration variants in the same resampling group. Select candidates on development data, freeze decisions, and evaluate the final test set without further tuning.

### 9.5 Separate tokenizer gains from alignment-data gains

Use the following complementary comparisons:

| Comparison | What it establishes |
|---|---|
| Existing embedding model with its original tokenizer, before and after alignment fine-tuning | Practical gain from adapting an available model |
| Matched Unigram and BPE encoders with the same corpus, vocabulary size, and objective | Effect of tokenizer algorithm under controlled training |
| Each matched encoder with baseline versus enriched alignment data | Whether alignment gains depend on tokenizer choice |
| Selected baseline with one sampling or morphology change at a time | Incremental effect of each intervention |

Hold tokenizer-training data fixed during the alignment-data comparison. Hold alignment data fixed during the tokenizer comparison. Compare corpus mixtures separately; use a factorial follow-up if interactions appear material. A custom model outperforming an existing pretrained model does not, by itself, attribute the gain to its tokenizer.

## 10. Embedding-training strategy

### 10.1 Stage 1: existing-model baseline and tokenizer screening

First evaluate an existing multilingual embedding model with its original tokenizer and required preprocessing, then fine-tune it on alignment data as a practical baseline. Train and evaluate the controlled tokenizer matrix alongside this baseline. Reject candidates with poor coverage, severe language imbalance, excessive truncation, or unacceptable speed, but retain both a viable Unigram and BPE candidate for matched downstream experiments. Intrinsic metrics screen candidates; they do not select the final winner.

### 10.2 Stage 2: encoder pretraining or adaptation

Depending on compute:

- train a new encoder with MLM, replaced-token detection, or another appropriate objective; or
- continue pretraining a compatible multilingual encoder on Bangla/Banglish/code-mixed text.

If adapting an existing encoder, retaining its tokenizer is the safest first baseline.

### 10.3 Stage 3: contrastive embedding learning

Train using:

- parallel Bangla–English data;
- Bangla–Banglish pairs;
- natural paraphrases;
- query–positive-document pairs;
- in-batch negatives;
- mined hard negatives;
- code-mixed augmentation.

Pair quality is more important than indiscriminate quantity. False negatives are especially harmful when multiple documents are valid answers.

### 10.4 Stage 4: distillation, if useful

A stronger multilingual teacher can help transfer semantic structure to a smaller student. Teacher quality should be checked on Bangla and Banglish rather than assumed from English benchmarks.

### 10.5 Stage 5: deployment optimization

After quality is established, evaluate:

- maximum sequence length;
- dynamic padding;
- pooling choice;
- quantization;
- embedding dimension reduction;
- approximate-nearest-neighbor index settings;
- CPU and GPU tokenization throughput.

## 11. Replacing an existing model's tokenizer

Replacing the tokenizer of a pretrained embedding model is not a drop-in optimization. Token IDs index learned embedding rows. A new vocabulary changes that mapping and may destroy learned lexical knowledge.

Safer options, in increasing order of cost, are:

1. keep the existing tokenizer and fine-tune on Bangla/Banglish pairs;
2. add a limited number of new tokens and initialize their embeddings from decompositions under the old tokenizer;
3. perform substantial continued pretraining after vocabulary expansion;
4. train a new encoder with the new tokenizer.

With limited compute, contrastive adaptation of a strong multilingual model may outperform a tokenizer-and-encoder project trained from scratch. Both paths should be benchmarked before committing major resources.

## 12. Proposed experiment plan

### Phase A: data audit

1. Inventory available Bangla, English, Banglish, code-mixed, and aligned data.
2. Verify licenses and provenance.
3. Deduplicate and split by source.
4. Measure script, domain, length, and spelling distributions.
5. Build frozen intrinsic and retrieval test sets.

### Phase B: normalization

1. Implement a versioned normalization pipeline.
2. Add Unicode and mixed-script unit tests.
3. Inspect high-frequency changes manually.
4. Measure how normalization affects distinct words and spelling variants.

### Phase C: baseline tokenizers

Train:

```text
Unigram: 16k, 32k, 48k
BPE:     16k, 32k, 48k
Character/byte baseline: at least one
```

Compare with existing BanglaBERT, XLM-R, and embedding-model tokenizers.

### Phase D: tokenizer evaluation

1. Compute per-language fertility and sequence distributions.
2. Compute unknown/fallback and truncation rates.
3. Annotate and score morphology boundaries.
4. Test Banglish spelling and mixed-script robustness.
5. Measure tokenization speed.

### Phase E: matched encoder experiments

1. Evaluate an existing multilingual embedding model before and after alignment fine-tuning, retaining its original tokenizer.
2. Select two or three viable custom tokenizer candidates, retaining a matched Unigram/BPE pair.
3. Train small matched encoders first, with tokenizer and alignment-data effects separated as in Section 9.5.
4. Use multiple random seeds and report uncertainty, parameter counts, memory, and measured cost.
5. Compare downstream retrieval on development data; do not rank different tokenizations by raw token-level perplexity or MLM loss.
6. Scale candidates with consistent gains, then evaluate the frozen choices on the final test collection.

### Phase F: morphology-aware ablations

Test one intervention at a time:

- grapheme awareness;
- soft morphology preferences;
- vocabulary seeding;
- subword regularization;
- alternate language sampling.

### Phase G: final embedding training

1. Train on multilingual contrastive pairs.
2. Add mined hard negatives.
3. Fine-tune on real Banglish-to-Bangla retrieval.
4. Evaluate quality, robustness, cost, and fairness by language/script.
5. Freeze the tokenizer, normalization specification, and model version together.

## 13. Recommended initial configuration

This is a starting hypothesis to test, not a guaranteed optimum:

```text
Algorithm:             SentencePiece-style Unigram
Vocabulary size:       32,000
Training languages:    Bangla + English + natural Banglish + code-mixed text
Normalization:         Conservative Unicode NFC plus tested whitespace rules
Sampling:              Explicitly balanced by language/domain
Character handling:    High character coverage with robust fallback
Training augmentation: Subword regularization as an ablation
Special evaluation:    Morphology, graphemes, transliteration, code-mixing
Final selection:       Held-out retrieval quality under cost constraints
```

The 32k vocabulary and the 40/25/25/10 mixture in Section 5.2 are provisional engineering choices, not published optima. Compare the planned vocabulary sizes and at least one alternative mixture using development retrieval results and measured cost. Do not infer that a larger vocabulary is better solely from lower fragmentation.

## 14. Success criteria

A new tokenizer should only be considered an improvement if it demonstrates a meaningful combination of:

- better Bangla and Banglish retrieval;
- no unacceptable English regression;
- lower or comparable truncation;
- robust handling of unseen inflections and spelling variants;
- acceptable tokenization/encoder latency;
- acceptable memory and training cost;
- consistent gains across random seeds and test domains.

Before inspecting final test results, specify the primary retrieval metric, the minimum practically useful gain, acceptable English regression, and memory/latency limits for the application. Report effect sizes with confidence intervals; statistical significance alone does not establish practical value.

Avoid claiming that it “outperforms other tokenizers” unless:

- the baselines are named;
- data and encoder controls are documented;
- test sets are held out and leakage-checked;
- uncertainty or seed variance is reported;
- both quality and efficiency are compared.

## 15. Risks and common mistakes

1. **Optimizing only fertility.** Fewer tokens do not guarantee better semantics.
2. **Calling Unigram morphology-aware by default.** Its pieces are learned statistically.
3. **Using only synthetic Banglish.** Real users exhibit broader variation.
4. **Letting English dominate vocabulary learning.** Balance must be explicit.
5. **Over-normalizing text.** Meaningful script and spelling distinctions can be destroyed.
6. **Transliterating all Latin text.** English/Banglish ambiguity makes this unsafe.
7. **Replacing a pretrained tokenizer without retraining.** Learned token embeddings no longer correspond.
8. **Testing tokenizer algorithms with different corpora.** Results become confounded.
9. **Ignoring sequence-length cost.** Tokenizers alter training and inference compute.
10. **Evaluating only monolingual tasks.** The target system requires cross-script retrieval.
11. **Leaking paired data across splits.** Near-duplicate transliterations can inflate scores.
12. **Assuming all morphological variants mean the same thing.** Negation, tense, person, and number can change semantics.
13. **Comparing raw perplexity across tokenizers.** Different token units and masking contexts prevent a direct comparison.
14. **Confounding vocabulary size with model capacity.** Larger embedding tables change parameter count and memory.
15. **Changing tokenizer and alignment data together.** The resulting gain cannot be attributed to either change alone.

## 16. Final recommendation

The proposed path to test is:

> Establish an existing multilingual embedding baseline with its original tokenizer, measure the gain from alignment fine-tuning, and then compare joint Unigram and BPE tokenizers in matched encoders. Treat 32k as a starting hypothesis. Add grapheme or morphology interventions only when they improve held-out retrieval at an acceptable cost; label character/byte architecture comparisons separately.

The system components to evaluate are:

```text
high-quality corpus
+ consistent normalization
+ robust joint tokenizer
+ Bangla/Banglish/English contrastive pairs
+ hard-negative training
+ cross-script retrieval evaluation
```

Tokenization is the first step, but the embedding model learns semantic alignment. Both must be designed and evaluated together.

## References

1. Arif Shahriar and Denilson Barbosa. 2024. **Improving Bengali and Hindi Large Language Models.** LREC-COLING 2024.  
   <https://aclanthology.org/2024.lrec-main.764/>

2. Kaj Bostrom and Greg Durrett. 2020. **Byte Pair Encoding is Suboptimal for Language Model Pretraining.** Findings of EMNLP 2020.  
   <https://aclanthology.org/2020.findings-emnlp.414/>

3. Taku Kudo. 2018. **Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates.** ACL 2018.  
   <https://aclanthology.org/P18-1007/>

4. Phillip Rust, Jonas Pfeiffer, Ivan Vulić, Sebastian Ruder, and Iryna Gurevych. 2021. **How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.** ACL-IJCNLP 2021.  
   <https://aclanthology.org/2021.acl-long.243/>

5. Md Fahim et al. 2024. **BanglaTLit: A Benchmark Dataset for Back-Transliteration of Romanized Bangla.** Findings of EMNLP 2024.  
   <https://aclanthology.org/2024.findings-emnlp.859/>  
   Repository: <https://github.com/farhanishmam/BanglaTLit>

6. Liang Wang et al. 2024. **Multilingual E5 Text Embeddings: A Technical Report.**  
   <https://arxiv.org/abs/2402.05672>

7. Xinyu Zhang et al. 2022. **Making a MIRACL: Multilingual Information Retrieval Across a Continuum of Languages.** arXiv preprint describing the WSDM 2023 Cup dataset.  
   <https://arxiv.org/abs/2210.09984>

8. Abhik Bhattacharjee et al. 2022. **BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla.** Findings of NAACL 2022.  
   <https://aclanthology.org/2022.findings-naacl.98/>  
   Repository: <https://github.com/csebuetnlp/banglabert>

9. **BanglaDual: A Multi-Source Dataset for Banglish Transliteration and Translation.** Dataset page.  
   <https://data.mendeley.com/datasets/pxw2bb4wyt/1>

10. **Banglakit Transliteration Data.**  
    <https://github.com/banglakit/transliteration-data>

11. **BdNC Romanized Corpus.** Unverified resource lead; linked page could not be retrieved during review.  
    <https://corpus.bangla.gov.bd/docs/romanized-corpus>

12. Telmo Pires, Eva Schlinger, and Dan Garrette. 2019. **How Multilingual is Multilingual BERT?** ACL 2019.  
    <https://aclanthology.org/P19-1493/>

13. Firoj Ahmmed Patwary and Abdullah Al Noman. 2025. **Evaluating Subword Tokenization Techniques for Bengali: A Benchmark Study with BengaliBPE.** arXiv preprint; not treated here as a peer-reviewed retrieval result.  
    <https://arxiv.org/abs/2511.05324>

14. Ahnaf Mozib Samin. 2025. **Byte Pair Encoding Is All You Need For Automatic Bengali Speech Recognition.** Second Workshop on Bangla Language Processing.  
    <https://aclanthology.org/2025.banglalp-1.1/>
