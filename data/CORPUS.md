# Corpus snapshot

Raw files are physically present under `data/raw/`; processed splits are under `data/processed/`. They are ignored by Git to avoid embedding large third-party corpora in source history. `sources.lock.json` records every download URL, exact byte count, and SHA-256; `bntok fetch` recreates matching files or rejects changed upstream content. `bntok prepare` verifies the manifest before reading data.

| Source | Included material | Attribution/license evidence |
|---|---|---|
| BanglaTLit, pinned Git commit | Romanized records plus nonempty Bengali paired fields; official held-out CSVs reserved through grouped splitting | Upstream repository includes MIT notice, retained at `raw/banglatlit/LICENSE`; source content aggregates third-party comments/datasets. MIT repository metadata is not an independent audit of every original contributor's rights. |
| Wikimedia Wikipedia, `20231101.bn` | Both raw shards downloaded; deterministic approximately 20% article sample, up to ten substantial paragraphs per article used | [Dataset card](https://huggingface.co/datasets/wikimedia/wikipedia): CC BY-SA 3.0 / GFDL. Original article URLs and contributor/license notice retained per record. Preserve applicable attribution/share-alike terms when distributing adapted corpus text. |
| Tatoeba via ManyThings | Bengali–English sentences and English side of French–English sentences | [Download page](https://www.manythings.org/anki/); each archive includes `_about.txt` identifying CC BY 2.0 and per-row attribution retained in every processed record. French text is not used. |

The raw snapshot has source-provided public text and is intended for local research. It has not undergone a comprehensive rights or privacy audit. Do not treat the repository's software licensing as a blanket license for corpus redistribution. No data or model has been uploaded elsewhere.

Preparation applies NFC and whitespace normalization; excludes text outside 12–2,000 normalized runes, replacement-character corruption, recognizable email addresses, Bangladeshi phone-number patterns, and simple credential assignments. These are conservative heuristic filters, **not complete PII removal**. It preserves punctuation, case, emoji, joiners, spelling variation, and meaningful combining marks in retained text.

Before splitting, paired records and paragraphs of an article form groups. Canonical duplicate keys ignore case, spacing, and punctuation solely for grouping/deduplication. Long texts (at least twelve whitespace words) additionally use 64-bit bag-of-words SimHash with Hamming distance at most three. This approximate heuristic can both miss variants and over-group common wording; there is no semantic/transliteration deduplication claim. Official test membership wins over validation membership; both win over deterministic 96/2/2 group hashing. Consequently official split sizes change through leakage exclusions and additional held-out groups.

No synthetic transliteration is generated. BanglaDual is excluded because its upstream sources overlap and natural/synthetic provenance needs further audit. `BanglaTLit-PT.txt` is not double-counted beside its training CSV. Raw licenses and attribution remain in the checksum-locked source files.

| Training bucket | Records | Normalized characters |
|---|---:|---:|
| Bangla | 173,108 | 43,368,936 |
| English | 166,936 | 5,252,680 |
| Banglish | 218,723 | 16,946,172 |
| Code-mixed script proxy | 4,709 | 400,072 |
| Total | 563,476 | 65,967,860 |

Validation contains 14,779 records; the frozen test split contains 16,712. See `reports/corpus-stats.json` and `reports/corpus-audit.json` for counts, source composition, and hashes. Test text is checked only for split integrity; reported tokenizer screening uses validation. These are grouped in-domain splits, not fully unseen-source evaluations.

The small code-mixed bucket is oversampled by the proposed weighted mixture. Additional naturally authored code-mixed and target-domain records are the highest-priority corpus expansion. Corpus sufficiency for final embeddings must be established by learning curves and downstream evaluation, not by the presence of a 32k vocabulary.
