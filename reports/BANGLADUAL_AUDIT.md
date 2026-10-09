# BanglaDual provenance and expansion audit — 9 October 2026

Use **original Vashantor v2** for the next natural-data expansion. Exclude BanglaDual's rule-based Wikipedia portion and repackaged BanglaTLit by default. BanglaDual's merged file is useful for provenance inspection, but its random splits and missing upstream IDs do not establish independent evaluation groups. The accompanying [machine-readable audit](bangladual-audit.json) records exact downloaded-file hashes, remote file metadata, counts, and limitations.

## Actual files and bounded inspection

The [BanglaDual v1 deposit](https://data.mendeley.com/datasets/pxw2bb4wyt/1) is published under CC BY 4.0. Its metadata endpoint reports `size: 0`, and the root **file** listing is empty because all files are nested in folders; neither observation means the deposit lacks downloadable files. The browser lists a 1.5 GB download-all archive. Folder metadata is available at `/public-api/datasets/pxw2bb4wyt/folders/1`; task files are enumerated by their folder UUID.

| File | Remote bytes | Inspection |
|---|---:|---|
| `banglish_bangla_transliteration.parquet` | 565,546,277 | Inventory only; exceeds the 400 MB cap |
| `banglish_merged_dual_task.parquet` | 576,401,227 | 9,456-byte Parquet footer and four source-tag chunks only |
| `banglish_english_translation.parquet` | 14,990,189 | Full download; SHA-256 verified against deposit metadata |
| Three sample CSVs | 68,218 | Full download; schema/source counts inspected |
| Three processing scripts | 36,519 | Full download; read, never executed |

Dataset payload transferred for these files plus the original Vashantor archive totals **16,297,297 bytes**. Small provenance metadata/cards are additional kilobytes. PyArrow was installed in an isolated temporary cache for inspection (a separate 34.2 MiB package). The full-data download cap was respected. Each ranged response was checked for HTTP 206, exact `Content-Length`, and exact `Content-Range` before reading. Remote full-file hashes for the undownloaded large files are recorded as **unverified**; hashes for the received ranges do not validate the complete files.

The small CSV schemas are `Banglish,Bangla,Source`, `Banglish,English,Source`, and `Banglish,Bangla,English,Source_bangla,Source_english`. The merged sample has a UTF-8 BOM: read it using `utf-8-sig`. Task samples contain ten examples **per source**, rather than a representative global sample; the scripts' `generate_samples` functions confirm stratified sampling. The 70-row merged sample contains 31 complete triples, 30 Bangla-only rows, and 9 English-only rows. These ratios must not be extrapolated to the corpus. [Bangla processing script](https://data.mendeley.com/public-files/datasets/pxw2bb4wyt/files/6ce23359-5a99-447b-a6a7-9abde3e6c4d0/file_downloaded), [English processing script](https://data.mendeley.com/public-files/datasets/pxw2bb4wyt/files/c1118714-3d28-4b2c-b8dc-2c892858cbe0/file_downloaded)

## Merged counts and join expansion

The downloaded footer establishes 1,122,998 rows in two row groups, with five nullable UTF-8 columns: `Banglish`, `Bangla`, `English`, `Source_bangla`, `Source_english`. Null counts are 0, 467, and 1,002,396 for the three text columns respectively. The source columns have matching target null counts within each row group. Reading the source-tag chunks produces zero rows with both source tags absent and **120,135 rows with both present**. Under the inspected clean-pair outer-join construction, this gives the complete-triple count. Text-column presence was checked through footer statistics, not by downloading all target text, so the per-row correspondence is an inference from construction and matching source statistics. [Merge script](https://data.mendeley.com/public-files/datasets/pxw2bb4wyt/files/88da4049-1c76-4467-9353-5adff5ddac01/file_downloaded), [merged file](https://data.mendeley.com/public-files/datasets/pxw2bb4wyt/files/d172dbab-8682-4956-8114-ee3e2ee3d4db/file_downloaded)

| Source, abbreviated | Merged Bangla-tagged rows | Merged English-tagged rows | English task rows |
|---|---:|---:|---:|
| Rule-based Wikipedia / Modeling Romanized Hindi and Bengali | 957,730 | 0 | 0 |
| Emotion | 80,266 | 80,280 | 79,407 |
| BanglaTLit / Updated BanglaTLit | 39,896 | 470 | 470 |
| BanTH | 37,350 | 37,352 | 37,350 |
| Automatic back transliteration benchmark | 4,789 | 0 | 0 |
| Vashantor | 2,500 | 2,500 | 2,500 |
| Missing tag | 467 | 1,002,396 | 0 |

These counts come from the inspected source chunks and fully downloaded [English task file](https://data.mendeley.com/public-files/datasets/pxw2bb4wyt/files/7e10dbc7-5454-4fb4-b61e-0ee8ce505f64/file_downloaded). The English task has 119,727 rows but only 119,554 distinct Banglish strings. The merged file has 120,602 nonnull English rows. Exact-pair deduplication does not make the join key unique; the inspected `pd.merge(..., how="outer")` has no key-uniqueness validation, so multiple targets can multiply joined rows. Do not use the merged row count as a unique-example count or assume every constructed triple is semantically correct. The supplied merge script also does **not** implement the advertised random 80/10/10 splitting; those partition files were not downloaded.

## Natural versus generated provenance

The largest source's [original dataset card](https://huggingface.co/datasets/sk-community/romanized_bangla/blob/main/README.md) explicitly identifies the **same `wikimedia/wikipedia` `20231101.bn` dump used by this project** and says its paragraphs were Romanized with a Python script. It links a [rule-based transliteration implementation](https://github.com/sk-research-community/multilingual-transliterator-llm-training/blob/main/src/rule_based_bengali_transliterate.py), which was downloaded and inspected. This material is natural Bengali with generated Romanization, not naturally authored Banglish. Exclude the entire source from natural Banglish expansion; reusing it would also risk reserved Wikipedia article groups. Its MIT card does not remove the original Wikipedia attribution/share-alike obligations.

[BanTH's source card](https://huggingface.co/datasets/aplycaebous/BanTH/blob/main/README.md) describes naturally collected transliterated comments from 26 YouTube channels and human annotation of **hate labels**. It does not establish human verification of every added Bengali/English target. The inspected translation samples contain visibly unreliable renderings. Natural input provenance alone does not validate those pairs for contrastive training. Retrieve upstream IDs/splits and audit translation creation/quality before including it; the aggregate strips these identifiers.

The [emotion source metadata](https://data.mendeley.com/datasets/4dnrwbxt8n/2) names existing emotion corpora plus YouTube/Twitter comments. This establishes mixed upstream provenance, but not the generation method of every Romanized form or English translation. Do not declare all 79k pairs natural, human-translated, or independent of project sources. The automatic-back-transliteration repository and BanglaTLit entries also need original-row/split mapping; source-name filtering alone cannot eliminate cross-source copies.

## Actionable expansion: original Vashantor v2

The [Vashantor author paper](https://arxiv.org/html/2311.11142), §§3.1–3.3, describes collection from social comments, newspapers, and student conversations, followed by human regional translation. This is stronger natural-input/human-translation evidence than the aggregate's source-name labels. Keep the natural base utterances and label regional renderings as human translations; do not describe the latter as independent user queries.

- Source deposit: [Vashantor v2](https://data.mendeley.com/datasets/bj5jgk878b/2), DOI `10.17632/bj5jgk878b.2`, CC BY 4.0.
- Direct archive: [Vashantor_CSV_Format.zip](https://data.mendeley.com/public-files/datasets/bj5jgk878b/files/ff221f8b-b38d-4db1-b579-fe310fc92b4a/file_downloaded).
- Local file: `data/raw/bangladual/Vashantor_CSV_Format.zip` (ignored); **1,073,445 bytes**; verified SHA-256 `5abb437220ad6fede5344b974f1abe48f4be8d5a9c250b807c0fea6fe4899919`.
- Original public metadata is retained in `data/raw/bangladual/vashantor-file-metadata.json`; a checksum/source manifest is also included in the audit JSON. Preserve author attribution, DOI, license link, and processing changes with imported records.

The archive has 15 CSVs: five regions, each with `Train`, `Validation`, `Test` directories and 1,875/250/375 rows. Trim **column-name whitespace** before mapping. All CSVs contain `bangla_speech`, `banglish_speech`, `{region}_bangla_speech`, `{region}_banglish_speech`, `region_name`, `english_speech`; none has empty fields in this inspection. There is no durable row ID. Use archive path/row number for raw provenance, and derive semantic grouping from all aligned forms. Each core appears across five dialect rows, so 12,500 rows are not 12,500 independent utterances.

The preliminary integrity scan finds 2,507 normalized core-Bangla keys (published base count is 2,500), one core shared by original train and validation, and 631 cores reserved by original validation/test membership. Removing the mixed-split core removes five train rows, leaving 9,370 regional train rows representing 1,876 core groups. Four candidate groups share a form with this project's existing validation/test; 123 share a form with its existing training data. A preliminary 1,872 core groups remain after heldout exclusion. These are **candidate** counts, not final accepted corpus counts: keying only on core Bengali misses connections through other forms and near duplicates.

For production import, connect **any** canonical multilingual form across all original rows, all regions, and the existing corpus before assignment. Original test membership must win over validation, which wins over train, and propagate through the whole component. Block current project heldout groups/keys, preserve matching existing-training group links, run quality/PII heuristics and existing near-duplicate exclusions, and report accepted counts and hashes. Read frozen-test text only to enforce integrity exclusions; do not score or select models from it. Preserve the original splits rather than adopting BanglaDual's random partitions. Then add clean train-only alignments to both contrastive training and adaptation, with all experiments consuming the same training examples.

This audit does not itself implement that importer. It supplies the public download, checksum, schema, provenance, and concrete leakage findings needed for a defensible expansion.
