# Additional evaluation datasets — v0.2.0

This directory contains only the additional split and external test dataset.
Original `data/train_dataset.csv` and `data/test_dataset.csv` remain unchanged.
New experiment code, predictions, logs, statistical exports and weights are not
added by this release. The original repository code and results remain available.

## grouped_split.csv

All 3,401 original accessions with their original label, original partition,
additional grouped partition and similarity-component identifier. Join `accession`
to the original training and test CSVs to obtain amino acid sequences. The
`grouped_split` column, not `original_split`, selects the additional holdout:
2,722 training sequences and 679 test sequences. Entire components were assigned
to a single partition using fixed split seed 42. Components are connected
components of MMseqs2-detected sequence pairs with identity >=80% and bidirectional
coverage >=80%, jointly across classes. There are 370 test components; none is
shared between the grouped partitions. Verification found no cross-partition
alignment satisfying these thresholds. This is sensitivity analysis on the same
cohort, not a new external or temporal dataset or exhaustive homology exclusion.

## external_test_dataset.csv

10,523 bacterial protein representatives covering the five target classes.
Columns identify the source accession, amino acid sequence, task label, source
and current EC annotations, current primary accession, source collections,
organism, taxonomy identifier and similarity component.

The source is the released EC-Bench collections at commit
5e98a2c632c6644f06975229ededbd09f4866809:
https://github.com/dsaeedeh/EC-Bench/tree/5e98a2c632c6644f06975229ededbd09f4866809 .
10,521 representatives occur in the released training collections; two come from
Price-149. This task-adapted set is external to this study's supervised cohort;
it is not the official EC-Bench test split or a temporal validation.

Ambiguous target classes were excluded. Accessions and exact sequences overlapping
the entire original 3,401-sequence cohort were excluded, along with detected
MMseqs2 matches of >=80% identity, >=80% bidirectional coverage and E<=0.001.
Source labels had to agree with fully interpretable current EC annotations and
the sequence version. Internal components used the same thresholds. The longest
eligible sequence per single-label component was selected, with accession order
breaking ties. Inclusion and representative rules were fixed before prediction.
Five-class counts are EC3.1: 4,211; EC3.2: 520; EC3.4: 1,583; EC3.5: 2,038;
Other EC3: 2,171. Annotation agreement does not imply direct experimental enzyme
characterization, and ESM-2 pretraining exposure was not excluded.

## Source terms and integrity

UniProt-derived records retain their applicable CC BY 4.0 attribution terms:
https://www.uniprot.org/help/license . EC-Bench source terms remain applicable:
https://github.com/dsaeedeh/EC-Bench . The root MIT LICENSE covers project code,
not a relicensing of third-party data, backbones or existing model archives.
The CSV byte sizes, SHA-256 hashes and row counts are listed in MANIFEST.csv.
