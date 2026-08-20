# Public train/test split

This directory contains the fixed sequence split used for the HydroProto study.

| File | Records | Purpose | SHA-256 |
|---|---:|---|---|
| `train_dataset.csv` | 2,721 | Model development, cross-validation and final training | `80bc0c5c8007c96b253a3c205a0b41689656330764eba8c53cdc9730db9055b0` |
| `test_dataset.csv` | 680 | Locked final evaluation | `3b592394728596439d5afd730b354b5f85dd7e002758cf9040b6ebcb9621da1b` |

The files use the following columns:

- `split`: fixed split assignment.
- `label`: one of the five hydrolase classes.
- `accession`: unique protein accession.
- `entry_name`, `organism`, `taxid`: source-record metadata.
- `length`: amino-acid sequence length.
- `sequence`: amino-acid sequence.
- `original_header`: original sequence-record header.

There are no duplicate accessions within either file and no accession overlap
between the training and test sets. The test set was kept locked during model
selection. See `../docs/DATA_FORMAT.md` and `../docs/REPRODUCIBILITY.md` for the
validation rules and evaluation protocol.

The software license for this repository does not replace any terms associated
with the underlying source database records. Users are responsible for checking
and following the applicable source-database attribution and reuse terms.
