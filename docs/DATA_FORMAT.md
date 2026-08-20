# Data format

Input is a UTF-8 CSV file with three required columns:

```text
accession,label,sequence
```

`accession` must be unique, `sequence` must contain valid amino-acid symbols, and
`label` must be one of:

- `EC3.1_ester_bonds`
- `EC3.2_glycosylases`
- `EC3.4_peptidases`
- `EC3.5_CN_nonpeptide`
- `EC3_other`

The published workflow fixes an 80/20 stratified train/test split. Model selection
uses only five-fold cross-validation within the training split. The fixed test set
is additionally filtered into nested `<80%`, `<60%` and `<40%` maximum train-set
identity subsets; these subsets are evaluation views, not separately trained
models.
