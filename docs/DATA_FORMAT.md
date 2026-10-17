# Data format

## Batch prediction

`hydroproto predict` accepts a UTF-8 CSV file with two required columns:

```text
accession,sequence
```

`accession` must be unique and non-empty. `sequence` is case-insensitive,
whitespace is removed, and only valid amino-acid symbols are accepted. A ground
truth label is not required. See `examples/predict_sequences.csv` for a ready-to-run
example.

The prediction output contains:

```text
accession,sequence_length,chunks,predicted_label,confidence,
p_EC3.1_ester_bonds,p_EC3.2_glycosylases,p_EC3.4_peptidases,
p_EC3.5_CN_nonpeptide,p_EC3_other
```

## Training and evaluation

Training and evaluation use three required columns:

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
uses only five-fold cross-validation within the training split. The historical
`lt80`, `lt60` and `lt40` result keys identify nested pooled evaluation groups:
sequences with measured maximum training-reference identity below the indicated
threshold together with sequences without qualified training-reference hits.
No measured identity is assigned to no-hit sequences. These are evaluation views,
not separately trained models. See [data provenance](DATA_PROVENANCE.md) for the
group counts and definitions.
