# Data provenance and filtering

The authors report retrieving hydrolase sequences from UniProtKB in January 2025,
restricting the search to bacterial, reviewed UniProtKB/Swiss-Prot entries and
excluding annotated fragments. The exact original query string and release
identifier were not preserved in the available download record; neither is
reconstructed here as an authenticated original query or release.

After download, EC second-level annotations were used to group EC 3.1, EC 3.2,
EC 3.4 and EC 3.5 separately. Other EC 3 pools the remaining hydrolase subclasses
because their individual sample sizes were limited. This pooled class is not a
non-enzyme or unknown-function category. Labels are inherited annotations, not
new measurements of enzyme activity. Reviewed status is not evidence that every
EC annotation was established by direct functional experiments.

The recorded input manifest contains 3,671 records. Filtering removes annotated
fragments, sequences shorter than 50 amino acids, and sequences containing residues
outside ACDEFGHIKLMNPQRSTVWY. Accessions or identical sequences appearing across
input target classes are excluded. Within-class identical sequences retain one
representative. The resulting 3,401 unique sequences are retained unchanged as
2,721 training and 680 held-out test records.

Original labels and records are retained. The filtering description documents
the original workflow; it does not establish direct experimental evidence for
every EC annotation or authenticate an exact historical UniProt release.

## Sequence similarity and no-hit status

Original `lt80`, `lt60` and `lt40` result keys are historical machine-readable
names for **pooled** groups. They contain measured identities below the indicated
threshold together with all 198 sequences without qualified training-reference
hits. The pooled groups contain 588, 496 and 334 sequences. The last contains
136 sequences with measured identity below 40% and 198 no-hit sequences. No
measured identity is assigned to no-hit sequences. The original keys are retained
to preserve correspondence with saved predictions; the manuscript revision names these groups explicitly.

## Reuse terms

The repository source code is covered by its MIT LICENSE. UniProt-derived records
retain the applicable UniProt CC BY 4.0 attribution requirements:
https://www.uniprot.org/help/license . EC-Bench resources retain their original
terms: https://github.com/dsaeedeh/EC-Bench . The existing model-weight archive
retains its CC BY 4.0 license; this code release does not relicense that archive or
the pretrained ESM backbones.
