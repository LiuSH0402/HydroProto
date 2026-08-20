# Model weights

Model weights are intentionally excluded from Git because the fine-tuned
checkpoints are large. The versioned Zenodo release contains the four locked final
checkpoints used in the study: ESM2-MLP, HydroLocal, HydroProto and HydroMPC.

The checkpoints store the trainable fine-tuned layers and task heads; they do not
redistribute the ESM-2 650M base model. Obtain
`facebook/esm2_t33_650M_UR50D` separately and provide its local directory through
the command-line interfaces documented in the project README.

The locked final checkpoints are archived at
[doi:10.5281/zenodo.22020952](https://doi.org/10.5281/zenodo.22020952). Compact
result tables and the archive description are available in
`../results/RESULT_ARCHIVE.md`.
