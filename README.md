# HydroProto

[![Model archive DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22020952.svg)](https://doi.org/10.5281/zenodo.22020952)

HydroProto is a prototype-guided protein language model for five-class bacterial
hydrolase prediction. It combines ESM-2 650M representations with class-conditional
online multi-prototype learning and is evaluated under both ordinary random-split
and progressively stricter train-test homology conditions.

This repository also contains three controlled comparators: **ESM2-MLP**,
**HydroLocal** and **HydroMPC**. The naming is fixed throughout new code and
documentation; historical aliases remain loadable only for old checkpoints.

## Method overview

| Model | Local positive selection | Online class prototypes | Prediction head |
|---|---:|---:|---|
| ESM2-MLP | No | No | MLP |
| HydroLocal | Yes | No | MLP |
| HydroProto | No | Yes | Prototype evidence |
| HydroMPC | Yes | Yes | MLP/prototype fusion |

The default HydroProto head learns five prototypes for each of five enzyme classes
(25 prototypes in total). This allows one enzyme category to be represented by
multiple sequence subfamilies instead of a single class center. See
[`docs/MODELS.md`](docs/MODELS.md) for the exact distinction among the four models.

## Repository layout

```text
HydroProto/
├── hydroproto/       installable model and training code
├── configs/          frozen-embedding development configurations
├── scripts/          statistical, error and prototype analyses
├── data/             fixed public 2,721/680 train-test split
├── examples/         five-record input example
├── docs/             model, data and reproducibility documentation
├── models/           placeholder for downloaded checkpoints
├── results/          compact index of the external result archive
├── tests/            lightweight model-head tests
├── environment.yml
├── requirements.txt
└── pyproject.toml
```

## Installation

```bash
conda env create -f environment.yml
conda activate hydroproto
pip install -e .
```

The reference server environment used Python 3.10.16, PyTorch 2.6.0+cu126 and
Transformers 4.49.0. HydroProto deliberately does not download ESM-2 weights;
provide the existing local `facebook/esm2_t33_650M_UR50D` snapshot through
`--model-dir`. More details are in [`INSTALL.md`](INSTALL.md).

## Input data

Sequence files are CSV tables with unique `accession`, categorical `label` and
amino-acid `sequence` columns. The five accepted labels and the exact validation
rules are described in [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md). The fixed
2,721-record training set and 680-record locked test set used in the study are
included as [`data/train_dataset.csv`](data/train_dataset.csv) and
[`data/test_dataset.csv`](data/test_dataset.csv); checksums and column definitions
are provided in [`data/README.md`](data/README.md).

## Five-fold model selection

The following command reproduces the main HydroProto cross-validation interface:

```bash
hydroproto proto \
  --train-csv /path/to/train_dataset.csv \
  --model-dir /path/to/esm2_t33_650M_UR50D \
  --output-root outputs/hydroproto_unfreeze4 \
  --folds 5 \
  --unfreeze-layers 4 \
  --num-prototypes 5 \
  --gradient-checkpointing \
  --amp bf16
```

Replace `proto` with `mlp`, `local` or `mpc` to run the corresponding comparator.
Use `hydroproto <command> --help` for the complete option list.

The earlier frozen-embedding experiments are also reproducible with `embed`,
`embedding-train` and `embedding-cv`; their YAML configurations are retained in
`configs/`. The sequence-similarity baseline is available through `mmseqs`.

## Locked final evaluation

Hyperparameters and epoch counts must be selected before test evaluation. One
final model is trained on the full training set, then the saved predictions are
filtered into homology subsets:

```bash
hydroproto final \
  --model-type hydroproto \
  --train-csv /path/to/train_dataset.csv \
  --test-csv /path/to/test_dataset.csv \
  --subset lt80=/path/to/test_nested_lt80.csv \
  --subset lt60=/path/to/test_nested_lt60.csv \
  --subset lt40=/path/to/test_nested_lt40.csv \
  --model-dir /path/to/esm2_t33_650M_UR50D \
  --output-dir outputs/hydroproto_final \
  --epoch-selection-source median_cv_best_epoch \
  --unfreeze-layers 4 \
  --epochs 23 \
  --num-prototypes 5 \
  --fusion-alpha 0.0 \
  --gradient-checkpointing \
  --amp bf16
```

The fixed experimental protocol is documented in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Results and large files

Git should contain code, configurations, compact tables and documentation—not the
ESM-2 base model or multi-gigabyte training checkpoints. The complete internal
archive is organized separately in `modeling/result/`, with manifests that map
every local and server artifact to its source. Before public release, upload the
selected weights and result bundle to a versioned data repository and replace the
placeholder instructions in [`models/README.md`](models/README.md).

The four locked final checkpoints are archived on Zenodo under
[doi:10.5281/zenodo.22020952](https://doi.org/10.5281/zenodo.22020952).

## Authors

- Shang-Hua Liu ([ORCID: 0009-0003-9000-7379](https://orcid.org/0009-0003-9000-7379))
- Ke-Jun Deng (corresponding author)

Affiliation: The Clinical Hospital of Chengdu Brain Science Institute, School of
Life Science and Technology and Center for Informational Biology, University of
Electronic Science and Technology of China, Chengdu 610054, China.

## Release checklist

- Select and add a software license; none is assumed by this preparation step.
- Run the tests in a clean environment and publish the exact data accession list.

## Acknowledgements

The project was methodologically inspired by contrastive enzyme annotation, while
the implemented architecture, local-positive ablations, online multi-prototype
head and homology-stratified evaluation were developed for the bacterial
hydrolase classification task.
