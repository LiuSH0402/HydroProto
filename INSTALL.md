# Installation

## Conda (recommended for reproducing the GPU experiments)

```bash
conda env create -f environment.yml
conda activate hydroproto
pip install -e .
```

The locked experiments used Python 3.10.16, PyTorch 2.6.0+cu126 and
Transformers 4.49.0. A CUDA build compatible with the host driver may be used.

## Existing environment

```bash
pip install -e ".[analysis,test]"
```

HydroProto does not download ESM-2 automatically. Pass a local Hugging Face
snapshot or model directory with `--model-dir`.

## Sanity checks

```bash
python -m compileall hydroproto scripts
pytest -q
hydroproto --help
```
