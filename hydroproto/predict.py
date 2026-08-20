"""Batch inference from the locked HydroProto model checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .data import ALLOWED_AA, LABELS


MODEL_ALIASES = {
    "mlp": "mlp",
    "esm2-mlp": "mlp",
    "esm2_mlp": "mlp",
    "local": "hydrolocal",
    "hlocal": "hydrolocal",
    "hydrolocal": "hydrolocal",
    "proto": "hydroproto",
    "hproto": "hydroproto",
    "hydroproto": "hydroproto",
    "mpc": "hydrompc",
    "hydrompc": "hydrompc",
}

DEFAULT_CHECKPOINT_DIRS = {
    "mlp": Path("models/esm2_mlp_unfreeze3_epoch11"),
    "hydrolocal": Path("models/hlocal_unfreeze6_epoch11"),
    "hydroproto": Path("models/hproto_unfreeze4_epoch23"),
    "hydrompc": Path("models/hydrompc_unfreeze6_epoch20"),
}


def normalize_model_type(value: str) -> str:
    key = str(value).strip().lower()
    if key not in MODEL_ALIASES:
        accepted = ", ".join(sorted(MODEL_ALIASES))
        raise ValueError(f"Unknown model type {value!r}; choose one of: {accepted}")
    return MODEL_ALIASES[key]


def load_prediction_csv(path: str | Path) -> pd.DataFrame:
    """Load an unlabeled accession/sequence table and validate its sequences."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"accession", "sequence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no sequences")
    if frame["accession"].isna().any() or frame["sequence"].isna().any():
        raise ValueError(f"{path} contains an empty accession or sequence")

    normalized = frame.copy()
    normalized["accession"] = normalized["accession"].astype(str).str.strip()
    normalized["sequence"] = (
        normalized["sequence"]
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
    )
    if (normalized["accession"] == "").any() or (normalized["sequence"] == "").any():
        raise ValueError(f"{path} contains an empty accession or sequence")
    if normalized["accession"].duplicated().any():
        duplicates = (
            normalized.loc[normalized["accession"].duplicated(), "accession"]
            .head(5)
            .tolist()
        )
        raise ValueError(f"{path} contains duplicate accessions: {duplicates}")
    invalid = normalized["sequence"].map(
        lambda sequence: not set(sequence).issubset(ALLOWED_AA)
    )
    if invalid.any():
        examples = normalized.loc[invalid, "accession"].head(5).tolist()
        raise ValueError(
            f"{path} contains invalid amino-acid symbols; "
            f"example accessions: {examples}"
        )
    return normalized


def build_prediction_frame(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    chunk_counts: list[int],
) -> pd.DataFrame:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    expected_shape = (len(frame), len(LABELS))
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"Probability shape {probabilities.shape} != {expected_shape}"
        )
    if len(chunk_counts) != len(frame):
        raise ValueError("Chunk counts do not match the number of predictions")
    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain NaN or Inf")
    if np.any(probabilities < 0):
        raise ValueError("Predictions contain negative probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Prediction probabilities do not sum to one")

    predicted = probabilities.argmax(axis=1)
    output = pd.DataFrame(
        {
            "accession": frame["accession"].astype(str).to_numpy(),
            "sequence_length": frame["sequence"].str.len().to_numpy(),
            "chunks": np.asarray(chunk_counts, dtype=np.int64),
            "predicted_label": [LABELS[index] for index in predicted],
            "confidence": probabilities.max(axis=1),
        }
    )
    for class_index, label in enumerate(LABELS):
        output[f"p_{label}"] = probabilities[:, class_index]
    return output


def resolve_device(value: str) -> torch.device:
    requested = str(value).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; use --device cpu or --device auto"
        )
    return device


def load_release_config(checkpoint_dir: Path, model_type: str) -> dict:
    config_path = checkpoint_dir / "config.json"
    checkpoint_path = checkpoint_dir / "final_trainable_state.pt"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing {config_path}. Copy the complete model directory from the "
            "Zenodo release into models/."
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing {checkpoint_path}. Download and extract "
            "HydroProto_models_v1.0.zip from Zenodo."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stored_type = normalize_model_type(config.get("model_type", ""))
    if stored_type != model_type:
        raise ValueError(
            f"Checkpoint config contains {stored_type!r}, but --model-type is "
            f"{model_type!r}"
        )
    return config


def inference_options(config: dict, cli_args, device: torch.device):
    amp = config.get("amp", "bf16") if cli_args.amp == "auto" else cli_args.amp
    if device.type != "cuda":
        amp = "none"
    elif amp == "bf16" and not torch.cuda.is_bf16_supported():
        amp = "fp16"
    return SimpleNamespace(
        chunk_size=(
            cli_args.chunk_size
            if cli_args.chunk_size is not None
            else int(config.get("chunk_size", 900))
        ),
        chunk_overlap=(
            cli_args.chunk_overlap
            if cli_args.chunk_overlap is not None
            else int(config.get("chunk_overlap", 100))
        ),
        chunk_batch_size=(
            cli_args.chunk_batch_size
            if cli_args.chunk_batch_size is not None
            else int(config.get("chunk_batch_size", 1))
        ),
        amp=amp,
        fusion_alpha=float(config.get("fusion_alpha", 0.0)),
    )


def _copy_nested_checkpoint(esm, head, checkpoint_path: Path, device: torch.device):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "esm_trainable_state" not in state or "head_state" not in state:
        raise ValueError(
            f"Unexpected checkpoint format in {checkpoint_path}; expected "
            "esm_trainable_state and head_state"
        )
    esm_parameters = dict(esm.named_parameters())
    for name, value in state["esm_trainable_state"].items():
        if name not in esm_parameters:
            raise KeyError(f"Checkpoint ESM parameter not found in backbone: {name}")
        esm_parameters[name].data.copy_(value.to(device))
    head.load_state_dict(
        {name: value.to(device) for name, value in state["head_state"].items()}
    )


def run_prediction(cli_args) -> tuple[pd.DataFrame, dict]:
    """Load one released checkpoint and predict every input sequence."""
    from transformers import AutoTokenizer, EsmModel

    from . import esm2_mlp as mlp
    from . import hydrolocal, hydrompc, hydroproto

    model_type = normalize_model_type(cli_args.model_type)
    checkpoint_dir = Path(
        cli_args.checkpoint_dir or DEFAULT_CHECKPOINT_DIRS[model_type]
    )
    model_dir = Path(cli_args.model_dir)
    output_csv = Path(cli_args.output_csv)
    if output_csv.exists() and not cli_args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_csv}. Use --overwrite to replace it."
        )
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"ESM-2 backbone directory not found: {model_dir}. Place "
            "facebook/esm2_t33_650M_UR50D there or pass --model-dir."
        )

    frame = load_prediction_csv(cli_args.input_csv)
    config = load_release_config(checkpoint_dir, model_type)
    device = resolve_device(cli_args.device)
    options = inference_options(config, cli_args, device)
    if options.chunk_overlap >= options.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if options.chunk_batch_size < 1:
        raise ValueError("chunk_batch_size must be positive")

    print(
        f"Loading {model_type} checkpoint from {checkpoint_dir} on {device}...",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    esm = EsmModel.from_pretrained(
        str(model_dir),
        local_files_only=True,
        add_pooling_layer=False,
        low_cpu_mem_usage=True,
    ).to(device)
    unfreeze_layers = int(config["unfreeze_layers"])
    mlp.freeze_esm_layers(esm, unfreeze_layers)
    input_dim = int(esm.config.hidden_size)
    hidden_dim = int(config.get("hidden_dim", 512))
    feature_dim = int(config.get("feature_dim", 256))
    projection_dim = int(config.get("projection_dim", 128))
    dropout = float(config.get("dropout", 0.2))
    checkpoint_path = checkpoint_dir / "final_trainable_state.pt"

    if model_type == "mlp":
        head = mlp.SequenceMLP(
            input_dim,
            hidden_dim,
            feature_dim,
            len(LABELS),
            dropout,
        ).to(device)
        _copy_nested_checkpoint(esm, head, checkpoint_path, device)
    elif model_type == "hydrolocal":
        head = hydrolocal.HydroLocalHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            projection_dim=projection_dim,
            num_classes=len(LABELS),
            dropout=dropout,
        ).to(device)
        hydrolocal.load_trainable_state(esm, head, checkpoint_path, device)
    elif model_type == "hydroproto":
        head = hydroproto.HydroProtoHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            projection_dim=projection_dim,
            num_classes=len(LABELS),
            num_prototypes=int(config.get("num_prototypes", 5)),
            prototype_temperature=float(
                config.get("prototype_temperature", 0.1)
            ),
            dropout=dropout,
        ).to(device)
        _copy_nested_checkpoint(esm, head, checkpoint_path, device)
    else:
        head = hydrompc.HydroMPC650MHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            projection_dim=projection_dim,
            num_classes=len(LABELS),
            num_prototypes=int(config.get("num_prototypes", 5)),
            prototype_temperature=float(
                config.get("prototype_temperature", 0.1)
            ),
            dropout=dropout,
        ).to(device)
        hydrompc.load_trainable_state(esm, head, checkpoint_path, device)

    indices = np.arange(len(frame), dtype=np.int64)
    print(f"Predicting {len(frame)} sequences...", flush=True)
    if model_type == "mlp":
        probabilities, chunk_counts = mlp.evaluate(
            esm, head, tokenizer, frame, None, indices, options, device
        )
    elif model_type == "hydrolocal":
        probabilities, chunk_counts = hydrolocal.evaluate(
            esm, head, tokenizer, frame, indices, options, device
        )
    elif model_type == "hydroproto":
        values = hydroproto.evaluate(
            esm, head, tokenizer, frame, indices, options, device
        )
        probabilities = values["final_probabilities"]
        chunk_counts = values["chunk_counts"]
    else:
        probabilities, _, _, chunk_counts = hydrompc.evaluate(
            esm, head, tokenizer, frame, indices, options, device
        )

    predictions = build_prediction_frame(frame, probabilities, chunk_counts)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_csv, index=False, encoding="utf-8-sig")
    metadata = {
        "model_type": model_type,
        "records": int(len(frame)),
        "input_csv": str(Path(cli_args.input_csv).resolve()),
        "output_csv": str(output_csv.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "esm2_model_dir": str(model_dir.resolve()),
        "device": str(device),
        "amp": options.amp,
        "chunk_size": options.chunk_size,
        "chunk_overlap": options.chunk_overlap,
        "chunk_batch_size": options.chunk_batch_size,
        "labels": LABELS,
    }
    metadata_path = output_csv.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return predictions, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict bacterial hydrolase classes from an accession/sequence CSV "
            "using a downloaded locked checkpoint; no training is performed."
        )
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument(
        "--model-type",
        default="hydroproto",
        choices=["hydroproto", "hydrolocal", "hydrompc", "mlp"],
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Released model directory; defaults to the selected directory under models/",
    )
    parser.add_argument(
        "--model-dir",
        default="hf_cache/esm2_t33_650M_UR50D",
        help="Local facebook/esm2_t33_650M_UR50D directory",
    )
    parser.add_argument("--output-csv", default="outputs/predictions.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp",
        choices=["auto", "bf16", "fp16", "none"],
        default="auto",
    )
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    parser.add_argument("--chunk-batch-size", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictions, metadata = run_prediction(args)
    print(
        f"Predicted {len(predictions)} sequences with {metadata['model_type']} "
        f"on {metadata['device']}."
    )
    print(f"Predictions: {metadata['output_csv']}")
    print(f"Metadata: {Path(metadata['output_csv']).with_suffix('.metadata.json')}")


if __name__ == "__main__":
    main()
