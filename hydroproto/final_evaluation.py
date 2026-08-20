from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, EsmModel

from . import esm2_mlp as mlp
from . import hydrolocal
from . import hydroproto
from . import hydrompc
from .data import (
    LABELS,
    LocalPositiveBatchSampler,
    build_local_positive_graph,
    encode_labels,
    load_sequence_csv,
    seed_everything,
)
from .metrics import classification_metrics


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_subsets(values: list[str]) -> dict[str, Path]:
    subsets: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Subset must use NAME=PATH syntax: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name == "test_all":
            raise ValueError(f"Invalid subset name: {name!r}")
        if name in subsets:
            raise ValueError(f"Duplicate subset name: {name}")
        subsets[name] = Path(raw_path)
    return subsets


def audit_datasets(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    subset_paths: dict[str, Path],
) -> tuple[dict[str, np.ndarray], dict]:
    for name, frame in (("train", train_frame), ("test", test_frame)):
        if frame["accession"].astype(str).duplicated().any():
            raise ValueError(f"{name} contains duplicate accessions")
    train_ids = set(train_frame["accession"].astype(str))
    test_ids = set(test_frame["accession"].astype(str))
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Train/test accession overlap: {len(overlap)}")

    test_lookup = test_frame.set_index(test_frame["accession"].astype(str))
    subset_indices: dict[str, np.ndarray] = {
        "test_all": np.arange(len(test_frame), dtype=np.int64)
    }
    subset_sets: dict[str, set[str]] = {"test_all": test_ids}
    audit = {
        "train_records": int(len(train_frame)),
        "test_records": int(len(test_frame)),
        "train_test_accession_overlap": 0,
        "subsets": {},
    }
    position = {
        accession: index
        for index, accession in enumerate(test_frame["accession"].astype(str))
    }
    for name, path in subset_paths.items():
        frame = load_sequence_csv(path)
        ids = frame["accession"].astype(str)
        if ids.duplicated().any():
            raise ValueError(f"Subset {name} contains duplicate accessions")
        id_set = set(ids)
        missing = id_set - test_ids
        if missing:
            raise ValueError(
                f"Subset {name} contains {len(missing)} accessions outside test"
            )
        subset_lookup = frame.set_index(ids)
        for accession in id_set:
            source = test_lookup.loc[accession]
            subset_row = subset_lookup.loc[accession]
            if str(source["label"]) != str(subset_row["label"]):
                raise ValueError(f"Label mismatch for {accession} in {name}")
            if str(source["sequence"]) != str(subset_row["sequence"]):
                raise ValueError(f"Sequence mismatch for {accession} in {name}")
        ordered = [position[accession] for accession in ids]
        subset_indices[name] = np.asarray(ordered, dtype=np.int64)
        subset_sets[name] = id_set
        audit["subsets"][name] = {
            "path": str(path.resolve()),
            "records": int(len(frame)),
            "all_members_in_test": True,
            "labels_and_sequences_match_test": True,
        }

    if {"lt80", "lt60", "lt40"}.issubset(subset_sets):
        nested = (
            subset_sets["lt40"]
            <= subset_sets["lt60"]
            <= subset_sets["lt80"]
            <= subset_sets["test_all"]
        )
        audit["nested_lt40_lt60_lt80_test"] = bool(nested)
        if not nested:
            raise ValueError("Similarity subsets are not properly nested")
    return subset_indices, audit


def build_esm(args, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
    )
    esm = EsmModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
        add_pooling_layer=False,
        low_cpu_mem_usage=True,
    ).to(device)
    trainable_esm = mlp.freeze_esm_layers(esm, args.unfreeze_layers)
    if args.gradient_checkpointing and args.unfreeze_layers > 0:
        esm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        esm.config.use_cache = False
    return tokenizer, esm, int(trainable_esm)


def optimizer_for(esm, head, args):
    esm_parameters = [
        parameter for parameter in esm.parameters() if parameter.requires_grad
    ]
    groups = [{"params": head.parameters(), "lr": args.head_learning_rate}]
    if esm_parameters:
        groups.append({"params": esm_parameters, "lr": args.learning_rate})
    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=args.weight_decay,
    )
    return optimizer, esm_parameters


def save_checkpoint(
    path: Path,
    esm: EsmModel,
    head: nn.Module,
    metadata: dict,
) -> None:
    torch.save(
        {
            "metadata": metadata,
            "esm_trainable_state": {
                name: parameter.detach().cpu().clone()
                for name, parameter in esm.named_parameters()
                if parameter.requires_grad
            },
            "head_state": {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            },
        },
        path,
    )


def train_mlp(
    train_frame,
    train_labels,
    test_frame,
    args,
    output_dir,
    device,
):
    tokenizer, esm, trainable_esm = build_esm(args, device)
    head = mlp.SequenceMLP(
        int(esm.config.hidden_size),
        args.hidden_dim,
        args.feature_dim,
        len(LABELS),
        args.dropout,
    ).to(device)
    optimizer, esm_parameters = optimizer_for(esm, head, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    train_idx = np.arange(len(train_frame), dtype=np.int64)
    history = []
    gradient_verified = not bool(esm_parameters)

    for epoch in range(1, args.epochs + 1):
        esm.train(args.unfreeze_layers > 0)
        head.train()
        total_loss = 0.0
        group_count = 0
        for group in mlp.balanced_groups(
            train_labels,
            args.samples_per_class,
            args.seed,
            epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for index in group:
                sequence = train_frame.iloc[int(index)]["sequence"]
                chunks = mlp.split_sequence(
                    sequence,
                    args.chunk_size,
                    args.chunk_overlap,
                )
                sequence_loss = torch.zeros((), device=device)
                for start in range(0, len(chunks), args.chunk_batch_size):
                    batch_chunks = chunks[start : start + args.chunk_batch_size]
                    encoded = mlp.token_chunks(tokenizer, batch_chunks, device)
                    with mlp.autocast_context(device, args.amp):
                        output = esm(**encoded)
                        embedding = mlp.residue_mean(
                            output.last_hidden_state,
                            [len(item) for item in batch_chunks],
                        )
                        logits = head(embedding)
                        target = torch.full(
                            (len(batch_chunks),),
                            int(train_labels[int(index)]),
                            dtype=torch.long,
                            device=device,
                        )
                        sequence_loss = (
                            sequence_loss
                            + criterion(logits, target) / len(chunks)
                        )
                (sequence_loss / len(group)).backward()
                group_loss += float(sequence_loss.detach().cpu())
                if not gradient_verified:
                    gradients = [
                        parameter.grad.detach().float().norm()
                        for parameter in esm_parameters
                        if parameter.grad is not None
                    ]
                    if not gradients:
                        raise RuntimeError(
                            "No gradients reached unfrozen ESM parameters"
                        )
                    norm = torch.linalg.vector_norm(torch.stack(gradients)).item()
                    if not np.isfinite(norm) or norm <= 0:
                        raise RuntimeError(f"Invalid ESM gradient norm: {norm}")
                    print(f"[gradient-check] esm_norm={norm:.6g}", flush=True)
                    gradient_verified = True
            torch.nn.utils.clip_grad_norm_(
                list(head.parameters()) + esm_parameters,
                args.max_grad_norm,
            )
            optimizer.step()
            total_loss += group_loss / len(group)
            group_count += 1
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(group_count, 1),
            "groups": group_count,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            output_dir / "history.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"[final] epoch={epoch:02d}/{args.epochs} "
            f"train_loss={row['train_loss']:.6f}",
            flush=True,
        )

    save_checkpoint(
        output_dir / "final_trainable_state.pt",
        esm,
        head,
        {"model_type": args.model_type, "epochs": args.epochs},
    )
    test_indices = np.arange(len(test_frame), dtype=np.int64)
    probabilities, chunks = mlp.evaluate(
        esm,
        head,
        tokenizer,
        test_frame,
        encode_labels(test_frame["label"]),
        test_indices,
        args,
        device,
    )
    return {"final": probabilities}, chunks, {
        "trainable_esm_parameters": trainable_esm,
        "gradient_verified": gradient_verified,
    }


def train_hydroproto(
    train_frame,
    train_labels,
    test_frame,
    args,
    output_dir,
    device,
):
    tokenizer, esm, trainable_esm = build_esm(args, device)
    head = hydroproto.HydroProtoHead(
        input_dim=int(esm.config.hidden_size),
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        projection_dim=args.projection_dim,
        num_classes=len(LABELS),
        num_prototypes=args.num_prototypes,
        prototype_temperature=args.prototype_temperature,
        dropout=args.dropout,
    ).to(device)
    optimizer, esm_parameters = optimizer_for(esm, head, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    train_idx = np.arange(len(train_frame), dtype=np.int64)
    history = []
    gradient_verified = not bool(esm_parameters)

    for epoch in range(1, args.epochs + 1):
        if (
            not bool(head.prototypes_initialized.item())
            and epoch > args.prototype_warmup_epochs
        ):
            hydroproto.initialize_online_prototypes(
                esm,
                head,
                tokenizer,
                train_frame,
                train_labels,
                train_idx,
                args,
                0,
                output_dir,
                device,
            )
        row, gradient_verified = hydroproto.train_epoch(
            esm,
            head,
            tokenizer,
            train_frame,
            train_labels,
            train_idx,
            optimizer,
            criterion,
            esm_parameters,
            args,
            0,
            epoch,
            device,
            gradient_verified,
        )
        row.update(
            {
                "epoch": epoch,
                "phase": (
                    "prototype"
                    if bool(head.prototypes_initialized.item())
                    else "warmup"
                ),
            }
        )
        history.append(row)
        pd.DataFrame(history).to_csv(
            output_dir / "history.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"[final] epoch={epoch:02d}/{args.epochs} "
            f"phase={row['phase']} train_loss={row['train_loss']:.6f}",
            flush=True,
        )
    if not bool(head.prototypes_initialized.item()):
        raise RuntimeError("HydroProto prototypes were never initialized")
    save_checkpoint(
        output_dir / "final_trainable_state.pt",
        esm,
        head,
        {"model_type": args.model_type, "epochs": args.epochs},
    )
    hydroproto.save_prototypes(output_dir / "learned_prototypes.npz", head)
    outputs = hydroproto.evaluate(
        esm,
        head,
        tokenizer,
        test_frame,
        np.arange(len(test_frame), dtype=np.int64),
        args,
        device,
    )
    return {
        "final": outputs["final_probabilities"],
        "classification_branch": outputs["class_probabilities"],
        "prototype_branch": outputs["prototype_probabilities"],
    }, outputs["chunk_counts"], {
        "trainable_esm_parameters": trainable_esm,
        "gradient_verified": gradient_verified,
        "prototypes_initialized": True,
    }


def train_hydrolocal(
    train_frame,
    train_labels,
    test_frame,
    args,
    output_dir,
    device,
):
    tokenizer, esm, trainable_esm = build_esm(args, device)
    head = hydrolocal.HydroLocalHead(
        input_dim=int(esm.config.hidden_size),
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        projection_dim=args.projection_dim,
        num_classes=len(LABELS),
        dropout=args.dropout,
    ).to(device)
    optimizer, esm_parameters = optimizer_for(esm, head, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    train_idx = np.arange(len(train_frame), dtype=np.int64)
    reference = hydrolocal.load_reference_embeddings(
        args.reference_embeddings,
        train_frame,
    )
    graph, graph_stats = build_local_positive_graph(
        reference,
        train_labels,
        neighbors_per_sample=args.local_knn,
        mutual=args.local_mutual_knn,
    )
    sampler = LocalPositiveBatchSampler(
        train_labels,
        graph,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    write_json(output_dir / "local_graph_stats.json", graph_stats)
    history = []
    gradient_verified = not bool(esm_parameters)

    for epoch in range(1, args.epochs + 1):
        row, gradient_verified = hydrolocal.train_epoch_grad_cache(
            esm,
            head,
            tokenizer,
            train_frame,
            train_labels,
            train_idx,
            sampler,
            graph,
            optimizer,
            criterion,
            args,
            device,
            epoch,
            gradient_verified,
        )
        row["epoch"] = epoch
        history.append(row)
        pd.DataFrame(history).to_csv(
            output_dir / "history.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"[final] epoch={epoch:02d}/{args.epochs} "
            f"train_loss={row['train_loss']:.6f}",
            flush=True,
        )
    hydrolocal.save_trainable_state(
        esm,
        head,
        output_dir / "final_trainable_state.pt",
    )
    probabilities, chunks = hydrolocal.evaluate(
        esm,
        head,
        tokenizer,
        test_frame,
        np.arange(len(test_frame), dtype=np.int64),
        args,
        device,
    )
    return {"final": probabilities}, chunks, {
        "trainable_esm_parameters": trainable_esm,
        "gradient_verified": gradient_verified,
        "local_graph": graph_stats,
    }


def train_hydrompc(
    train_frame,
    train_labels,
    test_frame,
    args,
    output_dir,
    device,
):
    tokenizer, esm, trainable_esm = build_esm(args, device)
    head = hydrompc.HydroMPC650MHead(
        input_dim=int(esm.config.hidden_size),
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        projection_dim=args.projection_dim,
        num_classes=len(LABELS),
        num_prototypes=args.num_prototypes,
        prototype_temperature=args.prototype_temperature,
        dropout=args.dropout,
    ).to(device)
    optimizer, esm_parameters = optimizer_for(esm, head, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    train_idx = np.arange(len(train_frame), dtype=np.int64)
    reference = hydrolocal.load_reference_embeddings(
        args.reference_embeddings,
        train_frame,
    )
    graph, graph_stats = build_local_positive_graph(
        reference,
        train_labels,
        neighbors_per_sample=args.local_knn,
        mutual=args.local_mutual_knn,
    )
    sampler = LocalPositiveBatchSampler(
        train_labels,
        graph,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    write_json(output_dir / "local_graph_stats.json", graph_stats)
    history = []
    gradient_verified = not bool(esm_parameters)

    for epoch in range(1, args.epochs + 1):
        if (
            not bool(head.prototypes_initialized.item())
            and epoch > args.prototype_warmup_epochs
        ):
            hydrompc.initialize_online_prototypes(
                esm,
                head,
                tokenizer,
                train_frame,
                train_labels,
                train_idx,
                args,
                device,
                output_dir,
            )
        row, gradient_verified = hydrompc.train_epoch_grad_cache(
            esm,
            head,
            tokenizer,
            train_frame,
            train_labels,
            train_idx,
            sampler,
            graph,
            optimizer,
            criterion,
            args,
            device,
            epoch,
            gradient_verified,
        )
        row.update(
            {
                "epoch": epoch,
                "prototypes_initialized": bool(
                    head.prototypes_initialized.item()
                ),
            }
        )
        history.append(row)
        pd.DataFrame(history).to_csv(
            output_dir / "history.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"[final] epoch={epoch:02d}/{args.epochs} "
            f"proto_on={row['prototypes_initialized']} "
            f"train_loss={row['train_loss']:.6f}",
            flush=True,
        )
    if not bool(head.prototypes_initialized.item()):
        raise RuntimeError("HydroMPC prototypes were never initialized")
    hydrompc.save_trainable_state(
        esm,
        head,
        output_dir / "final_trainable_state.pt",
    )
    np.savez_compressed(
        output_dir / "learned_prototypes.npz",
        prototypes=(
            head.normalized_prototypes().detach().float().cpu().numpy()
        ),
    )
    final_p, class_p, prototype_p, chunks = hydrompc.evaluate(
        esm,
        head,
        tokenizer,
        test_frame,
        np.arange(len(test_frame), dtype=np.int64),
        args,
        device,
    )
    if prototype_p is None:
        raise RuntimeError("HydroMPC prototype branch unavailable at test")
    return {
        "final": final_p,
        "classification_branch": class_p,
        "prototype_branch": prototype_p,
    }, chunks, {
        "trainable_esm_parameters": trainable_esm,
        "gradient_verified": gradient_verified,
        "prototypes_initialized": True,
        "local_graph": graph_stats,
    }


def save_test_results(
    output_dir: Path,
    test_frame: pd.DataFrame,
    test_labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    chunk_counts: list[int],
    subset_indices: dict[str, np.ndarray],
) -> dict:
    final_probabilities = probabilities["final"]
    predicted = final_probabilities.argmax(axis=1)
    prediction_frame = pd.DataFrame(
        {
            "accession": test_frame["accession"].astype(str).to_numpy(),
            "true_label": [LABELS[index] for index in test_labels],
            "predicted_label": [LABELS[index] for index in predicted],
            "correct": predicted == test_labels,
            "chunks": chunk_counts,
        }
    )
    for branch, values in probabilities.items():
        for class_index, label in enumerate(LABELS):
            prediction_frame[f"p_{branch}_{label}"] = values[:, class_index]
    prediction_frame.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_metrics: dict[str, dict] = {}
    summary_rows = []
    for subset_name, indices in subset_indices.items():
        subset_metrics: dict[str, dict] = {}
        for branch, values in probabilities.items():
            subset_metrics[branch] = classification_metrics(
                test_labels[indices],
                values[indices],
            )
        all_metrics[subset_name] = subset_metrics
        final = subset_metrics["final"]
        summary_rows.append(
            {
                "subset": subset_name,
                "n": int(len(indices)),
                "accuracy": final["accuracy"],
                "balanced_accuracy": final["balanced_accuracy"],
                "macro_f1": final["macro_f1"],
                "weighted_f1": final["weighted_f1"],
                "mcc": final["mcc"],
            }
        )
        prediction_frame.iloc[indices].to_csv(
            output_dir / f"test_predictions_{subset_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    write_json(output_dir / "test_metrics.json", all_metrics)
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "test_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return all_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one locked ESM2-650M hydrolase model on all training "
            "records for a fixed CV-selected epoch count, then evaluate the "
            "external test set exactly once."
        )
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["mlp", "hydroproto", "hproto", "hydrolocal", "hlocal", "hydrompc"],
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument(
        "--subset",
        action="append",
        default=[],
        help="Nested test subset in NAME=PATH form",
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--reference-embeddings")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epoch-selection-source", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-layers", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--local-knn", type=int, default=10)
    parser.add_argument(
        "--local-mutual-knn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--supcon-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num-prototypes", type=int, default=5)
    parser.add_argument("--prototype-temperature", type=float, default=0.1)
    parser.add_argument("--prototype-warmup-epochs", type=int, default=15)
    parser.add_argument("--prototype-loss-weight", type=float, default=0.1)
    parser.add_argument("--prototype-diversity-weight", type=float, default=0.01)
    parser.add_argument("--prototype-diversity-margin", type=float, default=0.3)
    parser.add_argument("--fusion-alpha", type=float, default=0.75)
    parser.add_argument("--max-prototype-records-per-class", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--chunk-batch-size", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-groups-per-epoch", type=int, default=0)
    parser.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.model_type == "hproto":
        args.model_type = "hydroproto"
    if args.model_type == "hlocal":
        args.model_type = "hydrolocal"
    if args.unfreeze_layers < 1:
        raise ValueError("Final partial fine-tuning requires unfreeze_layers >= 1")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.model_type in {"hydrolocal", "hydrompc"}:
        if not args.reference_embeddings:
            raise ValueError("This model requires --reference-embeddings")
        if args.samples_per_class < 2 or args.samples_per_class % 2:
            raise ValueError("Local models require an even samples_per_class >= 2")
    if args.model_type in {"hydroproto", "hydrompc"}:
        if args.epochs <= args.prototype_warmup_epochs:
            raise ValueError("Final epochs must extend beyond prototype warmup")
    if not 0.0 <= args.fusion_alpha <= 1.0:
        raise ValueError("fusion_alpha must be in [0, 1]")

    seed_everything(args.seed)
    train_frame = load_sequence_csv(args.train_csv)
    test_frame = load_sequence_csv(args.test_csv)
    train_labels = encode_labels(train_frame["label"])
    test_labels = encode_labels(test_frame["label"])
    subset_paths = parse_subsets(args.subset)
    subset_indices, data_audit = audit_datasets(
        train_frame,
        test_frame,
        subset_paths,
    )
    expected_counts = {
        "test_all": 680,
        "lt80": 588,
        "lt60": 496,
        "lt40": 334,
    }
    observed_counts = {
        name: int(len(indices)) for name, indices in subset_indices.items()
    }
    for name, expected in expected_counts.items():
        if name in observed_counts and observed_counts[name] != expected:
            raise ValueError(
                f"Unexpected {name} count: {observed_counts[name]} != {expected}"
            )

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "model_type": args.model_type,
                    "unfreeze_layers": args.unfreeze_layers,
                    "fixed_epochs": args.epochs,
                    "epoch_selection_source": args.epoch_selection_source,
                    "class_counts": {
                        LABELS[index]: int((train_labels == index).sum())
                        for index in range(len(LABELS))
                    },
                    "observed_test_counts": observed_counts,
                    "data_audit": data_audit,
                    "test_metrics_used_for_training": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    output_dir.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config["subsets"] = {
        name: str(path.resolve()) for name, path in subset_paths.items()
    }
    config["train_records"] = int(len(train_frame))
    config["test_records"] = int(len(test_frame))
    config["test_metrics_used_for_training"] = False
    config["test_evaluated_only_after_final_epoch"] = True
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "data_audit.json", data_audit)

    device = torch.device(
        "cuda"
        if args.device == "cuda" and torch.cuda.is_available()
        else args.device
    )
    print(
        f"[start] model={args.model_type} train={len(train_frame)} "
        f"test={len(test_frame)} layers={args.unfreeze_layers} "
        f"fixed_epochs={args.epochs} device={device}",
        flush=True,
    )
    trainers = {
        "mlp": train_mlp,
        "hydroproto": train_hydroproto,
        "hydrolocal": train_hydrolocal,
        "hydrompc": train_hydrompc,
    }
    probabilities, chunk_counts, training_summary = trainers[args.model_type](
        train_frame,
        train_labels,
        test_frame,
        args,
        output_dir,
        device,
    )
    metrics = save_test_results(
        output_dir,
        test_frame,
        test_labels,
        probabilities,
        chunk_counts,
        subset_indices,
    )
    summary = {
        "status": "complete",
        "model_type": args.model_type,
        "unfreeze_layers": args.unfreeze_layers,
        "fixed_epochs": args.epochs,
        "epoch_selection_source": args.epoch_selection_source,
        "train_records": int(len(train_frame)),
        "test_records": int(len(test_frame)),
        "test_metrics_used_for_training": False,
        "test_evaluated_only_after_final_epoch": True,
        "training": training_summary,
        "final_test_metrics": {
            name: branch_metrics["final"]
            for name, branch_metrics in metrics.items()
        },
    }
    write_json(output_dir / "final_run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
