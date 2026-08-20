from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import nn
from transformers import AutoTokenizer, EsmModel

from .esm2_mlp import (
    autocast_context,
    freeze_esm_layers,
    residue_mean,
    split_sequence,
    token_chunks,
)
from .data import (
    LABELS,
    LocalPositiveBatchSampler,
    build_local_positive_graph,
    encode_labels,
    load_sequence_csv,
    seed_everything,
)
from .losses import local_supervised_contrastive_loss
from .metrics import classification_metrics


class HydroLocalHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        feature_dim: int,
        projection_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.backbone(pooled)
        return {
            "logits": self.classifier(feature),
            "projection": self.projector(feature),
            "feature": feature,
        }


def encode_sequence(
    esm: EsmModel,
    tokenizer,
    sequence: str,
    chunk_size: int,
    overlap: int,
    chunk_batch_size: int,
    device: torch.device,
    amp: str,
) -> tuple[torch.Tensor, int]:
    chunks = split_sequence(sequence, chunk_size, overlap)
    pooled_chunks = []
    chunk_weights = []
    for start in range(0, len(chunks), chunk_batch_size):
        batch_chunks = chunks[start : start + chunk_batch_size]
        encoded = token_chunks(tokenizer, batch_chunks, device)
        with autocast_context(device, amp):
            output = esm(**encoded)
        pooled_chunks.append(
            residue_mean(
                output.last_hidden_state,
                [len(item) for item in batch_chunks],
            )
        )
        chunk_weights.extend(len(item) for item in batch_chunks)
    pooled = torch.cat(pooled_chunks, dim=0)
    weights = torch.as_tensor(
        chunk_weights,
        dtype=pooled.dtype,
        device=pooled.device,
    )
    sequence_embedding = (pooled * weights[:, None]).sum(dim=0) / weights.sum()
    return sequence_embedding, len(chunks)


def extract_reference_embeddings(args) -> None:
    output_path = Path(args.reference_embeddings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = load_sequence_csv(args.train_csv)
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else args.device
    )
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
    esm.eval()
    embeddings = []
    chunk_counts = []
    with torch.no_grad():
        for index, sequence in enumerate(frame["sequence"]):
            value, count = encode_sequence(
                esm,
                tokenizer,
                sequence,
                args.chunk_size,
                args.chunk_overlap,
                args.chunk_batch_size,
                device,
                args.amp,
            )
            embeddings.append(value.float().cpu().numpy())
            chunk_counts.append(count)
            if (index + 1) % 100 == 0 or index + 1 == len(frame):
                print(
                    f"[reference] encoded={index + 1}/{len(frame)}",
                    flush=True,
                )
    matrix = np.stack(embeddings).astype(np.float16)
    np.savez_compressed(
        output_path,
        embeddings=matrix,
        accessions=frame["accession"].astype(str).to_numpy(dtype=str),
        labels=frame["label"].astype(str).to_numpy(dtype=str),
        chunk_counts=np.asarray(chunk_counts, dtype=np.int16),
        model_dir=np.asarray(str(Path(args.model_dir).resolve())),
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output_path.resolve()),
                "records": int(len(frame)),
                "embedding_shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def load_reference_embeddings(path: str | Path, frame: pd.DataFrame) -> np.ndarray:
    with np.load(path, allow_pickle=False) as bundle:
        embeddings = bundle["embeddings"].astype(np.float32)
        accessions = bundle["accessions"].astype(str)
        labels = bundle["labels"].astype(str)
    if embeddings.shape[0] != len(frame):
        raise ValueError("Reference embedding count does not match training CSV")
    if accessions.tolist() != frame["accession"].astype(str).tolist():
        raise ValueError("Reference accession order does not match training CSV")
    if labels.tolist() != frame["label"].astype(str).tolist():
        raise ValueError("Reference labels do not match training CSV")
    if not np.isfinite(embeddings).all():
        raise ValueError("Reference embeddings contain non-finite values")
    return embeddings


def capture_rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu_state = torch.get_rng_state()
    cuda_state = None
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device)
    return cpu_state, cuda_state


def recompute_with_rng(
    esm: EsmModel,
    tokenizer,
    sequence: str,
    args,
    device: torch.device,
    rng_state: tuple[torch.Tensor, torch.Tensor | None],
) -> torch.Tensor:
    cpu_state, cuda_state = rng_state
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices, enabled=device.type == "cuda"):
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
        value, _ = encode_sequence(
            esm,
            tokenizer,
            sequence,
            args.chunk_size,
            args.chunk_overlap,
            args.chunk_batch_size,
            device,
            args.amp,
        )
    return value


def train_epoch_grad_cache(
    esm: EsmModel,
    head: HydroLocalHead,
    tokenizer,
    frame: pd.DataFrame,
    labels: np.ndarray,
    train_idx: np.ndarray,
    sampler: LocalPositiveBatchSampler,
    positive_graph: np.ndarray,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    args,
    device: torch.device,
    epoch: int,
    gradient_verified: bool,
) -> tuple[dict[str, float], bool]:
    esm.train(args.unfreeze_layers > 0)
    head.train()
    sampler.set_epoch(epoch)
    total_loss = 0.0
    total_ce = 0.0
    total_local = 0.0
    group_count = 0
    esm_parameters = [p for p in esm.parameters() if p.requires_grad]

    for group_number, local_indices in enumerate(sampler, start=1):
        if args.max_groups_per_epoch and group_number > args.max_groups_per_epoch:
            break
        optimizer.zero_grad(set_to_none=True)
        cached_embeddings = []
        rng_states = []
        global_indices = [int(train_idx[int(index)]) for index in local_indices]

        for global_index in global_indices:
            rng_state = capture_rng_state(device)
            with torch.no_grad():
                value, _ = encode_sequence(
                    esm,
                    tokenizer,
                    frame.iloc[global_index]["sequence"],
                    args.chunk_size,
                    args.chunk_overlap,
                    args.chunk_batch_size,
                    device,
                    args.amp,
                )
            cached_embeddings.append(value.detach())
            rng_states.append(rng_state)

        leaf_embeddings = (
            torch.stack(cached_embeddings)
            .float()
            .detach()
            .requires_grad_(True)
        )
        group_labels = torch.as_tensor(
            labels[global_indices],
            dtype=torch.long,
            device=device,
        )
        local_array = np.asarray(local_indices, dtype=np.int64)
        positive_mask = torch.as_tensor(
            positive_graph[np.ix_(local_array, local_array)],
            dtype=torch.bool,
            device=device,
        )
        with autocast_context(device, args.amp):
            output = head(leaf_embeddings)
            ce_loss = criterion(output["logits"], group_labels)
            local_loss = local_supervised_contrastive_loss(
                output["projection"],
                group_labels,
                positive_mask,
                temperature=args.temperature,
            )
            loss = ce_loss + args.supcon_weight * local_loss
        loss.backward()
        embedding_gradients = leaf_embeddings.grad.detach().clone()

        if esm_parameters:
            for row, (global_index, rng_state) in enumerate(
                zip(global_indices, rng_states)
            ):
                value = recompute_with_rng(
                    esm,
                    tokenizer,
                    frame.iloc[global_index]["sequence"],
                    args,
                    device,
                    rng_state,
                )
                torch.autograd.backward(
                    value,
                    grad_tensors=embedding_gradients[row].to(value.dtype),
                )

        if esm_parameters and not gradient_verified:
            gradients = [
                parameter.grad.detach().float().norm()
                for parameter in esm_parameters
                if parameter.grad is not None
            ]
            if not gradients:
                raise RuntimeError("No gradients reached the unfrozen ESM parameters")
            grad_norm = torch.linalg.vector_norm(torch.stack(gradients)).item()
            if not np.isfinite(grad_norm) or grad_norm <= 0:
                raise RuntimeError(f"Invalid unfrozen ESM gradient norm: {grad_norm}")
            print(
                f"[gradient-check] unfrozen_esm_norm={grad_norm:.6g}",
                flush=True,
            )
            gradient_verified = True

        parameters = list(head.parameters()) + esm_parameters
        torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_ce += float(ce_loss.detach().cpu())
        total_local += float(local_loss.detach().cpu())
        group_count += 1

    if group_count == 0:
        raise RuntimeError("No HydroLocal training groups were generated")
    return (
        {
            "train_loss": total_loss / group_count,
            "train_ce": total_ce / group_count,
            "train_local_supcon": total_local / group_count,
            "groups": int(group_count),
        },
        gradient_verified,
    )


def evaluate(
    esm: EsmModel,
    head: HydroLocalHead,
    tokenizer,
    frame: pd.DataFrame,
    indices: np.ndarray,
    args,
    device: torch.device,
) -> tuple[np.ndarray, list[int]]:
    esm.eval()
    head.eval()
    logits = []
    chunk_counts = []
    # no_grad is intentional: inference_mode can leave inference tensors in
    # the ESM rotary cache that cannot subsequently participate in autograd.
    with torch.no_grad():
        for index in indices:
            embedding, chunks = encode_sequence(
                esm,
                tokenizer,
                frame.iloc[int(index)]["sequence"],
                args.chunk_size,
                args.chunk_overlap,
                args.chunk_batch_size,
                device,
                args.amp,
            )
            with autocast_context(device, args.amp):
                value = head(embedding.unsqueeze(0))["logits"][0]
            logits.append(value.float().cpu())
            chunk_counts.append(chunks)
    logits_tensor = torch.stack(logits)
    return F.softmax(logits_tensor, dim=1).numpy(), chunk_counts


def save_trainable_state(
    esm: EsmModel,
    head: HydroLocalHead,
    path: Path,
) -> None:
    state = {
        **{
            f"esm.{name}": parameter.detach().cpu().clone()
            for name, parameter in esm.named_parameters()
            if parameter.requires_grad
        },
        **{
            f"head.{name}": parameter.detach().cpu().clone()
            for name, parameter in head.named_parameters()
        },
    }
    torch.save(state, path)


def load_trainable_state(
    esm: EsmModel,
    head: HydroLocalHead,
    path: Path,
    device: torch.device,
) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    esm_parameters = dict(esm.named_parameters())
    head_parameters = dict(head.named_parameters())
    for name, value in state.items():
        prefix, parameter_name = name.split(".", 1)
        target = esm_parameters if prefix == "esm" else head_parameters
        target[parameter_name].data.copy_(value.to(device))


def train_fold(
    frame: pd.DataFrame,
    labels: np.ndarray,
    reference_embeddings: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold: int,
    args,
    fold_dir: Path,
    device: torch.device,
) -> tuple[dict, dict]:
    seed_everything(args.seed + fold)
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
    trainable_esm = freeze_esm_layers(esm, args.unfreeze_layers)
    if args.gradient_checkpointing and args.unfreeze_layers > 0:
        esm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        esm.config.use_cache = False
    head = HydroLocalHead(
        input_dim=int(esm.config.hidden_size),
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        projection_dim=args.projection_dim,
        num_classes=len(LABELS),
        dropout=args.dropout,
    ).to(device)
    esm_parameters = [p for p in esm.parameters() if p.requires_grad]
    optimizer_groups = [{"params": head.parameters(), "lr": args.head_learning_rate}]
    if esm_parameters:
        optimizer_groups.append(
            {"params": esm_parameters, "lr": args.learning_rate}
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    positive_graph, graph_stats = build_local_positive_graph(
        reference_embeddings[train_idx],
        labels[train_idx],
        neighbors_per_sample=args.local_knn,
        mutual=args.local_mutual_knn,
    )
    sampler = LocalPositiveBatchSampler(
        labels[train_idx],
        positive_graph,
        samples_per_class=args.samples_per_class,
        seed=args.seed + fold,
    )
    np.savez_compressed(
        fold_dir / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
    )
    (fold_dir / "local_graph_stats.json").write_text(
        json.dumps(graph_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[fold_{fold}] train={len(train_idx)} val={len(val_idx)} "
        f"unfreeze_layers={args.unfreeze_layers} "
        f"trainable_esm_parameters={trainable_esm:,} "
        f"local_edges={graph_stats['undirected_edges']}",
        flush=True,
    )

    history = []
    best_macro_f1 = -1.0
    best_epoch = 0
    patience_count = 0
    gradient_verified = not bool(esm_parameters)
    checkpoint_path = fold_dir / "best_trainable_state.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics, gradient_verified = train_epoch_grad_cache(
            esm,
            head,
            tokenizer,
            frame,
            labels,
            train_idx,
            sampler,
            positive_graph,
            optimizer,
            criterion,
            args,
            device,
            epoch,
            gradient_verified,
        )
        evaluation_indices = val_idx
        if args.max_validation_records:
            evaluation_indices = val_idx[: args.max_validation_records]
        probabilities, _ = evaluate(
            esm,
            head,
            tokenizer,
            frame,
            evaluation_indices,
            args,
            device,
        )
        metrics = classification_metrics(
            labels[evaluation_indices],
            probabilities,
        )
        row = {
            "epoch": epoch,
            **train_metrics,
            "val_macro_f1": metrics["macro_f1"],
            "val_mcc": metrics["mcc"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            fold_dir / "history.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"[fold_{fold}] epoch={epoch:02d} "
            f"train_loss={row['train_loss']:.6f} "
            f"ce={row['train_ce']:.6f} "
            f"local={row['train_local_supcon']:.6f} "
            f"val_macro_f1={row['val_macro_f1']:.6f} "
            f"val_mcc={row['val_mcc']:.6f}",
            flush=True,
        )
        if metrics["macro_f1"] > best_macro_f1 + 1e-8:
            best_macro_f1 = float(metrics["macro_f1"])
            best_epoch = epoch
            patience_count = 0
            save_trainable_state(esm, head, checkpoint_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                break
        if args.smoke:
            break

    if args.smoke:
        summary = {
            "fold": fold,
            "smoke": True,
            "gradient_verified": gradient_verified,
            "gpu_memory_allocated_mb": (
                float(torch.cuda.max_memory_allocated(device) / 1024**2)
                if device.type == "cuda"
                else 0.0
            ),
        }
        (fold_dir / "smoke_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary, metrics

    load_trainable_state(esm, head, checkpoint_path, device)
    probabilities, chunk_counts = evaluate(
        esm,
        head,
        tokenizer,
        frame,
        val_idx,
        args,
        device,
    )
    metrics = classification_metrics(labels[val_idx], probabilities)
    prediction_frame = pd.DataFrame(
        {
            "accession": frame.iloc[val_idx]["accession"].to_numpy(),
            "true_label": [LABELS[index] for index in labels[val_idx]],
            "predicted_label": [
                LABELS[index] for index in probabilities.argmax(axis=1)
            ],
            "correct": probabilities.argmax(axis=1) == labels[val_idx],
            "chunks": chunk_counts,
        }
    )
    for class_index, label in enumerate(LABELS):
        prediction_frame[f"p_{label}"] = probabilities[:, class_index]
    prediction_frame.to_csv(
        fold_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (fold_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "fold": int(fold),
        "train_records": int(len(train_idx)),
        "validation_records": int(len(val_idx)),
        "best_epoch": int(best_epoch),
        "best_macro_f1": float(best_macro_f1),
        "trainable_esm_parameters": int(trainable_esm),
        "local_graph": graph_stats,
        "device": str(device),
    }
    (fold_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    del esm, head, tokenizer, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, metrics


def preflight(args) -> None:
    frame = load_sequence_csv(args.train_csv)
    labels = encode_labels(frame["label"])
    embeddings = load_reference_embeddings(args.reference_embeddings, frame)
    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.split_seed,
    )
    train_idx, val_idx = next(
        splitter.split(np.zeros(len(labels)), labels)
    )
    graph, stats = build_local_positive_graph(
        embeddings[train_idx],
        labels[train_idx],
        neighbors_per_sample=args.local_knn,
        mutual=args.local_mutual_knn,
    )
    sampler = LocalPositiveBatchSampler(
        labels[train_idx],
        graph,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    first_group = next(iter(sampler))
    print(
        json.dumps(
            {
                "status": "PASS",
                "records": int(len(frame)),
                "embedding_shape": list(embeddings.shape),
                "fold_train": int(len(train_idx)),
                "fold_validation": int(len(val_idx)),
                "local_graph": stats,
                "logical_batch_size": int(len(first_group)),
                "positive_edges_in_first_batch": int(
                    graph[np.ix_(first_group, first_group)].sum()
                ),
                "external_test_used": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Five-fold ESM-2 650M HydroLocal partial fine-tuning"
    )
    parser.add_argument(
        "--train-csv",
        default="random_split_similarity_audit/train_dataset.csv",
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--reference-embeddings",
        default="modeling/embeddings/train_esm2_650m_reference.npz",
    )
    parser.add_argument(
        "--output-root",
        default="modeling/finetune_650m_hydrolocal",
    )
    parser.add_argument("--extract-reference", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--local-knn", type=int, default=10)
    parser.add_argument(
        "--local-mutual-knn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--supcon-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--chunk-batch-size", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-groups-per-epoch", type=int, default=0)
    parser.add_argument("--max-validation-records", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    if args.extract_reference:
        extract_reference_embeddings(args)
        return
    if args.folds < 2:
        raise ValueError("folds must be at least 2")
    if args.samples_per_class < 2 or args.samples_per_class % 2:
        raise ValueError("samples_per_class must be an even integer >= 2")
    if not Path(args.reference_embeddings).exists():
        raise FileNotFoundError(
            f"Reference embeddings not found: {args.reference_embeddings}"
        )
    if args.preflight:
        preflight(args)
        return

    frame = load_sequence_csv(args.train_csv)
    labels = encode_labels(frame["label"])
    reference_embeddings = load_reference_embeddings(
        args.reference_embeddings,
        frame,
    )
    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.split_seed,
    )
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else args.device
    )
    output_root = (
        Path(args.output_root) / f"unfreeze_top{args.unfreeze_layers}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["model"] = "esm2_650m_hydrolocal"
    config["external_test_used"] = False
    (output_root / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fold_rows = []
    oof = []
    validation_counts = np.zeros(len(labels), dtype=np.int64)
    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels),
        start=1,
    ):
        validation_counts[val_idx] += 1
        fold_dir = output_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"Starting fold {fold}/{args.folds}", flush=True)
        summary, metrics = train_fold(
            frame,
            labels,
            reference_embeddings,
            train_idx,
            val_idx,
            fold,
            args,
            fold_dir,
            device,
        )
        if args.smoke:
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            return
        fold_rows.append(
            {
                "fold": fold,
                "train_n": len(train_idx),
                "validation_n": len(val_idx),
                "best_epoch": summary["best_epoch"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "mcc": metrics["mcc"],
                **{
                    f"f1_{label}": metrics["per_class"][label]["f1"]
                    for label in LABELS
                },
            }
        )
        fold_predictions = pd.read_csv(
            fold_dir / "validation_predictions.csv"
        )
        fold_predictions.insert(0, "fold", fold)
        oof.append(fold_predictions)
    if not np.all(validation_counts == 1):
        raise RuntimeError("Each record must be validated exactly once")
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(
        output_root / "fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(oof, ignore_index=True).to_csv(
        output_root / "out_of_fold_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "model": "esm2_650m_hydrolocal",
        "unfreeze_layers": args.unfreeze_layers,
        "folds": args.folds,
        "records": len(labels),
        "split_seed": args.split_seed,
        "local_knn": args.local_knn,
        "local_mutual_knn": args.local_mutual_knn,
        "supcon_weight": args.supcon_weight,
        "temperature": args.temperature,
        "external_test_used": False,
        "metrics": {
            name: {
                "mean": float(fold_metrics[name].mean()),
                "std": float(fold_metrics[name].std(ddof=1)),
            }
            for name in [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "mcc",
            ]
        },
    }
    (output_root / "cv_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
