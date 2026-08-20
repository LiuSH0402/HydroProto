from __future__ import annotations

import argparse
import gc
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from torch import nn
from transformers import AutoConfig, AutoTokenizer, EsmModel

from .esm2_mlp import (
    balanced_groups,
    freeze_esm_layers,
    residue_mean,
    split_sequence,
    token_chunks,
)
from .data import LABELS, encode_labels, load_sequence_csv, seed_everything
from .metrics import classification_metrics


class HydroProtoHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        feature_dim: int,
        projection_dim: int,
        num_classes: int,
        num_prototypes: int,
        prototype_temperature: float,
        dropout: float,
    ):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.projector = nn.Linear(feature_dim, projection_dim)
        self.num_classes = int(num_classes)
        self.num_prototypes = int(num_prototypes)
        self.prototype_temperature = float(prototype_temperature)
        initial = torch.randn(num_classes, num_prototypes, projection_dim)
        self.prototypes = nn.Parameter(F.normalize(initial, dim=2))
        self.register_buffer("prototypes_initialized", torch.tensor(False, dtype=torch.bool))

    def initialize_prototypes(self, centers: torch.Tensor) -> None:
        expected = tuple(self.prototypes.shape)
        if tuple(centers.shape) != expected:
            raise ValueError(f"Prototype shape {tuple(centers.shape)} != {expected}")
        with torch.no_grad():
            self.prototypes.copy_(F.normalize(centers.to(self.prototypes.device), dim=2))
            self.prototypes_initialized.fill_(True)

    def normalized_prototypes(self) -> torch.Tensor | None:
        if not bool(self.prototypes_initialized.item()):
            return None
        return F.normalize(self.prototypes, dim=2)

    def prototype_class_logits(self, projection: torch.Tensor) -> torch.Tensor | None:
        prototypes = self.normalized_prototypes()
        if prototypes is None:
            return None
        similarities = torch.einsum("bd,ckd->bck", projection, prototypes)
        return torch.logsumexp(
            similarities / self.prototype_temperature,
            dim=2,
        )

    def prototype_diversity_loss(self, margin: float) -> torch.Tensor:
        prototypes = self.normalized_prototypes()
        if prototypes is None or self.num_prototypes < 2:
            return self.classifier.weight.new_zeros(())
        similarities = torch.einsum("ckd,cmd->ckm", prototypes, prototypes)
        mask = ~torch.eye(
            self.num_prototypes,
            device=similarities.device,
            dtype=torch.bool,
        )
        off_diagonal = similarities[:, mask].reshape(self.num_classes, -1)
        return F.relu(off_diagonal - float(margin)).pow(2).mean()

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor | None]:
        features = self.adapter(embeddings)
        projection = F.normalize(self.projector(features), dim=1)
        return {
            "class_logits": self.classifier(features),
            "projection": projection,
            "prototype_logits": self.prototype_class_logits(projection),
        }


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def sequence_outputs(
    esm: EsmModel,
    head: HydroProtoHead,
    tokenizer,
    sequence: str,
    args,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, int]:
    chunks = split_sequence(sequence, args.chunk_size, args.chunk_overlap)
    class_logits = []
    prototype_logits = []
    projections = []
    for start in range(0, len(chunks), args.chunk_batch_size):
        batch_chunks = chunks[start:start + args.chunk_batch_size]
        encoded = token_chunks(tokenizer, batch_chunks, device)
        with autocast_context(device, args.amp):
            output = esm(**encoded)
            embeddings = residue_mean(
                output.last_hidden_state,
                [len(item) for item in batch_chunks],
            )
            values = head(embeddings)
        class_logits.append(values["class_logits"].float())
        projections.append(values["projection"].float())
        if values["prototype_logits"] is not None:
            prototype_logits.append(values["prototype_logits"].float())
    mean_class_logits = torch.cat(class_logits, dim=0).mean(dim=0)
    mean_projection = F.normalize(
        torch.cat(projections, dim=0).mean(dim=0),
        dim=0,
    )
    mean_prototype_logits = None
    if prototype_logits:
        mean_prototype_logits = torch.cat(prototype_logits, dim=0).mean(dim=0)
    return mean_class_logits, mean_prototype_logits, mean_projection, len(chunks)


def probabilities_from_logits(
    class_logits: np.ndarray,
    prototype_logits: np.ndarray | None,
    fusion_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    class_probabilities = F.softmax(
        torch.as_tensor(class_logits),
        dim=1,
    ).numpy()
    if prototype_logits is None:
        return class_probabilities, class_probabilities, None
    prototype_probabilities = F.softmax(
        torch.as_tensor(prototype_logits),
        dim=1,
    ).numpy()
    final_probabilities = (
        float(fusion_alpha) * class_probabilities
        + (1.0 - float(fusion_alpha)) * prototype_probabilities
    )
    return final_probabilities, class_probabilities, prototype_probabilities


def evaluate(
    esm: EsmModel,
    head: HydroProtoHead,
    tokenizer,
    frame: pd.DataFrame,
    indices: np.ndarray,
    args,
    device: torch.device,
):
    esm.eval()
    head.eval()
    class_logits = []
    prototype_logits = []
    projections = []
    chunk_counts = []
    # no_grad is intentional. inference_mode can leave inference tensors in
    # ESM rotary caches and break the next backward pass.
    with torch.no_grad():
        for index in indices:
            cls, proto, projection, chunks = sequence_outputs(
                esm,
                head,
                tokenizer,
                frame.iloc[int(index)]["sequence"],
                args,
                device,
            )
            class_logits.append(cls.cpu())
            projections.append(projection.cpu())
            chunk_counts.append(chunks)
            if proto is not None:
                prototype_logits.append(proto.cpu())
    class_logits_np = torch.stack(class_logits).numpy()
    prototype_logits_np = None
    if prototype_logits:
        prototype_logits_np = torch.stack(prototype_logits).numpy()
    final_p, class_p, prototype_p = probabilities_from_logits(
        class_logits_np,
        prototype_logits_np,
        args.fusion_alpha,
    )
    return {
        "final_probabilities": final_p,
        "class_probabilities": class_p,
        "prototype_probabilities": prototype_p,
        "projections": torch.stack(projections).numpy(),
        "chunk_counts": chunk_counts,
    }


def initialize_online_prototypes(
    esm: EsmModel,
    head: HydroProtoHead,
    tokenizer,
    frame: pd.DataFrame,
    labels: np.ndarray,
    train_idx: np.ndarray,
    args,
    fold_number: int,
    fold_dir: Path,
    device: torch.device,
) -> None:
    outputs = evaluate(
        esm,
        head,
        tokenizer,
        frame,
        train_idx,
        args,
        device,
    )
    projections = outputs["projections"]
    centers = []
    for class_index in range(len(LABELS)):
        class_values = projections[labels[train_idx] == class_index]
        if len(class_values) < args.num_prototypes:
            raise ValueError(
                f"Class {LABELS[class_index]} has only {len(class_values)} "
                f"samples for {args.num_prototypes} prototypes"
            )
        kmeans = KMeans(
            n_clusters=args.num_prototypes,
            random_state=args.seed + fold_number,
            n_init=20,
        )
        class_centers = kmeans.fit(class_values).cluster_centers_.astype(np.float32)
        class_centers /= np.clip(
            np.linalg.norm(class_centers, axis=1, keepdims=True),
            1e-12,
            None,
        )
        centers.append(class_centers)
    centers_np = np.stack(centers)
    head.initialize_prototypes(
        torch.as_tensor(centers_np, dtype=torch.float32, device=device)
    )
    arrays = {
        f"class_{index}_{LABELS[index]}": centers_np[index]
        for index in range(len(LABELS))
    }
    np.savez_compressed(fold_dir / "initial_prototypes.npz", **arrays)
    print(
        f"[{fold_dir.name}] initialized "
        f"{len(LABELS)}x{args.num_prototypes} online prototypes",
        flush=True,
    )


def train_epoch(
    esm: EsmModel,
    head: HydroProtoHead,
    tokenizer,
    frame: pd.DataFrame,
    labels: np.ndarray,
    train_idx: np.ndarray,
    optimizer,
    criterion,
    esm_parameters: list[torch.Tensor],
    args,
    fold_number: int,
    epoch: int,
    device: torch.device,
    esm_gradient_verified: bool,
) -> tuple[dict[str, float], bool]:
    esm.train(True)
    head.train()
    loss_sum = 0.0
    class_ce_sum = 0.0
    prototype_ce_sum = 0.0
    diversity_sum = 0.0
    sequences_seen = 0
    group_count = 0
    local_labels = labels[train_idx]
    for group in balanced_groups(
        local_labels,
        args.samples_per_class,
        args.seed + fold_number * 1000,
        epoch,
    ):
        optimizer.zero_grad(set_to_none=True)
        group_loss_value = 0.0
        for local_index in group:
            global_index = int(train_idx[local_index])
            sequence = frame.iloc[global_index]["sequence"]
            chunks = split_sequence(
                sequence,
                args.chunk_size,
                args.chunk_overlap,
            )
            sequence_loss = torch.zeros((), device=device)
            sequence_class_ce = 0.0
            sequence_prototype_ce = 0.0
            for start in range(0, len(chunks), args.chunk_batch_size):
                batch_chunks = chunks[start:start + args.chunk_batch_size]
                encoded = token_chunks(tokenizer, batch_chunks, device)
                with autocast_context(device, args.amp):
                    output = esm(**encoded)
                    embeddings = residue_mean(
                        output.last_hidden_state,
                        [len(item) for item in batch_chunks],
                    )
                    values = head(embeddings)
                    target = torch.full(
                        (len(batch_chunks),),
                        int(labels[global_index]),
                        dtype=torch.long,
                        device=device,
                    )
                    class_ce = criterion(values["class_logits"], target)
                    chunk_weight = len(batch_chunks) / len(chunks)
                    sequence_loss = sequence_loss + chunk_weight * class_ce
                    sequence_class_ce += (
                        chunk_weight * float(class_ce.detach().cpu())
                    )
                    if values["prototype_logits"] is not None:
                        prototype_ce = criterion(
                            values["prototype_logits"],
                            target,
                        )
                        sequence_loss = (
                            sequence_loss
                            + chunk_weight
                            * args.prototype_loss_weight
                            * prototype_ce
                        )
                        sequence_prototype_ce += (
                            chunk_weight * float(prototype_ce.detach().cpu())
                        )
            diversity = head.prototype_diversity_loss(
                args.prototype_diversity_margin
            )
            if bool(head.prototypes_initialized.item()):
                sequence_loss = (
                    sequence_loss
                    + args.prototype_diversity_weight * diversity
                )
            scaled_loss = sequence_loss / len(group)
            scaled_loss.backward()
            if not esm_gradient_verified:
                gradients = [
                    parameter.grad.detach().float().norm()
                    for parameter in esm_parameters
                    if parameter.grad is not None
                ]
                if not gradients:
                    raise RuntimeError(
                        "No gradients reached the unfrozen ESM parameters"
                    )
                esm_grad_norm = torch.linalg.vector_norm(
                    torch.stack(gradients)
                ).item()
                if not np.isfinite(esm_grad_norm) or esm_grad_norm <= 0:
                    raise RuntimeError(
                        f"Invalid unfrozen ESM gradient norm: {esm_grad_norm}"
                    )
                print(
                    f"verified ESM gradient norm={esm_grad_norm:.6g}",
                    flush=True,
                )
                esm_gradient_verified = True
            group_loss_value += float(sequence_loss.detach().cpu())
            class_ce_sum += sequence_class_ce
            prototype_ce_sum += sequence_prototype_ce
            diversity_sum += float(diversity.detach().cpu())
            sequences_seen += 1
        parameters = list(head.parameters()) + esm_parameters
        torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
        optimizer.step()
        loss_sum += group_loss_value
        group_count += 1
    return {
        "train_loss": loss_sum / max(sequences_seen, 1),
        "train_class_ce": class_ce_sum / max(sequences_seen, 1),
        "train_prototype_ce": prototype_ce_sum / max(sequences_seen, 1),
        "train_prototype_diversity": diversity_sum / max(sequences_seen, 1),
        "groups": int(group_count),
    }, esm_gradient_verified


def save_best_state(
    path: Path,
    esm: EsmModel,
    head: HydroProtoHead,
    epoch: int,
    metrics: dict,
) -> None:
    state = {
        "epoch": int(epoch),
        "metrics": metrics,
        "esm_trainable_state": {
            name: parameter.detach().cpu().clone()
            for name, parameter in esm.named_parameters()
            if parameter.requires_grad
        },
        "head_state": {
            name: value.detach().cpu().clone()
            for name, value in head.state_dict().items()
        },
    }
    torch.save(state, path)


def load_best_state(
    path: Path,
    esm: EsmModel,
    head: HydroProtoHead,
    device: torch.device,
) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    esm_parameters = dict(esm.named_parameters())
    for name, value in state["esm_trainable_state"].items():
        esm_parameters[name].data.copy_(value.to(device))
    head.load_state_dict(
        {name: value.to(device) for name, value in state["head_state"].items()}
    )
    return state


def save_prediction_frame(
    path: Path,
    frame: pd.DataFrame,
    labels: np.ndarray,
    indices: np.ndarray,
    outputs: dict,
) -> None:
    final_p = outputs["final_probabilities"]
    class_p = outputs["class_probabilities"]
    prototype_p = outputs["prototype_probabilities"]
    predicted = final_p.argmax(axis=1)
    prediction_frame = pd.DataFrame({
        "accession": frame.iloc[indices]["accession"].to_numpy(),
        "true_label": [LABELS[index] for index in labels[indices]],
        "predicted_label": [LABELS[index] for index in predicted],
        "correct": predicted == labels[indices],
        "chunks": outputs["chunk_counts"],
    })
    for class_index, label in enumerate(LABELS):
        prediction_frame[f"p_final_{label}"] = final_p[:, class_index]
        prediction_frame[f"p_cls_{label}"] = class_p[:, class_index]
        if prototype_p is not None:
            prediction_frame[f"p_proto_{label}"] = prototype_p[:, class_index]
    prediction_frame.to_csv(path, index=False, encoding="utf-8-sig")


def save_prototypes(path: Path, head: HydroProtoHead) -> None:
    prototypes = head.normalized_prototypes()
    if prototypes is None:
        return
    values = prototypes.detach().cpu().numpy()
    arrays = {
        f"class_{index}_{LABELS[index]}": values[index]
        for index in range(len(LABELS))
    }
    np.savez_compressed(path, **arrays)


def train_fold(
    frame: pd.DataFrame,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args,
    fold_number: int,
    fold_dir: Path,
    device: torch.device,
):
    seed_everything(args.seed + fold_number)
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
    if args.gradient_checkpointing:
        esm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        esm.config.use_cache = False
    head = HydroProtoHead(
        input_dim=int(esm.config.hidden_size),
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        projection_dim=args.projection_dim,
        num_classes=len(LABELS),
        num_prototypes=args.num_prototypes,
        prototype_temperature=args.prototype_temperature,
        dropout=args.dropout,
    ).to(device)
    esm_parameters = [
        parameter for parameter in esm.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": head.parameters(), "lr": args.head_learning_rate},
            {"params": esm_parameters, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    history = []
    best_macro_f1 = -1.0
    best_epoch = 0
    patience_count = 0
    esm_gradient_verified = False
    np.savez_compressed(
        fold_dir / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
    )
    print(
        f"[fold_{fold_number}] train={len(train_idx)} val={len(val_idx)} "
        f"unfreeze_layers={args.unfreeze_layers} "
        f"trainable_esm_parameters={trainable_esm:,}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        if (
            not bool(head.prototypes_initialized.item())
            and epoch > args.prototype_warmup_epochs
        ):
            initialize_online_prototypes(
                esm,
                head,
                tokenizer,
                frame,
                labels,
                train_idx,
                args,
                fold_number,
                fold_dir,
                device,
            )
        record, esm_gradient_verified = train_epoch(
            esm,
            head,
            tokenizer,
            frame,
            labels,
            train_idx,
            optimizer,
            criterion,
            esm_parameters,
            args,
            fold_number,
            epoch,
            device,
            esm_gradient_verified,
        )
        outputs = evaluate(
            esm,
            head,
            tokenizer,
            frame,
            val_idx,
            args,
            device,
        )
        metrics = classification_metrics(
            labels[val_idx],
            outputs["final_probabilities"],
        )
        phase = (
            "prototype"
            if bool(head.prototypes_initialized.item())
            else "warmup"
        )
        record.update({
            "epoch": epoch,
            "phase": phase,
            "val_macro_f1": metrics["macro_f1"],
            "val_mcc": metrics["mcc"],
        })
        history.append(record)
        pd.DataFrame(history).to_csv(
            fold_dir / "history.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"[fold_{fold_number}] epoch={epoch:02d} phase={phase} "
            f"train_loss={record['train_loss']:.6f} "
            f"val_macro_f1={metrics['macro_f1']:.6f} "
            f"val_mcc={metrics['mcc']:.6f}",
            flush=True,
        )
        checkpoint_eligible = bool(head.prototypes_initialized.item())
        if (
            checkpoint_eligible
            and metrics["macro_f1"] > best_macro_f1 + 1e-8
        ):
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            patience_count = 0
            save_best_state(
                fold_dir / "best_trainable_state.pt",
                esm,
                head,
                epoch,
                metrics,
            )
        elif checkpoint_eligible:
            patience_count += 1
            if patience_count >= args.patience:
                break

    checkpoint_path = fold_dir / "best_trainable_state.pt"
    if not checkpoint_path.exists():
        raise RuntimeError("No prototype-phase checkpoint was selected")
    best_state = load_best_state(checkpoint_path, esm, head, device)
    outputs = evaluate(
        esm,
        head,
        tokenizer,
        frame,
        val_idx,
        args,
        device,
    )
    metrics = classification_metrics(
        labels[val_idx],
        outputs["final_probabilities"],
    )
    save_prediction_frame(
        fold_dir / "validation_predictions.csv",
        frame,
        labels,
        val_idx,
        outputs,
    )
    save_prototypes(fold_dir / "prototypes.npz", head)
    (fold_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "fold": fold_number,
        "train_records": int(len(train_idx)),
        "validation_records": int(len(val_idx)),
        "best_epoch": int(best_state["epoch"]),
        "best_macro_f1": float(metrics["macro_f1"]),
        "trainable_esm_parameters": int(trainable_esm),
        "num_prototypes_per_class": int(args.num_prototypes),
        "fusion_alpha": float(args.fusion_alpha),
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
    config = AutoConfig.from_pretrained(
        args.model_dir,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
    )
    chunks = [
        split_sequence(sequence, args.chunk_size, args.chunk_overlap)
        for sequence in frame["sequence"].head(args.preflight_records)
    ]
    print(json.dumps({
        "status": "PASS",
        "model": "esm2_650m_hydroproto",
        "train_records": len(frame),
        "class_counts": {
            LABELS[index]: int((labels == index).sum())
            for index in range(len(LABELS))
        },
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "tokenizer": tokenizer.__class__.__name__,
        "unfreeze_layers": args.unfreeze_layers,
        "num_prototypes": args.num_prototypes,
        "prototype_warmup_epochs": args.prototype_warmup_epochs,
        "prototype_loss_weight": args.prototype_loss_weight,
        "prototype_diversity_weight": args.prototype_diversity_weight,
        "fusion_alpha": args.fusion_alpha,
        "preflight_chunk_counts": [len(items) for items in chunks],
        "external_test_used": False,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Five-fold ESM-2 650M partial fine-tuning with online "
            "prototype-guided HydroProto"
        )
    )
    parser.add_argument(
        "--train-csv",
        default="random_split_similarity_audit/train_dataset.csv",
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--output-root",
        default="modeling/finetune_650m_hydroproto",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-layers", type=int, required=True)
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
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--chunk-batch-size", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-prototypes", type=int, default=5)
    parser.add_argument("--prototype-warmup-epochs", type=int, default=15)
    parser.add_argument("--prototype-loss-weight", type=float, default=0.10)
    parser.add_argument(
        "--prototype-diversity-weight",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--prototype-diversity-margin",
        type=float,
        default=0.30,
    )
    parser.add_argument("--prototype-temperature", type=float, default=0.10)
    parser.add_argument(
        "--fusion-alpha",
        type=float,
        default=0.0,
        help="0 uses the HydroProto branch only; 1 uses the MLP branch only",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-records", type=int, default=20)
    args = parser.parse_args()

    if args.unfreeze_layers < 1:
        raise ValueError("HydroProto fine-tuning requires at least one unfrozen layer")
    if args.prototype_warmup_epochs >= args.epochs:
        raise ValueError("prototype_warmup_epochs must be smaller than epochs")
    if not 0.0 <= args.fusion_alpha <= 1.0:
        raise ValueError("fusion_alpha must be between 0 and 1")
    seed_everything(args.seed)
    if args.preflight:
        preflight(args)
        return

    frame = load_sequence_csv(args.train_csv)
    labels = encode_labels(frame["label"])
    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.split_seed,
    )
    device = torch.device(
        "cuda"
        if args.device == "cuda" and torch.cuda.is_available()
        else args.device
    )
    output_root = (
        Path(args.output_root) / f"unfreeze_top{args.unfreeze_layers}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["model"] = "esm2_650m_hydroproto"
    config["external_test_used"] = False
    (output_root / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fold_rows = []
    oof = []
    validation_counts = np.zeros(len(labels), dtype=np.int64)
    for fold_number, (train_idx, val_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels),
        start=1,
    ):
        validation_counts[val_idx] += 1
        fold_dir = output_root / f"fold_{fold_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"Starting fold {fold_number}/{args.folds}", flush=True)
        summary, metrics = train_fold(
            frame,
            labels,
            train_idx,
            val_idx,
            args,
            fold_number,
            fold_dir,
            device,
        )
        fold_rows.append({
            "fold": fold_number,
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
        })
        fold_predictions = pd.read_csv(
            fold_dir / "validation_predictions.csv"
        )
        fold_predictions.insert(0, "fold", fold_number)
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
        "model": "esm2_650m_hydroproto",
        "unfreeze_layers": args.unfreeze_layers,
        "num_prototypes_per_class": args.num_prototypes,
        "prototype_warmup_epochs": args.prototype_warmup_epochs,
        "prototype_loss_weight": args.prototype_loss_weight,
        "prototype_diversity_weight": args.prototype_diversity_weight,
        "prototype_temperature": args.prototype_temperature,
        "fusion_alpha": args.fusion_alpha,
        "folds": args.folds,
        "records": len(labels),
        "split_seed": args.split_seed,
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
