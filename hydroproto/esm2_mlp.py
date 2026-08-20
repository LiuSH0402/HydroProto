from __future__ import annotations

import argparse
import gc
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, EsmModel

from .data import LABELS, encode_labels, load_sequence_csv, seed_everything
from .metrics import classification_metrics


def split_sequence(sequence: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if len(sequence) <= chunk_size:
        return [sequence]
    step = chunk_size - overlap
    starts = list(range(0, len(sequence) - chunk_size + 1, step))
    last_start = len(sequence) - chunk_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return [sequence[start:start + chunk_size] for start in starts]


class SequenceMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def freeze_esm_layers(model: EsmModel, unfreeze_layers: int) -> int:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if unfreeze_layers <= 0:
        return 0
    layers = list(model.encoder.layer)
    if unfreeze_layers > len(layers):
        raise ValueError(f"unfreeze_layers={unfreeze_layers} exceeds {len(layers)} ESM layers")
    for layer in layers[-unfreeze_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    if hasattr(model.encoder, "emb_layer_norm_after"):
        for parameter in model.encoder.emb_layer_norm_after.parameters():
            parameter.requires_grad = True
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def balanced_groups(labels: np.ndarray, samples_per_class: int, seed: int, epoch: int):
    labels = np.asarray(labels, dtype=np.int64)
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    rng = np.random.default_rng(int(seed) + int(epoch))
    classes = np.unique(labels)
    pools = {cls: rng.permutation(np.flatnonzero(labels == cls)) for cls in classes}
    pointers = {cls: 0 for cls in classes}
    groups = int(np.ceil(max(len(indices) for indices in pools.values()) / samples_per_class))
    for _ in range(groups):
        group: list[int] = []
        for cls in classes:
            selected: list[int] = []
            while len(selected) < samples_per_class:
                pool = pools[cls]
                pointer = pointers[cls]
                if pointer == len(pool):
                    pool = rng.permutation(np.flatnonzero(labels == cls))
                    pointer = 0
                take = min(samples_per_class - len(selected), len(pool) - pointer)
                selected.extend(pool[pointer:pointer + take].tolist())
                pointer += take
                pools[cls] = pool
                pointers[cls] = pointer
            group.extend(selected)
        rng.shuffle(group)
        yield group


def token_chunks(tokenizer, chunks: list[str], device: torch.device):
    encoded = tokenizer(chunks, return_tensors="pt", padding=True, truncation=False)
    return {key: value.to(device) for key, value in encoded.items()}


def residue_mean(hidden: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    pooled = []
    for row, length in enumerate(lengths):
        # ESM tokenization is [BOS] + residues + [EOS].
        pooled.append(hidden[row, 1:1 + int(length)].mean(dim=0))
    return torch.stack(pooled, dim=0)


def chunk_embeddings(
    esm: EsmModel,
    tokenizer,
    sequence: str,
    chunk_size: int,
    overlap: int,
    chunk_batch_size: int,
    device: torch.device,
    amp: str,
    training: bool,
):
    chunks = split_sequence(sequence, chunk_size, overlap)
    pooled_chunks = []
    for start in range(0, len(chunks), chunk_batch_size):
        batch_chunks = chunks[start:start + chunk_batch_size]
        encoded = token_chunks(tokenizer, batch_chunks, device)
        with autocast_context(device, amp):
            output = esm(**encoded)
        pooled_chunks.append(residue_mean(output.last_hidden_state, [len(item) for item in batch_chunks]))
    return torch.cat(pooled_chunks, dim=0), len(chunks)


def predict_sequence(
    esm,
    classifier,
    tokenizer,
    sequence,
    args,
    device,
):
    embeddings, chunk_count = chunk_embeddings(
        esm,
        tokenizer,
        sequence,
        args.chunk_size,
        args.chunk_overlap,
        args.chunk_batch_size,
        device,
        args.amp,
        training=False,
    )
    with autocast_context(device, args.amp):
        logits = classifier(embeddings)
    return logits.float().mean(dim=0), chunk_count


def evaluate(esm, classifier, tokenizer, frame, labels, indices, args, device):
    esm.eval()
    classifier.eval()
    logits = []
    chunk_counts = []
    # no_grad is intentional. inference_mode can leave inference tensors in
    # ESM rotary caches and break the next backward pass after validation.
    with torch.no_grad():
        for index in indices:
            value, chunks = predict_sequence(esm, classifier, tokenizer, frame.iloc[int(index)]["sequence"], args, device)
            logits.append(value.cpu())
            chunk_counts.append(chunks)
    logits_np = torch.stack(logits).numpy()
    probabilities = F.softmax(torch.as_tensor(logits_np), dim=1).numpy()
    return probabilities, chunk_counts


def train_fold(frame, labels, train_idx, val_idx, args, fold_dir, device):
    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    esm = EsmModel.from_pretrained(
        str(model_dir),
        local_files_only=True,
        add_pooling_layer=False,
        low_cpu_mem_usage=True,
    ).to(device)
    trainable_esm = freeze_esm_layers(esm, args.unfreeze_layers)
    if args.gradient_checkpointing and args.unfreeze_layers > 0:
        # Non-reentrant checkpointing is required when the frozen lower layers
        # produce inputs without requires_grad but the upper ESM layers are trainable.
        esm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        esm.config.use_cache = False
    hidden_size = int(esm.config.hidden_size)
    classifier = SequenceMLP(hidden_size, args.hidden_dim, args.feature_dim, len(LABELS), args.dropout).to(device)
    esm_parameters = [p for p in esm.parameters() if p.requires_grad]
    optimizer_groups = [{"params": classifier.parameters(), "lr": args.head_learning_rate}]
    if esm_parameters:
        optimizer_groups.append({"params": esm_parameters, "lr": args.learning_rate})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    history = []
    best_macro_f1 = -1.0
    best_epoch = 0
    best_state = None
    patience_count = 0
    esm_gradient_verified = not bool(esm_parameters)

    print(
        f"[{fold_dir.name}] train={len(train_idx)} val={len(val_idx)} "
        f"unfreeze_layers={args.unfreeze_layers} "
        f"trainable_esm_parameters={trainable_esm:,}",
        flush=True,
    )

    fold_number = int(fold_dir.name.rsplit("_", maxsplit=1)[-1])
    sampling_seed = args.seed + fold_number
    for epoch in range(1, args.epochs + 1):
        esm.train(args.unfreeze_layers > 0)
        classifier.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        group_count = 0
        for group in balanced_groups(
            labels[train_idx], args.samples_per_class, sampling_seed, epoch
        ):
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for local_index in group:
                global_index = int(train_idx[local_index])
                sequence = frame.iloc[global_index]["sequence"]
                chunks = split_sequence(sequence, args.chunk_size, args.chunk_overlap)
                per_sequence_loss = 0.0
                for start in range(0, len(chunks), args.chunk_batch_size):
                    batch_chunks = chunks[start:start + args.chunk_batch_size]
                    encoded = token_chunks(tokenizer, batch_chunks, device)
                    if args.unfreeze_layers == 0:
                        with torch.no_grad():
                            with autocast_context(device, args.amp):
                                output = esm(**encoded)
                    else:
                        with autocast_context(device, args.amp):
                            output = esm(**encoded)
                    embedding = residue_mean(output.last_hidden_state, [len(item) for item in batch_chunks])
                    with autocast_context(device, args.amp):
                        chunk_logits = classifier(embedding)
                        target = torch.full((len(batch_chunks),), int(labels[global_index]), dtype=torch.long, device=device)
                        per_sequence_loss = per_sequence_loss + criterion(chunk_logits, target) / len(chunks)
                scaled_loss = per_sequence_loss / len(group)
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
                    esm_grad_norm = torch.linalg.vector_norm(torch.stack(gradients)).item()
                    if not np.isfinite(esm_grad_norm) or esm_grad_norm <= 0:
                        raise RuntimeError(
                            f"Invalid unfrozen ESM gradient norm: {esm_grad_norm}"
                        )
                    print(
                        f"[{fold_dir.name}] verified ESM gradient norm="
                        f"{esm_grad_norm:.6g}",
                        flush=True,
                    )
                    esm_gradient_verified = True
                group_loss += float(per_sequence_loss.detach().cpu())
            parameters = classifier.parameters()
            if esm_parameters:
                parameters = list(classifier.parameters()) + esm_parameters
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            total_loss += group_loss / len(group)
            group_count += 1

        probabilities, _ = evaluate(esm, classifier, tokenizer, frame, labels, val_idx, args, device)
        metrics = classification_metrics(labels[val_idx], probabilities)
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(group_count, 1),
            "val_macro_f1": metrics["macro_f1"],
            "val_mcc": metrics["mcc"],
        })
        print(
            f"[{fold_dir.name}] epoch={epoch:02d} "
            f"train_loss={history[-1]['train_loss']:.6f} "
            f"val_macro_f1={metrics['macro_f1']:.6f} "
            f"val_mcc={metrics['mcc']:.6f}",
            flush=True,
        )
        if metrics["macro_f1"] > best_macro_f1 + 1e-8:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            patience_count = 0
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in list(esm.named_parameters()) + list(classifier.named_parameters())
                if parameter.requires_grad
            }
            torch.save(best_state, fold_dir / "best_trainable_state.pt")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    current_state = dict(esm.named_parameters()) | dict(classifier.named_parameters())
    for name, value in best_state.items():
        current_state[name].data.copy_(value.to(device))
    probabilities, chunk_counts = evaluate(esm, classifier, tokenizer, frame, labels, val_idx, args, device)
    metrics = classification_metrics(labels[val_idx], probabilities)
    prediction_frame = pd.DataFrame({
        "accession": frame.iloc[val_idx]["accession"].to_numpy(),
        "true_label": [LABELS[index] for index in labels[val_idx]],
        "predicted_label": [LABELS[index] for index in probabilities.argmax(axis=1)],
        "correct": probabilities.argmax(axis=1) == labels[val_idx],
        "chunks": chunk_counts,
    })
    for class_index, label in enumerate(LABELS):
        prediction_frame[f"p_{label}"] = probabilities[:, class_index]
    prediction_frame.to_csv(fold_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(fold_dir / "split_indices.npz", train_idx=train_idx, val_idx=val_idx)
    (fold_dir / "validation_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "fold": int(fold_dir.name.split("_")[-1]),
        "train_records": int(len(train_idx)),
        "validation_records": int(len(val_idx)),
        "best_epoch": int(best_epoch),
        "best_macro_f1": float(best_macro_f1),
        "trainable_esm_parameters": int(trainable_esm),
        "device": str(device),
    }
    (fold_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    del esm, classifier, tokenizer, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, metrics


def preflight(args):
    frame = load_sequence_csv(args.train_csv)
    labels = encode_labels(frame["label"])
    config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    chunks = [split_sequence(sequence, args.chunk_size, args.chunk_overlap) for sequence in frame["sequence"].head(args.preflight_records)]
    print(json.dumps({
        "status": "PASS",
        "train_records": len(frame),
        "class_counts": {LABELS[index]: int((labels == index).sum()) for index in range(len(LABELS))},
        "model_dir": str(Path(args.model_dir).resolve()),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "tokenizer": tokenizer.__class__.__name__,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "preflight_records": args.preflight_records,
        "preflight_chunk_counts": [len(items) for items in chunks],
        "max_chunk_length": max(len(item) for items in chunks for item in items),
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Five-fold ESM-2 650M partial fine-tuning for hydrolase classification")
    parser.add_argument("--train-csv", default="random_split_similarity_audit/train_dataset.csv")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-root", default="modeling/finetune_650m")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-layers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--feature-dim", type=int, default=256)
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
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-records", type=int, default=20)
    args = parser.parse_args()
    seed_everything(args.seed)
    if args.preflight:
        preflight(args)
        return
    if args.folds < 2:
        raise ValueError("folds must be at least 2")
    frame = load_sequence_csv(args.train_csv)
    labels = encode_labels(frame["label"])
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.split_seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else args.device)
    output_root = Path(args.output_root) / f"unfreeze_top{args.unfreeze_layers}"
    output_root.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["external_test_used"] = False
    (output_root / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    fold_rows = []
    oof = []
    validation_counts = np.zeros(len(labels), dtype=np.int64)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        validation_counts[val_idx] += 1
        fold_dir = output_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"Starting fold {fold}/{args.folds}", flush=True)
        summary, metrics = train_fold(frame, labels, train_idx, val_idx, args, fold_dir, device)
        fold_rows.append({
            "fold": fold,
            "train_n": len(train_idx),
            "validation_n": len(val_idx),
            "best_epoch": summary["best_epoch"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "mcc": metrics["mcc"],
            **{f"f1_{label}": metrics["per_class"][label]["f1"] for label in LABELS},
        })
        fold_predictions = pd.read_csv(fold_dir / "validation_predictions.csv")
        fold_predictions.insert(0, "fold", fold)
        oof.append(fold_predictions)
    if not np.all(validation_counts == 1):
        raise RuntimeError("Each record must be validated exactly once")
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(output_root / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(oof, ignore_index=True).to_csv(output_root / "out_of_fold_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "model": "esm2_650m_mlp",
        "unfreeze_layers": args.unfreeze_layers,
        "folds": args.folds,
        "records": len(labels),
        "split_seed": args.split_seed,
        "external_test_used": False,
        "metrics": {
            name: {"mean": float(fold_metrics[name].mean()), "std": float(fold_metrics[name].std(ddof=1))}
            for name in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc"]
        },
    }
    (output_root / "cv_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
