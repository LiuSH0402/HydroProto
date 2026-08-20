from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from .data import (
    LABELS,
    BalancedBatchSampler,
    EmbeddingDataset,
    LocalPositiveBatchSampler,
    build_local_positive_graph,
    encode_labels,
    load_embedding_bundle,
    load_sequence_csv,
    make_dev_split,
    seed_everything,
    verify_bundle_matches_csv,
)
from .losses import local_supervised_contrastive_loss, supervised_contrastive_loss
from .metrics import classification_metrics
from .models import build_model
from .prototypes import fit_class_prototypes, prototype_logits, softmax_numpy


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "run_name", "model", "train_embeddings", "test_embeddings", "train_csv", "test_csv",
        "seed", "val_fraction", "epochs", "patience", "batch_samples_per_class",
        "learning_rate", "weight_decay", "hidden_dim", "feature_dim", "projection_dim",
        "supcon_weight", "temperature", "num_prototypes", "fusion_alpha", "prototype_temperature",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration is missing fields: {sorted(missing)}")
    return config


def torch_softmax(logits: np.ndarray) -> np.ndarray:
    return softmax_numpy(np.asarray(logits, dtype=np.float64))


def model_outputs(
    model: nn.Module,
    embeddings: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
    include_online_prototypes: bool = False,
):
    logits_list = []
    projections = []
    prototype_logits_list = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(embeddings), batch_size):
            batch = torch.as_tensor(embeddings[start:start + batch_size], dtype=torch.float32, device=device)
            output = model(batch)
            if output["logits"] is not None:
                logits_list.append(output["logits"].cpu().numpy())
            if output["projection"] is not None:
                projections.append(output["projection"].cpu().numpy())
            if output.get("prototype_logits") is not None:
                prototype_logits_list.append(output["prototype_logits"].cpu().numpy())
    logits = np.concatenate(logits_list) if logits_list else None
    projection = np.concatenate(projections) if projections else None
    prototype_logits_values = np.concatenate(prototype_logits_list) if prototype_logits_list else None
    if include_online_prototypes:
        return logits, projection, prototype_logits_values
    return logits, projection


def probabilities_for_model(
    model_kind: str,
    eval_logits: np.ndarray | None,
    eval_projection: np.ndarray | None,
    fit_projection: np.ndarray | None,
    fit_labels: np.ndarray,
    config: dict,
    eval_online_prototype_logits: np.ndarray | None = None,
):
    class_probabilities = torch_softmax(eval_logits) if eval_logits is not None else None
    prototype_probabilities = None
    prototypes = None
    if model_kind in {"clean", "hydrompc"} and eval_projection is not None:
        prototypes = fit_class_prototypes(
            fit_projection,
            fit_labels,
            num_classes=len(LABELS),
            num_prototypes=int(config["num_prototypes"]),
            seed=int(config["seed"]),
        )
        proto_logits = prototype_logits(
            eval_projection,
            prototypes,
            temperature=float(config["prototype_temperature"]),
        )
        prototype_probabilities = softmax_numpy(proto_logits)
    elif model_kind == "hydrompc_v2" and eval_online_prototype_logits is not None:
        prototype_probabilities = softmax_numpy(eval_online_prototype_logits)
    if model_kind in {"linear", "mlp"}:
        final_probabilities = class_probabilities
    elif model_kind == "clean":
        final_probabilities = prototype_probabilities
    elif model_kind == "hydrompc":
        alpha = float(config["fusion_alpha"])
        final_probabilities = alpha * class_probabilities + (1.0 - alpha) * prototype_probabilities
    elif model_kind == "hydrompc_v2":
        if prototype_probabilities is None:
            final_probabilities = class_probabilities
        else:
            alpha = float(config["fusion_alpha"])
            final_probabilities = alpha * class_probabilities + (1.0 - alpha) * prototype_probabilities
    else:
        raise ValueError(model_kind)
    return final_probabilities, class_probabilities, prototype_probabilities, prototypes


def train_one_epoch(model, loader, optimizer, ce_loss, config, device, local_positive_graph=None):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_supcon = 0.0
    total_prototype_ce = 0.0
    total_prototype_diversity = 0.0
    batches = 0
    for batch in loader:
        if len(batch) == 3:
            features, labels, batch_indices = batch
        else:
            features, labels = batch
            batch_indices = None
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(features)
        loss = torch.zeros((), device=device)
        ce_value = torch.zeros((), device=device)
        supcon_value = torch.zeros((), device=device)
        prototype_ce_value = torch.zeros((), device=device)
        prototype_diversity_value = torch.zeros((), device=device)
        if output["logits"] is not None:
            ce_value = ce_loss(output["logits"], labels)
            loss = loss + ce_value
        if output["projection"] is not None:
            if bool(config.get("local_supcon", False)):
                if local_positive_graph is None or batch_indices is None:
                    raise ValueError("Local contrastive training requires graph and dataset indices")
                index_array = np.asarray(batch_indices, dtype=np.int64)
                positive_mask = torch.as_tensor(
                    local_positive_graph[np.ix_(index_array, index_array)],
                    dtype=torch.bool,
                    device=device,
                )
                supcon_value = local_supervised_contrastive_loss(
                    output["projection"], labels, positive_mask, temperature=float(config["temperature"])
                )
            elif float(config["supcon_weight"]) > 0:
                supcon_value = supervised_contrastive_loss(
                    output["projection"], labels, temperature=float(config["temperature"])
                )
            loss = loss + float(config["supcon_weight"]) * supcon_value
        if output.get("prototype_logits") is not None:
            prototype_ce_value = ce_loss(output["prototype_logits"], labels)
            prototype_diversity_value = model.prototype_diversity_loss(
                margin=float(config.get("prototype_diversity_margin", 0.30))
            )
            loss = loss + float(config.get("prototype_loss_weight", 0.0)) * prototype_ce_value
            loss = loss + float(config.get("prototype_diversity_weight", 0.0)) * prototype_diversity_value
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_ce += float(ce_value.detach().cpu())
        total_supcon += float(supcon_value.detach().cpu())
        total_prototype_ce += float(prototype_ce_value.detach().cpu())
        total_prototype_diversity += float(prototype_diversity_value.detach().cpu())
        batches += 1
    return {
        "train_loss": total_loss / batches,
        "train_ce": total_ce / batches,
        "train_supcon": total_supcon / batches,
        "train_prototype_ce": total_prototype_ce / batches,
        "train_prototype_diversity": total_prototype_diversity / batches,
    }


def make_prediction_frame(accessions, true_labels, final_p, cls_p=None, proto_p=None):
    pred_idx = final_p.argmax(axis=1)
    frame = pd.DataFrame({
        "accession": accessions,
        "true_label": [LABELS[i] for i in true_labels],
        "predicted_label": [LABELS[i] for i in pred_idx],
        "correct": pred_idx == true_labels,
        "max_probability": final_p.max(axis=1),
    })
    for i, label in enumerate(LABELS):
        frame[f"p_final_{label}"] = final_p[:, i]
        if cls_p is not None:
            frame[f"p_cls_{label}"] = cls_p[:, i]
        if proto_p is not None:
            frame[f"p_proto_{label}"] = proto_p[:, i]
    return frame


def save_prototypes(path: Path, prototypes: list[np.ndarray] | None) -> None:
    if prototypes is None:
        return
    arrays = {f"class_{i}_{LABELS[i]}": centers for i, centers in enumerate(prototypes)}
    np.savez_compressed(path, **arrays)


def initialize_online_prototypes(model, embeddings, labels, config, device) -> list[np.ndarray]:
    _, projection = model_outputs(model, embeddings, device)
    centers = fit_class_prototypes(
        projection,
        labels,
        num_classes=len(LABELS),
        num_prototypes=int(config["num_prototypes"]),
        seed=int(config["seed"]),
    )
    model.initialize_prototypes(torch.as_tensor(np.stack(centers), dtype=torch.float32, device=device))
    return centers


def online_prototypes_as_numpy(model) -> list[np.ndarray] | None:
    if not hasattr(model, "normalized_prototypes"):
        return None
    prototypes = model.normalized_prototypes()
    if prototypes is None:
        return None
    values = prototypes.detach().cpu().numpy()
    return [values[index] for index in range(values.shape[0])]


def prepare_run_dir(config: dict, stage: str, requested: str | None, overwrite: bool) -> Path:
    run_dir = Path(requested) if requested else Path("modeling/runs") / config["run_name"] / f"{stage}_seed{config['seed']}"
    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Run directory exists and is not empty: {run_dir}; pass --overwrite explicitly")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train_dev(
    config: dict,
    run_dir: Path,
    device: torch.device,
    split_override: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    bundle = load_embedding_bundle(config["train_embeddings"])
    frame = load_sequence_csv(config["train_csv"])
    verify_bundle_matches_csv(bundle, frame, config["train_csv"])
    labels = encode_labels(bundle.labels)
    if split_override is None:
        train_idx, val_idx = make_dev_split(labels, float(config["val_fraction"]), int(config["seed"]))
    else:
        train_idx, val_idx = split_override
        train_idx = np.sort(np.asarray(train_idx, dtype=np.int64))
        val_idx = np.sort(np.asarray(val_idx, dtype=np.int64))
        if len(set(train_idx) & set(val_idx)):
            raise ValueError("Cross-validation training and validation indices overlap")
        if len(train_idx) + len(val_idx) != len(labels):
            raise ValueError("Cross-validation indices do not cover the full training data")
    np.savez_compressed(run_dir / "split_indices.npz", train_idx=train_idx, val_idx=val_idx)

    local_supcon = bool(config.get("local_supcon", False))
    local_positive_graph = None
    if local_supcon:
        local_positive_graph, graph_stats = build_local_positive_graph(
            bundle.embeddings[train_idx],
            labels[train_idx],
            neighbors_per_sample=int(config.get("local_knn", 10)),
            mutual=bool(config.get("local_mutual_knn", True)),
        )
        with (run_dir / "local_graph_stats.json").open("w", encoding="utf-8") as handle:
            json.dump(graph_stats, handle, ensure_ascii=False, indent=2)
    train_data = EmbeddingDataset(bundle.embeddings[train_idx], labels[train_idx], return_index=local_supcon)
    if local_supcon:
        sampler = LocalPositiveBatchSampler(
            labels[train_idx],
            local_positive_graph,
            int(config["batch_samples_per_class"]),
            int(config["seed"]),
        )
    else:
        sampler = BalancedBatchSampler(
            labels[train_idx], int(config["batch_samples_per_class"]), int(config["seed"])
        )
    loader = DataLoader(train_data, batch_sampler=sampler, num_workers=0)
    model = build_model(config, bundle.embeddings.shape[1], len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    ce_loss = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))
    history = []
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    online_prototypes = bool(config.get("online_prototypes", False))
    prototype_warmup_epochs = int(config.get("prototype_warmup_epochs", 0))

    progress = trange(1, int(config["epochs"]) + 1, desc=f"train {config['run_name']}")
    for epoch in progress:
        if online_prototypes and not bool(model.prototypes_initialized.item()) and epoch > prototype_warmup_epochs:
            initial_centers = initialize_online_prototypes(
                model, bundle.embeddings[train_idx], labels[train_idx], config, device
            )
            save_prototypes(run_dir / "initial_prototypes.npz", initial_centers)
        sampler.set_epoch(epoch)
        record = train_one_epoch(
            model, loader, optimizer, ce_loss, config, device, local_positive_graph=local_positive_graph
        )
        fit_logits, fit_projection, fit_online_proto_logits = model_outputs(
            model, bundle.embeddings[train_idx], device, include_online_prototypes=True
        )
        val_logits, val_projection, val_online_proto_logits = model_outputs(
            model, bundle.embeddings[val_idx], device, include_online_prototypes=True
        )
        val_p, _, _, _ = probabilities_for_model(
            config["model"],
            val_logits,
            val_projection,
            fit_projection,
            labels[train_idx],
            config,
            eval_online_prototype_logits=val_online_proto_logits,
        )
        metrics = classification_metrics(labels[val_idx], val_p)
        record.update({"epoch": epoch, "val_macro_f1": metrics["macro_f1"], "val_mcc": metrics["mcc"]})
        history.append(record)
        progress.set_postfix(loss=f"{record['train_loss']:.4f}", macro_f1=f"{metrics['macro_f1']:.4f}")
        checkpoint_eligible = not online_prototypes or bool(model.prototypes_initialized.item())
        if checkpoint_eligible and metrics["macro_f1"] > best_macro_f1 + 1e-8:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "metrics": metrics}, run_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(config["patience"]):
            break

    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    fit_logits, fit_projection, fit_online_proto_logits = model_outputs(
        model, bundle.embeddings[train_idx], device, include_online_prototypes=True
    )
    val_logits, val_projection, val_online_proto_logits = model_outputs(
        model, bundle.embeddings[val_idx], device, include_online_prototypes=True
    )
    val_p, cls_p, proto_p, prototypes = probabilities_for_model(
        config["model"],
        val_logits,
        val_projection,
        fit_projection,
        labels[train_idx],
        config,
        eval_online_prototype_logits=val_online_proto_logits,
    )
    if config["model"] == "hydrompc_v2":
        prototypes = online_prototypes_as_numpy(model)
    metrics = classification_metrics(labels[val_idx], val_p)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
    make_prediction_frame(bundle.accessions[val_idx], labels[val_idx], val_p, cls_p, proto_p).to_csv(
        run_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig"
    )
    np.savez_compressed(
        run_dir / "validation_model_outputs.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        train_labels=labels[train_idx],
        val_labels=labels[val_idx],
        train_logits=fit_logits if fit_logits is not None else np.empty((0, 0), dtype=np.float32),
        val_logits=val_logits if val_logits is not None else np.empty((0, 0), dtype=np.float32),
        train_projection=fit_projection if fit_projection is not None else np.empty((0, 0), dtype=np.float32),
        val_projection=val_projection if val_projection is not None else np.empty((0, 0), dtype=np.float32),
        train_online_prototype_logits=(
            fit_online_proto_logits
            if fit_online_proto_logits is not None
            else np.empty((0, 0), dtype=np.float32)
        ),
        val_online_prototype_logits=(
            val_online_proto_logits
            if val_online_proto_logits is not None
            else np.empty((0, 0), dtype=np.float32)
        ),
    )
    save_prototypes(run_dir / "prototypes.npz", prototypes)
    with (run_dir / "validation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    summary = {"stage": "dev", "best_epoch": best_epoch, "best_macro_f1": best_macro_f1, "device": str(device)}
    return summary


def train_final(config: dict, run_dir: Path, device: torch.device, final_epochs: int) -> dict:
    train_bundle = load_embedding_bundle(config["train_embeddings"])
    test_bundle = load_embedding_bundle(config["test_embeddings"])
    train_frame = load_sequence_csv(config["train_csv"])
    test_frame = load_sequence_csv(config["test_csv"])
    verify_bundle_matches_csv(train_bundle, train_frame, config["train_csv"])
    verify_bundle_matches_csv(test_bundle, test_frame, config["test_csv"])
    if set(train_bundle.accessions) & set(test_bundle.accessions):
        raise ValueError("Training and test accessions overlap")
    train_labels = encode_labels(train_bundle.labels)
    test_labels = encode_labels(test_bundle.labels)

    local_supcon = bool(config.get("local_supcon", False))
    local_positive_graph = None
    if local_supcon:
        local_positive_graph, graph_stats = build_local_positive_graph(
            train_bundle.embeddings,
            train_labels,
            neighbors_per_sample=int(config.get("local_knn", 10)),
            mutual=bool(config.get("local_mutual_knn", True)),
        )
        with (run_dir / "local_graph_stats.json").open("w", encoding="utf-8") as handle:
            json.dump(graph_stats, handle, ensure_ascii=False, indent=2)
    dataset = EmbeddingDataset(train_bundle.embeddings, train_labels, return_index=local_supcon)
    if local_supcon:
        sampler = LocalPositiveBatchSampler(
            train_labels, local_positive_graph, int(config["batch_samples_per_class"]), int(config["seed"])
        )
    else:
        sampler = BalancedBatchSampler(
            train_labels, int(config["batch_samples_per_class"]), int(config["seed"])
        )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    model = build_model(config, train_bundle.embeddings.shape[1], len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    ce_loss = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))
    history = []
    online_prototypes = bool(config.get("online_prototypes", False))
    prototype_warmup_epochs = int(config.get("prototype_warmup_epochs", 0))
    progress = trange(1, final_epochs + 1, desc=f"final {config['run_name']}")
    for epoch in progress:
        if online_prototypes and not bool(model.prototypes_initialized.item()) and epoch > prototype_warmup_epochs:
            initial_centers = initialize_online_prototypes(
                model, train_bundle.embeddings, train_labels, config, device
            )
            save_prototypes(run_dir / "initial_prototypes.npz", initial_centers)
        sampler.set_epoch(epoch)
        record = train_one_epoch(
            model, loader, optimizer, ce_loss, config, device, local_positive_graph=local_positive_graph
        )
        record["epoch"] = epoch
        history.append(record)
        progress.set_postfix(loss=f"{record['train_loss']:.4f}")
    torch.save({"state_dict": model.state_dict(), "epoch": final_epochs}, run_dir / "final_model.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")

    train_logits, train_projection, train_online_proto_logits = model_outputs(
        model, train_bundle.embeddings, device, include_online_prototypes=True
    )
    test_logits, test_projection, test_online_proto_logits = model_outputs(
        model, test_bundle.embeddings, device, include_online_prototypes=True
    )
    final_p, cls_p, proto_p, prototypes = probabilities_for_model(
        config["model"],
        test_logits,
        test_projection,
        train_projection,
        train_labels,
        config,
        eval_online_prototype_logits=test_online_proto_logits,
    )
    if config["model"] == "hydrompc_v2":
        prototypes = online_prototypes_as_numpy(model)
    predictions = make_prediction_frame(test_bundle.accessions, test_labels, final_p, cls_p, proto_p)
    predictions.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    save_prototypes(run_dir / "prototypes.npz", prototypes)

    subset_files = {
        "Test-all": None,
        "Test<80": Path(config["nested_dir"]) / "test_nested_lt80.csv",
        "Test<60": Path(config["nested_dir"]) / "test_nested_lt60.csv",
        "Test<40": Path(config["nested_dir"]) / "test_nested_lt40.csv",
    }
    metrics_by_subset = {}
    rows = []
    for name, subset_path in subset_files.items():
        if subset_path is None:
            mask = np.ones(len(test_bundle.accessions), dtype=bool)
        else:
            subset = load_sequence_csv(subset_path)
            subset_ids = set(subset["accession"].astype(str))
            mask = np.asarray([accession in subset_ids for accession in test_bundle.accessions])
        metrics = classification_metrics(test_labels[mask], final_p[mask])
        metrics_by_subset[name] = metrics
        rows.append({
            "subset": name,
            "n": metrics["n"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "mcc": metrics["mcc"],
        })
    pd.DataFrame(rows).to_csv(run_dir / "test_metrics_by_subset.csv", index=False, encoding="utf-8-sig")
    with (run_dir / "test_metrics_full.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_by_subset, handle, ensure_ascii=False, indent=2)
    return {"stage": "final", "epochs": final_epochs, "device": str(device), "test_macro_f1": metrics_by_subset["Test-all"]["macro_f1"]}


def parse_args():
    parser = argparse.ArgumentParser(description="Train unified hydrolase classification experiments")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["dev", "final"], required=True)
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    run_dir = prepare_run_dir(config, args.stage, args.run_dir, args.overwrite)
    with (run_dir / "config_resolved.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    if args.stage == "dev":
        summary = train_dev(config, run_dir, device)
    else:
        if args.final_epochs is None or args.final_epochs <= 0:
            raise ValueError("The final stage requires the development-locked epoch count via --final-epochs")
        summary = train_final(config, run_dir, device, args.final_epochs)
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"run_dir={run_dir.resolve()}")


if __name__ == "__main__":
    main()
