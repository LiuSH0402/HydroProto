from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from .data import encode_labels, load_embedding_bundle, load_sequence_csv, seed_everything, verify_bundle_matches_csv
from .train import load_config, train_dev


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}


def main():
    parser = argparse.ArgumentParser(description="Run stratified K-fold CV on the 2,721-record training set")
    parser.add_argument("--config", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output-root", default="modeling/cv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError("folds must be at least 2")
    config = load_config(args.config)
    bundle = load_embedding_bundle(config["train_embeddings"])
    frame = load_sequence_csv(config["train_csv"])
    verify_bundle_matches_csv(bundle, frame, config["train_csv"])
    labels = encode_labels(bundle.labels)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.split_seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    cv_dir = Path(args.output_root) / config["run_name"]
    cv_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    oof_frames = []
    validation_counts = np.zeros(len(labels), dtype=np.int64)
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(bundle.embeddings, labels), start=1):
        validation_counts[val_idx] += 1
        fold_dir = cv_dir / f"fold_{fold_index}"
        completed = (fold_dir / "run_summary.json").exists() and (fold_dir / "validation_model_outputs.npz").exists()
        if completed and not args.overwrite:
            summary = json.loads((fold_dir / "run_summary.json").read_text(encoding="utf-8"))
        else:
            if fold_dir.exists():
                shutil.rmtree(fold_dir)
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_config = dict(config)
            fold_config["seed"] = int(config["seed"]) + fold_index - 1
            fold_config["cv_fold"] = fold_index
            fold_config["cv_folds"] = args.folds
            fold_config["cv_split_seed"] = args.split_seed
            seed_everything(int(fold_config["seed"]))
            with (fold_dir / "config_resolved.json").open("w", encoding="utf-8") as handle:
                json.dump(fold_config, handle, ensure_ascii=False, indent=2)
            summary = train_dev(fold_config, fold_dir, device, split_override=(train_idx, val_idx))
            summary.update({
                "cv_fold": fold_index,
                "cv_folds": args.folds,
                "cv_split_seed": args.split_seed,
                "train_records": int(len(train_idx)),
                "validation_records": int(len(val_idx)),
            })
            with (fold_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)

        metrics = json.loads((fold_dir / "validation_metrics.json").read_text(encoding="utf-8"))
        fold_rows.append({
            "fold": fold_index,
            "train_n": int(len(train_idx)),
            "validation_n": int(len(val_idx)),
            "best_epoch": int(summary["best_epoch"]),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "mcc": metrics["mcc"],
            **{f"f1_{label}": metrics["per_class"][label]["f1"] for label in metrics["per_class"]},
        })
        predictions = pd.read_csv(fold_dir / "validation_predictions.csv")
        predictions.insert(0, "fold", fold_index)
        oof_frames.append(predictions)

    if not np.all(validation_counts == 1):
        raise RuntimeError("Cross-validation indices must validate every training record exactly once")
    fold_metrics = pd.DataFrame(fold_rows).sort_values("fold")
    fold_metrics.to_csv(cv_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    oof = pd.concat(oof_frames, ignore_index=True)
    if len(oof) != len(labels) or oof["accession"].duplicated().any():
        raise RuntimeError("Out-of-fold predictions must cover every training record exactly once")
    oof.to_csv(cv_dir / "out_of_fold_predictions.csv", index=False, encoding="utf-8-sig")

    metric_names = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc"]
    summary = {
        "run_name": config["run_name"],
        "model": config["model"],
        "folds": args.folds,
        "split_seed": args.split_seed,
        "records": int(len(labels)),
        "external_test_used": False,
        "fold_seed_rule": "base_seed + fold_index - 1",
        "metrics": {name: mean_std(fold_metrics[name].tolist()) for name in metric_names},
        "best_epoch": mean_std(fold_metrics["best_epoch"].tolist()),
        "fold_validation_sizes": fold_metrics["validation_n"].tolist(),
    }
    with (cv_dir / "cv_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(fold_metrics.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"cv_dir={cv_dir.resolve()}")


if __name__ == "__main__":
    main()
