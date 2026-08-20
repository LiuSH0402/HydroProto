from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, friedmanchisquare, wilcoxon
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def parse_run(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    return name, Path(path)


def stratified_indices(labels: np.ndarray, iterations: int, seed: int):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    for _ in range(iterations):
        sample = np.concatenate([
            rng.choice(group, size=len(group), replace=True) for group in groups
        ])
        rng.shuffle(sample)
        yield sample


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    result = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        result[index] = running
    return result.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="HydroProto prototype-count ablation statistics")
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--reference", default="K1")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    runs = dict(parse_run(value) for value in args.run)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_frames = {
        name: pd.read_csv(path / "out_of_fold_predictions.csv", dtype={"accession": str})
        .set_index("accession")
        for name, path in runs.items()
    }
    reference_frame = prediction_frames[args.reference]
    accessions = reference_frame.index.to_numpy()
    truth = reference_frame.loc[accessions, "true_label"].astype(str).to_numpy()
    predictions = {}
    for name, frame in prediction_frames.items():
        if set(frame.index) != set(accessions):
            raise RuntimeError(f"Accession mismatch for {name}")
        aligned_truth = frame.loc[accessions, "true_label"].astype(str).to_numpy()
        if not np.array_equal(truth, aligned_truth):
            raise RuntimeError(f"Truth mismatch for {name}")
        predictions[name] = frame.loc[accessions, "predicted_label"].astype(str).to_numpy()

    point_rows = []
    for name, pred in predictions.items():
        point_rows.append({
            "K": name,
            "n": len(truth),
            "oof_accuracy": accuracy_score(truth, pred),
            "oof_macro_f1": f1_score(truth, pred, average="macro"),
            "oof_mcc": matthews_corrcoef(truth, pred),
        })
    pd.DataFrame(point_rows).to_csv(output / "oof_point_metrics.csv", index=False)

    delta_values = {
        name: {"macro_f1": [], "accuracy": [], "mcc": []}
        for name in runs if name != args.reference
    }
    reference_pred = predictions[args.reference]
    for sample in stratified_indices(truth, args.iterations, args.seed):
        sampled_truth = truth[sample]
        reference_metrics = {
            "macro_f1": f1_score(sampled_truth, reference_pred[sample], average="macro"),
            "accuracy": accuracy_score(sampled_truth, reference_pred[sample]),
            "mcc": matthews_corrcoef(sampled_truth, reference_pred[sample]),
        }
        for name in delta_values:
            current = {
                "macro_f1": f1_score(sampled_truth, predictions[name][sample], average="macro"),
                "accuracy": accuracy_score(sampled_truth, predictions[name][sample]),
                "mcc": matthews_corrcoef(sampled_truth, predictions[name][sample]),
            }
            for metric in current:
                delta_values[name][metric].append(current[metric] - reference_metrics[metric])

    delta_rows = []
    for name, metric_values in delta_values.items():
        for metric, values in metric_values.items():
            array = np.asarray(values)
            low, high = np.percentile(array, [2.5, 97.5])
            delta_rows.append({
                "comparison": f"{name}-{args.reference}",
                "metric": metric,
                "mean_bootstrap_delta": array.mean(),
                "ci95_low": low,
                "ci95_high": high,
                "probability_delta_le_0": np.mean(array <= 0),
            })
    pd.DataFrame(delta_rows).to_csv(output / "paired_bootstrap_delta.csv", index=False)

    pair_rows = []
    for left, right in itertools.combinations(runs, 2):
        left_correct = predictions[left] == truth
        right_correct = predictions[right] == truth
        left_only = int(np.sum(left_correct & ~right_correct))
        right_only = int(np.sum(~left_correct & right_correct))
        discordant = left_only + right_only
        p = float(binomtest(left_only, discordant, 0.5).pvalue) if discordant else 1.0
        pair_rows.append({
            "left": left,
            "right": right,
            "left_only_correct": left_only,
            "right_only_correct": right_only,
            "discordant": discordant,
            "mcnemar_exact_p": p,
        })
    adjusted = holm([row["mcnemar_exact_p"] for row in pair_rows])
    for row, value in zip(pair_rows, adjusted):
        row["mcnemar_holm_p"] = value
        row["significant_0.05"] = value < 0.05
    pd.DataFrame(pair_rows).to_csv(output / "mcnemar_pairwise_holm.csv", index=False)

    fold_frames = {
        name: pd.read_csv(path / "fold_metrics.csv").sort_values("fold")
        for name, path in runs.items()
    }
    fold_rows = []
    for metric in ("accuracy", "macro_f1", "mcc"):
        values = [fold_frames[name][metric].to_numpy() for name in runs]
        friedman = friedmanchisquare(*values)
        reference_values = fold_frames[args.reference][metric].to_numpy()
        for name in runs:
            if name == args.reference:
                continue
            current = fold_frames[name][metric].to_numpy()
            test = wilcoxon(current, reference_values, alternative="two-sided", method="exact")
            fold_rows.append({
                "metric": metric,
                "comparison": f"{name}-{args.reference}",
                "mean_fold_delta": float(np.mean(current - reference_values)),
                "folds_positive": int(np.sum(current > reference_values)),
                "folds_negative": int(np.sum(current < reference_values)),
                "wilcoxon_exact_p": float(test.pvalue),
                "friedman_statistic_all_K": float(friedman.statistic),
                "friedman_p_all_K": float(friedman.pvalue),
            })
    pd.DataFrame(fold_rows).to_csv(output / "fold_paired_tests.csv", index=False)
    audit = {
        "runs": {name: str(path.resolve()) for name, path in runs.items()},
        "reference": args.reference,
        "records": len(truth),
        "iterations": args.iterations,
        "seed": args.seed,
        "warning": "OOF bootstrap conditions on the five fitted fold models; fold-level tests reflect training variability but have only five pairs.",
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", **audit}, indent=2))


if __name__ == "__main__":
    main()
