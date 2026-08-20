from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)


METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "mcc")


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Model must use NAME=DIR syntax: {value}")
    name, raw_path = value.split("=", 1)
    return name.strip(), Path(raw_path)


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def stratified_bootstrap_indices(
    labels: np.ndarray,
    iterations: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    samples = []
    for _ in range(iterations):
        values = [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        sample = np.concatenate(values)
        rng.shuffle(sample)
        samples.append(sample)
    return samples


def percentile_interval(values: list[float]) -> tuple[float, float]:
    low, high = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
    return float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def cochran_q(correctness: np.ndarray) -> tuple[float, int, float]:
    # Rows are samples and columns are models.
    matrix = correctness.astype(float)
    n_models = matrix.shape[1]
    model_totals = matrix.sum(axis=0)
    sample_totals = matrix.sum(axis=1)
    total = float(model_totals.sum())
    denominator = n_models * total - float(np.square(sample_totals).sum())
    if denominator <= 0:
        return float("nan"), n_models - 1, float("nan")
    numerator = (n_models - 1) * (
        n_models * float(np.square(model_totals).sum()) - total**2
    )
    statistic = numerator / denominator
    return float(statistic), n_models - 1, float(chi2.sf(statistic, n_models - 1))


def load_aligned_predictions(
    model_dirs: dict[str, Path],
    subset: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    frames = {}
    filename = f"test_predictions_{subset}.csv"
    for model, directory in model_dirs.items():
        path = directory / filename
        if subset == "test_all" and not path.exists():
            path = directory / "test_predictions.csv"
        frame = pd.read_csv(path, dtype={"accession": str})
        required = {"accession", "true_label", "predicted_label"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        if frame["accession"].duplicated().any():
            raise ValueError(f"Duplicate accession in {path}")
        frames[model] = frame.set_index("accession")

    reference_name = next(iter(frames))
    accessions = frames[reference_name].index.astype(str).to_numpy()
    truth = frames[reference_name].loc[accessions, "true_label"].astype(str).to_numpy()
    predictions = {}
    for model, frame in frames.items():
        if set(frame.index.astype(str)) != set(accessions):
            raise ValueError(f"Accession set mismatch for {model} in {subset}")
        aligned_truth = frame.loc[accessions, "true_label"].astype(str).to_numpy()
        if not np.array_equal(aligned_truth, truth):
            raise ValueError(f"True-label mismatch for {model} in {subset}")
        predictions[model] = (
            frame.loc[accessions, "predicted_label"].astype(str).to_numpy()
        )
    return truth, predictions, accessions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--reference-model", default="HydroProto")
    parser.add_argument(
        "--subset",
        action="append",
        default=[],
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    subsets = args.subset or ["test_all", "lt80", "lt60", "lt40"]

    display_aliases = {
        "H-Proto": "HydroProto",
        "H-Local": "HydroLocal",
        "Hydro-MPC": "HydroMPC",
    }
    model_dirs = {
        display_aliases.get(name, name): path
        for name, path in (parse_model(value) for value in args.model)
    }
    args.reference_model = display_aliases.get(
        args.reference_model,
        args.reference_model,
    )
    if args.reference_model not in model_dirs:
        raise ValueError(f"Unknown reference model: {args.reference_model}")
    if len(model_dirs) < 2:
        raise ValueError("At least two models are required")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ci_rows = []
    delta_rows = []
    mcnemar_rows = []
    q_rows = []
    audit = {
        "iterations": int(args.iterations),
        "seed": int(args.seed),
        "reference_model": args.reference_model,
        "models": {name: str(path.resolve()) for name, path in model_dirs.items()},
        "subsets": {},
        "paired_resampling": True,
        "stratified_by_true_class": True,
    }

    for subset_number, subset in enumerate(dict.fromkeys(subsets)):
        truth, predictions, accessions = load_aligned_predictions(model_dirs, subset)
        samples = stratified_bootstrap_indices(
            truth,
            args.iterations,
            args.seed + subset_number * 10000,
        )
        point = {model: metric_values(truth, pred) for model, pred in predictions.items()}
        bootstrap = {
            model: {metric: [] for metric in METRICS} for model in model_dirs
        }
        for indices in samples:
            sampled_truth = truth[indices]
            for model, pred in predictions.items():
                values = metric_values(sampled_truth, pred[indices])
                for metric in METRICS:
                    bootstrap[model][metric].append(values[metric])

        for model in model_dirs:
            for metric in METRICS:
                low, high = percentile_interval(bootstrap[model][metric])
                ci_rows.append(
                    {
                        "subset": subset,
                        "n": len(truth),
                        "model": model,
                        "metric": metric,
                        "estimate": point[model][metric],
                        "ci95_low": low,
                        "ci95_high": high,
                        "iterations": args.iterations,
                    }
                )

        reference = args.reference_model
        for comparator in model_dirs:
            if comparator == reference:
                continue
            for metric in METRICS:
                deltas = (
                    np.asarray(bootstrap[reference][metric])
                    - np.asarray(bootstrap[comparator][metric])
                )
                low, high = percentile_interval(deltas.tolist())
                delta_rows.append(
                    {
                        "subset": subset,
                        "n": len(truth),
                        "reference_model": reference,
                        "comparator": comparator,
                        "metric": metric,
                        "estimate_delta": point[reference][metric]
                        - point[comparator][metric],
                        "ci95_low": low,
                        "ci95_high": high,
                        "bootstrap_probability_delta_le_0": float(
                            np.mean(deltas <= 0)
                        ),
                    }
                )

        correctness = np.column_stack(
            [predictions[model] == truth for model in model_dirs]
        )
        q_stat, q_df, q_p = cochran_q(correctness)
        q_rows.append(
            {
                "subset": subset,
                "n": len(truth),
                "models": "|".join(model_dirs),
                "q_statistic": q_stat,
                "df": q_df,
                "p_value": q_p,
            }
        )

        subset_pair_rows = []
        for left, right in itertools.combinations(model_dirs, 2):
            left_correct = predictions[left] == truth
            right_correct = predictions[right] == truth
            left_only = int(np.sum(left_correct & ~right_correct))
            right_only = int(np.sum(~left_correct & right_correct))
            discordant = left_only + right_only
            p_value = (
                float(binomtest(left_only, discordant, 0.5).pvalue)
                if discordant
                else 1.0
            )
            subset_pair_rows.append(
                {
                    "subset": subset,
                    "n": len(truth),
                    "model_left": left,
                    "model_right": right,
                    "left_correct_right_wrong": left_only,
                    "left_wrong_right_correct": right_only,
                    "discordant": discordant,
                    "p_value_exact": p_value,
                }
            )
        adjusted = holm_adjust([row["p_value_exact"] for row in subset_pair_rows])
        for row, value in zip(subset_pair_rows, adjusted):
            row["p_value_holm"] = value
            row["significant_holm_0.05"] = bool(value < 0.05)
        mcnemar_rows.extend(subset_pair_rows)

        audit["subsets"][subset] = {
            "records": int(len(truth)),
            "unique_accessions": int(len(np.unique(accessions))),
            "class_counts": {
                str(label): int(np.sum(truth == label)) for label in np.unique(truth)
            },
            "aligned_across_models": True,
        }

    pd.DataFrame(ci_rows).to_csv(
        output_dir / "bootstrap_metric_ci.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(delta_rows).to_csv(
        output_dir / "bootstrap_paired_delta_ci.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(mcnemar_rows).to_csv(
        output_dir / "mcnemar_pairwise_holm.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(q_rows).to_csv(
        output_dir / "cochran_q.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "statistics_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", **audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
