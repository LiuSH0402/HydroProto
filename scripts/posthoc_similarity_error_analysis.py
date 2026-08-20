from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


LABELS = [
    "EC3.1_ester_bonds",
    "EC3.2_glycosylases",
    "EC3.4_peptidases",
    "EC3.5_CN_nonpeptide",
    "EC3_other",
]
SHORT_LABELS = ["EC3.1", "EC3.2", "EC3.4", "EC3.5", "EC3.other"]
MODELS = ["ESM2+MLP", "HydroLocal", "HydroMPC", "CLEAN", "HydroProto"]
COLORS = {
    "ESM2+MLP": "#484878",
    "HydroLocal": "#7884B4",
    "HydroMPC": "#A8A8A8",
    "CLEAN": "#B4C0E4",
    "HydroProto": "#B64342",
}
MARKERS = {"ESM2+MLP": "o", "HydroLocal": "s", "HydroMPC": "D", "CLEAN": "^", "HydroProto": "o"}
NONOVERLAP = ["ge80", "60_to_lt80", "40_to_lt60", "lt40"]
NONOVERLAP_LABEL = {"ge80": "≥80%", "60_to_lt80": "60–80%", "40_to_lt60": "40–60%", "lt40": "<40%"}
NESTED = ["test_all", "lt80", "lt60", "lt40"]
NESTED_LABEL = {"test_all": "All", "lt80": "<80%", "lt60": "<60%", "lt40": "<40%"}


def metric_values(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted")),
        "mcc": float(matthews_corrcoef(truth, prediction)),
    }


def bootstrap_macro_f1(
    truth: np.ndarray,
    prediction: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(truth == label) for label in np.unique(truth)]
    values = []
    for _ in range(iterations):
        sample = np.concatenate([
            rng.choice(group, len(group), replace=True) for group in groups
        ])
        values.append(f1_score(truth[sample], prediction[sample], average="macro"))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, value: str) -> None:
    ax.text(-0.13, 1.05, value, transform=ax.transAxes, fontsize=11, fontweight="bold")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--similarity-csv", required=True)
    parser.add_argument("--hydroproto", "--hproto", dest="hydroproto", required=True)
    parser.add_argument("--mlp", required=True)
    parser.add_argument("--hydrolocal", "--hlocal", dest="hydrolocal", required=True)
    parser.add_argument("--hydrompc", required=True)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output_dir)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    similarity = pd.read_csv(args.similarity_csv, dtype={"accession": str})
    similarity["max_identity_pct"] = pd.to_numeric(similarity["max_identity_pct"], errors="coerce")
    similarity["length"] = pd.to_numeric(similarity["length"], errors="raise").astype(int)
    similarity = similarity.set_index("accession", drop=False)
    paths = {
        "HydroProto": args.hydroproto,
        "ESM2+MLP": args.mlp,
        "HydroLocal": args.hydrolocal,
        "HydroMPC": args.hydrompc,
        "CLEAN": args.clean,
    }
    predictions = {}
    accessions = similarity.index.to_numpy()
    truth = similarity.loc[accessions, "label"].astype(str).to_numpy()
    for model, path in paths.items():
        frame = pd.read_csv(path, dtype={"accession": str}).set_index("accession")
        if set(frame.index) != set(accessions):
            raise RuntimeError(f"Accession mismatch for {model}")
        frame = frame.loc[accessions]
        if not np.array_equal(frame["true_label"].astype(str).to_numpy(), truth):
            raise RuntimeError(f"Truth mismatch for {model}")
        predictions[model] = frame

    # Preserve both scientific questions: nested stress tests and disjoint bins.
    nonoverlap_masks = {
        name: similarity["similarity_bin"].eq(name).to_numpy() for name in NONOVERLAP
    }
    nested_masks = {
        "test_all": np.ones(len(similarity), dtype=bool),
        "lt80": similarity["similarity_bin"].isin(["60_to_lt80", "40_to_lt60", "lt40"]).to_numpy(),
        "lt60": similarity["similarity_bin"].isin(["40_to_lt60", "lt40"]).to_numpy(),
        "lt40": similarity["similarity_bin"].eq("lt40").to_numpy(),
    }
    metric_rows = []
    per_class_rows = []
    for analysis, masks in (("nonoverlap", nonoverlap_masks), ("nested", nested_masks)):
        for bin_number, (bin_name, mask) in enumerate(masks.items()):
            local_truth = truth[mask]
            for model_number, model in enumerate(MODELS):
                local_pred = predictions[model]["predicted_label"].astype(str).to_numpy()[mask]
                values = metric_values(local_truth, local_pred)
                low, high = bootstrap_macro_f1(
                    local_truth, local_pred, args.bootstrap_iterations,
                    args.seed + bin_number * 100 + model_number,
                )
                metric_rows.append({
                    "analysis": analysis,
                    "bin": bin_name,
                    "bin_label": NONOVERLAP_LABEL.get(bin_name, NESTED_LABEL.get(bin_name, bin_name)),
                    "model": model,
                    "n": int(mask.sum()),
                    **values,
                    "macro_f1_ci95_low": low,
                    "macro_f1_ci95_high": high,
                })
                precision, recall, f1, support = precision_recall_fscore_support(
                    local_truth, local_pred, labels=LABELS, zero_division=0
                )
                for index, label in enumerate(LABELS):
                    per_class_rows.append({
                        "analysis": analysis,
                        "bin": bin_name,
                        "model": model,
                        "class": label,
                        "precision": precision[index],
                        "recall": recall[index],
                        "f1": f1[index],
                        "support": int(support[index]),
                    })
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(tables / "metrics_nested_and_nonoverlap.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_class_rows).to_csv(tables / "per_class_metrics_all_models.csv", index=False, encoding="utf-8-sig")

    counts = (
        similarity.groupby(["similarity_bin", "label"], observed=False)
        .size().rename("n").reset_index()
    )
    counts.to_csv(tables / "nonoverlap_counts_by_class.csv", index=False, encoding="utf-8-sig")

    # Confusion matrices and HydroProto per-class metrics requested by the user.
    confusion_rows = []
    hproto_class_rows = []
    hproto_pred = predictions["HydroProto"]["predicted_label"].astype(str).to_numpy()
    for subset_name, mask in (("test_all", nested_masks["test_all"]), ("lt40", nested_masks["lt40"])):
        cm = confusion_matrix(truth[mask], hproto_pred[mask], labels=LABELS)
        normalized = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        for i, true_label in enumerate(LABELS):
            for j, predicted_label in enumerate(LABELS):
                confusion_rows.append({
                    "subset": subset_name,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": int(cm[i, j]),
                    "row_fraction": float(normalized[i, j]),
                })
        precision, recall, f1, support = precision_recall_fscore_support(
            truth[mask], hproto_pred[mask], labels=LABELS, zero_division=0
        )
        for i, label in enumerate(LABELS):
            hproto_class_rows.append({
                "subset": subset_name,
                "class": label,
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": int(support[i]),
            })
    pd.DataFrame(confusion_rows).to_csv(tables / "hydroproto_confusion_matrices.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(hproto_class_rows).to_csv(tables / "hydroproto_per_class_metrics.csv", index=False, encoding="utf-8-sig")

    # Per-record error categories with biological metadata and confidence.
    record_frame = similarity.reset_index(drop=True).copy()
    record_truth = record_frame["label"].astype(str).to_numpy(copy=True)
    if not np.array_equal(record_truth, truth):
        mismatches = np.flatnonzero(record_truth != truth)
        raise RuntimeError(
            f"Record truth changed before error audit: {len(mismatches)} mismatches"
        )
    for model in MODELS:
        frame = predictions[model]
        model_prediction = frame["predicted_label"].astype(str).to_numpy(copy=True)
        record_frame[f"prediction_{model}"] = model_prediction
        record_frame[f"correct_{model}"] = model_prediction == record_truth
        expected_correct = int(round(accuracy_score(record_truth, model_prediction) * len(record_truth)))
        table_expected = int(round(
            metric_frame[
                metric_frame["analysis"].eq("nested")
                & metric_frame["bin"].eq("test_all")
                & metric_frame["model"].eq(model)
            ].iloc[0]["accuracy"] * len(record_truth)
        ))
        if expected_correct != table_expected:
            raise RuntimeError(
                f"Prediction alignment changed for {model}: "
                f"record={expected_correct}, table={table_expected}"
            )
        if int(record_frame[f"correct_{model}"].sum()) != expected_correct:
            raise RuntimeError(f"Per-record correctness audit failed for {model}")
        probability_columns = [
            column for column in frame.columns
            if column.startswith("p_final_") or (model == "CLEAN" and column.startswith("p_EC"))
        ]
        if probability_columns:
            record_frame[f"confidence_{model}"] = frame[probability_columns].max(axis=1).to_numpy()
    hproto_win = record_frame["correct_HydroProto"] & ~record_frame["correct_ESM2+MLP"]
    all_wrong = np.ones(len(record_frame), dtype=bool)
    for model in MODELS:
        all_wrong &= ~record_frame[f"correct_{model}"].to_numpy()
    # Copy is essential: in-place &= on a NumPy view would overwrite the
    # source correctness column in record_frame.
    unique_hproto = record_frame["correct_HydroProto"].to_numpy(copy=True)
    for model in MODELS:
        if model != "HydroProto":
            unique_hproto &= ~record_frame[f"correct_{model}"].to_numpy()
    record_frame.loc[hproto_win].to_csv(tables / "hydroproto_correct_mlp_wrong.csv", index=False, encoding="utf-8-sig")
    record_frame.loc[all_wrong].to_csv(tables / "all_five_models_wrong.csv", index=False, encoding="utf-8-sig")
    record_frame.loc[unique_hproto].to_csv(tables / "hydroproto_unique_correct_all_others_wrong.csv", index=False, encoding="utf-8-sig")
    record_frame.to_csv(tables / "test_record_error_audit.csv", index=False, encoding="utf-8-sig")

    # Sequence-length and maximum-identity relationships.
    length_labels = ["<200", "200–399", "400–599", "600–999", "≥1000"]
    record_frame["length_bin"] = pd.cut(
        record_frame["length"], bins=[0, 200, 400, 600, 1000, np.inf],
        right=False, labels=length_labels,
    )
    record_frame["identity_diagnostic_bin"] = np.select(
        [
            record_frame["has_coverage_qualified_hit"].astype(int).eq(0),
            record_frame["max_identity_pct"].lt(40),
            record_frame["max_identity_pct"].lt(60),
            record_frame["max_identity_pct"].lt(80),
        ],
        ["No qualified hit", "<40% hit", "40–60%", "60–80%"],
        default="≥80%",
    )
    relationship_rows = []
    for variable, order in (
        ("length_bin", length_labels),
        ("identity_diagnostic_bin", ["≥80%", "60–80%", "40–60%", "<40% hit", "No qualified hit"]),
    ):
        for value in order:
            mask = record_frame[variable].astype(str).eq(value).to_numpy()
            if not mask.any():
                continue
            for model in MODELS:
                pred = record_frame[f"prediction_{model}"].astype(str).to_numpy()[mask]
                values = metric_values(truth[mask], pred)
                relationship_rows.append({
                    "variable": variable,
                    "bin": value,
                    "model": model,
                    "n": int(mask.sum()),
                    **values,
                    "error_rate": 1.0 - values["accuracy"],
                })
    relationship_frame = pd.DataFrame(relationship_rows)
    relationship_frame.to_csv(tables / "performance_by_length_and_identity.csv", index=False, encoding="utf-8-sig")

    flow_rows = []
    for true_label in LABELS:
        mask = truth == true_label
        for model in MODELS:
            local_pred = record_frame.loc[mask, f"prediction_{model}"]
            for predicted_label, count in local_pred.value_counts().items():
                if predicted_label != true_label:
                    flow_rows.append({
                        "model": model,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(count),
                    })
    pd.DataFrame(flow_rows).to_csv(tables / "misclassification_flows.csv", index=False, encoding="utf-8-sig")

    # Figure 1: robustness and class-level error structure.
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), constrained_layout=True)
    for ax, analysis, order, label_map, panel in (
        (axes[0, 0], "nonoverlap", NONOVERLAP, NONOVERLAP_LABEL, "a"),
        (axes[0, 1], "nested", NESTED, NESTED_LABEL, "b"),
    ):
        subset = metric_frame[metric_frame["analysis"].eq(analysis)]
        for model in MODELS:
            values = subset[subset["model"].eq(model)].set_index("bin").loc[order]
            x = np.arange(len(order))
            estimate = values["macro_f1"].to_numpy()
            low = values["macro_f1_ci95_low"].to_numpy()
            high = values["macro_f1_ci95_high"].to_numpy()
            ax.errorbar(
                x, estimate, yerr=[estimate - low, high - estimate],
                color=COLORS[model], marker=MARKERS[model], ms=4,
                lw=1.4 if model == "HydroProto" else 1.0,
                capsize=2, label=model, zorder=5 if model == "HydroProto" else 2,
            )
        ax.set_xticks(np.arange(len(order)), [label_map[value] for value in order])
        ax.set_ylim(0.78, 1.0)
        ax.set_ylabel("Macro-F1")
        ax.set_title("Disjoint homology bins" if analysis == "nonoverlap" else "Nested stress tests")
        panel_label(ax, panel)
    axes[0, 1].legend(fontsize=6.5, ncol=2, loc="lower left")

    confusion_data = pd.DataFrame(confusion_rows)
    for ax, subset_name, title, panel in (
        (axes[0, 2], "test_all", "HydroProto: all test", "c"),
        (axes[1, 0], "lt40", "HydroProto: <40%", "d"),
    ):
        data = confusion_data[confusion_data["subset"].eq(subset_name)]
        matrix = data.pivot(index="true_label", columns="predicted_label", values="row_fraction").loc[LABELS, LABELS]
        sns.heatmap(
            matrix, ax=ax, cmap="Blues", vmin=0, vmax=1, annot=True, fmt=".2f",
            cbar=False, square=True, xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS,
            annot_kws={"fontsize": 6.5},
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        panel_label(ax, panel)

    class_data = pd.DataFrame(hproto_class_rows)
    x = np.arange(len(LABELS))
    width = 0.35
    for offset, subset_name, color, hatch, label in (
        (-width / 2, "test_all", "#7884B4", "", "All"),
        (width / 2, "lt40", "#B64342", "//", "<40%"),
    ):
        values = class_data[class_data["subset"].eq(subset_name)].set_index("class").loc[LABELS]
        axes[1, 1].bar(x + offset, values["f1"], width, color=color, hatch=hatch, label=label)
    axes[1, 1].set_xticks(x, SHORT_LABELS, rotation=30, ha="right")
    axes[1, 1].set_ylim(0.70, 1.01)
    axes[1, 1].set_ylabel("HydroProto class F1")
    axes[1, 1].legend(fontsize=7)
    panel_label(axes[1, 1], "e")

    length_data = relationship_frame[relationship_frame["variable"].eq("length_bin")]
    for model in ("ESM2+MLP", "CLEAN", "HydroProto"):
        values = length_data[length_data["model"].eq(model)].set_index("bin").loc[length_labels]
        axes[1, 2].plot(
            np.arange(len(length_labels)), values["accuracy"],
            marker=MARKERS[model], color=COLORS[model], label=model,
            lw=1.5 if model == "HydroProto" else 1.0,
        )
    axes[1, 2].set_xticks(np.arange(len(length_labels)), length_labels, rotation=25, ha="right")
    axes[1, 2].set_ylim(0.78, 1.0)
    axes[1, 2].set_xlabel("Sequence length (aa)")
    axes[1, 2].set_ylabel("Accuracy")
    axes[1, 2].legend(fontsize=7)
    panel_label(axes[1, 2], "f")
    save_figure(fig, figures / "figure_similarity_and_class_errors")

    # Figure 2: error relationships and sample-level complementarity.
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), constrained_layout=True)
    identity_order = ["≥80%", "60–80%", "40–60%", "<40% hit", "No qualified hit"]
    identity_data = relationship_frame[relationship_frame["variable"].eq("identity_diagnostic_bin")]
    for model in ("ESM2+MLP", "CLEAN", "HydroProto"):
        values = identity_data[identity_data["model"].eq(model)].set_index("bin").loc[identity_order]
        axes[0].plot(
            np.arange(len(identity_order)), values["error_rate"],
            marker=MARKERS[model], color=COLORS[model], label=model,
            lw=1.5 if model == "HydroProto" else 1.0,
        )
    axes[0].set_xticks(np.arange(len(identity_order)), identity_order, rotation=28, ha="right")
    axes[0].set_ylabel("Error rate")
    axes[0].set_title("Maximum train homology")
    axes[0].legend(fontsize=7)
    panel_label(axes[0], "a")

    comparison_counts = {
        "Both correct": int(np.sum(record_frame["correct_HydroProto"] & record_frame["correct_ESM2+MLP"])),
        "HydroProto only": int(np.sum(record_frame["correct_HydroProto"] & ~record_frame["correct_ESM2+MLP"])),
        "MLP only": int(np.sum(~record_frame["correct_HydroProto"] & record_frame["correct_ESM2+MLP"])),
        "Both wrong": int(np.sum(~record_frame["correct_HydroProto"] & ~record_frame["correct_ESM2+MLP"])),
    }
    bars = axes[1].bar(
        np.arange(4), list(comparison_counts.values()),
        color=["#D8D8D8", "#B64342", "#484878", "#606060"],
    )
    axes[1].set_xticks(np.arange(4), ["Both\ncorrect", "HydroProto\nonly", "MLP\nonly", "Both\nwrong"])
    axes[1].set_ylabel("Test sequences")
    axes[1].bar_label(bars, fontsize=7, padding=2)
    axes[1].set_title("Prediction complementarity")
    panel_label(axes[1], "b")

    error_counts = []
    for model in ("ESM2+MLP", "HydroProto"):
        error_counts.append([
            int(np.sum((truth == label) & ~record_frame[f"correct_{model}"].to_numpy()))
            for label in LABELS
        ])
    width = 0.36
    x = np.arange(len(LABELS))
    axes[2].bar(x - width / 2, error_counts[0], width, color=COLORS["ESM2+MLP"], label="ESM2+MLP")
    axes[2].bar(x + width / 2, error_counts[1], width, color=COLORS["HydroProto"], label="HydroProto")
    axes[2].set_xticks(x, SHORT_LABELS, rotation=30, ha="right")
    axes[2].set_ylabel("Errors")
    axes[2].set_title("Errors by true class")
    axes[2].legend(fontsize=7)
    panel_label(axes[2], "c")
    save_figure(fig, figures / "figure_error_diagnostics")

    audit = {
        "status": "complete",
        "records": len(similarity),
        "accessions_aligned_across_models": True,
        "nonoverlap_counts": {name: int(mask.sum()) for name, mask in nonoverlap_masks.items()},
        "nested_counts": {name: int(mask.sum()) for name, mask in nested_masks.items()},
        "hydroproto_correct_mlp_wrong": int(hproto_win.sum()),
        "all_five_models_wrong": int(all_wrong.sum()),
        "hydroproto_unique_correct_all_others_wrong": int(unique_hproto.sum()),
        "bootstrap_iterations_per_model_bin": args.bootstrap_iterations,
        "source_test_predictions_reused": True,
        "models_retrained": False,
    }
    (output / "analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
