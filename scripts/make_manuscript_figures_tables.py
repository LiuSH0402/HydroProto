"""Generate publication-ready main figures and tables for the HydroProto manuscript.

Outputs
-------
modeling/manuscript/图表/
    figures/Figure2..Figure5/{individual panels, combined JPG/PDF}
    tables/Table1..Table4.csv
    source_data/*.csv

All plots are generated with Python/matplotlib. JPG files are exported at 600 dpi;
PDF files retain vector text and drawing primitives (Type 42 fonts).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "modeling" / "result"
OUT = ROOT / "modeling" / "manuscript" / "图表"
FIG_OUT = OUT / "figures"
TABLE_OUT = OUT / "tables"
SOURCE_OUT = OUT / "source_data"

for directory in (FIG_OUT, TABLE_OUT, SOURCE_OUT):
    directory.mkdir(parents=True, exist_ok=True)


# EZSpecificity-inspired palette, adapted for the manuscript models.
COLORS = {
    "HydroProto": "#388D8F",
    "ESM2+MLP": "#4D4D4D",
    "HydroLocal": "#5FC3CE",
    "HydroMPC": "#78A604",
    "CLEAN": "#2788BE",
    "MMseqs2": "#B8B8B8",
    "coral": "#E76356",
    "gold": "#C8941B",
    "ink": "#262626",
    "grid": "#D9D9D9",
    "paper": "#FFFFFF",
}

# Canonical manuscript-wide model order. Every figure, legend, heatmap, and
# results table must preserve this order (or its relative order when only a
# subset of models is displayed).
MODEL_ORDER = ["HydroProto", "ESM2+MLP", "HydroLocal", "HydroMPC", "CLEAN", "MMseqs2"]
DL_MODEL_ORDER = [model for model in MODEL_ORDER if model != "MMseqs2"]
CORE_MODEL_ORDER = [model for model in MODEL_ORDER if model in {"HydroProto", "ESM2+MLP", "HydroLocal", "HydroMPC"}]
CLASS_ORDER = [
    "EC3.1_ester_bonds",
    "EC3.2_glycosylases",
    "EC3.4_peptidases",
    "EC3.5_CN_nonpeptide",
    "EC3_other",
]
CLASS_LABELS = {
    "EC3.1_ester_bonds": "EC 3.1\nEster bonds",
    "EC3.2_glycosylases": "EC 3.2\nGlycosylases",
    "EC3.4_peptidases": "EC 3.4\nPeptidases",
    "EC3.5_CN_nonpeptide": "EC 3.5\nC–N bonds",
    "EC3_other": "Other EC 3",
}
SHORT_CLASS = {
    "EC3.1_ester_bonds": "EC 3.1",
    "EC3.2_glycosylases": "EC 3.2",
    "EC3.4_peptidases": "EC 3.4",
    "EC3.5_CN_nonpeptide": "EC 3.5",
    "EC3_other": "Other EC 3",
}

NAME_MAP = {
    "H-Proto": "HydroProto",
    "H-Local": "HydroLocal",
    "Hydro-MPC": "HydroMPC",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    }
)

TEAL_CMAP = LinearSegmentedColormap.from_list(
    "paper_teal", ["#FFFFFF", "#D7EEF0", "#82C7CB", COLORS["HydroProto"]]
)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "paper_diverging", [COLORS["CLEAN"], "#F7F7F7", COLORS["coral"]]
)


def canonical_model(value: str) -> str:
    return NAME_MAP.get(value, value)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def clean_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS["ink"],
    )


def export_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        base.with_suffix(".jpg"),
        dpi=600,
        pil_kwargs={"quality": 96, "subsampling": 0},
        facecolor="white",
    )
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def export_panel(
    plotter: Callable[[plt.Axes], None],
    base: Path,
    size: tuple[float, float] = (3.5, 2.8),
) -> None:
    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    plotter(ax)
    export_figure(fig, base)


def write_source(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(SOURCE_OUT / filename, index=False, encoding="utf-8-sig")


def write_titled_csv(
    filename: str,
    title: str,
    notes: str,
    df: pd.DataFrame,
) -> None:
    path = TABLE_OUT / filename
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([title])
        writer.writerow([f"Notes: {notes}"])
        writer.writerow([])
        writer.writerow(list(df.columns))
        for row in df.itertuples(index=False, name=None):
            writer.writerow(row)


def fmt_mean_sd(mean: float, sd: float) -> str:
    return f"{mean:.4f} ± {sd:.4f}"


def fmt_ci(est: float, low: float, high: float) -> str:
    return f"{est:.4f} [{low:.4f}, {high:.4f}]"


def metric_row(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, float]:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y_true)
    class_indices = [np.flatnonzero(y_true == cls) for cls in classes]
    metrics = {name: [] for name in ("accuracy", "balanced_accuracy", "macro_f1", "mcc")}
    for _ in range(iterations):
        idx = np.concatenate([rng.choice(ids, size=len(ids), replace=True) for ids in class_indices])
        vals = metric_row(y_true[idx], y_pred[idx])
        for name in metrics:
            metrics[name].append(vals[name])
    point = metric_row(y_true, y_pred)
    return {
        name: (
            point[name],
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for name, values in metrics.items()
    }


def holm_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def paired_bootstrap_delta(
    y_true: np.ndarray,
    ref_pred: np.ndarray,
    cmp_pred: np.ndarray,
    iterations: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y_true)
    class_indices = [np.flatnonzero(y_true == cls) for cls in classes]
    values = []
    for _ in range(iterations):
        idx = np.concatenate([rng.choice(ids, size=len(ids), replace=True) for ids in class_indices])
        values.append(
            f1_score(y_true[idx], ref_pred[idx], average="macro", zero_division=0)
            - f1_score(y_true[idx], cmp_pred[idx], average="macro", zero_division=0)
        )
    point = f1_score(y_true, ref_pred, average="macro", zero_division=0) - f1_score(
        y_true, cmp_pred, average="macro", zero_division=0
    )
    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def load_cv_data() -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    folder_map = {
        "ESM2+MLP": "ESM2-MLP",
        "HydroLocal": "HydroLocal",
        "HydroProto": "HydroProto",
        "HydroMPC": "HydroMPC",
    }
    selected = {"ESM2+MLP": 3, "HydroLocal": 6, "HydroProto": 4, "HydroMPC": 6}
    locked_epochs = {"ESM2+MLP": 11, "HydroLocal": 11, "HydroProto": 23, "HydroMPC": 20}
    frames = []
    for model, folder in folder_map.items():
        for depth in range(1, 7):
            path = RESULT / "03_esm2_650m_cv_depth_sweep" / folder / f"unfreeze_top{depth}" / "fold_metrics.csv"
            if not path.exists() and depth == 4 and model in {"ESM2+MLP", "HydroProto"}:
                legacy = "esm2_mlp" if model == "ESM2+MLP" else "h_proto"
                path = (
                    RESULT
                    / "10_legacy_and_smoke_runs"
                    / "unfreeze4_four_models"
                    / legacy
                    / "unfreeze_top4"
                    / "fold_metrics.csv"
                )
            frame = read_csv(path)
            frame["model"] = model
            frame["unfreeze_layers"] = depth
            frames.append(frame)
    return pd.concat(frames, ignore_index=True), selected, locked_epochs


def load_prediction_audit() -> pd.DataFrame:
    path = (
        RESULT
        / "08_error_and_mechanism_analysis"
        / "similarity_error_canonical"
        / "tables"
        / "test_record_error_audit.csv"
    )
    audit = read_csv(path)
    if "true_label" not in audit.columns and "label" in audit.columns:
        audit = audit.rename(columns={"label": "true_label"})
    return audit


def identify_prediction_columns(audit: pd.DataFrame) -> dict[str, str]:
    candidates = {}
    for col in audit.columns:
        if not col.startswith("prediction_"):
            continue
        raw = col[len("prediction_") :]
        candidates[canonical_model(raw)] = col
    required = set(DL_MODEL_ORDER)
    missing = required - set(candidates)
    if missing:
        raise ValueError(f"Missing prediction columns for: {sorted(missing)}")
    return candidates


def augment_with_mmseq(audit: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    pred_cols = identify_prediction_columns(audit)
    mm = read_csv(RESULT / "06_external_baselines" / "MMseqs2_report" / "predictions.csv")
    mm = mm.rename(
        columns={
            "predicted_label": "prediction_MMseqs2",
            "has_coverage_qualified_hit": "mmseq_has_hit",
        }
    )
    cols = ["accession", "prediction_MMseqs2", "mmseq_has_hit"]
    merged = audit.merge(mm[cols], on="accession", how="left", validate="one_to_one")
    if merged["prediction_MMseqs2"].isna().any():
        raise ValueError("MMseqs2 predictions did not merge completely")
    pred_cols["MMseqs2"] = "prediction_MMseqs2"
    return merged, pred_cols


def figure2(cv: pd.DataFrame, selected: dict[str, int]) -> None:
    # Lead with the manuscript's proposed method, followed by the strong
    # baseline and the two architectural controls. Keep this order consistent
    # with the model ordering used in the downstream comparison figures/tables.
    model_panels = CORE_MODEL_ORDER

    def plot_depth(ax: plt.Axes, model: str) -> None:
        sub = cv[cv["model"] == model]
        summary = sub.groupby("unfreeze_layers")["macro_f1"].agg(["mean", "std"]).reset_index()
        for depth, group in sub.groupby("unfreeze_layers"):
            offsets = np.linspace(-0.09, 0.09, len(group))
            ax.scatter(
                depth + offsets,
                group["macro_f1"],
                s=15,
                facecolors="white",
                edgecolors=COLORS[model],
                linewidths=0.75,
                alpha=0.9,
                zorder=3,
            )
        ax.errorbar(
            summary["unfreeze_layers"],
            summary["mean"],
            yerr=summary["std"],
            color=COLORS[model],
            marker="o",
            markersize=4.3,
            linewidth=1.45,
            elinewidth=1.0,
            capsize=2.2,
            zorder=4,
        )
        chosen = selected[model]
        selected_row = summary[summary["unfreeze_layers"] == chosen].iloc[0]
        ax.scatter(
            [chosen],
            [selected_row["mean"]],
            s=58,
            marker="*",
            color=COLORS["gold"],
            edgecolor="white",
            linewidth=0.5,
            zorder=6,
        )
        ax.annotate(
            f"Selected: {chosen}",
            (chosen, selected_row["mean"]),
            xytext=(0, 16),
            textcoords="offset points",
            fontsize=6.8,
            color=COLORS["gold"],
            ha="center",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.7},
            zorder=8,
        )
        ax.set_xticks(range(1, 7))
        ax.set_ylim(0.91, 0.960)
        ax.set_xlabel("Unfrozen ESM-2 layers")
        ax.set_ylabel("Validation Macro-F1")
        ax.set_title(model, loc="left", fontweight="bold", color=COLORS[model])
        clean_axis(ax, "y")

    fig_dir = FIG_OUT / "Figure2"
    individual_dir = fig_dir / "individual"
    for suffix in ("jpg", "pdf"):
        for stale_panel in individual_dir.glob(f"Figure2[abcd]_*.{suffix}"):
            stale_panel.unlink()
    for idx, model in enumerate(model_panels):
        export_panel(
            lambda ax, m=model: plot_depth(ax, m),
            individual_dir / f"Figure2{chr(97 + idx)}_{model.replace('+', '_').replace(' ', '_')}",
        )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), constrained_layout=True, sharey=True)
    for idx, (ax, model) in enumerate(zip(axes.flat, model_panels)):
        plot_depth(ax, model)
        panel_label(ax, chr(97 + idx))
    export_figure(fig, fig_dir / "Figure2_combined")

    src = cv[
        ["model", "unfreeze_layers", "fold", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc"]
    ].copy()
    src["selected_depth"] = src.apply(lambda r: int(r["unfreeze_layers"]) == selected[r["model"]], axis=1)
    write_source(src, "Figure2_cv_depth_sweep.csv")


def build_overall_statistics(
    audit: pd.DataFrame, pred_cols: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    boot = read_csv(
        RESULT / "07_statistical_analysis" / "final_statistics_with_CLEAN" / "bootstrap_metric_ci.csv"
    )
    boot["model"] = boot["model"].map(canonical_model)
    boot = boot[boot["subset"] == "test_all"].copy()

    y_true = audit["true_label"].to_numpy()
    mm_pred = audit[pred_cols["MMseqs2"]].to_numpy()
    mm_ci = stratified_bootstrap_ci(y_true, mm_pred)
    mm_rows = []
    for metric, (estimate, low, high) in mm_ci.items():
        mm_rows.append(
            {
                "subset": "test_all",
                "n": len(audit),
                "model": "MMseqs2",
                "metric": metric,
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "iterations": 2000,
            }
        )
    boot = pd.concat([boot, pd.DataFrame(mm_rows)], ignore_index=True)

    class_rows = []
    for model in MODEL_ORDER:
        p, r, f, support = precision_recall_fscore_support(
            y_true,
            audit[pred_cols[model]].to_numpy(),
            labels=CLASS_ORDER,
            zero_division=0,
        )
        for cls, pp, rr, ff, nn in zip(CLASS_ORDER, p, r, f, support):
            class_rows.append(
                {"model": model, "class": cls, "precision": pp, "recall": rr, "f1": ff, "support": nn}
            )
    class_metrics = pd.DataFrame(class_rows)

    ref = audit[pred_cols["HydroProto"]].to_numpy()
    stats_dir = RESULT / "07_statistical_analysis" / "final_statistics_with_CLEAN"
    paired = read_csv(stats_dir / "bootstrap_paired_delta_ci.csv")
    paired["reference_model"] = paired["reference_model"].map(canonical_model)
    paired["comparator"] = paired["comparator"].map(canonical_model)
    paired = paired[
        (paired["subset"] == "test_all")
        & (paired["reference_model"] == "HydroProto")
        & (paired["metric"] == "macro_f1")
    ]
    mcnemar = read_csv(stats_dir / "mcnemar_pairwise_holm.csv")
    mcnemar["model_left"] = mcnemar["model_left"].map(canonical_model)
    mcnemar["model_right"] = mcnemar["model_right"].map(canonical_model)
    mcnemar = mcnemar[mcnemar["subset"] == "test_all"]

    comparison_rows = []
    for model in [m for m in DL_MODEL_ORDER if m != "HydroProto"]:
        delta_row = paired[paired["comparator"] == model].iloc[0]
        p_row = mcnemar[
            ((mcnemar["model_left"] == "HydroProto") & (mcnemar["model_right"] == model))
            | ((mcnemar["model_right"] == "HydroProto") & (mcnemar["model_left"] == model))
        ].iloc[0]
        comparison_rows.append(
            {
                "comparator": model,
                "macro_f1_delta": delta_row["estimate_delta"],
                "delta_ci95_low": delta_row["ci95_low"],
                "delta_ci95_high": delta_row["ci95_high"],
                "hydroproto_only_correct": np.nan,
                "comparator_only_correct": np.nan,
                "mcnemar_p_exact": p_row["p_value_exact"],
                "mcnemar_p_holm": p_row["p_value_holm"],
                "multiplicity_family": "All 10 pairwise comparisons among the five neural/deep-learning models",
            }
        )

    # MMseqs2 was not part of the archived five-model multiplicity family; report
    # its paired estimate and exact McNemar test as an explicitly exploratory row.
    mm_pred = audit[pred_cols["MMseqs2"]].to_numpy()
    delta, low, high = paired_bootstrap_delta(y_true, ref, mm_pred)
    ref_ok = ref == y_true
    mm_ok = mm_pred == y_true
    b = int(np.sum(ref_ok & ~mm_ok))
    c = int(np.sum(~ref_ok & mm_ok))
    pval = float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue) if b + c else 1.0
    comparison_rows.append(
        {
            "comparator": "MMseqs2",
            "macro_f1_delta": delta,
            "delta_ci95_low": low,
            "delta_ci95_high": high,
            "hydroproto_only_correct": b,
            "comparator_only_correct": c,
            "mcnemar_p_exact": pval,
            "mcnemar_p_holm": np.nan,
            "multiplicity_family": "Not adjusted; exploratory comparison",
        }
    )
    comparisons = pd.DataFrame(comparison_rows)
    return boot, class_metrics, comparisons


def figure3(
    audit: pd.DataFrame,
    pred_cols: dict[str, str],
    boot: pd.DataFrame,
    class_metrics: pd.DataFrame,
) -> None:
    comparison_model_order = MODEL_ORDER
    metric_names = ["accuracy", "macro_f1", "mcc"]
    metric_labels = {"accuracy": "Accuracy", "macro_f1": "Macro-F1", "mcc": "MCC"}
    metric_markers = {"accuracy": "o", "macro_f1": "s", "mcc": "^"}

    def plot_overall(ax: plt.Axes) -> None:
        y_positions = np.arange(len(comparison_model_order))[::-1]
        offsets = {"accuracy": 0.19, "macro_f1": 0.0, "mcc": -0.19}
        for model, y in zip(comparison_model_order, y_positions):
            for metric in metric_names:
                row = boot[(boot["model"] == model) & (boot["metric"] == metric)].iloc[0]
                ax.errorbar(
                    row["estimate"],
                    y + offsets[metric],
                    xerr=[[row["estimate"] - row["ci95_low"]], [row["ci95_high"] - row["estimate"]]],
                    fmt=metric_markers[metric],
                    color=COLORS[model],
                    markerfacecolor=COLORS[model],
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    markersize=4.4,
                    elinewidth=1.0,
                    capsize=2,
                    zorder=3,
                )
        ax.set_yticks(y_positions, comparison_model_order)
        for tick, model in zip(ax.get_yticklabels(), comparison_model_order):
            tick.set_color(COLORS[model])
            tick.set_fontweight("bold" if model == "HydroProto" else "normal")
        ax.set_xlim(0.72, 0.99)
        ax.set_xlabel("Performance (point estimate and 95% CI)")
        ax.set_title("Overall performance on the fixed test set (n = 680)", loc="left", fontweight="bold")
        handles = [
            Line2D([0], [0], marker=metric_markers[m], linestyle="none", color="#555555", label=metric_labels[m])
            for m in metric_names
        ]
        ax.legend(handles=handles, frameon=False, ncol=1, loc="lower right", handletextpad=0.4)
        clean_axis(ax, "x")

    def plot_class_heatmap(ax: plt.Axes) -> None:
        matrix = (
            class_metrics.pivot(index="model", columns="class", values="f1")
            .reindex(index=comparison_model_order, columns=CLASS_ORDER)
            .to_numpy()
        )
        mesh = ax.pcolormesh(matrix, cmap=TEAL_CMAP, vmin=0.72, vmax=1.0, edgecolors="white", linewidth=0.8)
        ax.set_xticks(np.arange(5) + 0.5, [SHORT_CLASS[c] for c in CLASS_ORDER], rotation=35, ha="right")
        ax.set_yticks(np.arange(6) + 0.5, comparison_model_order)
        ax.invert_yaxis()
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                color = "white" if matrix[i, j] > 0.945 else COLORS["ink"]
                ax.text(j + 0.5, i + 0.5, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=6.3, color=color)
        ax.set_title("Class-specific F1", loc="left", fontweight="bold")
        cb = plt.colorbar(mesh, ax=ax, fraction=0.045, pad=0.025)
        cb.set_label("F1")
        cb.outline.set_linewidth(0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cm = read_csv(
        RESULT
        / "08_error_and_mechanism_analysis"
        / "similarity_error_canonical"
        / "tables"
        / "hydroproto_confusion_matrices.csv"
    )

    def plot_confusion(ax: plt.Axes) -> None:
        sub = cm[cm["subset"] == "test_all"]
        matrix = (
            sub.pivot(index="true_label", columns="predicted_label", values="row_fraction")
            .reindex(index=CLASS_ORDER, columns=CLASS_ORDER)
            .fillna(0)
            .to_numpy()
        )
        counts = (
            sub.pivot(index="true_label", columns="predicted_label", values="count")
            .reindex(index=CLASS_ORDER, columns=CLASS_ORDER)
            .fillna(0)
            .to_numpy()
        )
        mesh = ax.pcolormesh(matrix, cmap=TEAL_CMAP, vmin=0, vmax=1, edgecolors="white", linewidth=0.8)
        ax.set_xticks(np.arange(5) + 0.5, [SHORT_CLASS[c] for c in CLASS_ORDER], rotation=35, ha="right")
        ax.set_yticks(np.arange(5) + 0.5, [SHORT_CLASS[c] for c in CLASS_ORDER])
        ax.invert_yaxis()
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        for i in range(5):
            for j in range(5):
                color = "white" if matrix[i, j] > 0.62 else COLORS["ink"]
                ax.text(j + 0.5, i + 0.5, f"{int(counts[i,j])}\n{matrix[i,j]:.2f}", ha="center", va="center", fontsize=6.0, color=color)
        ax.set_title("HydroProto confusion matrix", loc="left", fontweight="bold")
        cb = plt.colorbar(mesh, ax=ax, fraction=0.045, pad=0.025)
        cb.set_label("Row fraction")
        cb.outline.set_linewidth(0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def plot_complementarity(ax: plt.Axes) -> None:
        true = audit["true_label"].to_numpy()
        hp = audit[pred_cols["HydroProto"]].to_numpy() == true
        mlp = audit[pred_cols["ESM2+MLP"]].to_numpy() == true
        # Rows: MLP correct/wrong; columns: HydroProto correct/wrong.
        counts = np.array(
            [
                [int(np.sum(hp & mlp)), int(np.sum(~hp & mlp))],
                [int(np.sum(hp & ~mlp)), int(np.sum(~hp & ~mlp))],
            ]
        )
        cell_colors = np.array(
            [
                [mpl.colors.to_rgba(COLORS["HydroProto"]), mpl.colors.to_rgba(COLORS["CLEAN"])],
                [mpl.colors.to_rgba(COLORS["gold"]), mpl.colors.to_rgba(COLORS["coral"])],
            ]
        )
        for i in range(2):
            for j in range(2):
                ax.add_patch(
                    mpl.patches.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=cell_colors[i, j],
                        edgecolor="white",
                        linewidth=1.2,
                    )
                )
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)
        for i in range(2):
            for j in range(2):
                count = counts[i, j]
                text_color = COLORS["ink"] if (i, j) == (1, 0) else "white"
                ax.text(j, i, f"{count}\n({count/len(audit):.1%})", ha="center", va="center", fontsize=9, color=text_color, fontweight="bold")
        ax.set_xticks([0, 1], ["HydroProto\ncorrect", "HydroProto\nwrong"])
        ax.set_yticks([0, 1], ["MLP correct", "MLP wrong"])
        ax.set_xlabel("HydroProto outcome")
        ax.set_ylabel("ESM2+MLP outcome")
        for x in [-0.5, 0.5, 1.5]:
            ax.axvline(x, color="white", linewidth=1.2)
        for y in [-0.5, 0.5, 1.5]:
            ax.axhline(y, color="white", linewidth=1.2)
        ax.set_title("HydroProto–MLP prediction complementarity", loc="left", fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)

    plotters = [plot_overall, plot_class_heatmap, plot_confusion, plot_complementarity]
    sizes = [(4.0, 3.0), (4.0, 3.0), (4.0, 3.3), (4.0, 2.5)]
    fig_dir = FIG_OUT / "Figure3"
    for idx, (plotter, size) in enumerate(zip(plotters, sizes)):
        export_panel(plotter, fig_dir / "individual" / f"Figure3{chr(97 + idx)}", size)

    fig = plt.figure(figsize=(7.4, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.06, 1.0])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for idx, (ax, plotter) in enumerate(zip(axes, plotters)):
        plotter(ax)
        panel_label(ax, chr(97 + idx))
    export_figure(fig, fig_dir / "Figure3_combined")

    write_source(boot, "Figure3a_overall_metrics_ci.csv")
    write_source(class_metrics, "Figure3b_per_class_metrics.csv")
    write_source(cm, "Figure3c_confusion_matrices.csv")


def build_similarity_data(
    audit: pd.DataFrame, pred_cols: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = read_csv(
        RESULT
        / "08_error_and_mechanism_analysis"
        / "similarity_error_canonical"
        / "tables"
        / "metrics_nested_and_nonoverlap.csv"
    )
    original["model"] = original["model"].map(canonical_model)
    original = original[original["model"].isin(DL_MODEL_ORDER)].copy()

    bin_masks = {
        ("nonoverlap", "ge80"): audit["similarity_bin"].eq("ge80"),
        ("nonoverlap", "60_to_lt80"): audit["similarity_bin"].eq("60_to_lt80"),
        ("nonoverlap", "40_to_lt60"): audit["similarity_bin"].eq("40_to_lt60"),
        ("nonoverlap", "lt40"): audit["similarity_bin"].eq("lt40"),
        ("nested", "test_all"): np.ones(len(audit), dtype=bool),
        ("nested", "lt80"): audit["similarity_bin"].isin(["60_to_lt80", "40_to_lt60", "lt40"]),
        ("nested", "lt60"): audit["similarity_bin"].isin(["40_to_lt60", "lt40"]),
        ("nested", "lt40"): audit["similarity_bin"].eq("lt40"),
    }
    mm_rows = []
    for (analysis, bin_name), mask in bin_masks.items():
        sub = audit.loc[mask]
        vals = metric_row(sub["true_label"], sub[pred_cols["MMseqs2"]])
        mm_rows.append(
            {
                "analysis": analysis,
                "bin": bin_name,
                "bin_label": bin_name,
                "model": "MMseqs2",
                "n": len(sub),
                **vals,
                "macro_f1_ci95_low": np.nan,
                "macro_f1_ci95_high": np.nan,
            }
        )
    all_metrics = pd.concat([original, pd.DataFrame(mm_rows)], ignore_index=True)

    diagnostic_rows = []
    diagnostic_masks = {
        "≥80%": audit["similarity_bin"].eq("ge80"),
        "60–80%": audit["similarity_bin"].eq("60_to_lt80"),
        "40–60%": audit["similarity_bin"].eq("40_to_lt60"),
        "<40% with hit": audit["similarity_bin"].eq("lt40") & audit["has_coverage_qualified_hit"].astype(bool),
        "No qualified hit": audit["similarity_bin"].eq("lt40") & ~audit["has_coverage_qualified_hit"].astype(bool),
    }
    for label, mask in diagnostic_masks.items():
        for model in MODEL_ORDER:
            sub = audit.loc[mask]
            vals = metric_row(sub["true_label"], sub[pred_cols[model]])
            diagnostic_rows.append({"bin": label, "model": model, "n": len(sub), **vals})
    diagnostic = pd.DataFrame(diagnostic_rows)
    return all_metrics, diagnostic


def figure4(
    audit: pd.DataFrame,
    pred_cols: dict[str, str],
    similarity: pd.DataFrame,
    diagnostic: pd.DataFrame,
) -> None:
    plot_model_order = MODEL_ORDER
    non_bins = ["ge80", "60_to_lt80", "40_to_lt60", "lt40"]
    non_labels = ["≥80", "60–80", "40–60", "<40"]
    nested_bins = ["test_all", "lt80", "lt60", "lt40"]
    nested_labels = ["All", "<80", "<60", "<40"]

    def plot_lines(ax: plt.Axes, analysis: str) -> None:
        bins = non_bins if analysis == "nonoverlap" else nested_bins
        labels = non_labels if analysis == "nonoverlap" else nested_labels
        for model in plot_model_order:
            sub = similarity[(similarity["analysis"] == analysis) & (similarity["model"] == model)].set_index("bin").reindex(bins)
            ax.plot(
                range(len(bins)),
                sub["macro_f1"],
                marker="o",
                markersize=3.8,
                linewidth=1.35,
                color=COLORS[model],
                label=model,
                zorder=3 if model == "HydroProto" else 2,
            )
        counts = [int(similarity[(similarity["analysis"] == analysis) & (similarity["bin"] == b)]["n"].iloc[0]) for b in bins]
        ax.set_xticks(range(len(bins)), [f"{label}%\n(n={n})" if label != "All" else f"All\n(n={n})" for label, n in zip(labels, counts)])
        ax.set_ylim(0.45, 1.01)
        ax.set_ylabel("Macro-F1")
        ax.set_xlabel("Maximum train–test sequence identity")
        ax.set_title(
            "Non-overlapping identity intervals" if analysis == "nonoverlap" else "Nested low-identity stress tests",
            loc="left",
            fontweight="bold",
        )
        clean_axis(ax, "y")
        ax.legend(frameon=False, ncol=2, loc="lower left", fontsize=6.4)

    def plot_hit_split(ax: plt.Axes) -> None:
        labels = ["No qualified hit", "<40% with hit"]
        x = np.arange(len(labels))
        width = 0.12
        offsets = (np.arange(len(plot_model_order)) - (len(plot_model_order) - 1) / 2) * width

        # A true broken y-axis preserves the very low MMseqs2 no-hit result
        # without compressing differences among the neural models.
        ax.set_axis_off()
        upper = ax.inset_axes([0.0, 0.22, 1.0, 0.78])
        lower = ax.inset_axes([0.0, 0.0, 1.0, 0.13])
        for model, offset in zip(plot_model_order, offsets):
            values = [float(diagnostic[(diagnostic["bin"] == label) & (diagnostic["model"] == model)]["macro_f1"].iloc[0]) for label in labels]
            upper.bar(x + offset, values, width=width * 0.9, color=COLORS[model], label=model)
            lower.bar(x + offset, values, width=width * 0.9, color=COLORS[model])
            for xpos, value in zip(x + offset, values):
                if value >= 0.70:
                    upper.text(
                        xpos,
                        value + 0.004,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=5.3,
                        color="#222222",
                        fontweight="bold",
                    )

        mmseq_no_hit = float(
            diagnostic[
                (diagnostic["bin"] == "No qualified hit")
                & (diagnostic["model"] == "MMseqs2")
            ]["macro_f1"].iloc[0]
        )
        mmseq_offset = offsets[plot_model_order.index("MMseqs2")]
        lower.text(
            x[0] + mmseq_offset,
            mmseq_no_hit + 0.004,
            f"{mmseq_no_hit:.3f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=5.5,
            color="#222222",
            fontweight="bold",
        )

        counts = [int(diagnostic[diagnostic["bin"] == label]["n"].iloc[0]) for label in labels]
        upper.set_xlim(-0.55, 1.55)
        lower.set_xlim(-0.55, 1.55)
        upper.set_ylim(0.70, 1.02)
        lower.set_ylim(0.00, 0.12)
        upper.set_yticks([0.70, 0.80, 0.90, 1.00])
        lower.set_yticks([0.00, 0.10])
        upper.tick_params(axis="x", bottom=False, labelbottom=False)
        lower.set_xticks(x, [f"{label}\n(n={n})" for label, n in zip(labels, counts)])
        upper.set_title("Low-identity performance depends on hit availability", loc="left", fontweight="bold")
        upper.legend(
            frameon=False,
            ncol=2,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.98),
            fontsize=5.9,
            columnspacing=0.7,
            handlelength=1.2,
            labelspacing=0.35,
        )
        clean_axis(upper, "y")
        clean_axis(lower, "y")
        upper.spines["bottom"].set_visible(False)
        lower.spines["top"].set_visible(False)
        ax.text(-0.12, 0.50, "Macro-F1", transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=8.5)

        # Paired diagonal marks show the omitted interval (0.12–0.70).
        for x0 in (-0.020, 0.004):
            upper.plot(
                [x0, x0 + 0.014],
                [-0.012, 0.012],
                transform=upper.transAxes,
                color="#666666",
                linewidth=0.8,
                clip_on=False,
            )
            lower.plot(
                [x0, x0 + 0.014],
                [0.988, 1.012],
                transform=lower.transAxes,
                color="#666666",
                linewidth=0.8,
                clip_on=False,
            )

    cm = read_csv(
        RESULT
        / "08_error_and_mechanism_analysis"
        / "similarity_error_canonical"
        / "tables"
        / "hydroproto_confusion_matrices.csv"
    )

    def plot_lt40_confusion(ax: plt.Axes) -> None:
        sub = cm[cm["subset"] == "lt40"]
        matrix = sub.pivot(index="true_label", columns="predicted_label", values="row_fraction").reindex(index=CLASS_ORDER, columns=CLASS_ORDER).fillna(0).to_numpy()
        counts = sub.pivot(index="true_label", columns="predicted_label", values="count").reindex(index=CLASS_ORDER, columns=CLASS_ORDER).fillna(0).to_numpy()
        mesh = ax.pcolormesh(matrix, cmap=TEAL_CMAP, vmin=0, vmax=1, edgecolors="white", linewidth=0.8)
        ax.set_xticks(np.arange(5) + 0.5, [SHORT_CLASS[c] for c in CLASS_ORDER], rotation=35, ha="right")
        ax.set_yticks(np.arange(5) + 0.5, [SHORT_CLASS[c] for c in CLASS_ORDER])
        ax.invert_yaxis()
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        for i in range(5):
            for j in range(5):
                color = "white" if matrix[i, j] > 0.62 else COLORS["ink"]
                ax.text(j + 0.5, i + 0.5, f"{int(counts[i,j])}\n{matrix[i,j]:.2f}", ha="center", va="center", fontsize=5.8, color=color)
        ax.set_title("HydroProto confusion matrix (<40%)", loc="left", fontweight="bold")
        cb = plt.colorbar(mesh, ax=ax, fraction=0.045, pad=0.025)
        cb.set_label("Row fraction")
        cb.outline.set_linewidth(0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    per_class = read_csv(
        RESULT
        / "08_error_and_mechanism_analysis"
        / "similarity_error_canonical"
        / "tables"
        / "per_class_metrics_all_models.csv"
    )
    per_class["model"] = per_class["model"].map(canonical_model)

    def plot_class_delta(ax: plt.Axes) -> None:
        hp = per_class[per_class["model"] == "HydroProto"]
        all_rows = hp[(hp["analysis"] == "nested") & (hp["bin"] == "test_all")].set_index("class")
        low_rows = hp[(hp["analysis"] == "nested") & (hp["bin"] == "lt40")].set_index("class")
        matrix = np.column_stack(
            [low_rows.loc[CLASS_ORDER, metric].to_numpy() - all_rows.loc[CLASS_ORDER, metric].to_numpy() for metric in ["precision", "recall", "f1"]]
        )
        lim = max(0.12, float(np.nanmax(np.abs(matrix))))
        mesh = ax.pcolormesh(matrix, cmap=DIVERGING_CMAP, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), edgecolors="white", linewidth=0.8)
        ax.set_xticks(np.arange(3) + 0.5, ["Precision", "Recall", "F1"])
        ax.set_yticks(np.arange(5) + 0.5, [SHORT_CLASS[c] for c in CLASS_ORDER])
        ax.invert_yaxis()
        for i in range(5):
            for j in range(3):
                ax.text(j + 0.5, i + 0.5, f"{matrix[i,j]:+.3f}", ha="center", va="center", fontsize=6.4, color=COLORS["ink"])
        ax.set_title("Class-wise change at <40% identity", loc="left", fontweight="bold")
        cb = plt.colorbar(mesh, ax=ax, fraction=0.045, pad=0.025)
        cb.set_label("<40% minus all")
        cb.outline.set_linewidth(0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def plot_mmseq(ax: plt.Axes) -> None:
        masks = [
            np.ones(len(audit), dtype=bool),
            audit["similarity_bin"].isin(["60_to_lt80", "40_to_lt60", "lt40"]),
            audit["similarity_bin"].isin(["40_to_lt60", "lt40"]),
            audit["similarity_bin"].eq("lt40"),
        ]
        coverage, hit_accuracy, full_accuracy = [], [], []
        for mask in masks:
            sub = audit.loc[mask]
            hit = sub["mmseq_has_hit"].astype(bool)
            coverage.append(float(hit.mean()))
            hit_accuracy.append(accuracy_score(sub.loc[hit, "true_label"], sub.loc[hit, pred_cols["MMseqs2"]]))
            full_accuracy.append(accuracy_score(sub["true_label"], sub[pred_cols["MMseqs2"]]))
        x = np.arange(4)
        ax.plot(x, coverage, marker="o", color=COLORS["gold"], linewidth=1.5, label="Qualified-hit coverage")
        ax.plot(x, hit_accuracy, marker="s", color=COLORS["CLEAN"], linewidth=1.5, label="Hit-only accuracy")
        ax.plot(x, full_accuracy, marker="^", color=COLORS["MMseqs2"], markeredgecolor="#777777", linewidth=1.5, label="Full-set accuracy (fallback)")
        ax.set_xticks(x, nested_labels)
        ax.set_ylim(0.35, 1.02)
        ax.set_ylabel("Fraction")
        ax.set_xlabel("Nested test subset")
        ax.set_title("MMseqs2 coverage–accuracy trade-off", loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=6.4, loc="lower left")
        clean_axis(ax, "y")

    # Keep the panel order aligned with the Results narrative:
    # global stress tests -> interval localization -> class-level errors ->
    # hit availability -> MMseqs2 coverage/accuracy interpretation.
    plotters = [
        lambda ax: plot_lines(ax, "nested"),
        lambda ax: plot_lines(ax, "nonoverlap"),
        plot_lt40_confusion,
        plot_class_delta,
        plot_hit_split,
        plot_mmseq,
    ]
    sizes = [(4.0, 3.0), (4.0, 3.0), (4.0, 3.3), (4.0, 3.0), (4.0, 3.0), (4.0, 3.0)]
    fig_dir = FIG_OUT / "Figure4"
    for idx, (plotter, size) in enumerate(zip(plotters, sizes)):
        export_panel(plotter, fig_dir / "individual" / f"Figure4{chr(97 + idx)}", size)

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4), constrained_layout=True)
    for idx, (ax, plotter) in enumerate(zip(axes.flat, plotters)):
        plotter(ax)
        panel_label(ax, chr(97 + idx))
    export_figure(fig, fig_dir / "Figure4_combined")

    write_source(similarity, "Figure4ab_similarity_strata_metrics.csv")
    write_source(diagnostic, "Figure4e_identity_hit_diagnostics.csv")


def load_k_ablation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for k in [1, 3, 5, 7]:
        frame = read_csv(RESULT / "05_prototype_number_ablation" / "local_statistics" / f"k{k}" / "fold_metrics.csv")
        frame["K"] = k
        frames.append(frame)
    folds = pd.concat(frames, ignore_index=True)
    tests = read_csv(
        RESULT / "05_prototype_number_ablation" / "local_statistics" / "statistics" / "fold_paired_tests.csv"
    )
    boot = read_csv(
        RESULT / "05_prototype_number_ablation" / "local_statistics" / "statistics" / "paired_bootstrap_delta.csv"
    )
    return folds, tests, boot


def figure5(kfolds: pd.DataFrame, ktests: pd.DataFrame, kboot: pd.DataFrame) -> None:
    mechanism = RESULT / "08_error_and_mechanism_analysis" / "HydroProto_mechanism_canonical" / "tables"
    matrix_df = read_csv(mechanism / "prototype_cosine_matrix.csv")
    matrix_labels = matrix_df.iloc[:, 0].astype(str).tolist()
    matrix = matrix_df.iloc[:, 1:].to_numpy(float)
    geometry = read_csv(mechanism / "prototype_geometry_summary.csv")
    occupancy = read_csv(mechanism / "prototype_occupancy_train_test_lt40.csv")
    assignments = read_csv(mechanism / "test_assignments_with_outcomes.csv")

    def plot_k(ax: plt.Axes) -> None:
        summary = kfolds.groupby("K")["macro_f1"].agg(["mean", "std"]).reset_index()
        for k, group in kfolds.groupby("K"):
            offsets = np.linspace(-0.07, 0.07, len(group))
            ax.scatter(k + offsets, group["macro_f1"], s=18, facecolors="white", edgecolors=COLORS["HydroProto"], linewidths=0.8, zorder=3)
        ax.errorbar(summary["K"], summary["mean"], yerr=summary["std"], color=COLORS["HydroProto"], marker="o", markersize=4.5, linewidth=1.5, capsize=2.5, zorder=4)
        k5 = summary[summary["K"] == 5].iloc[0]
        ax.scatter([5], [k5["mean"]], marker="*", s=65, color=COLORS["gold"], edgecolor="white", linewidth=0.5, zorder=5)
        p = float(ktests[(ktests["metric"] == "macro_f1") & (ktests["comparison"] == "K5-K1")]["wilcoxon_exact_p"].iloc[0])
        friedman = float(ktests[ktests["metric"] == "macro_f1"]["friedman_p_all_K"].iloc[0])
        delta = kboot[(kboot["metric"] == "macro_f1") & (kboot["comparison"] == "K5-K1")].iloc[0]
        ax.text(0.03, 0.04, f"Friedman P = {friedman:.3f}\nK=5 − K=1: {delta['mean_bootstrap_delta']:+.4f}\n95% CI [{delta['ci95_low']:+.4f}, {delta['ci95_high']:+.4f}]", transform=ax.transAxes, fontsize=6.7, va="bottom")
        ax.set_xticks([1, 3, 5, 7])
        ax.set_ylim(0.91, 0.96)
        ax.set_xlabel("Prototypes per class (K)")
        ax.set_ylabel("Validation Macro-F1")
        ax.set_title("Prototype-number ablation", loc="left", fontweight="bold")
        clean_axis(ax, "y")

    def plot_matrix(ax: plt.Axes) -> None:
        mesh = ax.pcolormesh(matrix, cmap=DIVERGING_CMAP, norm=TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=1.0), edgecolors="none")
        centers = np.arange(2.5, 25, 5)
        ax.set_xticks(centers, ["EC 3.1", "EC 3.2", "EC 3.4", "EC 3.5", "Other"], rotation=35, ha="right")
        ax.set_yticks(centers, ["EC 3.1", "EC 3.2", "EC 3.4", "EC 3.5", "Other"])
        ax.invert_yaxis()
        for boundary in [5, 10, 15, 20]:
            ax.axhline(boundary, color="white", linewidth=1.1)
            ax.axvline(boundary, color="white", linewidth=1.1)
        ax.set_xlabel("Prototype")
        ax.set_ylabel("Prototype")
        ax.set_title("Learned prototype cosine similarity", loc="left", fontweight="bold")
        cb = plt.colorbar(mesh, ax=ax, fraction=0.045, pad=0.025)
        cb.set_label("Cosine similarity")
        cb.outline.set_linewidth(0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def plot_geometry(ax: plt.Axes) -> None:
        rng = np.random.default_rng(13)
        within, between = [], []
        for i in range(25):
            for j in range(i + 1, 25):
                if i // 5 == j // 5:
                    within.append(matrix[i, j])
                else:
                    between.append(matrix[i, j])
        data = [within, between]
        bp = ax.boxplot(data, widths=0.52, patch_artist=True, showfliers=False, medianprops={"color": COLORS["ink"], "linewidth": 1.1})
        for patch, color in zip(bp["boxes"], [COLORS["HydroProto"], COLORS["CLEAN"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor(color)
        for i, values in enumerate(data, 1):
            sample = np.asarray(values)
            jitter = rng.normal(i, 0.055, len(sample))
            ax.scatter(jitter, sample, s=7, color=[COLORS["HydroProto"], COLORS["CLEAN"]][i - 1], alpha=0.45, linewidth=0)
        ax.axhline(0, color="#777777", linewidth=0.6, linestyle="--")
        ax.set_xticks([1, 2], [f"Within class\n(n={len(within)})", f"Between classes\n(n={len(between)})"])
        ax.set_ylabel("Cosine similarity")
        ax.set_title("Within- and between-class geometry", loc="left", fontweight="bold")
        clean_axis(ax, "y")

    def occupancy_plot(ax: plt.Axes, subset: str, title: str) -> None:
        sub = occupancy[occupancy["subset"] == subset]
        mat = sub.pivot(index="class", columns="prototype", values="class_fraction").reindex(index=CLASS_ORDER, columns=[1, 2, 3, 4, 5]).fillna(0).to_numpy()
        mesh = ax.pcolormesh(mat, cmap=TEAL_CMAP, vmin=0, vmax=1, edgecolors="white", linewidth=0.8)
        ax.set_xticks(np.arange(5) + 0.5, [f"P{i}" for i in range(1, 6)])
        ax.set_yticks(np.arange(5) + 0.5, [SHORT_CLASS[c] for c in CLASS_ORDER])
        ax.invert_yaxis()
        for i in range(5):
            for j in range(5):
                ax.text(j + 0.5, i + 0.5, f"{mat[i,j]:.1%}", ha="center", va="center", fontsize=6.0, color="white" if mat[i,j] > 0.58 else COLORS["ink"])
        ax.set_xlabel("Nearest prototype within the true class")
        ax.set_title(title, loc="left", fontweight="bold")
        cb = plt.colorbar(mesh, ax=ax, fraction=0.045, pad=0.025)
        cb.set_label("Class fraction")
        cb.outline.set_linewidth(0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def plot_distance(ax: plt.Axes) -> None:
        order = [
            "Both correct",
            "HydroProto correct / MLP wrong",
            "MLP correct / HydroProto wrong",
            "Both wrong",
        ]
        colors = [COLORS["HydroProto"], COLORS["gold"], COLORS["CLEAN"], COLORS["coral"]]
        data = [assignments.loc[assignments["outcome_group"] == label, "within_true_cosine_distance"].dropna().to_numpy() for label in order]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops={"color": COLORS["ink"], "linewidth": 1.1})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor(color)
        rng = np.random.default_rng(19)
        for i, (values, color) in enumerate(zip(data, colors), 1):
            take = values if len(values) <= 90 else rng.choice(values, 90, replace=False)
            ax.scatter(rng.normal(i, 0.06, len(take)), take, s=6, color=color, alpha=0.38, linewidth=0)
        ax.set_xticks(range(1, 5), [f"Both\ncorrect\n(n={len(data[0])})", f"HydroProto\nonly\n(n={len(data[1])})", f"MLP\nonly\n(n={len(data[2])})", f"Both\nwrong\n(n={len(data[3])})"])
        ax.set_ylabel("Distance to nearest true-class prototype")
        ax.set_title("Prototype distance and prediction outcome", loc="left", fontweight="bold")
        clean_axis(ax, "y")

    plotters = [
        plot_k,
        plot_matrix,
        plot_geometry,
        lambda ax: occupancy_plot(ax, "train", "Training-set prototype occupancy"),
        lambda ax: occupancy_plot(ax, "test_lt40", "Prototype occupancy at <40% identity"),
        plot_distance,
    ]
    sizes = [(4.0, 3.0), (4.0, 3.3), (4.0, 3.0), (4.0, 3.0), (4.0, 3.0), (4.0, 3.0)]
    fig_dir = FIG_OUT / "Figure5"
    for idx, (plotter, size) in enumerate(zip(plotters, sizes)):
        export_panel(plotter, fig_dir / "individual" / f"Figure5{chr(97 + idx)}", size)

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4), constrained_layout=True)
    for idx, (ax, plotter) in enumerate(zip(axes.flat, plotters)):
        plotter(ax)
        panel_label(ax, chr(97 + idx))
    export_figure(fig, fig_dir / "Figure5_combined")

    write_source(kfolds, "Figure5a_prototype_number_folds.csv")
    write_source(matrix_df, "Figure5b_prototype_cosine_matrix.csv")
    write_source(geometry, "Figure5c_prototype_geometry_summary.csv")
    write_source(occupancy, "Figure5de_prototype_occupancy.csv")
    write_source(assignments[["accession", "true_label", "similarity_bin", "within_true_cosine_distance", "outcome_group"]], "Figure5f_outcome_prototype_distance.csv")


def make_tables(
    cv: pd.DataFrame,
    selected: dict[str, int],
    locked_epochs: dict[str, int],
    boot: pd.DataFrame,
    comparisons: pd.DataFrame,
    similarity: pd.DataFrame,
    audit: pd.DataFrame,
    pred_cols: dict[str, str],
    kfolds: pd.DataFrame,
) -> None:
    # Remove obsolete pre-renumbering files after the manuscript gained a
    # dataset-composition table as the new Table 1.
    stale_names = [
        "Table1_model_selection_and_locked_configurations.csv",
        "Table2_model_selection_and_locked_configurations.csv",
        "Table2_fixed_test_performance_and_statistics.csv",
        "Table3_sequence_identity_stratified_performance.csv",
        "Table4_computational_cost.csv",
        "Table4_sequence_identity_stratified_performance.csv",
        "Table5_computational_cost.csv",
    ]
    for stale_name in stale_names:
        stale_path = TABLE_OUT / stale_name
        if stale_path.exists():
            stale_path.unlink()

    # Table 1: dataset composition and non-overlapping identity intervals.
    split_summary_path = (
        RESULT
        / "01_data_split_and_similarity"
        / "random_split_similarity_audit"
        / "summary.json"
    )
    split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
    display_class = {
        "EC3.1_ester_bonds": "EC 3.1 (ester bonds)",
        "EC3.2_glycosylases": "EC 3.2 (glycosylases)",
        "EC3.4_peptidases": "EC 3.4 (peptidases)",
        "EC3.5_CN_nonpeptide": "EC 3.5 (C–N bonds, excluding peptide bonds)",
        "EC3_other": "Other EC 3 hydrolases",
    }
    dataset_rows = []
    for class_name in CLASS_ORDER:
        train_n = int(split_summary["train_counts_by_class"][class_name])
        test_n = int(split_summary["test_counts_by_class"][class_name])
        total_n = train_n + test_n
        nested = split_summary["nested_test_counts_by_class"][class_name]
        lt80 = int(nested["lt80"])
        lt60 = int(nested["lt60"])
        lt40 = int(nested["lt40"])
        dataset_rows.append(
            {
                "Hydrolase class": display_class[class_name],
                "Full dataset n": total_n,
                "Dataset share (%)": round(100.0 * total_n / int(split_summary["total_records"]), 1),
                "Training set n": train_n,
                "Fixed test set n": test_n,
                "Test ≥80% n": test_n - lt80,
                "Test 60–80% n": lt80 - lt60,
                "Test 40–60% n": lt60 - lt40,
                "Test <40% n": lt40,
                "No qualified hit n": int(split_summary["no_hit_counts_by_class"][class_name]),
            }
        )
    dataset_rows.append(
        {
            "Hydrolase class": "Total",
            "Full dataset n": int(split_summary["total_records"]),
            "Dataset share (%)": 100.0,
            "Training set n": int(split_summary["train_records"]),
            "Fixed test set n": int(split_summary["test_records"]),
            "Test ≥80% n": int(split_summary["test_records"] - split_summary["nested_test_counts"]["lt80"]),
            "Test 60–80% n": int(split_summary["nested_test_counts"]["lt80"] - split_summary["nested_test_counts"]["lt60"]),
            "Test 40–60% n": int(split_summary["nested_test_counts"]["lt60"] - split_summary["nested_test_counts"]["lt40"]),
            "Test <40% n": int(split_summary["nested_test_counts"]["lt40"]),
            "No qualified hit n": int(split_summary["test_sequences_without_coverage_qualified_hit"]),
        }
    )
    table1 = pd.DataFrame(dataset_rows)
    write_titled_csv(
        "Table1_dataset_composition_and_similarity_strata.csv",
        "Table 1. Composition of the bacterial hydrolase dataset and distribution of the fixed test set across train–test sequence-identity intervals.",
        "The 80/20 split was stratified by hydrolase class. The ≥80%, 60–80%, 40–60%, and <40% columns are mutually exclusive intervals defined by the maximum MMseqs2 sequence identity to the training set. 'No qualified hit' is a subset of the <40% group and denotes sequences without an alignment satisfying the bidirectional 80% coverage requirement; it must not be added to the four identity intervals.",
        table1,
    )

    # Table 2: complete fold-level results for the locked model-specific CV
    # configurations. Architectural distinctions are described in Methods and
    # are intentionally not repeated as Yes/No columns in the Results table.
    rows = []
    metric_columns = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro-F1"),
        ("weighted_f1", "Weighted F1"),
        ("mcc", "MCC"),
    ]
    for model in CORE_MODEL_ORDER:
        depth = selected[model]
        sub = cv[(cv["model"] == model) & (cv["unfreeze_layers"] == depth)].sort_values("fold")
        for _, fold_row in sub.iterrows():
            row = {"Model": model, "Fold": f"Fold {int(fold_row['fold'])}"}
            for source_column, display_column in metric_columns:
                row[display_column] = f"{fold_row[source_column]:.4f}"
            rows.append(row)

        summary_row = {"Model": model, "Fold": "Mean ± SD"}
        for source_column, display_column in metric_columns:
            summary_row[display_column] = fmt_mean_sd(
                sub[source_column].mean(), sub[source_column].std(ddof=1)
            )
        rows.append(summary_row)
    table2_cv = pd.DataFrame(rows)
    write_titled_csv(
        "Table2_five_fold_cross_validation_results.csv",
        "Table 2. Five-fold cross-validation performance of the four ESM-2-based hydrolase classifiers.",
        "Values for folds 1–5 are validation-set metrics for the locked model-specific configuration selected using the 2,721-sequence training set. The final row for each model reports mean ± standard deviation across the five folds. The fixed 680-sequence test set was not used in cross-validation or model selection.",
        table2_cv,
    )

    # Table 3: overall fixed-test performance and paired comparisons.
    table2_rows = []
    comparison_lookup = comparisons.set_index("comparator")
    for model in MODEL_ORDER:
        row = {"Model": model, "Test n": 680}
        for metric, label in [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro-F1"), ("mcc", "MCC")]:
            value = boot[(boot["model"] == model) & (boot["metric"] == metric)].iloc[0]
            row[f"{label} [95% CI]"] = fmt_ci(value["estimate"], value["ci95_low"], value["ci95_high"])
        if model == "HydroProto":
            row["HydroProto − model Macro-F1 [95% CI]"] = "Reference"
            row["McNemar exact P vs HydroProto"] = "—"
            row["Holm-adjusted P vs HydroProto"] = "—"
        else:
            comp = comparison_lookup.loc[model]
            row["HydroProto − model Macro-F1 [95% CI]"] = fmt_ci(comp["macro_f1_delta"], comp["delta_ci95_low"], comp["delta_ci95_high"])
            row["McNemar exact P vs HydroProto"] = f"{comp['mcnemar_p_exact']:.6g}"
            row["Holm-adjusted P vs HydroProto"] = (
                "Not adjusted" if pd.isna(comp["mcnemar_p_holm"]) else f"{comp['mcnemar_p_holm']:.6g}"
            )
        table2_rows.append(row)
    table2 = pd.DataFrame(table2_rows)
    write_titled_csv(
        "Table3_fixed_test_performance_and_statistics.csv",
        "Table 3. Performance on the fixed independent test set and paired comparison with HydroProto.",
        "All rows evaluate the same 680 sequences. MMseqs2 uses its majority-class fallback when no coverage-qualified hit is available. Confidence intervals are stratified bootstrap intervals (2,000 resamples; seed 42). Macro-F1 deltas are paired stratified-bootstrap estimates. For the five neural/deep-learning models, McNemar P values retain the archived Holm correction across all 10 pairwise comparisons. The MMseqs2 comparison was added post hoc and its exact P value is not multiplicity-adjusted.",
        table2,
    )

    # Tables 4 and 5: separate the nested low-identity stress tests from the
    # mutually exclusive identity intervals. This keeps each main table focused
    # on one scientific question and gives both tables the same metric set.
    stratified_model_order = MODEL_ORDER
    metric_columns = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro-F1"),
        ("weighted_f1", "Weighted F1"),
        ("mcc", "MCC"),
    ]

    def stratified_table(analysis: str, bins: list[tuple[str, str]]) -> pd.DataFrame:
        rows = []
        for bin_name, display_label in bins:
            sub_metrics = similarity[
                (similarity["analysis"] == analysis) & (similarity["bin"] == bin_name)
            ]
            for model in stratified_model_order:
                metric = sub_metrics[sub_metrics["model"] == model].iloc[0]
                row = {
                    "Test subset": display_label,
                    "Model": model,
                    "n": int(metric["n"]),
                }
                for source_column, display_column in metric_columns:
                    row[display_column] = f"{metric[source_column]:.4f}"
                rows.append(row)
        return pd.DataFrame(rows)

    nested_table = stratified_table(
        "nested",
        [
            ("test_all", "All"),
            ("lt80", "<80%"),
            ("lt60", "<60%"),
            ("lt40", "<40%"),
        ],
    )
    write_titled_csv(
        "Table4_nested_sequence_identity_stress_tests.csv",
        "Table 4. Model performance on progressively stricter nested low-sequence-identity test subsets.",
        "The <80%, <60%, and <40% subsets are nested: each stricter subset is contained within the preceding subset. Similarity denotes the maximum MMseqs2 sequence identity between each test sequence and the training set.",
        nested_table,
    )

    nonoverlap_table = stratified_table(
        "nonoverlap",
        [
            ("ge80", "≥80%"),
            ("60_to_lt80", "60–<80%"),
            ("40_to_lt60", "40–<60%"),
            ("lt40", "<40%"),
        ],
    )
    write_titled_csv(
        "Table5_non_overlapping_sequence_identity_intervals.csv",
        "Table 5. Model performance across mutually exclusive train–test sequence-identity intervals.",
        "The four intervals are non-overlapping and together partition the fixed 680-sequence test set. Similarity denotes the maximum MMseqs2 sequence identity between each test sequence and the training set.",
        nonoverlap_table,
    )

    # Also preserve the K ablation summary as figure source data, since it supports the main mechanistic figure.
    k_summary = kfolds.groupby("K").agg(
        folds=("fold", "count"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        mcc_mean=("mcc", "mean"),
        mcc_sd=("mcc", "std"),
    ).reset_index()
    write_source(k_summary, "Table_source_K_ablation_summary.csv")


def write_manifest() -> None:
    records = []
    panel_counts = {2: 4, 3: 4, 4: 6, 5: 6}
    for fig_num, count in panel_counts.items():
        for idx in range(count):
            label = chr(97 + idx)
            records.append(
                {
                    "figure": f"Figure {fig_num}",
                    "panel": label,
                    "jpg": str((FIG_OUT / f"Figure{fig_num}" / "individual" / f"Figure{fig_num}{label}.jpg").relative_to(ROOT)) if fig_num != 2 else "See model-named individual file",
                    "pdf": str((FIG_OUT / f"Figure{fig_num}" / "individual" / f"Figure{fig_num}{label}.pdf").relative_to(ROOT)) if fig_num != 2 else "See model-named individual file",
                }
            )
    pd.DataFrame(records).to_csv(OUT / "figure_file_manifest.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    cv, selected, locked_epochs = load_cv_data()
    audit_raw = load_prediction_audit()
    audit, pred_cols = augment_with_mmseq(audit_raw)
    boot, class_metrics, comparisons = build_overall_statistics(audit, pred_cols)
    similarity, diagnostic = build_similarity_data(audit, pred_cols)
    kfolds, ktests, kboot = load_k_ablation()

    figure2(cv, selected)
    figure3(audit, pred_cols, boot, class_metrics)
    figure4(audit, pred_cols, similarity, diagnostic)
    figure5(kfolds, ktests, kboot)
    make_tables(cv, selected, locked_epochs, boot, comparisons, similarity, audit, pred_cols, kfolds)
    write_source(comparisons, "Table2_paired_comparisons_source.csv")
    write_manifest()

    print(json.dumps({
        "root": str(ROOT),
        "figure_files": len(list(FIG_OUT.rglob("*.jpg"))) + len(list(FIG_OUT.rglob("*.pdf"))),
        "table_files": len(list(TABLE_OUT.glob("*.csv"))),
        "source_files": len(list(SOURCE_OUT.glob("*.csv"))),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
