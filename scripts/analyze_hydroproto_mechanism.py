from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
SHORT = ["EC3.1", "EC3.2", "EC3.4", "EC3.5", "EC3.other"]


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, value: str) -> None:
    ax.text(-0.13, 1.05, value, transform=ax.transAxes, fontsize=11, fontweight="bold")


def protein_description(value: str) -> str:
    match = re.search(r"^[^\s]+\s+(.+?)\s+OS=", str(value))
    return match.group(1) if match else str(value)


def effective_prototypes(fractions: np.ndarray) -> float:
    nonzero = fractions[fractions > 0]
    return float(np.exp(-np.sum(nonzero * np.log(nonzero))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-dir", required=True)
    parser.add_argument("--test-audit", required=True)
    parser.add_argument("--initial-prototypes")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    mechanism = Path(args.mechanism_dir)
    output = Path(args.output_dir)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    arrays = np.load(mechanism / "mechanism_arrays.npz")
    prototypes = arrays["prototypes"]
    flattened = prototypes.reshape(-1, prototypes.shape[-1])
    cosine = flattened @ flattened.T
    prototype_names = [f"{SHORT[c]}-P{k + 1}" for c in range(5) for k in range(5)]
    pd.DataFrame(cosine, index=prototype_names, columns=prototype_names).to_csv(
        tables / "prototype_cosine_matrix.csv", encoding="utf-8-sig"
    )

    initial_within_mean = None
    same_index_drift_mean = None
    if args.initial_prototypes:
        initial_file = np.load(args.initial_prototypes)
        initial = np.stack([initial_file[key] for key in initial_file.files])
        initial /= np.clip(np.linalg.norm(initial, axis=2, keepdims=True), 1e-12, None)
        drift_rows = []
        initial_within = []
        for class_index, label in enumerate(LABELS):
            initial_similarity = initial[class_index] @ initial[class_index].T
            learned_initial = prototypes[class_index] @ initial[class_index].T
            initial_within.extend(initial_similarity[np.triu_indices(5, k=1)].tolist())
            for prototype_index in range(5):
                drift_rows.append({
                    "class": label,
                    "prototype": prototype_index + 1,
                    "same_index_initial_learned_cosine": float(
                        learned_initial[prototype_index, prototype_index]
                    ),
                    "nearest_initial_prototype": int(
                        learned_initial[prototype_index].argmax()
                    ) + 1,
                    "nearest_initial_cosine": float(
                        learned_initial[prototype_index].max()
                    ),
                })
        drift_frame = pd.DataFrame(drift_rows)
        drift_frame.to_csv(
            tables / "prototype_initial_to_learned_drift.csv",
            index=False,
            encoding="utf-8-sig",
        )
        initial_within_mean = float(np.mean(initial_within))
        same_index_drift_mean = float(
            drift_frame["same_index_initial_learned_cosine"].mean()
        )

    geometry = pd.read_csv(mechanism / "prototype_pairwise_geometry.csv")
    geometry_summary = (
        geometry.groupby("pair_type")["cosine_similarity"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    geometry_summary.to_csv(tables / "prototype_geometry_summary.csv", index=False, encoding="utf-8-sig")
    within_summary = (
        geometry[geometry["pair_type"].eq("within_class")]
        .groupby("left_class")["cosine_similarity"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index().rename(columns={"left_class": "class"})
    )
    within_summary.to_csv(tables / "within_class_prototype_geometry.csv", index=False, encoding="utf-8-sig")

    train = pd.read_csv(mechanism / "train_prototype_assignments.csv", dtype={"accession": str})
    test = pd.read_csv(mechanism / "test_prototype_assignments.csv", dtype={"accession": str})
    audit = pd.read_csv(args.test_audit, dtype={"accession": str})
    audit = audit.rename(columns={
        "correct_H-Proto": "correct_HydroProto",
        "prediction_H-Proto": "prediction_HydroProto",
    })
    test = test.merge(
        audit[[
            "accession", "similarity_bin", "has_coverage_qualified_hit", "max_identity_pct",
            "correct_HydroProto", "correct_ESM2+MLP", "prediction_HydroProto", "prediction_ESM2+MLP",
        ]],
        on="accession", how="left", validate="one_to_one",
    )
    subsets = {
        "train": train,
        "test": test,
        "test_lt40": test[test["similarity_bin"].eq("lt40")],
    }
    occupancy_rows = []
    summary_rows = []
    for subset_name, frame in subsets.items():
        for label in LABELS:
            local = frame[frame["true_label"].eq(label)]
            counts = local["within_true_prototype"].value_counts().reindex(range(5), fill_value=0)
            fractions = counts.to_numpy(dtype=float) / max(len(local), 1)
            summary_rows.append({
                "subset": subset_name,
                "class": label,
                "n": len(local),
                "dominant_prototype": int(np.argmax(fractions)) + 1,
                "dominant_fraction": float(fractions.max()),
                "occupied_prototypes": int(np.sum(fractions > 0)),
                "effective_prototypes": effective_prototypes(fractions),
            })
            for prototype, (count, fraction) in enumerate(zip(counts, fractions), start=1):
                occupancy_rows.append({
                    "subset": subset_name,
                    "class": label,
                    "prototype": prototype,
                    "n": int(count),
                    "class_fraction": float(fraction),
                })
    occupancy = pd.DataFrame(occupancy_rows)
    occupancy.to_csv(tables / "prototype_occupancy_train_test_lt40.csv", index=False, encoding="utf-8-sig")
    occupancy_summary = pd.DataFrame(summary_rows)
    occupancy_summary.to_csv(tables / "prototype_occupancy_summary.csv", index=False, encoding="utf-8-sig")

    representatives = pd.read_csv(
        mechanism / "prototype_representative_proteins.csv", dtype={"accession": str}
    )
    representatives["protein_description"] = representatives["original_header"].map(protein_description)
    representatives.to_csv(tables / "prototype_representatives_annotated.csv", index=False, encoding="utf-8-sig")
    rep_summary_rows = []
    overlap_rows = []
    for label in LABELS:
        local = representatives[representatives["class"].eq(label)]
        for prototype in range(5):
            values = local[local["prototype"].eq(prototype)].sort_values("rank")
            rep_summary_rows.append({
                "class": label,
                "prototype": prototype + 1,
                "top_accession": values.iloc[0]["accession"],
                "top_description": values.iloc[0]["protein_description"],
                "top_cosine_similarity": values.iloc[0]["cosine_similarity"],
                "unique_accessions_top10": values["accession"].nunique(),
                "unique_descriptions_top10": values["protein_description"].nunique(),
                "median_length_top10": float(values["length"].median()),
            })
        for left in range(5):
            left_values = local[local["prototype"].eq(left)]
            left_ids = set(left_values["accession"])
            left_descriptions = set(left_values["protein_description"])
            for right in range(left + 1, 5):
                right_values = local[local["prototype"].eq(right)]
                right_ids = set(right_values["accession"])
                right_descriptions = set(right_values["protein_description"])
                overlap_rows.append({
                    "class": label,
                    "left_prototype": left + 1,
                    "right_prototype": right + 1,
                    "accession_jaccard_top10": len(left_ids & right_ids) / max(len(left_ids | right_ids), 1),
                    "description_jaccard_top10": len(left_descriptions & right_descriptions) / max(len(left_descriptions | right_descriptions), 1),
                    "shared_accessions": len(left_ids & right_ids),
                    "shared_descriptions": len(left_descriptions & right_descriptions),
                })
    pd.DataFrame(rep_summary_rows).to_csv(tables / "prototype_representative_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(overlap_rows).to_csv(tables / "prototype_representative_overlap.csv", index=False, encoding="utf-8-sig")

    test["outcome_group"] = np.select(
        [
            test["correct_HydroProto"].astype(bool) & ~test["correct_ESM2+MLP"].astype(bool),
            ~test["correct_HydroProto"].astype(bool) & test["correct_ESM2+MLP"].astype(bool),
            test["correct_HydroProto"].astype(bool) & test["correct_ESM2+MLP"].astype(bool),
        ],
        ["HydroProto correct / MLP wrong", "MLP correct / HydroProto wrong", "Both correct"],
        default="Both wrong",
    )
    test.to_csv(tables / "test_assignments_with_outcomes.csv", index=False, encoding="utf-8-sig")
    gain_distance = (
        test.groupby("outcome_group")
        .agg(
            n=("accession", "size"),
            mean_true_prototype_distance=("within_true_cosine_distance", "mean"),
            median_true_prototype_distance=("within_true_cosine_distance", "median"),
            mean_prototype_class_margin=("prototype_class_margin", "mean"),
            median_prototype_class_margin=("prototype_class_margin", "median"),
        )
        .reset_index()
    )
    gain_distance.to_csv(tables / "outcome_prototype_distance_summary.csv", index=False, encoding="utf-8-sig")

    # Mechanism figure: geometry, occupancy, and relation to corrected samples.
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), constrained_layout=True)
    sns.heatmap(
        cosine, ax=axes[0, 0], cmap="vlag", vmin=-0.5, vmax=0.5, center=0,
        xticklabels=False, yticklabels=False, cbar_kws={"label": "Cosine similarity"},
    )
    for boundary in (5, 10, 15, 20):
        axes[0, 0].axhline(boundary, color="white", lw=1)
        axes[0, 0].axvline(boundary, color="white", lw=1)
    axes[0, 0].set_title("All 25 learned prototypes")
    axes[0, 0].text(-0.13, 1.05, "a", transform=axes[0, 0].transAxes, fontsize=11, fontweight="bold")

    sns.boxplot(
        data=geometry, x="pair_type", y="cosine_similarity", ax=axes[0, 1],
        order=["within_class", "between_class"],
        palette=["#B64342", "#7884B4"], width=0.55, fliersize=2,
    )
    axes[0, 1].set_xticklabels(["Within class", "Between class"])
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Prototype cosine similarity")
    panel_label(axes[0, 1], "b")

    for ax, subset_name, title, panel in (
        (axes[0, 2], "train", "Train occupancy", "c"),
        (axes[1, 0], "test_lt40", "<40% test occupancy", "d"),
    ):
        values = occupancy[occupancy["subset"].eq(subset_name)].pivot(
            index="class", columns="prototype", values="class_fraction"
        ).reindex(LABELS).fillna(0)
        sns.heatmap(
            values, ax=ax, cmap="Reds", vmin=0, vmax=1, annot=True, fmt=".2f",
            cbar=False, xticklabels=["P1", "P2", "P3", "P4", "P5"],
            yticklabels=SHORT, annot_kws={"fontsize": 6.5},
        )
        ax.set_xlabel("Nearest within-class prototype")
        ax.set_ylabel("")
        ax.set_title(title)
        panel_label(ax, panel)

    plot_order = [
        "Both correct", "HydroProto correct / MLP wrong",
        "MLP correct / HydroProto wrong", "Both wrong",
    ]
    sns.boxplot(
        data=test, x="outcome_group", y="within_true_cosine_distance",
        order=plot_order, ax=axes[1, 1], color="#B4C0E4", width=0.6,
        showfliers=False,
    )
    axes[1, 1].set_xticklabels(["Both\ncorrect", "HydroProto\nonly", "MLP\nonly", "Both\nwrong"])
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Distance to nearest true-class prototype")
    panel_label(axes[1, 1], "e")

    sns.boxplot(
        data=test, x="outcome_group", y="prototype_class_margin",
        order=plot_order, ax=axes[1, 2], color="#E4CCD8", width=0.6,
        showfliers=False,
    )
    axes[1, 2].axhline(0, color="#767676", lw=0.8, ls="--")
    axes[1, 2].set_xticklabels(["Both\ncorrect", "HydroProto\nonly", "MLP\nonly", "Both\nwrong"])
    axes[1, 2].set_xlabel("")
    axes[1, 2].set_ylabel("Top-vs-second prototype-class margin")
    panel_label(axes[1, 2], "f")
    save_figure(fig, figures / "figure_hydroproto_mechanism")

    train_summary = occupancy_summary[occupancy_summary["subset"].eq("train")]
    audit_values = {
        "status": "complete",
        "within_class_cosine_mean": float(
            geometry.loc[geometry["pair_type"].eq("within_class"), "cosine_similarity"].mean()
        ),
        "between_class_cosine_mean": float(
            geometry.loc[geometry["pair_type"].eq("between_class"), "cosine_similarity"].mean()
        ),
        "train_dominant_fraction_min": float(train_summary["dominant_fraction"].min()),
        "train_dominant_fraction_max": float(train_summary["dominant_fraction"].max()),
        "train_effective_prototypes_mean": float(train_summary["effective_prototypes"].mean()),
        "initial_within_class_cosine_mean": initial_within_mean,
        "same_index_initial_learned_cosine_mean": same_index_drift_mean,
        "prototype_collapse_detected": bool((train_summary["dominant_fraction"] > 0.95).all()),
        "mechanistic_claim_supported": False,
        "reason": "All classes assign >95% of training sequences to one prototype; representative sets overlap and inactive prototypes lack stable occupancy.",
    }
    (output / "mechanism_analysis_audit.json").write_text(
        json.dumps(audit_values, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit_values, indent=2))


if __name__ == "__main__":
    main()
