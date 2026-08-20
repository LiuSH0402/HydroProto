from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


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


def percent(value: float, reference: float) -> float:
    return (value / reference - 1.0) * 100.0


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp", required=True)
    parser.add_argument("--hydroproto", "--hproto", dest="hydroproto", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mlp = json.loads(Path(args.mlp).read_text(encoding="utf-8"))
    hproto = json.loads(Path(args.hydroproto).read_text(encoding="utf-8"))
    values = []
    for name, item in (("ESM2+MLP", mlp), ("HydroProto", hproto)):
        values.append({
            "model": name,
            "unfreeze_layers": item["unfreeze_layers"],
            "locked_epochs": item["locked_epochs"],
            "total_parameters": item["total_parameters"],
            "head_parameters": item["head_parameters"],
            "total_trainable_parameters": item["total_trainable_parameters"],
            "checkpoint_mib": item["checkpoint_bytes"] / 1024**2,
            "one_epoch_minutes": item["one_epoch_train_seconds"] / 60,
            "projected_total_hours": item["projected_locked_total_train_seconds"] / 3600,
            "train_peak_gib": item["train_peak_gpu_memory_bytes"] / 1024**3,
            "inference_seconds_per_100": item["inference_seconds_per_100"],
            "inference_peak_gib": item["inference_peak_gpu_memory_bytes"] / 1024**3,
        })
    frame = pd.DataFrame(values)
    frame.to_csv(output / "model_cost_summary.csv", index=False, encoding="utf-8-sig")

    head_delta = hproto["head_parameters"] - mlp["head_parameters"]
    same_l4_mlp_trainable = hproto["esm_trainable_parameters"] + mlp["head_parameters"]
    deltas = {
        "prototype_head_extra_parameters": head_delta,
        "prototype_head_percent_increase": percent(hproto["head_parameters"], mlp["head_parameters"]),
        "total_model_parameter_percent_increase": percent(hproto["total_parameters"], mlp["total_parameters"]),
        "same_backbone_l4_trainable_parameter_percent_increase": percent(
            hproto["total_trainable_parameters"], same_l4_mlp_trainable
        ),
        "locked_final_trainable_parameter_percent_increase": percent(
            hproto["total_trainable_parameters"], mlp["total_trainable_parameters"]
        ),
        "one_epoch_time_percent_increase": percent(
            hproto["one_epoch_train_seconds"], mlp["one_epoch_train_seconds"]
        ),
        "locked_total_time_percent_increase": percent(
            hproto["projected_locked_total_train_seconds"], mlp["projected_locked_total_train_seconds"]
        ),
        "training_peak_memory_percent_increase": percent(
            hproto["train_peak_gpu_memory_bytes"], mlp["train_peak_gpu_memory_bytes"]
        ),
        "inference_time_percent_increase": percent(
            hproto["inference_seconds_per_100"], mlp["inference_seconds_per_100"]
        ),
        "inference_peak_memory_percent_increase": percent(
            hproto["inference_peak_gpu_memory_bytes"], mlp["inference_peak_gpu_memory_bytes"]
        ),
        "total_training_time_interpretation": (
            "Projected from one standardized full epoch; HydroProto uses 23 locked epochs "
            "and MLP uses 11, so total cost combines architecture, unfreeze depth, and epoch selection."
        ),
    }
    (output / "model_cost_deltas.json").write_text(json.dumps(deltas, indent=2), encoding="utf-8")

    colors = ["#484878", "#B64342"]
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 2.8), constrained_layout=True)
    metrics = [
        ("total_trainable_parameters", 1e6, "Trainable parameters\n(million)"),
        ("one_epoch_minutes", 1, "Time per epoch\n(min)"),
        ("train_peak_gib", 1, "Training peak memory\n(GiB)"),
        ("inference_seconds_per_100", 1, "Inference time\n(s per 100)"),
    ]
    for panel, (ax, (column, scale, ylabel)) in enumerate(zip(axes, metrics)):
        plotted = frame[column] / scale
        bars = ax.bar([0, 1], plotted, color=colors, width=0.62)
        ax.set_xticks([0, 1], ["MLP", "HydroProto"])
        ax.set_ylabel(ylabel)
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in plotted], padding=2, fontsize=7)
        ax.set_ylim(0, max(plotted) * 1.20)
        ax.text(-0.18, 1.05, chr(ord("a") + panel), transform=ax.transAxes, fontsize=11, fontweight="bold")
    save_figure(fig, output / "figure_model_cost")
    print(json.dumps(deltas, indent=2))


if __name__ == "__main__":
    main()
