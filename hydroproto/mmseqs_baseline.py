from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LABELS, LABEL_TO_INDEX, load_sequence_csv
from .metrics import classification_metrics


def one_hot(indices: np.ndarray) -> np.ndarray:
    result = np.zeros((len(indices), len(LABELS)), dtype=np.float64)
    result[np.arange(len(indices)), indices] = 1.0
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate MMseqs2 nearest-neighbor label baseline")
    parser.add_argument("--train", default="random_split_similarity_audit/train_dataset.csv")
    parser.add_argument("--test-similarity", default="random_split_similarity_audit/test_dataset_with_train_similarity.csv")
    parser.add_argument("--nested-dir", default="random_split_similarity_audit")
    parser.add_argument("--output-dir", default="modeling/reports/mmseqs_baseline")
    args = parser.parse_args()

    train = load_sequence_csv(args.train)
    test = load_sequence_csv(args.test_similarity)
    train_label = train.set_index("accession")["label"].to_dict()
    has_hit = test["has_coverage_qualified_hit"].astype(bool).to_numpy()
    nearest = test["nearest_train_accession"].astype("string")
    missing_mapped = sorted({str(acc) for acc in nearest[has_hit] if str(acc) not in train_label})
    if missing_mapped:
        raise ValueError(f"Nearest-neighbor accessions are absent from the training set: {missing_mapped[:5]}")

    majority_label = train["label"].value_counts().idxmax()
    predicted = []
    for hit, accession in zip(has_hit, nearest):
        predicted.append(train_label[str(accession)] if hit else majority_label)
    true_idx = np.asarray([LABEL_TO_INDEX[label] for label in test["label"]], dtype=np.int64)
    pred_idx = np.asarray([LABEL_TO_INDEX[label] for label in predicted], dtype=np.int64)
    probabilities = one_hot(pred_idx)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame({
        "accession": test["accession"],
        "true_label": test["label"],
        "predicted_label": predicted,
        "has_coverage_qualified_hit": has_hit,
        "nearest_train_accession": nearest,
        "fallback_used": ~has_hit,
        "fallback_label": majority_label,
    })
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    subset_files = {
        "Test-all": None,
        "Test<80": Path(args.nested_dir) / "test_nested_lt80.csv",
        "Test<60": Path(args.nested_dir) / "test_nested_lt60.csv",
        "Test<40": Path(args.nested_dir) / "test_nested_lt40.csv",
    }
    detailed = {}
    rows = []
    test_accessions = test["accession"].astype(str).to_numpy()
    for subset_name, subset_file in subset_files.items():
        if subset_file is None:
            subset_mask = np.ones(len(test), dtype=bool)
        else:
            subset_ids = set(load_sequence_csv(subset_file)["accession"].astype(str))
            subset_mask = np.asarray([accession in subset_ids for accession in test_accessions])
        full_metrics = classification_metrics(true_idx[subset_mask], probabilities[subset_mask])
        hit_mask = subset_mask & has_hit
        hit_metrics = classification_metrics(true_idx[hit_mask], probabilities[hit_mask]) if hit_mask.any() else None
        detailed[subset_name] = {
            "full_with_majority_fallback": full_metrics,
            "coverage_qualified_hits_only": hit_metrics,
            "hit_count": int(hit_mask.sum()),
            "total_count": int(subset_mask.sum()),
            "hit_coverage": float(hit_mask.sum() / subset_mask.sum()),
            "fallback_label": majority_label,
        }
        rows.append({
            "subset": subset_name,
            "n": int(subset_mask.sum()),
            "hit_count": int(hit_mask.sum()),
            "hit_coverage": float(hit_mask.sum() / subset_mask.sum()),
            "full_macro_f1": full_metrics["macro_f1"],
            "full_accuracy": full_metrics["accuracy"],
            "hit_only_macro_f1": None if hit_metrics is None else hit_metrics["macro_f1"],
            "hit_only_accuracy": None if hit_metrics is None else hit_metrics["accuracy"],
        })
    pd.DataFrame(rows).to_csv(output_dir / "metrics_by_subset.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "metrics_full.json").open("w", encoding="utf-8") as handle:
        json.dump(detailed, handle, ensure_ascii=False, indent=2)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"output_dir={output_dir.resolve()}")


if __name__ == "__main__":
    main()
