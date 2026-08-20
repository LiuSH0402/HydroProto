# Result archive

The Git repository contains only compact summary tables and documentation. The
complete internal archive, including fold-level predictions, logs, figures and
development checkpoints, is intentionally stored outside this repository. The
public Zenodo release contains the four locked final checkpoints together with
their configurations, metrics, predictions and SHA-256 checksums. Its DOI will be
added here after the first Zenodo record is created.

## Compact tables included here

- `final_test_summary.csv`: locked final evaluation on the full test set and nested
  `<80%`, `<60%` and `<40%` subsets.
- `final_test_bootstrap_ci.csv`: stratified bootstrap confidence intervals.
- `final_test_paired_delta_ci.csv`: paired bootstrap differences among models.
- `prototype_number_ablation_oof.csv`: out-of-fold K=1/3/5/7 comparison.
- `prototype_number_ablation_statistics.csv`: fold-paired significance tests.
- `model_cost_summary.csv`: trainable parameters, runtime and memory measurements.

On the fixed 680-sequence test set, HydroProto achieved Accuracy 0.9515,
Macro-F1 0.9501 and MCC 0.9378. These values come from the locked final model and
must not be used for hyperparameter selection.
