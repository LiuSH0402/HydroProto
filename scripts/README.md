# Post-hoc analyses

These scripts consume saved prediction, metric and prototype files; they do not
train a model or select hyperparameters from the fixed test set.

- `final_paired_statistics.py` — stratified bootstrap intervals, paired deltas,
  Cochran's Q and Holm-adjusted McNemar tests.
- `posthoc_similarity_error_analysis.py` — nested and non-overlapping identity
  strata, confusion matrices, classwise metrics and error-case tables.
- `hydroproto_k_ablation_statistics.py` — K=1/3/5/7 out-of-fold comparisons and
  paired tests.
- `analyze_hydroproto_mechanism.py` — prototype geometry, occupancy,
  representatives and distance/outcome analyses.
- `summarize_model_cost.py` — parameter, training time, peak memory and inference
  cost tables and figures.

Run any script with `--help` to see its required locked-result paths. Keep raw
predictions immutable and write regenerated reports to a new output directory.
