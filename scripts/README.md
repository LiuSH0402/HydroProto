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

The analysis CLIs document their required saved-result paths with `--help`.
`make_manuscript_figures_tables.py` is an original plotting-workspace script rather
than a standalone CLI; it expects the original result-directory layout. Its
historical `lt40` key includes no-hit sequences, as explained in
`../docs/DATA_PROVENANCE.md`. Keep raw predictions immutable and write regenerated
reports to a new output directory.
