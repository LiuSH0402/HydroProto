# Reproducibility protocol

1. Keep the fixed train/test CSV files unchanged and record their SHA-256 hashes.
2. Select the model, unfreezing depth and epoch count using five-fold stratified
   cross-validation on the training split only.
3. Train one final model on all 2,721 training sequences for the locked epoch count.
4. Run inference once on the 680-sequence test set.
5. Derive the nested and non-overlapping homology analyses by filtering the saved
   test predictions; do not retrain for a test subset.
6. Report Accuracy, Macro-F1 and MCC, classwise metrics, paired bootstrap confidence
   intervals and paired significance tests.

The full internal result archive includes the exact configurations, training
histories, fold predictions, final predictions, logs and checkpoints. The compact
Git repository excludes large model artifacts by design.
