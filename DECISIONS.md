# Decisions

1. Preserve the supplied notebook as provenance, remove broken internal citation tokens and transient upload widget output. Do not claim its absent outputs were measured.
2. Keep its aggregate daily sales task and LSTM architecture; explain the difference from Kaggle's store × family task.
3. Use causal forward-fill instead of whole-series interpolation, preventing look-ahead across missing dates.
4. Fit MinMaxScaler after the chronological split; recursive evaluation receives no held-out actuals.
5. Add weekly seasonal-naive and last-value forecasts to put the LSTM performance in context. Use MAE/RMSE in original sales units.
6. Keep fixed hyperparameters without selecting on the final holdout. Additional rolling-origin validation is future work.
7. Save scaler parameters with weights and require consecutive future dates starting immediately after history, then select the first 15.
8. Attribute Tuwaiq Academy using its public training organization and website. Keep certificate and screenshots outside the public repository because they contain private identifiers/account information.
9. Use Bayan-style organization/documentation, not NLP-specific grading requirements that do not apply to this course. No official rubric beyond the supplied project workflow was available.

Horizon correction: the Kaggle test file has 16 dates (16–31 August 2017). The original notebook used all test dates while labeling them 15 days. The revised pipeline explicitly selects the first 15 (16–30 August) to match the project goal.
