# Presentation outline

1. Problem: forecast total daily retail sales 15 days ahead.
2. Data: Kaggle Favorita sales aggregated across stores and families.
3. Preparation: causal missing-date fill, chronological split, training-only scaling, 30-day sequences.
4. Model: two-layer LSTM, 64 hidden units, Adam/MSE, 30 epochs.
5. Evaluation: show validation figure and measured MAE/RMSE against two baselines.
6. Forecast: show next-15-days figure; distinguish predictions from known outcomes.
7. Limitations: short holdout, recursive drift, no promotion/holiday features, aggregate rather than store/family scope.
8. Next steps: rolling-origin validation, exogenous covariates, per-series forecasting.
9. Acknowledgement: Tuwaiq Academy; disclose ChatGPT assistance and source notebook provenance.
