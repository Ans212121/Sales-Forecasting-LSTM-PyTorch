# Model card

Model: PyTorch LSTM, univariate input, 64 hidden units, 2 layers, dropout 0.2, linear scalar output. Input shape: batch × 30 × 1. Target: next-day scaled aggregate sales. Optimizer: Adam, lr 0.001; objective: MSE; 30 epochs; seed 42.

Forecasting recursively uses prior predictions after the first step. Scaling is fit on training history only during evaluation. Final training uses all known data with a fresh scaler/model. Negative inverse-scaled outputs are clipped to zero. Checkpoint includes weights and scaler min/scale values.

Intended use: educational forecasting and visualization. Not intended for inventory automation, financial decisions or official Kaggle submission. Metrics and baselines are in EVALUATION_REPORT.md. No causal explanations or guaranteed accuracy are claimed. CPU/GPU and package differences can change numerical results. Load only trusted checkpoints.
