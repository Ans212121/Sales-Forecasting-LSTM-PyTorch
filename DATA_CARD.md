# Data card

Source: Kaggle Store Sales — Time Series Forecasting, Corporación Favorita, Ecuador. https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

Verification download: public third-party mirror https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting/tree/5768b1d . This is not the original publisher; SHA-256 hashes are recorded in reports/metrics.json for traceability. Users should obtain the original competition data and check its rules. Raw files are not included.

Used columns: train `date`, `sales`; test `date`. The pipeline sums sales across all stores and families; sales are units of products, not revenue. Missing calendar dates are forward-filled causally and counted. Invalid dates, missing/nonfinite/negative sales and unexpected future-date ranges fail validation.

The last 15 historical days are held out before scaler fitting and model training. Other competition variables (promotions, oil, stores, holidays, transactions) are not used. The aggregation loses store/family detail. No personal or government identification data is included.

Horizon correction: the Kaggle test file has 16 dates (16–31 August 2017). The original notebook used all test dates while labeling them 15 days. The revised pipeline explicitly selects the first 15 (16–30 August) to match the project goal.
