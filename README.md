# Energy Consumption Prediction ⚡

A reproducible machine-learning project predicting household appliance energy use (Wh per 10-minute interval) from UCI Appliances Energy Prediction data. **The data comes from one household**, so predictions do not generalize automatically to other homes.

## Structure
- `data/`: original UCI CSV (19,735 rows)
- `notebooks/`: Analysis notebook: EDA, feature engineering and comparison of five regression models
- `src/features.py`: shared feature engineering for training and app
- `src/train_and_export.py`: trains and exports Extra Trees
- `models/`: serialized trained model (created by training)
- `app/app.py`: Streamlit user interface
- `reports/`: random-split and chronological evaluation metrics

## Run
From the **project root**:
```bash
python -m pip install -r requirements.txt
python src/train_and_export.py
python -m streamlit run app/app.py
```

## Modeling
- Target: `Appliances` (Wh / 10 minutes).
- Removed random features `rv1`, `rv2`.
- Cyclical time features `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, plus `month`; raw hour/day-of-week are not retained.
- The notebook compares Linear, Log-Linear, Random Forest, Gradient Boosting and Extra Trees.
- The app uses the **Extra Trees** model exported by `src/train_and_export.py` with exactly the same feature transformations.
- `reports/metrics.json` reproduces a random 80/20 split. `reports/chronological_metrics.json` evaluates a separate earlier-to-later holdout. **Random-split performance is not a reliable estimate of future forecasting performance.**

## Notes
- `models/energy_model.joblib` is a Python joblib artifact. Only load artifacts from trusted sources.
- Default sensor readings in the app are training medians and are illustrative, not live measurements.
- Source: [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction).

## Updated conclusions and recommendations
See [`reports/FINAL_REPORT_AR.md`](reports/FINAL_REPORT_AR.md) for the final Arabic report, five-line summary, limitations and recommendations. A chronological holdout **has already been run** and showed poor future-period generalization (R² ≈ -0.931). Next steps are time-aware validation, fair model tuning, temporal feature ablations and broader household/season coverage. `reports/README_AR_legacy_archive.md` is archived historical documentation, **not** the current run guide.
