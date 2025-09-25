# Linear Regression Module

This module trains and evaluates a Linear Regression model for the Cancer Mortality dataset using the preprocessing pipeline from `src/PreProcessing.py`.

## Train and Test

From the project root:

```bash
# Train, evaluate, save artifacts, and generate a performance plot
python3 src/LinearRegression/linearregression.py
# or
python -m src.LinearRegression.linearregression
```

## What the code does

- Uses `CancerDataPreprocessor` to:
  - Handle missing values, categorical variables, outliers, log-transform skewed features
  - Split into Train/Val/Test (70/15/15)
  - Scale features using RobustScaler (train-only fit)
- Trains `sklearn.linear_model.LinearRegression`
- Saves artifacts under `<project_root>/models/`:
  - `linear_regression.joblib` (weights)
  - `linear_regression_metrics.json` (Train/Val/Test metrics)
  - `linear_regression_test_predictions.npy` (test predictions)
  - `linear_regression_performance.png` (parity plot for Step 6)
- Provides a `test_model()` function that loads the saved model and evaluates it on the test set

## Functions

- `train_model(data_csv=None, save_model_path=None)`: Train and save the model, return metrics
- `test_model(model_path=None, data_csv=None)`: Load saved model and compute Test MSE and R²
