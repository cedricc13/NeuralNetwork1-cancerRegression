import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Simple import that works when running this file directly
try:
    from PreProcessing import CancerDataPreprocessor
except ModuleNotFoundError:
    import sys
    from pathlib import Path as _Path
    sys.path.append(str(_Path(__file__).resolve().parents[1]))  # add project/src to path
    from PreProcessing import CancerDataPreprocessor


def get_default_paths() -> dict:
    """Return simple absolute paths for data and artifacts (project root based)."""
    base_dir = Path(__file__).resolve().parents[2]
    models_dir = base_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    return {
        'data_csv': str(base_dir / 'cancer_reg-1.csv'),
        'model_path': str(models_dir / 'linear_regression.joblib'),
        'metrics_path': str(models_dir / 'linear_regression_metrics.json'),
        'predictions_path': str(models_dir / 'linear_regression_test_predictions.npy'),
        'plot_path': str(models_dir / 'linear_regression_performance.png'),
    }


def prepare_data(data_csv: str) -> CancerDataPreprocessor:
    """
    Load and preprocess the dataset using the pipeline defined in PreProcessing.

    Returns a fitted preprocessor object exposing:
      - X_train_scaled, X_val_scaled, X_test_scaled
      - y_train, y_val, y_test
    """
    preprocessor = CancerDataPreprocessor(data_file=data_csv)
    preprocessor.run_all_steps(
        missing_strategy='median',
        keep_geography=False,
        keep_binned_inc=True,
        scaling_method='robust',
        test_size=0.15,
        val_size=0.15,
        shuffle=True,
        random_state=42,
    )
    return preprocessor


def train_model(data_csv: str | None = None, save_model_path: str | None = None) -> dict:
    """
    Train a Linear Regression model on preprocessed data and save the trained model.

    Returns a metrics dictionary.
    """
    paths = get_default_paths()
    data_csv = data_csv or paths['data_csv']
    save_model_path = save_model_path or paths['model_path']

    preprocessor = prepare_data(data_csv)

    model = LinearRegression()
    model.fit(preprocessor.X_train_scaled, preprocessor.y_train)

    y_train_pred = model.predict(preprocessor.X_train_scaled)
    y_val_pred = model.predict(preprocessor.X_val_scaled)
    y_test_pred = model.predict(preprocessor.X_test_scaled)

    metrics = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'model': 'LinearRegression',
        'train_mse': float(mean_squared_error(preprocessor.y_train, y_train_pred)),
        'val_mse': float(mean_squared_error(preprocessor.y_val, y_val_pred)),
        'test_mse': float(mean_squared_error(preprocessor.y_test, y_test_pred)),
        'train_r2': float(r2_score(preprocessor.y_train, y_train_pred)),
        'val_r2': float(r2_score(preprocessor.y_val, y_val_pred)),
        'test_r2': float(r2_score(preprocessor.y_test, y_test_pred)),
        'n_features': int(preprocessor.X_train_scaled.shape[1]),
        'n_train': int(preprocessor.X_train_scaled.shape[0]),
        'n_val': int(preprocessor.X_val_scaled.shape[0]),
        'n_test': int(preprocessor.X_test_scaled.shape[0]),
    }

    # Save artifacts
    joblib.dump(model, save_model_path)
    with open(paths['metrics_path'], 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    np.save(paths['predictions_path'], y_test_pred)

    # Simple performance plot for Step 6 (Predicted vs Actual on Test)
    plot_performance(preprocessor.y_test, y_test_pred, paths['plot_path'])
    metrics['performance_plot'] = paths['plot_path']

    print("Linear Regression trained and saved")
    print(json.dumps(metrics, indent=2))
    return metrics


def test_model(model_path: str | None = None, data_csv: str | None = None) -> dict:
    """
    Load a trained Linear Regression model and evaluate on the test set.

    This re-runs the preprocessing pipeline with the same defaults to produce
    the Train/Val/Test splits and scaling, then evaluates the loaded model on
    the test split.
    """
    paths = get_default_paths()
    model_path = model_path or paths['model_path']
    data_csv = data_csv or paths['data_csv']

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run with --train first.")

    model: LinearRegression = joblib.load(model_path)
    preprocessor = prepare_data(data_csv)

    y_test_pred = model.predict(preprocessor.X_test_scaled)

    metrics = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'model': 'LinearRegression',
        'test_mse': float(mean_squared_error(preprocessor.y_test, y_test_pred)),
        'test_r2': float(r2_score(preprocessor.y_test, y_test_pred)),
        'n_test': int(preprocessor.X_test_scaled.shape[0]),
    }

    print("Loaded model evaluated on the test set")
    print(json.dumps(metrics, indent=2))
    return metrics

def plot_performance(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> None:
    """Create a simple parity plot (y_true vs y_pred) and save it."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=12, alpha=0.6, label='Test samples')
    min_v = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_v = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=1, label='y = x')
    plt.xlabel('Actual TARGET_deathRate')
    plt.ylabel('Predicted TARGET_deathRate')
    plt.title('Linear Regression — Test Set Predictions')
    plt.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    # Simple default: train, save, and produce Step 6 plot; also print metrics.
    train_model()


