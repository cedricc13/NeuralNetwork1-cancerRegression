import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow import keras
from tensorflow.keras import layers

try:
    from PreProcessing import CancerDataPreprocessor
except ModuleNotFoundError:
    import sys
    from pathlib import Path as _Path
    sys.path.append(str(_Path(__file__).resolve().parents[1]))
    from PreProcessing import CancerDataPreprocessor
class ConsoleLossLogger(keras.callbacks.Callback):
    def __init__(self, max_epochs: int):
        super().__init__()
        self.max_epochs = int(max_epochs)
    def on_epoch_end(self, epoch: int, logs=None):  # type: ignore[override]
        logs = logs or {}
        train = logs.get('loss')
        val = logs.get('val_loss')
        if train is None:
            return
        if val is None:
            print(f"Epoch {epoch+1}/{self.max_epochs} – Train Loss: {float(train):.6f}")
        else:
            print(f"Epoch {epoch+1}/{self.max_epochs} – Train Loss: {float(train):.6f} – Val Loss: {float(val):.6f}")


ARCH_NAME = "DNN-30-16-8-4"
ARCH = [30, 16, 8, 4]
LEARNING_RATES = [0.1, 0.01, 0.001, 0.0001]


def _lr_tag(lr: float) -> str:
    return str(lr).replace('.', 'p')


def get_paths(lr: float) -> dict:
    base_dir = Path(__file__).resolve().parents[2]
    models_dir = base_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"dnn30_16_8_4_lr_{_lr_tag(lr)}"
    return {
        'data_csv': str(base_dir / 'cancer_reg-1.csv'),
        'model_path': str(models_dir / f'{prefix}.keras'),
        'metrics_path': str(models_dir / f'{prefix}_metrics.json'),
        'predictions_path': str(models_dir / f'{prefix}_test_predictions.npy'),
        'plot_path': str(models_dir / f'{prefix}_performance.png'),
    }


def prepare_data(data_csv: str) -> CancerDataPreprocessor:
    pp = CancerDataPreprocessor(data_file=data_csv)
    pp.run_all_steps()
    return pp


def build_model(input_dim: int, learning_rate: float) -> keras.Sequential:
    model = keras.Sequential()
    model.add(layers.Dense(ARCH[0], activation='relu', input_shape=(input_dim,)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(ARCH[1], activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(ARCH[2], activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(ARCH[3], activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1, activation='linear'))
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    return model


def plot_performance(history: 'keras.callbacks.History', out_path: str, title: str) -> None:
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history.history['loss'], label='train')
    plt.plot(epochs, history.history.get('val_loss', []), label='val')
    plt.xlabel('epoch')
    plt.ylabel('mean squared error')
    plt.title('Model loss' if not title else title)
    plt.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train_model(learning_rate: float, data_csv: str | None = None) -> dict:
    paths = get_paths(learning_rate)
    data_csv = data_csv or paths['data_csv']

    pp = prepare_data(data_csv)
    model = build_model(pp.X_train_scaled.shape[1], learning_rate)

    early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    max_epochs = 1000
    history = model.fit(
        pp.X_train_scaled, pp.y_train,
        validation_data=(pp.X_val_scaled, pp.y_val),
        epochs=max_epochs,
        batch_size=128,
        verbose=0,
        callbacks=[early, ConsoleLossLogger(max_epochs)],
    )

    y_train_pred = model.predict(pp.X_train_scaled, verbose=0).flatten()
    y_val_pred = model.predict(pp.X_val_scaled, verbose=0).flatten()
    y_test_pred = model.predict(pp.X_test_scaled, verbose=0).flatten()
    if np.isnan(y_test_pred).any():
        raise ValueError("NaN predictions encountered. Lower LR or review preprocessing.")

    metrics = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'model': ARCH_NAME,
        'learning_rate': learning_rate,
        'epochs_trained': len(history.history['loss']),
        'train_mse': float(mean_squared_error(pp.y_train, y_train_pred)),
        'val_mse': float(mean_squared_error(pp.y_val, y_val_pred)),
        'test_mse': float(mean_squared_error(pp.y_test, y_test_pred)),
        'train_r2': float(r2_score(pp.y_train, y_train_pred)),
        'val_r2': float(r2_score(pp.y_val, y_val_pred)),
        'test_r2': float(r2_score(pp.y_test, y_test_pred)),
    }

    model.save(paths['model_path'])
    with open(paths['metrics_path'], 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    np.save(paths['predictions_path'], y_test_pred)

    plot_performance(history, paths['plot_path'], f'{ARCH_NAME} — Model loss (LR={learning_rate})')
    metrics['performance_plot'] = paths['plot_path']

    print(f"{ARCH_NAME} trained and saved")
    print(json.dumps(metrics, indent=2))
    return metrics


def test_model(model_path: str | None = None, data_csv: str | None = None) -> dict:
    paths = get_paths()
    model_path = model_path or paths['model_path']
    data_csv = data_csv or paths['data_csv']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train first.")
    model = keras.models.load_model(model_path)
    pp = prepare_data(data_csv)
    y_test_pred = model.predict(pp.X_test_scaled, verbose=0).flatten()
    metrics = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'model': ARCH_NAME,
        'test_mse': float(mean_squared_error(pp.y_test, y_test_pred)),
        'test_r2': float(r2_score(pp.y_test, y_test_pred)),
    }
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == '__main__':
    for lr in LEARNING_RATES:
        train_model(lr)


