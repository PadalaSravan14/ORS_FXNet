
import numpy as np


def inverse_minmax(values, y_min, y_max):
    
    return y_min + np.asarray(values) * (y_max - y_min)


def root_mean_squared_error(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))  # Eq. (45)


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))  # Eq. (46)


def mean_absolute_percentage_error(y_true, y_pred, eps=1e-8):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / (y_true + eps))))  # Eq. (47)


def directional_accuracy(y_true, y_pred):
   
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) < 2:
        return float("nan")
    true_prev = y_true[:-1]
    true_next = y_true[1:]
    pred_next = y_pred[1:]
    correct = ((true_next - true_prev) * (pred_next - true_prev)) > 0
    return float(np.mean(correct) * 100.0)


def volatility_adjusted_error(y_true, y_pred, rolling_window=21, eps=1e-8):
   
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    series = np.asarray(y_true, dtype=float)
    vol = np.zeros_like(series)
    for i in range(len(series)):
        start = max(0, i - rolling_window + 1)
        window = series[start:i + 1]
        vol[i] = np.std(window) if len(window) > 1 else eps
    vol = np.where(vol < eps, eps, vol)
    return float(np.mean(np.abs(y_true - y_pred) / vol))


def r_squared(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return float(1 - ss_res / ss_tot)


def compute_all_metrics(y_true, y_pred, rolling_window=21):
    return {
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "Directional_Accuracy": directional_accuracy(y_true, y_pred),
        "VAE": volatility_adjusted_error(y_true, y_pred, rolling_window),
        "R2": r_squared(y_true, y_pred),
    }
