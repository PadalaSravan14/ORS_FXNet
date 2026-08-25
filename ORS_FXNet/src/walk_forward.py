
import copy
import numpy as np
import pandas as pd

from config import DEFAULT_TRAINING
from src.data_preparation import build_sliding_window_samples
from src.datasets import WindowedExchangeRateDataset, make_dataloader
from src.model import CurrencyForecastingNetwork
from src.train import train_neural_forecaster, generate_base_predictions
from src.evaluation_metrics import compute_all_metrics


def walk_forward_folds(df, date_col, fold_boundaries):
  
    folds = []
    for train_end, test_start, test_end in fold_boundaries:
        train_df = df[df[date_col].dt.year <= train_end].reset_index(drop=True)
        test_df = df[(df[date_col].dt.year >= test_start) & (df[date_col].dt.year <= test_end)].reset_index(drop=True)
        folds.append((train_df, test_df))
    return folds


def run_walk_forward_cv(df, date_col, target_col, feature_cols, pair, model_builder,
                         fold_boundaries, window, macro_idx, tech_idx, device="cpu",
                         quick_epochs=None):
    training_cfg = copy.deepcopy(DEFAULT_TRAINING)
    if quick_epochs is not None:
        training_cfg.epochs = quick_epochs

    fold_results = []
    for i, (train_df, test_df) in enumerate(walk_forward_folds(df, date_col, fold_boundaries)):
        if len(train_df) <= window or len(test_df) <= window:
            continue
        X_train, y_train, _ = build_sliding_window_samples(train_df, feature_cols, target_col, window)
        X_test, y_test, _ = build_sliding_window_samples(test_df, feature_cols, target_col, window)
        if len(X_train) == 0 or len(X_test) == 0:
            continue

        train_ds = WindowedExchangeRateDataset(X_train, y_train)
        test_ds = WindowedExchangeRateDataset(X_test, y_test)
        train_loader = make_dataloader(train_ds, batch_size=training_cfg.batch_size, shuffle=True)
        test_loader = make_dataloader(test_ds, batch_size=training_cfg.batch_size, shuffle=False)

        model = model_builder()
        model, _ = train_neural_forecaster(
            model, train_loader, test_loader, pair, training_cfg,
            short_window=min(21, window), long_window=window,
            macro_idx=macro_idx, tech_idx=tech_idx, device=device, verbose=False,
        )
        y_pred, y_true, _, _ = generate_base_predictions(
            model, test_loader, pair, min(21, window), window, macro_idx, tech_idx, device=device,
        )
        metrics = compute_all_metrics(y_true, y_pred)
        fold_results.append({
            "fold": i + 1,
            "train_period": f"{train_df[date_col].dt.year.min()}-{train_df[date_col].dt.year.max()}",
            "test_period": f"{test_df[date_col].dt.year.min()}-{test_df[date_col].dt.year.max()}",
            **metrics,
        })
    return pd.DataFrame(fold_results)


def run_multi_horizon_forecast(df, target_col, feature_cols, pair, model_builder,
                                window, horizons, macro_idx, tech_idx, device="cpu",
                                quick_epochs=None):
    training_cfg = copy.deepcopy(DEFAULT_TRAINING)
    if quick_epochs is not None:
        training_cfg.epochs = quick_epochs

    results = []
    for horizon in horizons:
        X, y, _ = build_sliding_window_samples(df, feature_cols, target_col, window, horizon=horizon)
        if len(X) < 50:
            continue
        split = int(len(X) * 0.8)
        train_ds = WindowedExchangeRateDataset(X[:split], y[:split])
        test_ds = WindowedExchangeRateDataset(X[split:], y[split:])
        train_loader = make_dataloader(train_ds, batch_size=training_cfg.batch_size, shuffle=True)
        test_loader = make_dataloader(test_ds, batch_size=training_cfg.batch_size, shuffle=False)

        model = model_builder()
        model, _ = train_neural_forecaster(
            model, train_loader, test_loader, pair, training_cfg,
            short_window=min(21, window), long_window=window,
            macro_idx=macro_idx, tech_idx=tech_idx, device=device, verbose=False,
        )
        y_pred, y_true, _, _ = generate_base_predictions(
            model, test_loader, pair, min(21, window), window, macro_idx, tech_idx, device=device,
        )
        metrics = compute_all_metrics(y_true, y_pred)
        results.append({"forecast_horizon_days": horizon, **metrics})
    return pd.DataFrame(results)
