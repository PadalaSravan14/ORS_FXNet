
import argparse
import json
import os
import pickle
import time
import copy

import numpy as np
import pandas as pd
import torch

from config import (
    CURRENCY_PAIRS, SLIDING_WINDOW, SHORT_TERM_WINDOW, LONG_TERM_WINDOW,
    FORECAST_HORIZONS, RANDOM_SEED, TABLES_DIR, MODELS_DIR, LOGS_DIR,
    DEFAULT_SHORT_TERM, DEFAULT_LONG_TERM, DEFAULT_ATTENTION, DEFAULT_MACRO_AE,
    DEFAULT_FUSION_HEAD, DEFAULT_DQN, DEFAULT_DDPG, DEFAULT_PPO, DEFAULT_TRAINING,
    DATE_COLUMN,
)
from src.data_preparation import (
    full_dataset_pipeline, add_technical_indicators, FeatureScalingBundle,
    chronological_split, build_sliding_window_samples, feature_column_set,
)
from src.datasets import WindowedExchangeRateDataset, make_dataloader
from src.model import CurrencyForecastingNetwork
from src.train import (
    train_neural_forecaster, generate_base_predictions,
    train_correction_ensemble, apply_correction_ensemble,
)
from src.baselines import (
    LSTMForecaster, BiLSTMForecaster, GRUForecaster,
    ConvolutionalLSTMForecaster, TransformerForecaster,
    fit_arima_forecast, fit_var_forecast,
)
from src.evaluation_metrics import compute_all_metrics
from src.statistical_tests import run_significance_tests
from src.ablation import run_ablation_study
from src.walk_forward import run_walk_forward_cv, run_multi_horizon_forecast
from src.visualization_eda import generate_all_eda_plots
from src.visualization_architecture import architecture_pipeline_diagram
from src.visualization_results import (
    actual_vs_predicted_plot, residual_curve_plot, scatter_fidelity_plot,
    directional_confusion_matrix_plot, error_distribution_plot,
    rmse_stability_across_pairs_plot, training_time_comparison_plot,
    ablation_radar_chart,
)


def set_random_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="ORS-FXNet end-to-end pipeline")
    parser.add_argument("--quick", action="store_true",
                         help="Run a fast smoke test with reduced epochs/episodes.")
    parser.add_argument("--pairs", nargs="+", default=CURRENCY_PAIRS,
                         help="Currency pairs to model (subset of USD EUR GBP JPY AUD).")
    parser.add_argument("--epochs", type=int, default=None, help="Override neural training epochs.")
    parser.add_argument("--rl-episodes", type=int, default=None, help="Override RL training episodes.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_ors_fxnet(n_features, n_macro, n_tech, pairs, ablation=None):
    return CurrencyForecastingNetwork(
        n_features=n_features, n_macro_features=n_macro, n_tech_features=n_tech,
        currency_pairs=pairs,
        short_cfg=DEFAULT_SHORT_TERM, long_cfg=DEFAULT_LONG_TERM, attn_cfg=DEFAULT_ATTENTION,
        macro_cfg=DEFAULT_MACRO_AE, fusion_cfg=DEFAULT_FUSION_HEAD, ablation=ablation,
    )


def main():
    args = parse_args()
    set_random_seed()

    training_cfg = copy.deepcopy(DEFAULT_TRAINING)
    if args.quick:
        training_cfg.epochs = 3
        training_cfg.rl_episodes = 20
        training_cfg.early_stopping_patience = 2
    if args.epochs is not None:
        training_cfg.epochs = args.epochs
    rl_episodes = args.rl_episodes or training_cfg.rl_episodes

    run_log = {"start_time": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args)}


    print("== Step 1: dataset construction and preprocessing ==")
    inr_df = full_dataset_pipeline()
    inr_df = add_technical_indicators(inr_df, price_col="USD_INR")

    target_col = "USD_INR"
    pair_cols = [f"{p}_INR" if p != "USD" else "USD_INR" for p in args.pairs]
    macro_cols = ["Gold_Price", "Crude_Oil", "Inflation_CPI"]
    tech_cols = [c for c in inr_df.columns if c.startswith("USD_INR_") and c != "USD_INR_target_col"]

    scaler_bundle = FeatureScalingBundle()
    minmax_cols = [c for c in inr_df.columns if c not in [DATE_COLUMN, "USD_INR_target_col"] + macro_cols]
    scaled_df = scaler_bundle.fit_transform(inr_df, minmax_cols=minmax_cols, robust_cols=macro_cols)

    y_min = inr_df[target_col].min()
    y_max = inr_df[target_col].max()

    with open(os.path.join(MODELS_DIR, "scaler_bundle.pkl"), "wb") as f:
        pickle.dump(scaler_bundle, f)


    print("== Step 2: exploratory data analysis plots ==")
    eda_paths = generate_all_eda_plots(inr_df, DATE_COLUMN, target_col, pair_cols, macro_cols)
    eda_paths.append(architecture_pipeline_diagram())
    run_log["eda_figures"] = eda_paths

    print("== Step 3: sliding-window construction and chronological split ==")
    feature_cols = feature_column_set(scaled_df)
    train_df, val_df, test_df = chronological_split(scaled_df)

    def make_loaders(target_pair_col):
        X_tr, y_tr, _ = build_sliding_window_samples(train_df, feature_cols, target_pair_col, SLIDING_WINDOW)
        X_va, y_va, _ = build_sliding_window_samples(val_df, feature_cols, target_pair_col, SLIDING_WINDOW)
        X_te, y_te, d_te = build_sliding_window_samples(test_df, feature_cols, target_pair_col, SLIDING_WINDOW)
        tr = make_dataloader(WindowedExchangeRateDataset(X_tr, y_tr), training_cfg.batch_size, shuffle=True)
        va = make_dataloader(WindowedExchangeRateDataset(X_va, y_va), training_cfg.batch_size, shuffle=False)
        te = make_dataloader(WindowedExchangeRateDataset(X_te, y_te), training_cfg.batch_size, shuffle=False)
        return tr, va, te, d_te, (X_va, y_va), (X_te, y_te)

    macro_idx = [feature_cols.index(c) for c in macro_cols if c in feature_cols]
    tech_idx = [feature_cols.index(c) for c in tech_cols if c in feature_cols]
    n_features = len(feature_cols)

    all_pair_metrics = {}
    all_pair_predictions = {}
    baseline_training_times = {}

    print("== Step 4: baseline model training (ARIMA, VAR, LSTM, BiLSTM, GRU, CNN-LSTM, Transformer) ==")
    primary_pair = "USD"
    primary_col = "USD_INR"
    train_loader, val_loader, test_loader, test_dates, val_arrays, test_arrays = make_loaders(primary_col)

    baseline_errors = {}
    baseline_metrics_table = {}

    t0 = time.time()
    try:
        arima_pred = fit_arima_forecast(train_df[primary_col].values, len(test_arrays[1]))
        m = compute_all_metrics(test_arrays[1], arima_pred)
        baseline_metrics_table["ARIMA"] = m
        baseline_errors["ARIMA"] = test_arrays[1] - arima_pred
    except Exception as e:
        print("ARIMA failed:", e)
    baseline_training_times["ARIMA"] = (time.time() - t0) / 60.0

    t0 = time.time()
    try:
        var_input_cols = [primary_col] + [c for c in pair_cols if c != primary_col][:2] + macro_cols
        var_input_cols = [c for c in var_input_cols if c in train_df.columns]
        var_pred = fit_var_forecast(train_df[var_input_cols], len(test_arrays[1]), primary_col)
        m = compute_all_metrics(test_arrays[1], var_pred)
        baseline_metrics_table["VAR"] = m
        baseline_errors["VAR"] = test_arrays[1] - var_pred
    except Exception as e:
        print("VAR failed:", e)
    baseline_training_times["VAR"] = (time.time() - t0) / 60.0

    neural_baselines = {
        "LSTM": LSTMForecaster, "BiLSTM": BiLSTMForecaster, "GRU": GRUForecaster,
        "CNN-LSTM": ConvolutionalLSTMForecaster, "Transformer": TransformerForecaster,
    }
    for name, cls in neural_baselines.items():
        print(f"  training baseline: {name}")
        t0 = time.time()
        model = cls(input_dim=n_features).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_val, best_state, patience = float("inf"), copy.deepcopy(model.state_dict()), 0
        for epoch in range(training_cfg.epochs):
            model.train()
            for batch in train_loader:
                seq = batch["sequence"].to(args.device)
                target = batch["target"].to(args.device)
                pred = model(seq)
                loss = torch.nn.functional.huber_loss(pred, target, delta=training_cfg.huber_delta)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                v_losses = []
                for batch in val_loader:
                    seq = batch["sequence"].to(args.device); target = batch["target"].to(args.device)
                    v_losses.append(torch.nn.functional.huber_loss(model(seq), target).item())
                val_loss = float(np.mean(v_losses)) if v_losses else float("inf")
            if val_loss < best_val - 1e-6:
                best_val, best_state, patience = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                patience += 1
                if patience >= training_cfg.early_stopping_patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                seq = batch["sequence"].to(args.device)
                preds.append(model(seq).cpu().numpy())
                trues.append(batch["target"].numpy())
        preds, trues = np.concatenate(preds), np.concatenate(trues)
        baseline_metrics_table[name] = compute_all_metrics(trues, preds)
        baseline_errors[name] = trues - preds
        baseline_training_times[name] = (time.time() - t0) / 60.0

    print("== Step 5: ORS-FXNet training (Dual-Phase LSTM + Attention Fusion + Macro AE + RL Ensemble) ==")
    t0 = time.time()
    ors_fxnet_errors_primary = None
    for pair_code, pair_col in zip(args.pairs, pair_cols):
        print(f"  -- currency pair: {pair_col} --")
        tr, va, te, dates_te, val_arrays_p, test_arrays_p = make_loaders(pair_col)

        model = build_ors_fxnet(n_features, len(macro_idx), len(tech_idx), [pair_code])
        model, history = train_neural_forecaster(
            model, tr, va, pair_code, training_cfg,
            SHORT_TERM_WINDOW, LONG_TERM_WINDOW, macro_idx, tech_idx, device=args.device, verbose=False,
        )

        # base predictions on validation (for RL training) and test (for final eval)
        y_base_val, y_true_val, temporal_val, macro_val = generate_base_predictions(
            model, va, pair_code, SHORT_TERM_WINDOW, LONG_TERM_WINDOW, macro_idx, tech_idx, device=args.device,
        )
        y_base_test, y_true_test, temporal_test, macro_test = generate_base_predictions(
            model, te, pair_code, SHORT_TERM_WINDOW, LONG_TERM_WINDOW, macro_idx, tech_idx, device=args.device,
        )

        ensemble, reward_history = train_correction_ensemble(
            y_base_val, y_true_val, temporal_val, macro_val,
            DEFAULT_DQN, DEFAULT_DDPG, DEFAULT_PPO,
            episodes=rl_episodes, episode_length=training_cfg.rl_episode_length,
            device=args.device, verbose=False,
        )
        y_final_test = apply_correction_ensemble(ensemble, y_base_test, y_true_test, temporal_test, macro_test)

        metrics = compute_all_metrics(y_true_test, y_final_test)
        all_pair_metrics[pair_col] = metrics
        all_pair_predictions[pair_col] = {
            "dates": dates_te, "y_true": y_true_test, "y_pred": y_final_test, "y_base": y_base_test,
        }

        torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"ors_fxnet_{pair_col}.pt"))
        ensemble.save(os.path.join(MODELS_DIR, f"rl_ensemble_{pair_col}.pt"))

        if pair_col == primary_col:
            ors_fxnet_errors_primary = y_true_test - y_final_test
            baseline_metrics_table["Proposed ORS-FXNet"] = metrics

    baseline_training_times["ORS-FXNet"] = (time.time() - t0) / 60.0

    print("== Step 6: statistical significance analysis ==")
    significance_results = run_significance_tests(ors_fxnet_errors_primary, baseline_errors)
    pd.DataFrame(significance_results).to_csv(
        os.path.join(TABLES_DIR, "statistical_significance_tests.csv"), index=False)


    print("== Step 7: ablation study ==")
    ablation_results, _ = run_ablation_study(
        n_features, len(macro_idx), len(tech_idx), primary_pair,
        train_loader, val_loader, test_loader,
        SHORT_TERM_WINDOW, LONG_TERM_WINDOW, macro_idx, tech_idx,
        device=args.device, quick_epochs=training_cfg.epochs,
    )
    ablation_df = pd.DataFrame(ablation_results).T
    ablation_df.to_csv(os.path.join(TABLES_DIR, "ablation_study.csv"))

    
    print("== Step 8: walk-forward cross-validation and multi-horizon forecasting ==")
    fold_boundaries = [
        (2010, 2011, 2012), (2012, 2013, 2014), (2014, 2015, 2016),
        (2016, 2017, 2018), (2018, 2019, 2019),
    ]
    walk_forward_df = run_walk_forward_cv(
        scaled_df, DATE_COLUMN, primary_col, feature_cols, primary_pair,
        lambda: build_ors_fxnet(n_features, len(macro_idx), len(tech_idx), [primary_pair]),
        fold_boundaries, SLIDING_WINDOW, macro_idx, tech_idx,
        device=args.device, quick_epochs=training_cfg.epochs,
    )
    walk_forward_df.to_csv(os.path.join(TABLES_DIR, "walk_forward_cross_validation.csv"), index=False)

    multi_horizon_df = run_multi_horizon_forecast(
        scaled_df, primary_col, feature_cols, primary_pair,
        lambda: build_ors_fxnet(n_features, len(macro_idx), len(tech_idx), [primary_pair]),
        SLIDING_WINDOW, FORECAST_HORIZONS, macro_idx, tech_idx,
        device=args.device, quick_epochs=training_cfg.epochs,
    )
    multi_horizon_df.to_csv(os.path.join(TABLES_DIR, "multi_horizon_forecasting.csv"), index=False)

    print("== Step 9: writing result tables ==")
    pd.DataFrame(baseline_metrics_table).T.to_csv(
        os.path.join(TABLES_DIR, "baseline_vs_proposed_comparison.csv"))
    pd.DataFrame(all_pair_metrics).T.to_csv(
        os.path.join(TABLES_DIR, "per_currency_pair_performance.csv"))
    pd.DataFrame(baseline_training_times.items(), columns=["Model", "Training_Time_Minutes"]).to_csv(
        os.path.join(TABLES_DIR, "training_time_comparison.csv"), index=False)

    print("== Step 10: result plots ==")
    for pair_col, result in all_pair_predictions.items():
        safe_name = pair_col.replace("/", "_")
        actual_vs_predicted_plot(result["dates"], result["y_true"], result["y_pred"], pair_col,
                                  f"actual_vs_predicted_{safe_name}.png")
        residual_curve_plot(result["dates"], result["y_true"], result["y_pred"], pair_col,
                             f"residual_curve_{safe_name}.png")
        scatter_fidelity_plot(result["y_true"], result["y_pred"], pair_col,
                               f"scatter_fidelity_{safe_name}.png")

    primary_result = all_pair_predictions[primary_col]
    directional_confusion_matrix_plot(primary_result["y_true"], primary_result["y_pred"])
    error_distribution_plot(primary_result["y_true"], primary_result["y_pred"])
    rmse_stability_across_pairs_plot({k: v["RMSE"] for k, v in all_pair_metrics.items()})
    training_time_comparison_plot(baseline_training_times)

    metric_names = ["RMSE", "MAE", "MAPE", "Directional_Accuracy", "VAE"]
    ablation_radar_chart(ablation_results, metric_names)

    run_log["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(LOGS_DIR, "run_log.json"), "w") as f:
        json.dump(run_log, f, indent=2, default=str)

    metadata = {
        "feature_cols": feature_cols,
        "macro_idx": macro_idx,
        "tech_idx": tech_idx,
        "macro_cols": macro_cols,
        "tech_cols": tech_cols,
        "pair_cols": pair_cols,
        "pairs": args.pairs,
        "primary_pair": primary_pair,
        "primary_col": primary_col,
        "sliding_window": SLIDING_WINDOW,
        "short_term_window": SHORT_TERM_WINDOW,
        "long_term_window": LONG_TERM_WINDOW,
        "n_features": n_features,
        "n_macro": len(macro_idx),
        "n_tech": len(tech_idx),
        "target_col": target_col,
    }
    with open(os.path.join(MODELS_DIR, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nPipeline complete. See outputs/figures, outputs/tables, outputs/models.")


if __name__ == "__main__":
    main()
