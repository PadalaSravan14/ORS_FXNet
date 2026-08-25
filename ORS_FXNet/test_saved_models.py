
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch

from config import (
    MODELS_DIR, TABLES_DIR, DATE_COLUMN,
    DEFAULT_SHORT_TERM, DEFAULT_LONG_TERM, DEFAULT_ATTENTION, DEFAULT_MACRO_AE,
    DEFAULT_FUSION_HEAD, DEFAULT_DQN, DEFAULT_DDPG, DEFAULT_PPO,
)
from src.data_preparation import (
    full_dataset_pipeline, add_technical_indicators, chronological_split,
    build_sliding_window_samples,
)
from src.datasets import WindowedExchangeRateDataset, make_dataloader
from src.model import CurrencyForecastingNetwork
from src.rl_agents import MultiAgentCorrectionEnsemble
from src.train import generate_base_predictions, apply_correction_ensemble
from src.evaluation_metrics import compute_all_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Test saved ORS-FXNet models on sample test data.")
    parser.add_argument("--n-samples", type=int, default=10,
                         help="Number of held-out test samples to display per currency pair.")
    parser.add_argument("--pairs", nargs="+", default=None,
                         help="Currency pair columns to test, e.g. USD_INR EUR_INR. Defaults to all trained pairs.")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_metadata():
    path = os.path.join(MODELS_DIR, "run_metadata.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "outputs/models/run_metadata.json not found. Run `python run_pipeline.py` "
            "at least once before testing saved models."
        )
    with open(path) as f:
        return json.load(f)


def load_scaler_bundle():
    path = os.path.join(MODELS_DIR, "scaler_bundle.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def rebuild_model(metadata, pair_code):
    return CurrencyForecastingNetwork(
        n_features=metadata["n_features"], n_macro_features=metadata["n_macro"],
        n_tech_features=metadata["n_tech"], currency_pairs=[pair_code],
        short_cfg=DEFAULT_SHORT_TERM, long_cfg=DEFAULT_LONG_TERM, attn_cfg=DEFAULT_ATTENTION,
        macro_cfg=DEFAULT_MACRO_AE, fusion_cfg=DEFAULT_FUSION_HEAD,
    )


def main():
    args = parse_args()
    metadata = load_metadata()
    scaler_bundle = load_scaler_bundle()

    pair_cols = args.pairs or metadata["pair_cols"]
    pair_lookup = dict(zip(metadata["pair_cols"], metadata["pairs"]))

    # Rebuild the same preprocessed dataset used during training (uses only
    # the real Foreign_Exchange_Rates.csv / GoldUP.csv files in data/raw/).
    inr_df = full_dataset_pipeline()
    inr_df = add_technical_indicators(inr_df, price_col="USD_INR")
    scaled_df = inr_df.copy()
    for col in metadata["feature_cols"]:
        if col in scaler_bundle.minmax_scalers:
            scaled_df[col] = scaler_bundle.minmax_scalers[col].transform(scaled_df[[col]]).flatten()
        elif col in scaler_bundle.robust_scalers:
            scaled_df[col] = scaler_bundle.robust_scalers[col].transform(scaled_df[[col]]).flatten()

    _, _, test_df = chronological_split(scaled_df)

    all_samples = []
    for pair_col in pair_cols:
        pair_code = pair_lookup.get(pair_col, pair_col.split("_")[0])
        model_path = os.path.join(MODELS_DIR, f"ors_fxnet_{pair_col}.pt")
        ensemble_path = os.path.join(MODELS_DIR, f"rl_ensemble_{pair_col}.pt")
        if not os.path.exists(model_path):
            print(f"[skip] no saved checkpoint for {pair_col} at {model_path}")
            continue

        print(f"\n=== Testing saved model for {pair_col} ===")
        model = rebuild_model(metadata, pair_code)
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        model.to(args.device)

        X_te, y_te, d_te = build_sliding_window_samples(
            test_df, metadata["feature_cols"], pair_col, metadata["sliding_window"]
        )
        test_loader = make_dataloader(WindowedExchangeRateDataset(X_te, y_te), batch_size=64, shuffle=False)

        y_base, y_true, temporal_summary, macro_summary = generate_base_predictions(
            model, test_loader, pair_code,
            metadata["short_term_window"], metadata["long_term_window"],
            metadata["macro_idx"], metadata["tech_idx"], device=args.device,
        )

        if os.path.exists(ensemble_path):
            ensemble = MultiAgentCorrectionEnsemble.load(
                ensemble_path, DEFAULT_DQN, DEFAULT_DDPG, DEFAULT_PPO, device=args.device,
            )
            y_pred = apply_correction_ensemble(ensemble, y_base, y_true, temporal_summary, macro_summary)
        else:
            print(f"[info] no RL ensemble checkpoint found for {pair_col}; reporting base forecast only.")
            y_pred = y_base

        # Inverse-transform back to real exchange-rate units (Eq. 43)
        y_true_real = scaler_bundle.inverse_transform_target(y_true, target_col=pair_col)
        y_pred_real = scaler_bundle.inverse_transform_target(y_pred, target_col=pair_col)

        n = min(args.n_samples, len(y_true_real))
        sample_idx = np.linspace(0, len(y_true_real) - 1, n, dtype=int)

        metrics_full = compute_all_metrics(y_true_real, y_pred_real)
        print(f"Full held-out test set metrics for {pair_col}: {metrics_full}")

        print(f"{'Date':<12}{'Actual':>12}{'Predicted':>12}{'AbsError':>12}")
        for idx in sample_idx:
            date_val = pd.Timestamp(d_te[idx]).strftime("%Y-%m-%d")
            actual = y_true_real[idx]
            predicted = y_pred_real[idx]
            print(f"{date_val:<12}{actual:>12.4f}{predicted:>12.4f}{abs(actual - predicted):>12.4f}")
            all_samples.append({
                "pair": pair_col, "date": date_val, "actual": actual, "predicted": predicted,
                "abs_error": abs(actual - predicted),
            })

    if all_samples:
        out_path = os.path.join(TABLES_DIR, "sample_test_predictions.csv")
        pd.DataFrame(all_samples).to_csv(out_path, index=False)
        print(f"\nSample predictions saved to {out_path}")


if __name__ == "__main__":
    main()
