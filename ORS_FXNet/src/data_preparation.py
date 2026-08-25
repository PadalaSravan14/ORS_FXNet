
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from config import (
    FX_RATES_FILE, GOLD_MACRO_FILE, DATE_COLUMN, FX_SOURCE_COLUMNS,
    CURRENCY_PAIRS, TRAIN_START, TRAIN_END, VAL_START, VAL_END,
    TEST_START, TEST_END, SLIDING_WINDOW,
)


def verify_raw_datasets_exist():

    missing = [p for p in (FX_RATES_FILE, GOLD_MACRO_FILE) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Required dataset file(s) not found:\n  " + "\n  ".join(missing) +
            "\n\nPlace the real 'Foreign_Exchange_Rates.csv' (Foreign Exchange Rates "
            "per Dollar, 2000-2019) and 'GoldUP.csv' (Gold Forecasting Dataset) files "
            "inside data/raw/ before running the pipeline."
        )

def load_fx_rates(path=FX_RATES_FILE):
    df = pd.read_csv(path)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    return df


def load_macro_indicators(path=GOLD_MACRO_FILE):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def clean_missing_markers(df, columns, missing_token="ND"):
    
    out = df.copy()
    for col in columns:
        out[col] = out[col].replace(missing_token, np.nan).astype(float)
    return out


def hybrid_impute(df, columns, max_gap=5):

    out = df.copy()
    for col in columns:
        series = out[col]
        is_na = series.isna()
        if not is_na.any():
            continue
        # short gaps -> forward fill
        filled = series.ffill(limit=1)
        still_na = filled.isna()
        if still_na.any():
            # longer gaps -> causal spline interpolation (no look-ahead
            # beyond the fold, achieved via 'from_derivatives'-free simple
            # interpolation with limit_direction='forward')
            filled = filled.interpolate(method="linear", limit_direction="forward")
        # remaining leading NaNs (start of series) -> backward fill once
        filled = filled.bfill()
        out[col] = filled
    return out


def reconstruct_inr_cross_rates(fx_df, macro_df):

    fx_sorted = fx_df.sort_values(DATE_COLUMN).reset_index(drop=True)
    macro_sorted = macro_df.sort_values("Date").reset_index(drop=True)
    merged = pd.merge_asof(
        fx_sorted, macro_sorted[["Date", "USD_INR", "Gold_Price", "Crude_Oil",
                                  "Interest_Rate", "CPI", "USD_Index"]],
        left_on=DATE_COLUMN, right_on="Date", direction="backward",
    )
    merged = merged.sort_values(DATE_COLUMN).reset_index(drop=True)
    for col in ["USD_INR", "Gold_Price", "Crude_Oil", "Interest_Rate", "CPI", "USD_Index"]:
        merged[col] = merged[col].ffill().bfill()

    inr_df = pd.DataFrame({DATE_COLUMN: merged[DATE_COLUMN]})
    inr_df["USD_INR"] = merged["USD_INR"]
    for pair in CURRENCY_PAIRS:
        if pair == "USD":
            continue
        src_col = FX_SOURCE_COLUMNS[pair]
        c_per_usd = merged[src_col]  # C/USD
        inr_df[f"{pair}_INR"] = merged["USD_INR"] * c_per_usd  # C/INR, Eq. (1)

    inr_df["Gold_Price"] = merged["Gold_Price"]
    inr_df["Crude_Oil"] = merged["Crude_Oil"]
    inr_df["Inflation_CPI"] = merged["CPI"]
    return inr_df


def full_dataset_pipeline():

    verify_raw_datasets_exist()
    fx_df = load_fx_rates()
    macro_df = load_macro_indicators()

    fx_cols = list(FX_SOURCE_COLUMNS.values())
    fx_df = clean_missing_markers(fx_df, fx_cols)
    fx_df = hybrid_impute(fx_df, fx_cols)

    inr_df = reconstruct_inr_cross_rates(fx_df, macro_df)
    inr_df = hybrid_impute(inr_df, [c for c in inr_df.columns if c != DATE_COLUMN])
    inr_df = inr_df.rename(columns={"USD_INR": "USD_INR"})
    inr_df["USD_INR_target_col"] = "USD_INR"
    return inr_df


def add_technical_indicators(df, price_col="USD_INR"):
    out = df.copy()
    out[f"{price_col}_ma7"] = out[price_col].rolling(7, min_periods=1).mean()
    out[f"{price_col}_ma21"] = out[price_col].rolling(21, min_periods=1).mean()
    out[f"{price_col}_roc"] = out[price_col].pct_change().fillna(0.0)
    out[f"{price_col}_logret"] = np.log(out[price_col]).diff().fillna(0.0)
    out[f"{price_col}_volatility"] = out[f"{price_col}_logret"].rolling(21, min_periods=1).std().fillna(0.0)
    out[f"{price_col}_diff"] = out[price_col].diff().fillna(0.0)
    return out


class FeatureScalingBundle:
 

    def __init__(self):
        self.minmax_scalers = {}
        self.robust_scalers = {}

    def fit_transform(self, df, minmax_cols, robust_cols):
        out = df.copy()
        for col in minmax_cols:
            scaler = MinMaxScaler()
            out[col] = scaler.fit_transform(out[[col]]).flatten()
            self.minmax_scalers[col] = scaler
        for col in robust_cols:
            scaler = RobustScaler()
            out[col] = scaler.fit_transform(out[[col]]).flatten()
            self.robust_scalers[col] = scaler
        return out

    def inverse_transform_target(self, values, target_col="USD_INR"):
        scaler = self.minmax_scalers[target_col]
        values = np.asarray(values).reshape(-1, 1)
        return scaler.inverse_transform(values).flatten()

def chronological_split(df, date_col=DATE_COLUMN):
    train = df[(df[date_col] >= TRAIN_START) & (df[date_col] <= TRAIN_END)].reset_index(drop=True)
    val = df[(df[date_col] >= VAL_START) & (df[date_col] <= VAL_END)].reset_index(drop=True)
    test = df[(df[date_col] >= TEST_START) & (df[date_col] <= TEST_END)].reset_index(drop=True)
    return train, val, test


def build_sliding_window_samples(df, feature_cols, target_col, window=SLIDING_WINDOW, horizon=1):

    values = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    dates = df[DATE_COLUMN].values if DATE_COLUMN in df.columns else np.arange(len(df))

    X, y, y_dates = [], [], []
    n = len(df)
    for t in range(window, n - horizon + 1):
        X.append(values[t - window:t])
        y.append(target[t + horizon - 1])
        y_dates.append(dates[t + horizon - 1])
    return np.array(X), np.array(y), np.array(y_dates)


def feature_column_set(df, exclude=(DATE_COLUMN, "USD_INR_target_col")):
    return [c for c in df.columns if c not in exclude]
