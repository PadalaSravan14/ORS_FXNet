
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd

from config import FIGURES_DIR, TRAIN_END, VAL_END

sns.set_theme(style="whitegrid")


def exchange_rate_trend_plot(df, date_col, target_col="USD_INR", filename="usd_inr_trend.png"):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df[date_col], df[target_col], color="#1f4e79", linewidth=1)
    ax.set_title("USD/INR Exchange Rate Trend (2000-2019)")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD/INR")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def multi_currency_trend_plot(df, date_col, pair_columns, filename="multi_currency_trends.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in pair_columns:
        ax.plot(df[date_col], df[col], linewidth=0.9, label=col)
    ax.set_title("Multi-Currency INR Exchange Rates Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rate vs INR")
    ax.legend(loc="upper left", ncol=len(pair_columns))
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def macro_variables_plot(df, date_col, macro_columns, filename="macro_variables.png"):
    fig, axes = plt.subplots(len(macro_columns), 1, figsize=(11, 2.2 * len(macro_columns)), sharex=True)
    if len(macro_columns) == 1:
        axes = [axes]
    for ax, col in zip(axes, macro_columns):
        ax.plot(df[date_col], df[col], color="#b5651d", linewidth=1)
        ax.set_ylabel(col)
    axes[0].set_title("Macroeconomic Variables Over Time: Gold, Crude Oil, Inflation")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def correlation_heatmap_plot(df, columns, filename="correlation_heatmap.png"):
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[columns].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                cbar_kws={"label": "Correlation"})
    ax.set_title("Correlation Heatmap of FX and Macroeconomic Variables")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def temporal_split_plot(df, date_col, target_col="USD_INR", filename="temporal_split.png"):
    fig, ax = plt.subplots(figsize=(11, 4))
    train_end = pd.Timestamp(TRAIN_END)
    val_end = pd.Timestamp(VAL_END)

    train_mask = df[date_col] <= train_end
    val_mask = (df[date_col] > train_end) & (df[date_col] <= val_end)
    test_mask = df[date_col] > val_end

    ax.plot(df.loc[train_mask, date_col], df.loc[train_mask, target_col], color="#2e7d32", label="Train (2000-2012)")
    ax.plot(df.loc[val_mask, date_col], df.loc[val_mask, target_col], color="#f9a825", label="Validation (2013-2016)")
    ax.plot(df.loc[test_mask, date_col], df.loc[test_mask, target_col], color="#c62828", label="Test (2017-2019)")
    ax.axvline(train_end, color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(val_end, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("Train-Validation-Test Split for USD/INR Time Series")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD/INR")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_all_eda_plots(df, date_col, target_col, pair_columns, macro_columns):
    paths = []
    paths.append(exchange_rate_trend_plot(df, date_col, target_col))
    paths.append(multi_currency_trend_plot(df, date_col, pair_columns))
    paths.append(macro_variables_plot(df, date_col, macro_columns))
    paths.append(correlation_heatmap_plot(df, pair_columns + macro_columns))
    paths.append(temporal_split_plot(df, date_col, target_col))
    return paths
