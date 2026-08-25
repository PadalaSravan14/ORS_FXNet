
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import FIGURES_DIR

sns.set_theme(style="whitegrid")


def actual_vs_predicted_plot(dates, y_true, y_pred, pair_label, filename):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(dates, y_true, color="#1f4e79", label="Actual", linewidth=1.1)
    ax.plot(dates, y_pred, color="#e07b00", label="Predicted", linewidth=1.0, linestyle="--")
    ax.set_title(f"Actual vs Predicted {pair_label} (ORS-FXNet), Test Period")
    ax.set_xlabel("Date")
    ax.set_ylabel(pair_label)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def residual_curve_plot(dates, y_true, y_pred, pair_label, filename):
    residual = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.plot(dates, residual, color="#6a1b9a", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title(f"Residual Error Curve for {pair_label} Forecasting (Actual - Predicted)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def scatter_fidelity_plot(y_true, y_pred, pair_label, filename):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.5, color="#00695c")
    lo, hi = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1)
    ax.set_title(f"Scatter Plot of Actual vs Predicted {pair_label} Values")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def directional_confusion_matrix_plot(y_true, y_pred, filename="directional_confusion_matrix.png"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_dir = (np.diff(y_true) > 0).astype(int)
    pred_dir = (np.diff(y_pred) > 0).astype(int)
    cm = confusion_matrix(true_dir, pred_dir, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Down", "Up"], yticklabels=["Down", "Up"], ax=ax)
    ax.set_title("Confusion Matrix for Directional Movement Prediction")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def error_distribution_plot(y_true, y_pred, filename="error_distribution.png"):
    errors = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    sns.histplot(errors, bins=40, kde=True, color="#455a64", ax=ax)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Distribution of Forecasting Errors (ORS-FXNet)")
    ax.set_xlabel("Prediction Error")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def rmse_stability_across_pairs_plot(pair_rmse_dict, filename="rmse_stability_across_pairs.png"):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    pairs = list(pair_rmse_dict.keys())
    values = list(pair_rmse_dict.values())
    ax.bar(pairs, values, color="#00838f")
    ax.set_title("RMSE Stability of ORS-FXNet Across INR-Based Currency Pairs")
    ax.set_ylabel("RMSE")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def training_time_comparison_plot(model_time_dict, filename="training_time_comparison.png"):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    models = list(model_time_dict.keys())
    times = list(model_time_dict.values())
    colors = ["#90a4ae"] * (len(models) - 1) + ["#c62828"]
    ax.bar(models, times, color=colors)
    ax.set_title("Training Time Comparison of Baseline Models vs ORS-FXNet")
    ax.set_ylabel("Training Time (minutes)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def ablation_radar_chart(variant_metrics, metric_names, filename="ablation_radar_chart.png"):

    variants = list(variant_metrics.keys())
    raw = np.array([[variant_metrics[v][m] for m in metric_names] for v in variants], dtype=float)

    normalized = np.zeros_like(raw)
    error_like = {"RMSE", "MAE", "MAPE", "VAE"}
    for j, m in enumerate(metric_names):
        col = raw[:, j]
        lo, hi = col.min(), col.max()
        span = hi - lo if hi > lo else 1.0
        scaled = (col - lo) / span
        normalized[:, j] = (1 - scaled) if m in error_like else scaled

    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    palette = sns.color_palette("husl", len(variants))
    for i, variant in enumerate(variants):
        values = normalized[i].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2 if i == 0 else 1, label=variant, color=palette[i])
        ax.fill(angles, values, alpha=0.08, color=palette[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_title("Ablation Study: Component Contribution Across Evaluation Metrics")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
