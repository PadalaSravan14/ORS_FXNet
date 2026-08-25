
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from config import FIGURES_DIR

STAGE_LABELS = [
    "INR-Centric\nMulti-Currency Dataset\n(Triangular Cross-Rates +\nGold / Crude / Inflation)",
    "Dual-Phase\nTemporal Encoder\n(Short-Term + Long-Term LSTM)",
    "Attention\nFusion Layer\n(Multi-Head)",
    "Feature-Sensitivity\nAutoencoder\n(Macro Latent Code)",
    "Fusion Head\n(Base Forecast)",
    "Multi-Agent RL\nCorrection Ensemble\n(DQN + DDPG + PPO)",
    "Final INR\nExchange-Rate\nForecast",
]

STAGE_COLORS = [
    "#cfe8f3", "#a9d6e5", "#89c2d9", "#61a5c2", "#468faf", "#f9a825", "#2e7d32",
]


def architecture_pipeline_diagram(filename="ors_fxnet_architecture_overview.png"):
    fig, ax = plt.subplots(figsize=(15, 3.6))
    ax.set_xlim(0, len(STAGE_LABELS))
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_width, box_height = 0.86, 0.62
    for i, (label, color) in enumerate(zip(STAGE_LABELS, STAGE_COLORS)):
        x = i + 0.5
        box = FancyBboxPatch(
            (x - box_width / 2, 0.19), box_width, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.2, edgecolor="#263238", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=8.3, wrap=True)

        if i < len(STAGE_LABELS) - 1:
            arrow = FancyArrowPatch(
                (x + box_width / 2, 0.5), (x + 1 - box_width / 2, 0.5),
                arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="#263238",
            )
            ax.add_patch(arrow)

    ax.set_title("ORS-FXNet: End-to-End Multimodal Time-Series Fusion Architecture for INR Prediction",
                  fontsize=11, pad=14)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
