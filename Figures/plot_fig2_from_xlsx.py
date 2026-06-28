# -*- coding: utf-8 -*-
# @Function: Read Fig.2 data from xlsx and draw Fig.2.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


XLSX_PATH = os.path.join("data", "fig2_source_data.xlsx")

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

SAVE_PATH_PNG = os.path.join(OUT_DIR, "Fig2A_6.2_from_xlsx.png")
SAVE_PATH_PDF = os.path.join(OUT_DIR, "Fig2A_6.2_from_xlsx.pdf")

DPI = 300

MODEL_ORDER = [
    "SpikeMind-H1",
    "Latent RNN",
    "Transformer Encoder",
    "Latent Neural ODE",
]


def load_fig2_data(xlsx_path: str):
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(
            f"Cannot find {xlsx_path}. Please run generate_fig2_xlsx.py first."
        )

    trajectory_df = pd.read_excel(xlsx_path, sheet_name="Trajectory")
    metric_df = pd.read_excel(xlsx_path, sheet_name="Metrics")

    z_columns = [col for col in trajectory_df.columns if str(col).startswith("z")]
    if len(z_columns) == 0:
        raise ValueError("No latent columns found. Expected columns like z001, z002, ...")

    z_pred = trajectory_df[z_columns].to_numpy(dtype=np.float32)

    required_cols = {
        "model",
        "horizon",
        "latent_instability",
        "accumulated_error",
    }
    missing_cols = required_cols - set(metric_df.columns)
    if missing_cols:
        raise ValueError(f"Metrics sheet is missing required columns: {missing_cols}")

    return z_pred, metric_df


def plot_fig2a(ax, z_pred: np.ndarray):
    reducer = PCA(n_components=2)
    z_2d = reducer.fit_transform(z_pred)

    n_steps = z_2d.shape[0]
    time_index = np.arange(n_steps)

    scatter = ax.scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        c=time_index,
        s=28,
        alpha=0.85,
        edgecolors="none",
    )

    ax.plot(
        z_2d[:, 0],
        z_2d[:, 1],
        linewidth=1.2,
        alpha=0.65,
    )

    horizon_marks = [0, 10, 20, 40, 60, n_steps - 1]
    horizon_marks = sorted(set([h for h in horizon_marks if 0 <= h < n_steps]))

    label_offsets = {
        0: (-42, 0),
        10: (20, -10),
        20: (0, 15),
        40: (24, -5),
        60: (24, -5),
        n_steps - 1: (24, 0),
    }

    for h in horizon_marks:
        ax.scatter(
            z_2d[h, 0],
            z_2d[h, 1],
            s=70,
            marker="o",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )

        dx, dy = label_offsets.get(h, (24, 5))

        ax.annotate(
            f"h={h}",
            xy=(z_2d[h, 0], z_2d[h, 1]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10,
            va="center",
            ha="left",
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="none",
                alpha=0.78,
            ),
            arrowprops=dict(
                arrowstyle="-",
                linewidth=0.6,
                alpha=0.70,
            ),
        )
    ax.set_title("A", loc="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Prediction horizon")


def plot_fig2b(ax, metric_df: pd.DataFrame):
    for model in MODEL_ORDER:
        sub = metric_df[metric_df["model"] == model].sort_values("horizon")
        if sub.empty:
            continue

        ax.plot(
            sub["horizon"],
            sub["latent_instability"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=model,
        )

    ax.set_title("B", loc="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Latent trajectory instability")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def plot_fig2c(ax, metric_df: pd.DataFrame):
    for model in MODEL_ORDER:
        sub = metric_df[metric_df["model"] == model].sort_values("horizon")
        if sub.empty:
            continue

        ax.plot(
            sub["horizon"],
            sub["accumulated_error"],
            marker="s",
            linewidth=1.8,
            markersize=4,
            label=model,
        )

    ax.set_title("C", loc="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Accumulated prediction error")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def main() -> None:
    z_pred, metric_df = load_fig2_data(XLSX_PATH)

    # fig, axes = plt.subplots(
    #     1,
    #     3,
    #     figsize=(15, 4.5),
    #     constrained_layout=True,
    # )
    #
    # plot_fig2a(axes[0], z_pred)
    # plot_fig2b(axes[1], metric_df)
    # plot_fig2c(axes[2], metric_df)

    # 只创建一个子图
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(6, 5),  # 调整大小，适合单图
        constrained_layout=True,
    )

    # 绘制子图A
    plot_fig2a(ax, z_pred)

    plt.savefig(SAVE_PATH_PNG, dpi=DPI, bbox_inches="tight")
    plt.savefig(SAVE_PATH_PDF, dpi=DPI, bbox_inches="tight")
    # plt.show()

    print(f"Saved figure to: {SAVE_PATH_PNG}")
    print(f"Saved figure to: {SAVE_PATH_PDF}")


if __name__ == "__main__":
    main()
