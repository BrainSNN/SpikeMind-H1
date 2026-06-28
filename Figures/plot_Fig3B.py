import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Global style
# ============================================================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Color palette
RED = "#c9282d"
BLUE = "#2d78b7"
GRAY = "#7f7f7f"
BLACKISH = "#444444"


def clean_axes(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10, length=3, width=0.8)
    if grid:
        ax.grid(True, color="0.88", lw=0.8, zorder=0)


def save_single_panel(fig, save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".svg"), bbox_inches="tight")
    plt.show()


def plot_panel_C_from_xlsx(
    xlsx_path="Fig3B.xlsx",
    save_path="Fig3B.png",
):
    """Read Fig. 3C data from xlsx and draw recursive prediction error curves."""
    df = pd.read_excel(xlsx_path, sheet_name="Fig3C_Data")

    horizon = df["Prediction_horizon"].to_numpy()

    curves = [
        ("SpikeMind_H1", "SpikeMind_H1_band", RED, "SpikeMind-H1", 2.2),
        ("Continuous_latent", "Continuous_latent_band", BLUE, "Continuous latent", 1.8),
        ("Neural_ODE", "Neural_ODE_band", GRAY, "Neural ODE", 1.8),
        ("Transformer_baseline", "Transformer_baseline_band", BLACKISH, "Transformer baseline", 1.8),
    ]

    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=300)

    for y_col, band_col, color, label, line_width in curves:
        y = df[y_col].to_numpy(dtype=float)
        band = df[band_col].to_numpy(dtype=float)
        ax.plot(horizon, y, color=color, lw=line_width, label=label)
        ax.fill_between(horizon, y - band, y + band, color=color, alpha=0.13, lw=0)

    ax.set_xlabel("Prediction horizon", fontsize=11)
    ax.set_ylabel("Recursive prediction error", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.62)

    clean_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    save_single_panel(fig, save_path)


if __name__ == "__main__":
    plot_panel_C_from_xlsx(
        xlsx_path="Fig3B.xlsx",
        save_path="Fig3B_from_xlsx.png",
    )
