from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

DATA_PATH = Path("Fig1_B_to_E_data_modified.xlsx")
OUT_DIR = Path("fig1_outputs_modified")
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["svg.fonttype"] = "none"

BLUE = "#1f77b4"
TOKEN_BLUE = "#2d78b7"
TOKEN_EDGE = "#111111"


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)


def save_fig_multi(fig, filename_stem, bbox_inches=None):
    for ext in ["png", "svg"]:
        fig.savefig(
            OUT_DIR / f"{filename_stem}.{ext}",
            dpi=300,
            bbox_inches=bbox_inches,
        )


def add_time_colored_line(ax, x, y, time, cmap="viridis", linewidth=1.6, alpha=0.9):
    """Draw a continuous trajectory as a time-colored line without point markers."""
    x = np.asarray(x)
    y = np.asarray(y)
    time = np.asarray(time)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidth=linewidth, alpha=alpha)
    lc.set_array(time[:-1])
    ax.add_collection(lc)
    ax.autoscale_view()
    return lc


def plot_fig1b(data_path=DATA_PATH):
    """Fig.1B: Continuous neural latent trajectories during cognitive-state evolution."""
    df = pd.read_excel(data_path, sheet_name="Fig1B_latent_trajectories")

    fig, ax = plt.subplots(figsize=(5.4, 4.4))

    lc = add_time_colored_line(
        ax,
        df["latent_z1"],
        df["latent_z2"],
        df["time"],
        cmap="viridis",
        linewidth=1.7,
        alpha=0.95,
    )

    cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Time step", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax.set_xlabel("Latent dimension 1")
    ax.set_ylabel("Latent dimension 2")
    clean_axes(ax)
    fig.tight_layout()

    save_fig_multi(fig, "fig1B_continuous_neural_latent_trajectories")
    plt.close(fig)




def main():
    plot_fig1b()
    print(f"Saved revised Fig.1B-E plots as PNG and SVG to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
