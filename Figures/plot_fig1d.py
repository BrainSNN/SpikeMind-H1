from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DATA_XLSX = Path("fig1D.xlsx")
OUT_PNG = Path("fig1D.png")
OUT_PDF = Path("fig1D.pdf")


def main():
    observed = pd.read_excel(DATA_XLSX, sheet_name="observed_states")
    predicted = pd.read_excel(DATA_XLSX, sheet_name="predicted_states")
    tokens = pd.read_excel(DATA_XLSX, sheet_name="symbolic_tokens")

    fig, ax = plt.subplots(figsize=(8.8, 6.6), dpi=300)

    # Predicted neural states: clearer than hollow circles, but visually secondary.
    ax.scatter(
        predicted["latent1"],
        predicted["latent2"],
        marker="x",
        s=20,
        linewidths=0.9,
        c="#8A8A8A",
        alpha=0.9,
        zorder=1,
        label="predicted neural state",
    )

    # Observed neural-state trajectory: continuous UMAP manifold-like evolution.
    scatter = ax.scatter(
        observed["latent1"],
        observed["latent2"],
        c=observed["predictive_time"],
        cmap="viridis",
        s=34,
        edgecolors="white",
        linewidths=0.35,
        alpha=0.98,
        zorder=2,
        label="observed neural state",
    )

    # Light temporal path helps show recursive evolution without making the plot crowded.
    ax.plot(
        observed["latent1"],
        observed["latent2"],
        color="#B8B8B8",
        linewidth=0.7,
        alpha=0.42,
        zorder=0,
    )

    # Symbolic cognitive tokens: discrete prototypes organizing the continuous trajectory.
    token_colors = {
        "C0": "#1f77b4",
        "C1": "#2ca02c",
        "C2": "#9467bd",
        "C3": "#e377c2",
        "C4": "#bcbd22",
        "C5": "#17becf",
    }
    for _, row in tokens.iterrows():
        token = row["token"]
        ax.scatter(
            row["latent1"],
            row["latent2"],
            marker="D",
            s=190,
            c=token_colors.get(token, "#1f77b4"),
            edgecolors="black",
            linewidths=1.5,
            zorder=4,
        )
        ax.text(
            row["latent1"] + row["label_dx"],
            row["latent2"] + row["label_dy"],
            token,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="center",
            color="black",
            zorder=5,
        )

    # Axes and colorbar.
    ax.set_xlabel("latent1", fontsize=14)
    ax.set_ylabel("latent2", fontsize=14)
    ax.set_xlim(-3.75, 3.35)
    ax.set_ylim(-2.8, 2.75)
    ax.tick_params(axis="both", labelsize=11, width=1.2, length=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.grid(False)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("predictive time", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    cbar.outline.set_linewidth(1.2)

    # Three legend entries with comfortable vertical spacing.
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=8,
               markerfacecolor="#1f77b4", markeredgecolor="#1f77b4",
               label="observed neural state"),
        Line2D([0], [0], marker="x", linestyle="None", markersize=7,
               markeredgewidth=1.1, color="#8A8A8A",
               label="predicted neural state"),
        Line2D([0], [0], marker="D", linestyle="None", markersize=10,
               markerfacecolor="#1f77b4", markeredgecolor="black",
               markeredgewidth=1.4, label="symbolic cognitive token"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=False,
        fontsize=11,
        handletextpad=0.9,
        labelspacing=0.95,
        borderpad=0.5,
        columnspacing=1.0,
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
