# -*- coding: utf-8 -*-
# @Function: Read Fig.4 source data from xlsx and draw Fig.4.


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_XLSX = os.path.join("data", "fig4_source_data.xlsx")
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

SAVE_PATH_PNG = os.path.join(OUT_DIR, "Fig4_5.18.png")
SAVE_PATH_PDF = os.path.join(OUT_DIR, "Fig4_5.18.pdf")
DPI = 300


def load_fig4_data(xlsx_path=DATA_XLSX):
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(
            f"Cannot find {xlsx_path}. Please run generate_fig4_xlsx.py first."
        )

    sheets = {
        "a": "fig4a_symbolic_trajectory",
        "b": "fig4b_transition_matrix",
        "c": "fig4c_neural_symbolic_alignment",
        "d": "fig4d_repeated_stability",
        "e": "fig4e_perturbation_variability",
        "f": "fig4f_depression_occupancy",
        "g": "fig4g_sleep_token_evolution",
        "h": "fig4h_affective_occupancy",
    }
    return {key: pd.read_excel(xlsx_path, sheet_name=sheet) for key, sheet in sheets.items()}


def plot_a(ax, df):
    sc = ax.scatter(
        df["traj_x"],
        df["traj_y"],
        c=df["time_step"],
        s=18,
        alpha=0.85,
        edgecolors="none",
    )
    ax.plot(df["traj_x"], df["traj_y"], linewidth=0.8, alpha=0.55)
    ax.set_title("A", fontsize=12, fontweight="bold")
    ax.set_xlabel("Symbolic dim. 1")
    ax.set_ylabel("Symbolic dim. 2")
    ax.text(
        0.02,
        0.96,
        "Symbolic trajectory",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
    )
    return sc


def plot_b(ax, df):
    mat = df.pivot(index="to_token", columns="from_token", values="transition_prob").sort_index(ascending=True)
    mat = mat.reindex(sorted(mat.columns), axis=1)

    im = ax.imshow(mat.values, aspect="auto", origin="lower")
    ax.set_title("A", fontsize=12, fontweight="bold")
    ax.set_xlabel("Current token $s_t$")
    ax.set_ylabel("Next token $s_{t+1}$")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_xticklabels(mat.columns.astype(int), fontsize=7)
    ax.set_yticklabels(mat.index.astype(int), fontsize=7)
    return im


def plot_c(ax, df):

    # Sort by time to ensure a correct trajectory.
    df = df.sort_values("time_step").reset_index(drop=True)

    ax.plot(
        df["neural_x"],
        df["neural_y"],
        linewidth=1.2,
        alpha=0.55,
        label="Continuous neural trajectory",
        zorder=1,
    )

    ax.scatter(
        df["neural_x"],
        df["neural_y"],
        s=10,
        alpha=0.45,
        edgecolors="none",
        zorder=2,
    )

    sc = ax.scatter(
        df["symbolic_x"],
        df["symbolic_y"],
        c=df["token_id"],
        s=28,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.35,
        zorder=4,
    )

    ax.plot(
        df["symbolic_x"],
        df["symbolic_y"],
        linewidth=2.0,
        alpha=0.75,
        label="Symbolic cognitive-state trajectory",
        zorder=3,
    )

    link_indices = np.linspace(0, len(df) - 1, 12, dtype=int)

    for idx in link_indices:
        ax.plot(
            [df.loc[idx, "neural_x"], df.loc[idx, "symbolic_x"]],
            [df.loc[idx, "neural_y"], df.loc[idx, "symbolic_y"]],
            linestyle="--",
            linewidth=0.8,
            alpha=0.45,
            zorder=0,
        )

    transition_df = df[df["is_transition"] == 1]

    if len(transition_df) > 0:
        ax.scatter(
            transition_df["symbolic_x"],
            transition_df["symbolic_y"],
            s=80,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.95,
            label="Symbolic transition",
            zorder=5,
        )


    ax.set_title("B", fontsize=12, fontweight="bold")
    ax.set_xlabel("Continuous neural trajectory dim. 1")
    ax.set_ylabel("Symbolic organization dim. 2")

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.35,
    )

    ax.legend(
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
    )

    return sc


def plot_d(ax, df):

    method_order = ["SpikeMind-H1", "Continuous clustering", "K-means tokenization", "VQ baseline"]

    for method in method_order:
        sub = df[df["method"] == method].sort_values("perturbation_magnitude")
        ax.plot(
            sub["perturbation_magnitude"],
            sub["assignment_consistency"],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=method,
        )

    ax.set_title("C", fontsize=12, fontweight="bold")
    ax.set_xlabel("Latent perturbation magnitude")
    ax.set_ylabel("Assignment consistency (%)")
    ax.set_xlim(-0.005, 0.205)
    ax.set_ylim(60, 100)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(frameon=False, fontsize=7, loc="lower left")


def plot_e(ax, df):
    for method, sub in df.groupby("method"):
        sub = sub.sort_values("perturbation_intensity")
        ax.plot(
            sub["perturbation_intensity"],
            sub["assignment_variability"],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=method,
        )
    ax.set_title("D", fontsize=12, fontweight="bold")
    ax.set_xlabel("Latent perturbation intensity")
    ax.set_ylabel("Assignment variability")
    ax.set_ylim(0, 0.45)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(frameon=False, fontsize=7, loc="upper left")


def plot_f(ax, df):
    x = np.arange(len(df))
    width = 0.38
    ax.bar(
        x - width / 2,
        df["depressive_occupancy"],
        width=width,
        label="Depressive",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        df["non_depressive_occupancy"],
        width=width,
        label="Non-depressive",
        alpha=0.85,
    )
    ax.set_title("D", fontsize=12, fontweight="bold")
    ax.set_xlabel("Cognitive token")
    ax.set_ylabel("Token occupancy ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(df["token_id"].astype(int), fontsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(
        frameon=True,
        fontsize=8,
        loc="upper left",
        framealpha=0.75,
        facecolor="white",
        edgecolor="none",
    )


def plot_g(ax, df):
    stage_to_code = {"Wake": 0, "NREM": 1, "REM": 2}
    stage_codes = df["sleep_stage"].map(stage_to_code).values
    times = df["time_window"].values

    start_idx = 0
    for idx in range(1, len(df) + 1):
        if idx == len(df) or stage_codes[idx] != stage_codes[start_idx]:
            ax.axvspan(times[start_idx], times[idx - 1], alpha=0.10)
            mid = (times[start_idx] + times[idx - 1]) / 2
            ax.text(
                mid,
                11.8,
                df.loc[start_idx, "sleep_stage"],
                ha="center",
                va="top",
                fontsize=8,
            )
            start_idx = idx

    ax.plot(
        df["time_window"],
        df["token_id"],
        linewidth=1.3,
        marker="o",
        markersize=2.2,
        alpha=0.85,
    )
    ax.set_title("E", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time window")
    ax.set_ylabel("Cognitive token ID")
    ax.set_ylim(-0.5, 12.2)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.40)


def plot_h(ax, df):
    mat = df.pivot(index="affective_state", columns="token_id", values="occupancy_ratio")
    state_order = [
        "High V / High A",
        "High V / Low A",
        "Low V / High A",
        "Low V / Low A",
    ]
    mat = mat.loc[state_order]

    im = ax.imshow(mat.values, aspect="auto")
    ax.set_title("F", fontsize=12, fontweight="bold")
    ax.set_xlabel("Cognitive token")
    ax.set_ylabel("Valence-arousal state")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns.astype(int), fontsize=8)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    return im


def main():
    data = load_fig4_data()

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.2), constrained_layout=True)
    axes = axes.ravel()

    # sc_a = plot_a(axes[0], data["a"])
    im_b = plot_b(axes[0], data["b"])
    sc_c = plot_c(axes[1], data["c"])
    plot_d(axes[2], data["d"])
    # plot_e(axes[4], data["e"])
    plot_f(axes[3], data["f"])
    plot_g(axes[4], data["g"])
    im_h = plot_h(axes[5], data["h"])

    # cbar_a = fig.colorbar(sc_a, ax=axes[0], fraction=0.046, pad=0.04)
    # cbar_a.set_label("Time step")

    cbar_b = fig.colorbar(im_b, ax=axes[0], fraction=0.046, pad=0.04)
    cbar_b.set_label("Transition probability")

    # cbar_c = fig.colorbar(sc_c, ax=axes[2], fraction=0.046, pad=0.04)
    # cbar_c.set_label("Symbolic cognitive token")

    cbar_h = fig.colorbar(im_h, ax=axes[5], fraction=0.046, pad=0.04)
    cbar_h.set_label("Occupancy ratio")

    plt.savefig(SAVE_PATH_PNG, dpi=DPI, bbox_inches="tight")
    plt.savefig(SAVE_PATH_PDF, dpi=DPI, bbox_inches="tight")
    print(f"Saved figure to: {SAVE_PATH_PNG}")
    print(f"Saved figure to: {SAVE_PATH_PDF}")


if __name__ == "__main__":
    main()
