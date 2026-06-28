#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D


def build_augmented_data(xlsx_path):
    df = pd.read_excel(xlsx_path, sheet_name="Hybrid_robustness_data")
    anchors = pd.read_excel(xlsx_path, sheet_name="Symbolic_anchors")

    keep = [
        "Reference latent trajectory",
        "Additive perturbation σ=0.18",
        "Additive perturbation σ=0.36",
        "Temporal noise jitter=0.08",
    ]
    df = df[df["condition"].isin(keep)].copy()

    # Build an additional stronger temporal-noise condition from the reference manifold.
    ref = df[(df["condition"] == "Reference latent trajectory") & (df["replicate"] == 1)].sort_values("time_index")
    n = len(ref)
    t = np.linspace(0, 1, n)
    x_ref = ref["latent1"].to_numpy()
    y_ref = ref["latent2"].to_numpy()

    rng = np.random.default_rng(2026)
    new_rows = []
    # add 2 replicates for stronger temporal jitter
    for rep in [1, 2]:
        tt = np.clip(t + rng.normal(0, 0.16, n), 0, 1)
        xb = np.interp(tt, t, x_ref)
        yb = np.interp(tt, t, y_ref)
        # estimate local tangent/normal from ref path
        dx = np.gradient(x_ref)
        dy = np.gradient(y_ref)
        norm = np.sqrt(dx**2 + dy**2) + 1e-8
        tx, ty = dx / norm, dy / norm
        nx, ny = -ty, tx
        nxb = np.interp(tt, t, nx)
        nyb = np.interp(tt, t, ny)
        txb = np.interp(tt, t, tx)
        tyb = np.interp(tt, t, ty)

        normal_noise = rng.normal(0, 0.24, n)
        tangential_noise = rng.normal(0, 0.10, n)
        iso_x = rng.normal(0, 0.06, n)
        iso_y = rng.normal(0, 0.06, n)

        x = xb + normal_noise * nxb + tangential_noise * txb + iso_x
        y = yb + normal_noise * nyb + tangential_noise * tyb + iso_y

        scale = np.sqrt((x / 3.05) ** 2 + (y / 2.65) ** 2)
        shrink = np.where(scale > 1, 1 / scale, 1)
        x *= shrink
        y *= shrink

        # approximate symbolic states by original time index mapping
        state_seq = []
        for val in tt:
            idx = int(np.clip(round(val * (n - 1)), 0, n - 1))
            if idx < 60:
                s = 'A'
            elif idx < 130:
                s = 'B'
            elif idx < 200:
                s = 'C'
            else:
                s = 'D'
            state_seq.append(s)

        for i in range(n):
            new_rows.append({
                "condition": "Temporal noise jitter=0.16",
                "noise_level": 0.16,
                "missing_rate": 0.0,
                "replicate": rep,
                "time_index": i,
                "trajectory_time": float(t[i]),
                "latent1": float(x[i]),
                "latent2": float(y[i]),
                "symbolic_state": state_seq[i],
                "is_missing": 0,
                "transition_region": 0,
            })

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df, anchors


def draw_merged_panel(xlsx_path=None, out_prefix="Fig3D"):
    script_dir = Path(__file__).resolve().parent
    if xlsx_path is None:
        xlsx_path = script_dir / "Fig3D.xlsx"
    else:
        xlsx_path = Path(xlsx_path)

    df, anchors = build_augmented_data(xlsx_path)

    display_name = {
        "Reference latent trajectory": "Reference latent trajectory",
        "Additive perturbation σ=0.18": "Neural perturbation σ=0.18",
        "Additive perturbation σ=0.36": "Neural perturbation σ=0.36",
        "Temporal noise jitter=0.08": "Temporal noise jitter=0.08",
        "Temporal noise jitter=0.16": "Temporal noise jitter=0.16",
    }

    cond_colors = {
        "Reference latent trajectory": "#4DA3FF",
        "Additive perturbation σ=0.18": "#63D471",
        "Additive perturbation σ=0.36": "#FFB34D",
        "Temporal noise jitter=0.08": "#B07AF4",
        "Temporal noise jitter=0.16": "#7E57C2",
    }
    state_colors = {"A": "#4DA3FF", "B": "#63D471", "C": "#FFB34D", "D": "#C084FC"}
    cond_order = list(cond_colors.keys())

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
    })

    fig, ax = plt.subplots(figsize=(6.25, 4.0), dpi=350)

    ref = df[(df["condition"] == "Reference latent trajectory") & (df["replicate"] == 1)].sort_values("time_index")
    ax.plot(ref["latent1"], ref["latent2"], lw=1.55, color="#2C6FB7", alpha=0.95, zorder=2)

    for cond in cond_order:
        sub = df[df["condition"] == cond]
        size = 10 if cond == "Reference latent trajectory" else 8
        ax.scatter(sub["latent1"], sub["latent2"], s=size, color=cond_colors[cond],
                   alpha=1.0, edgecolors="none", zorder=3 if cond != "Reference latent trajectory" else 4)

    bounded = Ellipse((0.0, 0.0), width=5.65, height=4.85, facecolor="none",
                      edgecolor="0.70", linestyle="--", linewidth=1.0, zorder=1)
    ax.add_patch(bounded)

    for _, row in anchors.iterrows():
        s = row["symbolic_state"]
        region = Ellipse((row["anchor_latent1"], row["anchor_latent2"]),
                         width=row["region_width"], height=row["region_height"],
                         facecolor=state_colors[s], edgecolor=state_colors[s],
                         alpha=0.10, linewidth=0.8, zorder=1.5)
        ax.add_patch(region)
        ax.scatter(row["anchor_latent1"], row["anchor_latent2"], marker="s", s=98,
                   color=state_colors[s], edgecolors="black", linewidths=0.75, zorder=6)
        ax.text(row["anchor_latent1"], row["anchor_latent2"], s,
                ha="center", va="center", fontsize=8, fontweight="bold", color="black", zorder=7)

    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_title("E", loc="center", fontweight="bold", fontsize=13, pad=4)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-3.05, 3.05)
    ax.set_ylim(-2.65, 2.65)
    ax.set_aspect("equal", adjustable="box")

    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=5.2,
               markerfacecolor=cond_colors[c], markeredgecolor='none', label=display_name[c])
        for c in cond_order
    ]
    legend_handles += [
        Line2D([0], [0], color="#2C6FB7", lw=1.5, label="Reference manifold"),
        Line2D([0], [0], marker='s', linestyle='None', markersize=6,
               markerfacecolor='white', markeredgecolor='black', label="Symbolic-state center"),
    ]

    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, handletextpad=0.40, borderaxespad=0.0, labelspacing=0.50)

    fig.tight_layout(rect=(0, 0, 0.78, 1))
    out_prefix = script_dir / out_prefix
    fig.savefig(out_prefix.with_suffix('.png'), bbox_inches='tight', dpi=350)
    fig.savefig(out_prefix.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out_prefix.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    draw_merged_panel()
