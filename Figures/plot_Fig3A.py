import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap

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

# ============================================================
# Utility functions
# ============================================================
def clean_axes(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10, length=3, width=0.8)
    if grid:
        ax.grid(True, color="0.88", lw=0.8, zorder=0)


def colored_line(ax, x, y, c, cmap="coolwarm", lw=2.0, alpha=1.0, zorder=3):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, linewidth=lw, alpha=alpha, zorder=zorder)
    lc.set_array(c)
    ax.add_collection(lc)
    return lc


def save_single_panel(fig, save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".svg"), bbox_inches="tight")
    plt.show()


def read_style(style_df):
    """Convert Style sheet with columns item/value into a dictionary."""
    return dict(zip(style_df["item"], style_df["value"]))


# ============================================================
# Panel A: read xlsx data and plot
# ============================================================
def plot_panel_A_from_xlsx(
    xlsx_path="Fig3A.xlsx",
    save_path="Fig3A_from_xlsx.png"
):
    # 1) Read data from xlsx
    traj = pd.read_excel(xlsx_path, sheet_name="Fig3A_Trajectory")
    centers = pd.read_excel(xlsx_path, sheet_name="State_Centers")
    style = read_style(pd.read_excel(xlsx_path, sheet_name="Style"))

    t = traj["time"].to_numpy()
    z1 = traj["SpikeMind_latent1"].to_numpy()
    z2 = traj["SpikeMind_latent2"].to_numpy()
    zb1 = traj["Baseline_latent1"].to_numpy()
    zb2 = traj["Baseline_latent2"].to_numpy()

    # 2) Draw figure
    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=300)

    # Symbolic cognitive-state basins / centers
    for _, row in centers.iterrows():
        color = row["color_hex"]
        cx = row["center_latent1"]
        cy = row["center_latent2"]
        label = row["label"]

        ax.add_patch(
            Circle(
                (cx, cy), 0.72,
                facecolor=color,
                edgecolor="none",
                alpha=0.08,
                zorder=0
            )
        )
        ax.scatter(
            cx, cy,
            s=70,
            color=color,
            edgecolor=style["edge_color"],
            linewidth=0.6,
            zorder=4
        )
        ax.text(cx + 0.07, cy + 0.07, f"${label}$", fontsize=9)

    # Outer baseline trajectory
    ax.plot(
        zb1, zb2,
        color=style["baseline_color"],
        lw=1.5,
        alpha=0.55,
        clip_on=False
    )

    # Main SpikeMind-H1 latent trajectory
    cmap = LinearSegmentedColormap.from_list(
        "state_path",
        [
            style["spikemind_path_cmap_start"],
            style["spikemind_path_cmap_mid"],
            style["spikemind_path_cmap_end"],
        ]
    )
    lc = colored_line(ax, z1, z2, t, cmap=cmap, lw=2.5)

    # Start and end markers
    ax.scatter(z1[0], z2[0], s=45, color=style["start_marker_face"],
               edgecolor=style["edge_color"], zorder=6)
    ax.scatter(z1[-1], z2[-1], s=50, color=style["end_marker_color"],
               edgecolor=style["edge_color"], zorder=6)

    ax.set_xlabel(r"latent1", fontsize=11)
    ax.set_ylabel(r"latent2", fontsize=11)
    ax.set_xlim(float(style["xlim_min"]), float(style["xlim_max"]))
    ax.set_ylim(float(style["ylim_min"]), float(style["ylim_max"]))

    clean_axes(ax)

    # Leave space for right colorbar
    fig.subplots_adjust(left=0.13, right=0.84, bottom=0.13, top=0.95)

    # Right vertical colorbar
    cax = fig.add_axes([0.88, 0.25, 0.035, 0.50])
    cb = plt.colorbar(lc, cax=cax, orientation="vertical")
    cb.set_ticks([t.min(), t.max()])
    cb.set_ticklabels(["early", "late"])
    cb.ax.tick_params(labelsize=8, length=2)
    cb.outline.set_linewidth(0.5)

    save_single_panel(fig, save_path)


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    plot_panel_A_from_xlsx(
        xlsx_path="Fig3A.xlsx",
        save_path="Fig3A_from_xlsx.png"
    )
