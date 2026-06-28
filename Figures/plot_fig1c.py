import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("fig1c.csv")
OUT_PATH = Path("Fig1C.png")

df = pd.read_csv(DATA_PATH)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"

fig, ax = plt.subplots(figsize=(3.45, 3.05), dpi=300)

# points only; no connecting thin lines
scatter = ax.scatter(
    df["symbolic_dim_1"],
    df["symbolic_dim_2"],
    c=df["symbolic_state_value"],
    cmap="rainbow",
    s=6.4,
    alpha=0.95,
    linewidths=0
)

ax.set_xlabel("Symbolic dimension 1", fontsize=7)
ax.set_ylabel("Symbolic dimension 2", fontsize=7)
ax.tick_params(labelsize=6.5, length=2.5, width=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_aspect("equal", adjustable="box")

x_pad, y_pad = 0.18, 0.18
ax.set_xlim(df["symbolic_dim_1"].min() - x_pad, df["symbolic_dim_1"].max() + x_pad)
ax.set_ylim(df["symbolic_dim_2"].min() - y_pad, df["symbolic_dim_2"].max() + y_pad)

cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.035)
cbar.set_label("Symbolic state value", fontsize=7)
cbar.ax.tick_params(labelsize=6.5, length=2.5, width=0.8)
cbar.set_ticks([0, 1, 2, 3, 4, 5])

fig.tight_layout(pad=0.8)
fig.savefig(OUT_PATH, dpi=600, bbox_inches="tight")
fig.savefig(OUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
plt.show()
