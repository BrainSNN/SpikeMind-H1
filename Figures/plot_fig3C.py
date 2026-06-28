import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

# ============================================================
# Global style
# ============================================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

RED = '#c9282d'
BLUE = '#2d78b7'
PURPLE = '#8e2a8d'
GREEN = '#0b9b5a'
CYAN = '#27a9d8'
YELLOW = '#f0b23c'
GRAY = '#777777'
DARK_GRAY = '#4d4d4d'
LIGHT_GRAY = '#d8d8d8'
BLACK = '#111111'
TOKEN_COLORS = {1: PURPLE, 2: GREEN, 3: CYAN, 4: YELLOW}

# ============================================================
# Utility functions
# ============================================================
def clean_axes(ax, grid=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10, length=3, width=0.8)
    if grid:
        ax.grid(True, color='0.88', lw=0.8, zorder=0)




def save_panel(fig, out_dir, stem):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out_dir / f'{stem}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def colored_line(ax, x, y, c, cmap, lw=2.0, alpha=1.0, zorder=3):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, linewidth=lw, alpha=alpha, zorder=zorder)
    lc.set_array(np.asarray(c))
    ax.add_collection(lc)
    return lc

# ============================================================
# Plot functions
# ============================================================
OUT_DIR = Path("fig3_outputs")


def plot_fig3C(xlsx_path, out_dir='fig3_outputs'):
    df = pd.read_excel(xlsx_path, sheet_name='Fig3D')
    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=300)

    ax.plot(df['t'], df['realized_latent'], color=BLACK, lw=1.9, label='Realized latent')
    ax.plot(df['t'], df['spikemind_predicted'], color=RED, lw=2.2, label='SpikeMind-H1 predicted')
    ax.plot(df['t'], df['baseline_predicted'], color=BLUE, lw=1.9, label='Baseline predicted')
    ax.fill_between(df['t'], df['realized_latent'], df['spikemind_predicted'], color=RED, alpha=0.17, lw=0)
    ax.fill_between(df['t'], df['realized_latent'], df['baseline_predicted'], color=BLUE, alpha=0.12, lw=0)

    ax.set_xlim(0, 34)
    ax.set_ylim(-1.08, 1.16)
    ax.set_xlabel(r'time $t$', fontsize=11)
    ax.set_ylabel('Latent coordinate', fontsize=11)
    clean_axes(ax)

    ax.legend(frameon=False, fontsize=8, loc='lower left')
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.13, top=0.87)
    save_panel(fig, out_dir, 'Fig3C')




def plot_all_fig3(xlsx_path='Fig3_C.xlsx', out_dir='fig3_outputs'):
    plot_fig3C(xlsx_path, out_dir)


if __name__ == '__main__':
    plot_all_fig3('Fig3_C.xlsx', 'fig3_outputs')
