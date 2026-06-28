import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpecFromSubplotSpec

# =========================
# Load saved plot data
# =========================
data_path = Path("./Fig5_Fig6_new_style_plot_data.npy")
plot_data = np.load(data_path, allow_pickle=True).item()

out_dir = Path("./")
fig5_path = out_dir / "Fig5_from_npy.png"
fig6_path = out_dir / "Fig6_from_npy.png"
pdf_path = out_dir / "Fig5_Fig6_from_npy.pdf"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 15,
    "axes.labelsize": 9,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

blue = "#03AED2"
orange = "#F45B26"
red = "#D12052"
green = "#48B3AF"
dark_blue = "#476EAE"
light_green = "#A7E399"
yellow = "#FFD65A"
dark_green = "#5B7E3C"
deep_orange = "#FF9D23"
deep_red = "#EA5252"

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ======================================================
# Figure 5
# ======================================================
fig5 = plt.figure(figsize=(22, 15))
gs5 = fig5.add_gridspec(3, 4)

axA = fig5.add_subplot(gs5[0, :])
axB = fig5.add_subplot(gs5[1, 0:3])
axC = fig5.add_subplot(gs5[1, 3])
axD = fig5.add_subplot(gs5[2, 0])
axE = fig5.add_subplot(gs5[2, 1])
axF = fig5.add_subplot(gs5[2, 2])
axG = fig5.add_subplot(gs5[2, 3])

axs5 = [axA, axB, axC, axD, axE, axF, axG]

# ---------- Fig.5A ----------
ax = axA
ax.axis("off")
ax.set_title("A", fontweight="bold", loc="left")

if "A" in plot_data["fig5"] and "image_paths" in plot_data["fig5"]["A"]:
    image_paths = plot_data["fig5"]["A"]["image_paths"]
    inner_gs = GridSpecFromSubplotSpec(
        1, len(image_paths),
        subplot_spec=gs5[0, :],
        wspace=0.05
    )

    for i, path in enumerate(image_paths):
        ax_img = fig5.add_subplot(inner_gs[0, i])
        try:
            img = mpimg.imread(path)
            ax_img.imshow(img)
        except FileNotFoundError:
            ax_img.text(
                0.5, 0.5,
                f"Missing:\n{path}",
                ha="center", va="center",
                fontsize=10, color="gray"
            )
        ax_img.axis("off")

# ---------- Fig.5B ----------
ax = axB
B = plot_data["fig5"].get("B", None)
if B is not None:
    activation = B["activation"]
    im = ax.imshow(activation, aspect="auto")
    ax.set_title("B", fontweight="bold")
    ax.set_ylabel("Brain regions")
    ax.set_xlabel("Cognitive tokens")

    if "yticklabels" in B:
        ax.set_yticks(np.arange(len(B["yticklabels"])))
        ax.set_yticklabels(B["yticklabels"])

    ax.set_xticks(np.arange(0, activation.shape[1], 4))
    cbar = fig5.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

# ---------- Fig.5C ----------
# 如果你的 npy 里有 C，就画 C；否则尝试画 D 中保存的 latent / symbolic state
ax = axC
C = plot_data["fig5"].get("C", None)
D5 = plot_data["fig5"].get("D", None)

if C is not None and "pts" in C:
    pts = C["pts"]
    token_ids = C["token_ids"]

    sc = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c=token_ids,
        s=20,
        alpha=0.85,
        cmap="rainbow"
    )
    ax.set_title("C", fontweight="bold")
    ax.set_xlabel("Latent dimension 1")
    ax.set_ylabel("Latent dimension 2")
    cbar = fig5.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Token ID")

elif D5 is not None:
    latent_x = D5["latent_x"]
    latent_y = D5["latent_y"]
    symbolic_states = D5["symbolic_states"]

    sc = ax.scatter(
        latent_x,
        latent_y,
        c=symbolic_states,
        s=10,
        cmap="rainbow"
    )
    ax.set_title("C", fontweight="bold")
    ax.set_xlabel("Latent dimension 1")
    ax.set_ylabel("Latent dimension 2")
    cbar = fig5.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Symbolic state")

clean_axis(ax)

# ---------- Fig.5D ----------
ax = axD
E5 = plot_data["fig5"].get("E", None)
if E5 is not None:
    t = E5["t"]
    transition_rate = E5["transition_rate"]
    unc = E5["unc"]
    thr = E5["threshold"]
    act = E5["act"]

    ax.plot(t, transition_rate, lw=2, color=blue, label="Neural transition rate")
    ax.plot(t, unc, lw=2, color=orange, alpha=0.85, label="Predictive uncertainty")

    if "event_steps" in E5:
        for b in E5["event_steps"]:
            ax.axvline(b, ls=":", lw=1.5, ymin=0., ymax=0.85)

    ax.scatter(
        t[act == 1],
        unc[act == 1],
        s=28,
        color=red,
        label="Symbolic regulation"
    )

    ax.set_ylim(0, 1.05)
    ax.set_title("D", fontweight="bold")
    ax.set_xlabel("Inference step")
    ax.set_ylabel("Normalized value")
    ax.legend(frameon=False)
    clean_axis(ax)

# ---------- Fig.5E ----------
ax = axE
F5 = plot_data["fig5"].get("F", None)
if F5 is not None:
    centers = F5["centers"]
    freq = F5["freq"]

    ax.plot(centers, freq, marker="o", lw=2, color=blue)
    ax.set_title("E", fontweight="bold")
    ax.set_xlabel("Predictive uncertainty bin")
    ax.set_ylabel("Activation frequency")
    ax.set_ylim(-0.05, 1.05)
    clean_axis(ax)

# ---------- Fig.5F ----------
ax = axF
G5 = plot_data["fig5"].get("G", None)
if G5 is not None:
    noise = G5["noise"]
    spike = G5["spike"]
    base_model = G5["base_model"]

    ax.plot(noise, spike, marker="o", lw=2, color=blue, label="SpikeMind-H1")
    ax.plot(noise, base_model, marker="s", lw=2, color=red, label="Continuous latent model")
    ax.set_title("F", fontweight="bold")
    ax.set_xlabel("Perturbation intensity")
    ax.set_ylabel("Latent trajectory variance")
    ax.legend(frameon=False)
    clean_axis(ax)

# ---------- Fig.5G ----------
ax = axG
H5 = plot_data["fig5"].get("H", None)
if H5 is not None:
    conditions = H5["conditions"]
    instability = H5["instability"]
    err = H5["err"]

    ax.bar(
        conditions,
        instability,
        yerr=err,
        capsize=3,
        color=[dark_blue, green, light_green]
    )
    ax.set_title("G", fontweight="bold")
    ax.set_ylabel("Recursive instability index")
    ax.set_ylim(0, 0.7)
    clean_axis(ax)

fig5.savefig(fig5_path, dpi=300, bbox_inches="tight")

# ======================================================
# Figure 6
# Layout:
# A B C
# D   E
# F G H
# ======================================================
fig6 = plt.figure(figsize=(18, 14))
gs6 = fig6.add_gridspec(3, 3)

axA = fig6.add_subplot(gs6[0, 0])
axB = fig6.add_subplot(gs6[0, 1])
axC = fig6.add_subplot(gs6[0, 2])
axD = fig6.add_subplot(gs6[1, 0:2])
axE = fig6.add_subplot(gs6[1, 2])
axF = fig6.add_subplot(gs6[2, 0])
axG = fig6.add_subplot(gs6[2, 1])
axH = fig6.add_subplot(gs6[2, 2])

axs6 = [axA, axB, axC, axD, axE, axF, axG, axH]

task_colors = plot_data["fig6"].get(
    "task_colors",
    {
        "Depression": blue,
        "Sleep": green,
        "Affective": red,
    }
)

# ---------- Fig.6A ----------
ax = axA
A6 = plot_data["fig6"].get("A", None)
if A6 is not None:
    x = A6["x"]
    y = A6["y"]
    states = A6["states"]

    sc = ax.scatter(x, y, c=states, s=8, cmap="viridis")
    ax.scatter(x[0], y[0], marker="o", s=90, color=dark_blue, label="Start")
    ax.scatter(x[-1], y[-1], marker="*", s=100, color=red, label="Late state")
    ax.set_title("A", fontweight="bold")
    ax.set_xlabel("Latent dimension 1")
    ax.set_ylabel("Latent dimension 2")
    ax.legend(frameon=False)
    clean_axis(ax)

# ---------- Fig.6B ----------
ax = axB
B6 = plot_data["fig6"].get("B", None)
if B6 is not None:
    latent_x = B6["latent_x"]
    latent_y = B6["latent_y"]
    sleep_stage = B6["sleep_stage"]
    stage_names = B6.get(
        "stage_names",
        np.array(["Wake", "N1", "N2", "N3", "REM"], dtype=object)
    )

    stage_colors = {
        0: "#476EAE",
        1: "#48B3AF",
        2: "#FFD65A",
        3: "#FF9D23",
        4: "#D12052",
    }

    for i in range(len(latent_x) - 1):
        ax.plot(
            latent_x[i:i+2],
            latent_y[i:i+2],
            color=stage_colors[int(sleep_stage[i])],
            lw=2.0,
            alpha=0.6
        )

    for s in np.unique(sleep_stage):
        idx = np.where(sleep_stage == s)[0]
        ax.scatter(
            latent_x[idx],
            latent_y[idx],
            s=16,
            color=stage_colors[int(s)],
            alpha=0.85,
            label=stage_names[int(s)]
        )

    ax.set_title("B", fontweight="bold")
    ax.set_xlabel("Latent dimension 1")
    ax.set_ylabel("Latent dimension 2")
    ax.legend(frameon=False, fontsize=7, loc="best")
    clean_axis(ax)

# ---------- Fig.6C ----------
ax = axC
C6 = plot_data["fig6"].get("C", None)
if C6 is not None:
    aff_x = C6["aff_x"]
    aff_y = C6["aff_y"]
    aff_states = C6["aff_states"]
    state_names = C6.get(
        "state_names",
        np.array(["Negative", "Neutral", "Positive"], dtype=object)
    )

    aff_colors = {
        0: "#D12052",
        1: "#48B3AF",
        2: "#03AED2",
    }

    for s in np.unique(aff_states):
        idx = np.where(aff_states == s)[0]
        ax.scatter(
            aff_x[idx],
            aff_y[idx],
            s=13,
            color=aff_colors[int(s)],
            alpha=0.85,
            label=state_names[int(s)]
        )

    for boundary in [50, 100]:
        if boundary < len(aff_x):
            ax.scatter(
                aff_x[boundary],
                aff_y[boundary],
                s=80,
                marker="*",
                color="#FF9D23",
                edgecolor="black",
                linewidth=0.5,
                zorder=4,
                label="Transition" if boundary == 50 else None
            )

    ax.set_title("C", fontweight="bold")
    ax.set_xlabel("Latent dimension 1")
    ax.set_ylabel("Latent dimension 2")
    ax.legend(frameon=False, loc="best")
    clean_axis(ax)

# ---------- Fig.6D ----------
ax = axD
D6 = plot_data["fig6"].get("D", None)
if D6 is not None and "horizon" in D6:
    horizon = D6["horizon"]

    curves = [
        ("Depression", D6["depression_coherence"], D6["depression_err"], task_colors["Depression"], "o"),
        ("Sleep", D6["sleep_coherence"], D6["sleep_err"], task_colors["Sleep"], "s"),
        ("Affect", D6["affect_coherence"], D6["affect_err"], task_colors["Affective"], "^"),
    ]

    for name, coh, err, color, marker in curves:
        ax.plot(horizon, coh, marker=marker, lw=2.2, color=color, label=name)
        ax.fill_between(horizon, coh - err, coh + err, color=color, alpha=0.14, lw=0)

    ax.set_title("D", fontweight="bold")
    ax.set_xlabel("Time horizon")
    ax.set_ylabel("Trajectory coherence score")
    ax.set_ylim(0.78, 1.00)
    ax.set_xticks(horizon)
    ax.legend(frameon=False, loc="lower left", ncol=3)
    clean_axis(ax)

# ---------- Fig.6E ----------
ax = axE
E6 = plot_data["fig6"].get("E", None)
if E6 is not None and "top1_alignment" in E6:
    tasks = E6["tasks"]
    top1 = E6["top1_alignment"]
    top5 = E6["top5_alignment"]
    top1_err = E6["top1_err"]
    top5_err = E6["top5_err"]

    y = np.arange(len(tasks))
    height = 0.34

    ax.barh(
        y - height / 2,
        top1,
        height,
        xerr=top1_err,
        capsize=3,
        color=blue,
        label="Top-1 alignment"
    )
    ax.barh(
        y + height / 2,
        top5,
        height,
        xerr=top5_err,
        capsize=3,
        color=deep_orange,
        label="Top-5 alignment"
    )

    ax.set_title("E", fontweight="bold")
    ax.set_xlabel("Neural-symbolic alignment score")
    ax.set_yticks(y)
    ax.set_yticklabels(tasks)
    ax.set_xlim(0.75, 1.00)
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)

# ---------- Fig.6F ----------
ax = axF
F6 = plot_data["fig6"].get("F", None)
if F6 is not None and "MSE" in F6:
    h = F6["h"]
    MSE = F6["MSE"]
    labels = F6["labels"]

    for mse, lab in zip(MSE, labels):
        ax.plot(h, mse, marker="o", lw=1.8, label=lab)

    ax.set_title("F", fontweight="bold")
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("MSE")
    ax.legend(frameon=False)
    clean_axis(ax)

# ---------- Fig.6G ----------
ax = axG
G6 = plot_data["fig6"].get("G", None)
if G6 is not None:
    models = G6["models"]
    no_noise = G6["no_noise"]
    added_noise = G6["added_noise"]
    err_no_noise = G6["err_no_noise"]
    err_added_noise = G6["err_added_noise"]

    x = np.arange(len(models))
    width = 0.34

    ax.bar(
        x - width / 2,
        no_noise,
        width,
        yerr=err_no_noise,
        capsize=3,
        label="No noise",
        color=blue
    )
    ax.bar(
        x + width / 2,
        added_noise,
        width,
        yerr=err_added_noise,
        capsize=3,
        label="Added noise",
        color=deep_orange
    )

    ax.set_title("G", fontweight="bold")
    ax.set_ylabel("Latent-space variance")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax)

# ---------- Fig.6H ----------
ax = axH
H6 = plot_data["fig6"].get("H", None)
if H6 is not None:
    variants = H6["variants"]
    metrics = H6["metrics"]
    values = H6["values"]
    err = H6["err"]

    x = np.arange(len(variants))
    width = 0.23
    colors = [dark_green, yellow, deep_orange]

    for i, metric in enumerate(metrics):
        ax.bar(
            x + (i - 1) * width,
            values[:, i],
            width,
            yerr=err[:, i],
            capsize=3,
            label=metric,
            color=colors[i]
        )

    ax.set_title("H", fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.68, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend(frameon=False, ncol=1, fontsize=8)
    clean_axis(ax)

fig6.savefig(fig6_path, dpi=300, bbox_inches="tight")

# ======================================================
# Save combined PDF
# ======================================================
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig5, bbox_inches="tight")
    pdf.savefig(fig6, bbox_inches="tight")

plt.close(fig5)
plt.close(fig6)

print(f"Generated:")
print(f"- {fig5_path}")
print(f"- {fig6_path}")
print(f"- {pdf_path}")