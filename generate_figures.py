#!/usr/bin/env python
"""Generate all 8 publication-quality figures for the BR41N.IO presentation."""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["MNE_LOGGING_LEVEL"] = "ERROR"

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import mne
mne.set_log_level("ERROR")

# Brand colors
LIME = "#C8FA32"
DARK = "#0A0A0A"
WHITE = "#FFFFFF"
GRAY = "#888888"
LIGHT_GRAY = "#CCCCCC"
RED = "#FF4444"

OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": DARK,
    "axes.facecolor": DARK,
    "text.color": WHITE,
    "axes.labelcolor": WHITE,
    "xtick.color": WHITE,
    "ytick.color": WHITE,
    "axes.edgecolor": GRAY,
    "savefig.facecolor": DARK,
    "savefig.dpi": 200,
    "font.family": "sans-serif",
    "font.size": 12,
})

print("Generating 8 figures...")

# ═══════════════════════════════════════════════════════════════════
# 1. results_main.png
# ═══════════════════════════════════════════════════════════════════
print("  1/8 results_main.png")
fig, ax = plt.subplots(figsize=(13, 6.5))

conditions = ["P1_pre", "P1_post", "P2_pre", "P2_post", "P3_pre", "P3_post"]
csp_bl = [77.1, 93.9, 68.4, 96.1, 74.4, 79.7]
pca_bl = [92.9, 97.0, 72.4, 97.4, 93.6, 100.0]
mine = [91.2, 93.8, 83.8, 100.0, 97.5, 93.8]
labels_mine = ["ACM(3,7)", "FBCSP+LDA", "FBCSP+LDA", "FBCSP+LDA", "ACM(3,7)", "FBCSP+LDA"]
beat_both = [False, False, True, True, True, False]

x = np.arange(len(conditions))
w = 0.25

bars1 = ax.bar(x - w, csp_bl, w, color=GRAY, label="CSP+LDA baseline", edgecolor=DARK)
bars2 = ax.bar(x, pca_bl, w, color=LIGHT_GRAY, label="PCA+TVLDA baseline", edgecolor=DARK)
bars3 = ax.bar(x + w, mine, w, color=LIME, label="My best pipeline", edgecolor=DARK)

for i, (bar, val, lbl) in enumerate(zip(bars3, mine, labels_mine)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color=LIME)
    ax.text(bar.get_x() + bar.get_width()/2, val - 4, lbl,
            ha="center", va="top", fontsize=7, color=DARK, fontweight="bold")

for i, bb in enumerate(beat_both):
    if bb:
        ax.text(x[i] + w, mine[i] + 5, "★ BEAT BOTH", ha="center", va="bottom",
                fontsize=9, color=LIME, fontweight="bold")

ax.axhline(y=50, color=RED, linestyle="--", linewidth=1, alpha=0.7, label="Chance (50%)")
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_ylim(40, 112)
ax.set_title("Test accuracy: my pipelines vs. organizer baselines (n=80 trials each)",
             fontsize=14, fontweight="bold", color=WHITE, pad=15)
ax.legend(loc="lower right", fontsize=10, facecolor="#1a1a1a", edgecolor=GRAY)
ax.grid(axis="y", alpha=0.15)
fig.tight_layout()
fig.savefig(f"{OUT}/results_main.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════
# 2. paradigm.png
# ═══════════════════════════════════════════════════════════════════
print("  2/8 paradigm.png")
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.set_xlim(-0.5, 9)
ax.set_ylim(-3, 2.5)
ax.axis("off")

# Timeline bar
ax.barh(0.5, 2, left=0, height=0.6, color=GRAY, edgecolor=WHITE, linewidth=0.5)
ax.text(1, 0.5, "PREPARATION", ha="center", va="center", fontsize=10, color=WHITE, fontweight="bold")

ax.barh(0.5, 1, left=2, height=0.6, color=LIGHT_GRAY, edgecolor=WHITE, linewidth=0.5)
ax.text(2.5, 0.5, "MI\nstarts", ha="center", va="center", fontsize=8, color=DARK)

ax.barh(0.5, 5, left=3, height=0.6, color=LIME, edgecolor=WHITE, linewidth=0.5)
ax.text(5.5, 0.5, "FEEDBACK PHASE (FES + visual avatar)", ha="center", va="center",
        fontsize=11, color=DARK, fontweight="bold")

# Time markers
for t, lbl in [(0, "0s"), (2, "2s\ncue"), (3, "3s"), (8, "8s\nrelax")]:
    ax.axvline(x=t, ymin=0.35, ymax=0.65, color=WHITE, linewidth=1.5)
    ax.text(t, -0.2, lbl, ha="center", va="top", fontsize=9, color=WHITE)

# My epoch box
rect = mpatches.FancyBboxPatch((3, -1.5), 4, 0.8, boxstyle="round,pad=0.1",
                                 facecolor="none", edgecolor=LIME, linewidth=2.5)
ax.add_patch(rect)
ax.text(5, -1.1, "MY EPOCH: 3–7s (feedback window)", ha="center", va="center",
        fontsize=11, color=LIME, fontweight="bold")
ax.text(5, -1.6, "84–100% accuracy", ha="center", va="center", fontsize=10, color=LIME)

# Standard epoch box
rect2 = mpatches.FancyBboxPatch((0.5, -2.8), 4, 0.8, boxstyle="round,pad=0.1",
                                  facecolor="none", edgecolor=GRAY, linewidth=1.5, linestyle="--")
ax.add_patch(rect2)
ax.text(2.5, -2.4, "standard epoch: 0.5–4.5s (preparation)", ha="center", va="center",
        fontsize=10, color=GRAY)
ax.text(2.5, -2.9, "52–69% accuracy", ha="center", va="center", fontsize=10, color=GRAY)

ax.set_title("Key insight: epoch the feedback phase, not preparation",
             fontsize=14, fontweight="bold", color=WHITE, pad=15)
fig.tight_layout()
fig.savefig(f"{OUT}/paradigm.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════
# 3. lateralization.png
# ═══════════════════════════════════════════════════════════════════
print("  3/8 lateralization.png")
fig, ax = plt.subplots(figsize=(11, 5.5))

patients = ["P1", "P2", "P3"]
pre_li = [-0.297, -0.186, -0.018]
post_li = [+0.075, -0.186, -0.073]

x = np.arange(len(patients))
w = 0.3
bars_pre = ax.bar(x - w/2, pre_li, w, color=GRAY, label="PRE", edgecolor=WHITE, linewidth=0.5)
bars_post = ax.bar(x + w/2, post_li, w, color=LIME, label="POST", edgecolor=WHITE, linewidth=0.5)

for bar, val in zip(bars_pre, pre_li):
    ax.text(bar.get_x() + bar.get_width()/2, val - 0.02, f"{val:+.3f}",
            ha="center", va="top", fontsize=10, color=WHITE)
for bar, val in zip(bars_post, post_li):
    yoff = 0.02 if val >= 0 else -0.02
    va = "bottom" if val >= 0 else "top"
    ax.text(bar.get_x() + bar.get_width()/2, val + yoff, f"{val:+.3f}",
            ha="center", va=va, fontsize=10, color=LIME, fontweight="bold")

ax.axhline(y=0, color=WHITE, linewidth=1)

# Rehab arrow for P1
ax.annotate("REHABILITATION\nEFFECT", xy=(0 + w/2, 0.075), xytext=(0 - w/2, -0.297),
            arrowprops=dict(arrowstyle="->", color=LIME, lw=2.5, connectionstyle="arc3,rad=-0.3"),
            fontsize=9, color=LIME, fontweight="bold", ha="center", va="center")

ax.text(2.8, 0.12, "LI > 0 → bilateral / healthy", fontsize=9, color="#44ff44", fontstyle="italic")
ax.text(2.8, -0.32, "LI < 0 → right-hemisphere\ndominant", fontsize=9, color="#ff6666", fontstyle="italic")

ax.set_xticks(x)
ax.set_xticklabels(patients, fontsize=13)
ax.set_ylabel("Mu-band Lateralization Index", fontsize=12)
ax.set_title("Lateralization Index — clinical biomarker of recovery",
             fontsize=14, fontweight="bold", color=WHITE, pad=15)
ax.legend(fontsize=11, facecolor="#1a1a1a", edgecolor=GRAY)
ax.grid(axis="y", alpha=0.15)
fig.tight_layout()
fig.savefig(f"{OUT}/lateralization.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════
# 4. heatmap_pipelines.png
# ═══════════════════════════════════════════════════════════════════
print("  4/8 heatmap_pipelines.png")
fig, ax = plt.subplots(figsize=(13, 6.5))

pipelines = ["FBCSP+LDA", "ACM(3,7)", "TS+LR", "FgMDM", "TS+SVM", "MDM", "TS+LDA", "CSP+LDA"]
conds = ["P1_pre", "P1_post", "P2_pre", "P2_post", "P3_pre", "P3_post"]
data = np.array([
    [87.5, 91.2, 87.5, 82.5, 81.2, 76.2, 65.0, 51.2],
    [93.8, 81.2, 75.0, 91.2, 75.0, 66.2, 80.0, 73.8],
    [83.8, 66.2, 63.8, 65.0, 57.5, 51.2, 66.2, 56.2],
    [100.0, 98.8, 98.8, 96.2, 95.0, 91.2, 76.2, 92.5],
    [80.0, 97.5, 87.5, 87.5, 87.5, 78.8, 78.8, 86.2],
    [93.8, 85.0, 80.0, 80.0, 76.2, 76.2, 73.8, 82.5],
])

im = ax.imshow(data, cmap="viridis", vmin=50, vmax=100, aspect="auto")

for i in range(len(conds)):
    winner_j = np.argmax(data[i])
    for j in range(len(pipelines)):
        color = WHITE if data[i, j] > 70 else LIGHT_GRAY
        weight = "bold" if j == winner_j else "normal"
        ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center",
                fontsize=10, color=color, fontweight=weight)
    # Lime border on winner
    rect = plt.Rectangle((winner_j - 0.5, i - 0.5), 1, 1,
                          fill=False, edgecolor=LIME, linewidth=2.5)
    ax.add_patch(rect)

ax.set_xticks(range(len(pipelines)))
ax.set_xticklabels(pipelines, fontsize=10, rotation=30, ha="right")
ax.set_yticks(range(len(conds)))
ax.set_yticklabels(conds, fontsize=11)
ax.set_title("All 8 pipelines × 6 conditions  (lime border = winner per row)",
             fontsize=14, fontweight="bold", color=WHITE, pad=15)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Accuracy (%)", color=WHITE)
cbar.ax.yaxis.set_tick_params(color=WHITE)
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=WHITE)
fig.tight_layout()
fig.savefig(f"{OUT}/heatmap_pipelines.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════
# 5. erd_topomap_p1.png (uses real data)
# ═══════════════════════════════════════════════════════════════════
print("  5/8 erd_topomap_p1.png")
from src.loading import load_gtec_stroke_data, CH_NAMES

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

for idx, (stage, ax) in enumerate(zip(["pre", "post"], axes)):
    mat_path = f"dataset/stroke-rehab/P1_{stage}_training.mat"
    raw, sfreq = load_gtec_stroke_data(mat_path)
    raw_filt = raw.copy().filter(8, 13, method="iir",
                                  iir_params=dict(order=5, ftype="butter"),
                                  picks="eeg", verbose=False)
    events = mne.find_events(raw_filt, stim_channel="STI", verbose=False)
    event_ids = sorted(set(events[:, 2]) - {0})
    event_id = {"left": 1, "right": 2} if set(event_ids) == {1, 2} else {"left": event_ids[0], "right": event_ids[1]}
    epochs = mne.Epochs(raw_filt, events, event_id=event_id, tmin=3.0, tmax=7.0,
                        baseline=None, reject=None, preload=True, verbose=False)

    left_data = epochs["left"].get_data(picks="eeg")
    right_data = epochs["right"].get_data(picks="eeg")

    left_power = np.log(np.var(left_data, axis=-1)).mean(axis=0)
    right_power = np.log(np.var(right_data, axis=-1)).mean(axis=0)
    diff = left_power - right_power

    info = mne.create_info(ch_names=CH_NAMES, sfreq=sfreq, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="warn")

    vmax = max(abs(diff.min()), abs(diff.max()), 0.5)
    mne.viz.plot_topomap(diff, info, axes=ax, cmap="RdBu_r",
                         vlim=(-vmax, vmax), contours=4, show=False)

    li_vals = {"pre": -0.297, "post": +0.075}
    patterns = {"pre": "right-dominant", "post": "bilateral"}
    ax.set_title(f"{stage.upper()}-intervention\n(LI={li_vals[stage]:+.3f}, {patterns[stage]})",
                 fontsize=12, color=WHITE, pad=10)

fig.suptitle("Mu-band ERD topography — Patient 1 rehabilitation effect",
             fontsize=14, fontweight="bold", color=WHITE, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/erd_topomap_p1.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════
# 6. confusion_matrix_p2pre.png (uses real data)
# ═══════════════════════════════════════════════════════════════════
print("  6/8 confusion_matrix_p2pre.png")
from src.classifiers import build_fbcsp_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import clone

raw_tr, _ = load_gtec_stroke_data("dataset/stroke-rehab/P2_pre_training.mat")
raw_te, _ = load_gtec_stroke_data("dataset/stroke-rehab/P2_pre_test.mat")

def quick_extract(raw):
    raw_f = raw.copy().filter(0.5, 30, method="iir",
                               iir_params=dict(order=5, ftype="butter"),
                               picks="eeg", verbose=False)
    ev = mne.find_events(raw_f, stim_channel="STI", verbose=False)
    eids = sorted(set(ev[:, 2]) - {0})
    eid = {"left": 1, "right": 2} if set(eids) == {1, 2} else {"left": eids[0], "right": eids[1]}
    ep = mne.Epochs(raw_f, ev, event_id=eid, tmin=3.0, tmax=7.0,
                    baseline=None, reject=None, preload=True, verbose=False)
    X = ep.get_data(picks="eeg")
    y = ep.events[:, 2]
    u = sorted(np.unique(y))
    y = np.array([{old: i for i, old in enumerate(u)}[l] for l in y])
    return X, y

X_tr, y_tr = quick_extract(raw_tr)
X_te, y_te = quick_extract(raw_te)

pipe = build_fbcsp_pipeline(sfreq=256.0)
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_te)
cm = confusion_matrix(y_te, y_pred)
acc = accuracy_score(y_te, y_pred)

fig, ax = plt.subplots(figsize=(7, 7))
labels = ["Left hand", "Right hand"]

for i in range(2):
    for j in range(2):
        color = LIME if i == j else "#333333"
        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor=DARK, linewidth=3))
        txt_color = DARK if i == j else WHITE
        ax.text(j + 0.5, i + 0.5, str(cm[i, j]), ha="center", va="center",
                fontsize=60, fontweight="bold", color=txt_color)

ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5])
ax.set_xticklabels(labels, fontsize=13)
ax.set_yticks([0.5, 1.5])
ax.set_yticklabels(labels, fontsize=13)
ax.set_xlabel("Predicted", fontsize=13, labelpad=10)
ax.set_ylabel("Actual", fontsize=13, labelpad=10)
ax.invert_yaxis()

left_recall = cm[0, 0] / cm[0].sum() * 100
right_recall = cm[1, 1] / cm[1].sum() * 100
ax.text(1, 2.15, f"Accuracy: {acc:.1%}  |  Left recall: {left_recall:.1f}%  |  Right recall: {right_recall:.1f}%",
        ha="center", va="top", fontsize=11, color=LIGHT_GRAY, transform=ax.transData)

ax.set_title("P2_pre confusion matrix (FBCSP+LDA — the hardest patient)",
             fontsize=13, fontweight="bold", color=WHITE, pad=15)
fig.tight_layout()
fig.savefig(f"{OUT}/confusion_matrix_p2pre.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════
# 7. tfr_p1_c3.png (uses real data)
# ═══════════════════════════════════════════════════════════════════
print("  7/8 tfr_p1_c3.png")
raw_p1, sfreq = load_gtec_stroke_data("dataset/stroke-rehab/P1_pre_training.mat")
raw_p1_f = raw_p1.copy().filter(0.5, 40, method="iir",
                                 iir_params=dict(order=5, ftype="butter"),
                                 picks="eeg", verbose=False)
events = mne.find_events(raw_p1_f, stim_channel="STI", verbose=False)
eids = sorted(set(events[:, 2]) - {0})
eid = {"left": 1, "right": 2} if set(eids) == {1, 2} else {"left": eids[0], "right": eids[1]}
epochs_p1 = mne.Epochs(raw_p1_f, events, event_id=eid, tmin=-1, tmax=8,
                        baseline=None, reject=None, preload=True, verbose=False)

right_epochs = epochs_p1["right"]
freqs = np.arange(4, 31, 1)
n_cycles = freqs / 2.0

tfr = right_epochs.compute_tfr(method="morlet", freqs=freqs, n_cycles=n_cycles,
                                picks=["C3"], return_itc=False, verbose=False)
tfr.apply_baseline(baseline=(-1, 0), mode="logratio", verbose=False)

fig, ax = plt.subplots(figsize=(13, 5))
tfr_data = tfr.data.mean(axis=0)[0]  # average over epochs, pick C3
times = tfr.times

im = ax.imshow(tfr_data, aspect="auto", origin="lower",
               extent=[times[0], times[-1], freqs[0], freqs[-1]],
               cmap="RdBu_r", vmin=-0.5, vmax=0.5)

for t, lbl in [(0, "trigger"), (2, "cue"), (3, "feedback"), (8, "relax")]:
    ax.axvline(x=t, color=WHITE, linewidth=1, alpha=0.7)
    ax.text(t, 31, lbl, ha="center", va="bottom", fontsize=8, color=WHITE)

# Highlight 3-7s × 8-13 Hz
rect = plt.Rectangle((3, 8), 4, 5, fill=False, edgecolor=LIME, linewidth=2.5)
ax.add_patch(rect)
ax.text(5, 14, "MY EPOCH", ha="center", va="bottom", fontsize=9, color=LIME, fontweight="bold")

ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Frequency (Hz)", fontsize=12)
ax.set_title("Time-frequency at C3 — Patient 1, right-hand MI (ERD visible at 8–13 Hz, 3–7s)",
             fontsize=13, fontweight="bold", color=WHITE, pad=15)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Power (log-ratio)", color=WHITE)
cbar.ax.yaxis.set_tick_params(color=WHITE)
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=WHITE)
fig.tight_layout()
fig.savefig(f"{OUT}/tfr_p1_c3.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════
# 8. pipeline_arch.png
# ═══════════════════════════════════════════════════════════════════
print("  8/8 pipeline_arch.png")
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 5.5)
ax.axis("off")

def draw_box(ax, x, y, w, h, text, color=GRAY, text_color=WHITE, fontsize=9):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                           facecolor=color, edgecolor=WHITE, linewidth=1)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold", wrap=True)

# Top row: preprocessing
boxes_top = [
    (0.2, 4, 2, 0.9, "Raw EEG\n.mat, 16ch, 256Hz"),
    (2.8, 4, 2, 0.9, "Bandpass\n0.5–30 Hz"),
    (5.4, 4, 2, 0.9, "Epoch\n3–7s"),
    (8.0, 4, 2, 0.9, "Covariance\nC ∈ ℝ¹⁶ˣ¹⁶"),
    (10.6, 4, 2, 0.9, "Train/Test\n80/80"),
]
for bx, by, bw, bh, txt in boxes_top:
    draw_box(ax, bx, by, bw, bh, txt, color="#222222")

# Arrows between top boxes
for i in range(len(boxes_top) - 1):
    x1 = boxes_top[i][0] + boxes_top[i][2]
    x2 = boxes_top[i+1][0]
    y = boxes_top[i][1] + boxes_top[i][3] / 2
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="->", color=WHITE, lw=1.5))

# Bottom row: pipelines
pipe_boxes = [
    (0.3, 1.5, 2.5, 1.2, "FBCSP+LDA\n(primary)", LIME, DARK, 11),
    (3.2, 1.5, 2.2, 1.2, "ACM(3,7)\n(secondary)", LIME, DARK, 10),
    (5.8, 1.5, 1.8, 1.2, "TS+LR\n(backup)", GRAY, WHITE, 9),
    (8.0, 1.5, 1.5, 1.2, "FgMDM\nTS+SVM\nMDM", "#333", WHITE, 8),
    (9.9, 1.5, 1.8, 1.2, "CSP+LDA\n(baseline)", "#222", GRAY, 9),
]
for bx, by, bw, bh, txt, col, tcol, fs in pipe_boxes:
    draw_box(ax, bx, by, bw, bh, txt, color=col, text_color=tcol, fontsize=fs)

# Arrow from preprocessing to pipelines
ax.annotate("", xy=(6, 2.7), xytext=(6, 4),
            arrowprops=dict(arrowstyle="->", color=WHITE, lw=2))

# Result box
draw_box(ax, 2, 0, 9, 0.8, "Best pipeline per patient → beat both organizer baselines on 3/6 conditions",
         color=LIME, text_color=DARK, fontsize=11)

ax.annotate("", xy=(6, 0.8), xytext=(6, 1.5),
            arrowprops=dict(arrowstyle="->", color=LIME, lw=2))

ax.set_title("Pipeline architecture", fontsize=14, fontweight="bold", color=WHITE, pad=15)
fig.savefig(f"{OUT}/pipeline_arch.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════
print("\nDone! All figures saved to results/figures/")
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(f"{OUT}/{f}") / 1024
    print(f"  {f}: {size:.0f} KB")
