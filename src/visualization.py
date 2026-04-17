"""Publication-quality visualizations for stroke MI-BCI results.

All plots use seaborn style and can optionally save to results/ directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Global style
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})


def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> None:
    """Save figure to disk if path given, otherwise show."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info("Saved figure to %s", save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_pipeline_comparison(
    results_df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart comparing pipeline accuracies with error bars.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of evaluate_all() with columns: pipeline, mean_acc, std_acc.
    save_path : str or Path, optional
        If given, save figure to this path.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    df = results_df.sort_values("mean_acc", ascending=True)
    colors = sns.color_palette("viridis", n_colors=len(df))

    bars = ax.barh(
        df["pipeline"],
        df["mean_acc"],
        xerr=df["std_acc"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=4,
    )

    # Add value labels
    for bar, acc in zip(bars, df["mean_acc"]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.3f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Chance level line
    ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1.5, label="Chance (50%)")

    ax.set_xlabel("Accuracy (5-fold CV)")
    ax.set_title("Pipeline Comparison — Stroke MI Classification")
    ax.set_xlim(0, 1.0)
    ax.legend(loc="lower right")
    fig.tight_layout()

    _save_or_show(fig, save_path)
    return fig


def plot_confusion_matrix(
    y_true: NDArray,
    y_pred: NDArray,
    title: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot confusion matrix heatmap.

    Parameters
    ----------
    y_true : NDArray
        True labels.
    y_pred : NDArray
        Predicted labels.
    title : str
        Plot title.
    save_path : str or Path, optional
        If given, save figure to this path.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    from sklearn.metrics import confusion_matrix as cm_func

    cm = cm_func(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Left MI", "Right MI"],
        yticklabels=["Left MI", "Right MI"],
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion Matrix")
    fig.tight_layout()

    _save_or_show(fig, save_path)
    return fig


def plot_topographic_erd(
    epochs: Any,
    sfreq: float,
    ch_names: list[str],
    band: tuple[float, float] = (8, 13),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot topographic map of Event-Related Desynchronization.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE Epochs object with montage set.
    sfreq : float
        Sampling frequency.
    ch_names : list[str]
        Channel names.
    band : tuple[float, float]
        Frequency band for ERD computation (default: mu 8–13 Hz).
    save_path : str or Path, optional
        If given, save figure to this path.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    import mne

    # Compute PSD for the specified band
    spectrum = epochs.compute_psd(
        method="welch",
        fmin=band[0],
        fmax=band[1],
        picks="eeg",
        verbose=False,
    )

    fig = spectrum.plot_topomap(
        bands={f"{band[0]}-{band[1]} Hz": band},
        ch_type="eeg",
        normalize=True,
        show=False,
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved topomap to %s", save_path)
        plt.close(fig)

    return fig


def plot_laterality_comparison(
    li_results: dict[str, dict[str, float]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Grouped bar chart of mu vs beta LI per patient.

    Parameters
    ----------
    li_results : dict
        Mapping patient_id → dict with 'mu_li' and 'beta_li' keys.
    save_path : str or Path, optional
        If given, save figure to this path.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    patients = sorted(li_results.keys())
    mu_vals = [li_results[p]["mu_li"] for p in patients]
    beta_vals = [li_results[p]["beta_li"] for p in patients]

    x = np.arange(len(patients))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_mu = ax.bar(x - width / 2, mu_vals, width, label="Mu (8–13 Hz)",
                     color="#2196F3", edgecolor="black", linewidth=0.5)
    bars_beta = ax.bar(x + width / 2, beta_vals, width, label="Beta (13–30 Hz)",
                       color="#FF9800", edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar in list(bars_mu) + list(bars_beta):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005 if height >= 0 else height - 0.02,
            f"{height:.3f}",
            ha="center", va="bottom" if height >= 0 else "top",
            fontsize=9,
        )

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
    ax.axhline(y=0.1, color="green", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Lateralization threshold (±0.1)")
    ax.axhline(y=-0.1, color="green", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Patient")
    ax.set_ylabel("Laterality Index")
    ax.set_title("Laterality Index — Mu vs Beta Band")
    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()

    _save_or_show(fig, save_path)
    return fig


def plot_csp_patterns(
    csp_fitted: Any,
    epochs_info: Any,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot CSP spatial patterns as topographic maps.

    Parameters
    ----------
    csp_fitted : mne.decoding.CSP
        Fitted CSP object.
    epochs_info : mne.Info
        MNE Info object from epochs (for channel positions).
    save_path : str or Path, optional
        If given, save figure to this path.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    import mne

    # CSP patterns_ shape: (n_components, n_channels)
    patterns = csp_fitted.patterns_
    n_components = patterns.shape[0]

    fig, axes = plt.subplots(1, n_components, figsize=(3 * n_components, 4))
    if n_components == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        mne.viz.plot_topomap(
            patterns[idx],
            epochs_info,
            axes=ax,
            show=False,
        )
        ax.set_title(f"CSP {idx + 1}")

    fig.suptitle("CSP Spatial Patterns", fontsize=14, fontweight="bold")
    fig.tight_layout()

    _save_or_show(fig, save_path)
    return fig
