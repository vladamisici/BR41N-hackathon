"""Channel selection via CSP ranking.

Identifies the most discriminative channels for stroke MI classification
using CSP spatial filter analysis.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def csp_rank_channels(
    X: NDArray,
    y: NDArray,
    n_components: int = 4,
) -> tuple[NDArray, NDArray]:
    """Rank channels by CSP filter importance.

    Fits CSP and computes per-channel importance as the sum of
    absolute spatial filter weights across selected components.

    Parameters
    ----------
    X : NDArray, shape (n_epochs, n_channels, n_times)
        Epoch data.
    y : NDArray, shape (n_epochs,)
        Labels.
    n_components : int
        Number of CSP components to use.

    Returns
    -------
    ranked_indices : NDArray, shape (n_channels,)
        Channel indices sorted by importance (most important first).
    importance_scores : NDArray, shape (n_channels,)
        Importance score per channel (sorted same as ranked_indices).
    """
    from mne.decoding import CSP

    csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True)
    csp.fit(X, y)

    # Spatial filters shape: (n_components, n_channels)
    filters = csp.filters_[:n_components]

    # Per-channel importance = sum of absolute weights across components
    importance = np.abs(filters).sum(axis=0)

    ranked_indices = np.argsort(importance)[::-1]
    importance_sorted = importance[ranked_indices]

    logger.info("CSP channel ranking (top 5): %s", ranked_indices[:5])
    return ranked_indices, importance_sorted
