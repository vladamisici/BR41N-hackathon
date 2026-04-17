"""Channel selection via CSP ranking and genetic algorithm optimization.

Identifies the most discriminative channel subsets for stroke MI classification
using CSP component analysis and DEAP-based evolutionary search.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression

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


def _evaluate_channel_subset(
    X: NDArray,
    y: NDArray,
    channel_mask: list[int],
) -> float:
    """Evaluate a channel subset using TS+LR with 3-fold CV.

    Parameters
    ----------
    X : NDArray
        Full epoch data.
    y : NDArray
        Labels.
    channel_mask : list[int]
        Binary mask (1 = include, 0 = exclude) for each channel.

    Returns
    -------
    float
        Mean cross-validation accuracy.
    """
    selected = [i for i, m in enumerate(channel_mask) if m == 1]
    if len(selected) < 2:
        return 0.0  # Need at least 2 channels for covariance

    X_sub = X[:, selected, :]

    pipe = Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("lr", LogisticRegression(C=1.0, max_iter=500)),
    ])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(pipe, X_sub, y, cv=cv, scoring="accuracy")
        return float(scores.mean())
    except Exception:
        return 0.0


def ga_channel_selection(
    X: NDArray,
    y: NDArray,
    n_gen: int = 30,
    pop_size: int = 50,
) -> dict[str, Any]:
    """Genetic algorithm channel selection using DEAP.

    Evolves binary channel masks to maximize cross-validation accuracy.

    Parameters
    ----------
    X : NDArray, shape (n_epochs, n_channels, n_times)
        Epoch data.
    y : NDArray, shape (n_epochs,)
        Labels.
    n_gen : int
        Number of generations.
    pop_size : int
        Population size.

    Returns
    -------
    dict
        Keys: best_channels (list[int] indices), best_accuracy (float),
              best_mask (list[int]), history (list of per-gen best fitness).
    """
    from deap import base, creator, tools, algorithms

    n_channels = X.shape[1]

    # DEAP setup — maximize accuracy
    if not hasattr(creator, "FitnessMax_ChanSel"):
        creator.create("FitnessMax_ChanSel", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual_ChanSel"):
        creator.create("Individual_ChanSel", list, fitness=creator.FitnessMax_ChanSel)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual_ChanSel,
        toolbox.attr_bool,
        n=n_channels,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_individual(individual: list[int]) -> tuple[float]:
        acc = _evaluate_channel_subset(X, y, individual)
        return (acc,)

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / n_channels)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run GA
    random.seed(42)
    np.random.seed(42)

    pop = toolbox.population(n=pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    hof = tools.HallOfFame(1)

    logger.info("Starting GA channel selection: %d gen, %d pop", n_gen, pop_size)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.7,
        mutpb=0.2,
        ngen=n_gen,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )

    best = hof[0]
    best_channels = [i for i, m in enumerate(best) if m == 1]
    best_acc = best.fitness.values[0]

    history = [record["max"] for record in logbook]

    logger.info(
        "GA result: %d/%d channels selected, accuracy=%.3f",
        len(best_channels), n_channels, best_acc,
    )
    logger.info("Selected channels: %s", best_channels)

    return {
        "best_channels": best_channels,
        "best_accuracy": best_acc,
        "best_mask": list(best),
        "history": history,
    }
