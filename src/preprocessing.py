"""Data augmentation for EEG epochs.

Provides noise injection, sliding window, Fourier-transform surrogates,
and hemisphere recombination strategies tailored for stroke MI data.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def augment_gaussian_noise(
    X: NDArray,
    y: NDArray,
    std: float = 0.1,
    n_copies: int = 2,
) -> tuple[NDArray, NDArray]:
    """Add Gaussian noise to create augmented copies.

    Parameters
    ----------
    X : NDArray, shape (n_epochs, n_channels, n_times)
        Original epoch data.
    y : NDArray, shape (n_epochs,)
        Labels.
    std : float
        Standard deviation of noise relative to per-epoch std.
    n_copies : int
        Number of noisy copies per original epoch.

    Returns
    -------
    X_aug : NDArray
        Concatenation of original + augmented data.
    y_aug : NDArray
        Concatenation of original + augmented labels.
    """
    augmented_X = [X]
    augmented_y = [y]

    for _ in range(n_copies):
        noise = np.random.randn(*X.shape) * std * X.std(axis=-1, keepdims=True)
        augmented_X.append(X + noise)
        augmented_y.append(y.copy())

    X_aug = np.concatenate(augmented_X, axis=0)
    y_aug = np.concatenate(augmented_y, axis=0)
    logger.info("Gaussian noise augmentation: %d → %d epochs", len(y), len(y_aug))
    return X_aug, y_aug


def augment_sliding_window(
    X: NDArray,
    y: NDArray,
    window_size: int,
    stride: int,
) -> tuple[NDArray, NDArray]:
    """Create augmented epochs via sliding window cropping.

    Parameters
    ----------
    X : NDArray, shape (n_epochs, n_channels, n_times)
        Original epoch data.
    y : NDArray, shape (n_epochs,)
        Labels.
    window_size : int
        Size of each window in samples.
    stride : int
        Stride between consecutive windows in samples.

    Returns
    -------
    X_aug : NDArray, shape (n_new, n_channels, window_size)
        Augmented epochs.
    y_aug : NDArray, shape (n_new,)
        Augmented labels.
    """
    n_times = X.shape[2]
    if window_size > n_times:
        raise ValueError(
            f"window_size ({window_size}) > n_times ({n_times})"
        )

    crops_X: list[NDArray] = []
    crops_y: list[int] = []

    for i in range(X.shape[0]):
        start = 0
        while start + window_size <= n_times:
            crops_X.append(X[i, :, start : start + window_size])
            crops_y.append(y[i])
            start += stride

    X_aug = np.array(crops_X)
    y_aug = np.array(crops_y)
    logger.info("Sliding window augmentation: %d → %d epochs", len(y), len(y_aug))
    return X_aug, y_aug


def augment_ft_surrogate(
    X: NDArray,
    y: NDArray,
    phase_noise_mag: float = 0.3,
    n_copies: int = 2,
) -> tuple[NDArray, NDArray]:
    """Create Fourier-transform surrogate epochs.

    Preserves power spectrum while randomizing phase by a controlled amount.

    Parameters
    ----------
    X : NDArray, shape (n_epochs, n_channels, n_times)
        Original epoch data.
    y : NDArray, shape (n_epochs,)
        Labels.
    phase_noise_mag : float
        Maximum phase perturbation in radians (scaled by pi).
    n_copies : int
        Number of surrogate copies per original epoch.

    Returns
    -------
    X_aug : NDArray
        Concatenation of original + surrogate data.
    y_aug : NDArray
        Concatenation of original + surrogate labels.
    """
    augmented_X = [X]
    augmented_y = [y]

    for _ in range(n_copies):
        X_fft = np.fft.rfft(X, axis=-1)
        n_freq = X_fft.shape[-1]

        # Random phase perturbation — same across channels per epoch for consistency
        phase_noise = (
            np.random.uniform(-phase_noise_mag * np.pi, phase_noise_mag * np.pi,
                              size=(X.shape[0], 1, n_freq))
        )
        X_fft_perturbed = X_fft * np.exp(1j * phase_noise)
        X_surrogate = np.fft.irfft(X_fft_perturbed, n=X.shape[-1], axis=-1)

        augmented_X.append(X_surrogate)
        augmented_y.append(y.copy())

    X_aug = np.concatenate(augmented_X, axis=0)
    y_aug = np.concatenate(augmented_y, axis=0)
    logger.info("FT surrogate augmentation: %d → %d epochs", len(y), len(y_aug))
    return X_aug, y_aug


def augment_hemisphere_recombination(
    X: NDArray,
    y: NDArray,
    left_idx: list[int],
    right_idx: list[int],
    n_copies: int = 2,
) -> tuple[NDArray, NDArray]:
    """Recombine hemispheres from different epochs of the same class.

    Takes left-hemisphere channels from one epoch and right-hemisphere
    channels from another epoch of the same class, creating chimeric epochs.

    Parameters
    ----------
    X : NDArray, shape (n_epochs, n_channels, n_times)
        Original epoch data.
    y : NDArray, shape (n_epochs,)
        Labels.
    left_idx : list[int]
        Channel indices for left hemisphere.
    right_idx : list[int]
        Channel indices for right hemisphere.
    n_copies : int
        Number of recombined copies per original epoch.

    Returns
    -------
    X_aug : NDArray
        Concatenation of original + recombined data.
    y_aug : NDArray
        Concatenation of original + recombined labels.
    """
    augmented_X = [X]
    augmented_y = [y]

    classes = np.unique(y)

    for _ in range(n_copies):
        new_epochs = []
        new_labels = []
        for cls in classes:
            cls_mask = y == cls
            cls_indices = np.where(cls_mask)[0]
            if len(cls_indices) < 2:
                continue

            # Random pairing within the class
            shuffled = np.random.permutation(cls_indices)
            partner = np.random.permutation(cls_indices)

            for i, j in zip(shuffled, partner):
                if i == j:
                    continue
                chimera = X[i].copy()
                chimera[right_idx] = X[j, right_idx]
                new_epochs.append(chimera)
                new_labels.append(cls)

        if new_epochs:
            augmented_X.append(np.array(new_epochs))
            augmented_y.append(np.array(new_labels))

    X_aug = np.concatenate(augmented_X, axis=0)
    y_aug = np.concatenate(augmented_y, axis=0)
    logger.info(
        "Hemisphere recombination augmentation: %d → %d epochs",
        len(y), len(y_aug),
    )
    return X_aug, y_aug
