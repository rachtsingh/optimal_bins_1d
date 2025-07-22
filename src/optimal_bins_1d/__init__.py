import warnings

import numpy as np

from optimal_bins_1d import linear, loglinear, quadratic
from optimal_bins_1d.utils import check_if_n_bins_possible


def optimal_bins_quadratic(
    data: np.ndarray,
    n_bins: int,
    assume_sorted: bool = False,
) -> np.ndarray:
    """
    Compute optimal bin edges for 1D data using quadratic ckmeans algorithm.

    This implements the QUADRATIC_CKMEANS algorithm (quadratic time solution)
    using dynamic programming without speedups.

    Parameters
    ----------
    data : np.ndarray
        1D array of data to bin
    n_bins : int
        Number of bins to create

    Returns
    -------
    np.ndarray
        Bin edges (length n_bins + 1)
    """
    assert len(data.shape) == 1, "optimal_bins_1d requires a 1D array"
    assert not np.isnan(data).any(), (
        "please remove NaNs before calling optimal_bins_quadratic"
    )
    assert n_bins > 0, "n_bins must be positive"

    if len(data) == 0:
        return np.array([0.0, 1.0])

    if n_bins == 1:
        return np.array([np.min(data), np.max(data)])

    if not assume_sorted:
        data = np.sort(data)

    if (early_ret := check_if_n_bins_possible(data, n_bins)) is not None:
        n_unique = np.unique(data).shape[0]
        warnings.warn(
            f"Not possible to assign {n_bins} bins to this problem, "
            f"number of unique values is {n_unique}. "
            f"Creating {n_unique} bins instead, one for each unique value.",
            UserWarning,
        )
        return early_ret

    # Run the quadratic ckmeans algorithm
    return quadratic.ckmeans_1d_dp_quadratic(data, n_bins)


def optimal_bins_loglinear(
    data: np.ndarray,
    n_bins: int,
    assume_sorted: bool = False,
) -> np.ndarray:
    """
    Compute optimal bin edges for 1D data using loglinear ckmeans algorithm.

    This implements the LOGLINEAR_CKMEANS algorithm using convex Monge properties
    for O(k * n log n) time complexity.

    Parameters
    ----------
    data : np.ndarray
        1D array of data to bin
    n_bins : int
        Number of bins to create

    Returns
    -------
    np.ndarray
        Bin edges (length n_bins + 1)
    """
    assert len(data.shape) == 1, "optimal_bins_1d requires a 1D array"
    assert not np.isnan(data).any(), (
        "please remove NaNs before calling optimal_bins_loglinear"
    )
    assert n_bins > 0, "n_bins must be positive"

    if len(data) == 0:
        return np.array([0.0, 1.0])

    if n_bins == 1:
        return np.array([np.min(data), np.max(data)])

    if not assume_sorted:
        data = np.sort(data)

    if (early_ret := check_if_n_bins_possible(data, n_bins)) is not None:
        n_unique = np.unique(data).shape[0]
        warnings.warn(
            f"Not possible to assign {n_bins} bins to this problem, "
            f"number of unique values is {n_unique}. "
            f"Creating {n_unique} bins instead, one for each unique value.",
            UserWarning,
        )
        return early_ret

    # Run the loglinear ckmeans algorithm
    return loglinear.ckmeans_1d_dp_loglinear(data, n_bins)


def optimal_bins_linear(
    data: np.ndarray,
    n_bins: int,
    assume_sorted: bool = False,
) -> np.ndarray:
    """
    Compute optimal bin edges for 1D data using linear ckmeans algorithm.

    This implements the LINEAR_CKMEANS algorithm using Song & Zhong's
    in-place search space reduction for O(k*n) time complexity.

    Parameters
    ----------
    data : np.ndarray
        1D array of data to bin
    n_bins : int
        Number of bins to create

    Returns
    -------
    np.ndarray
        Bin edges (length n_bins + 1)
    """
    assert len(data.shape) == 1, "optimal_bins_1d requires a 1D array"
    assert not np.isnan(data).any(), (
        "please remove NaNs before calling optimal_bins_linear"
    )
    assert n_bins > 0, "n_bins must be positive"

    if len(data) == 0:
        return np.array([0.0, 1.0])

    if n_bins == 1:
        return np.array([np.min(data), np.max(data)])

    if not assume_sorted:
        data = np.sort(data)

    if (early_ret := check_if_n_bins_possible(data, n_bins)) is not None:
        n_unique = np.unique(data).shape[0]
        warnings.warn(
            f"Not possible to assign {n_bins} bins to this problem, "
            f"number of unique values is {n_unique}. "
            f"Creating {n_unique} bins instead, one for each unique value.",
            UserWarning,
        )
        return early_ret

    # Run the linear ckmeans algorithm
    return linear.ckmeans_1d_dp_smawk(data, n_bins)
