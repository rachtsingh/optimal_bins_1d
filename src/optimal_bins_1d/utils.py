import numba as nb
import numpy as np


@nb.njit(cache=True, inline="always")
def wiggle_left(x: float) -> float:
    return -0.1 if x == 0.0 else x - abs(x / 10.0)


@nb.njit(cache=True, inline="always")
def wiggle_right(x: float) -> float:
    return 0.1 if x == 0.0 else x + abs(x / 10.0)


@nb.njit(cache=True)
def check_if_n_bins_possible(data: np.ndarray, n_bins: int) -> np.ndarray | None:
    """Returns artificial edges if insufficient unique values, None otherwise."""
    unique_vals = np.unique(data)
    n_unique = len(unique_vals)

    if n_unique >= n_bins:
        return None

    if n_unique == 1:
        val = unique_vals[0]
        return np.array([wiggle_left(val), wiggle_right(val)])

    edges = np.zeros(n_unique + 1)
    edges[0] = wiggle_left(unique_vals[0])
    edges[1:-1] = (unique_vals[1:] + unique_vals[:-1]) / 2.0
    edges[-1] = wiggle_right(unique_vals[-1])

    return edges


@nb.njit(cache=True)
def breaks_if_k_is_greater_than_n(x_sorted: np.ndarray) -> np.ndarray:
    N = x_sorted.shape[0]
    breaks = np.empty(N + 1, dtype=x_sorted.dtype)
    breaks[0] = x_sorted[0]
    for i in range(1, N):
        breaks[i] = 0.5 * (x_sorted[i - 1] + x_sorted[i])
    breaks[N] = x_sorted[-1]
    return breaks


@nb.njit(cache=True, inline="always")
def ssq(i: int, j: int, c_x: np.ndarray, c_x_sq: np.ndarray) -> float:
    """
    Within-cluster SSE for half-open slice [i, j).
    prefix_* arrays have length n+1 with prefix_*[0] = 0.
    """
    if j <= i:
        return 0.0
    w = j - i
    s = c_x[j] - c_x[i]
    sq = c_x_sq[j] - c_x_sq[i]
    ssq = sq - (s * s) / w
    if ssq < 0.0:
        ssq = 0.0
    return ssq


@nb.njit(cache=True)
def calculate_cumulative_ssq(
    x: np.ndarray, c_x: np.ndarray, c_x_sq: np.ndarray
) -> None:
    """
    Compute cumulative sums and squares for the sorted array x
    c_x and c_x_sq must be preallocated with length N + 1.
    """
    N = x.shape[0]
    c_x[0] = 0.0
    c_x_sq[0] = 0.0
    for i in range(N):
        v = x[i]
        c_x[i + 1] = c_x[i] + v
        c_x_sq[i + 1] = c_x_sq[i] + v * v
