import numba as nb
import numpy as np

from optimal_bins_1d.utils import (
    breaks_if_k_is_greater_than_n,
    calculate_cumulative_ssq,
    ssq,
    wiggle_left,
    wiggle_right,
)


@nb.njit(cache=True)
def _fill_row_loglinear(
    j: int,
    left: int,
    right: int,
    opt_left: int,
    opt_right: int,
    cost_prev: np.ndarray,
    arg_row: np.ndarray,
    c_x: np.ndarray,
    c_xsq: np.ndarray,
) -> None:
    """
    Divide-and-conquer optimization for DP row j (number of clusters = j).
    Fills arg_row[i] (optimal split index p) for i in [left, right] (inclusive),
    where valid i satisfy i >= j and i <= m, with recurrence:
        cost_curr[i] = min_{p in [j-1, i-1]} cost_prev[p] + SSQ(p, i)
    Monotone (Knuth / quadrangle inequality / SMAWK-style) property assumed.
    """
    if left > right:
        return

    mid = (left + right) // 2

    best_p = -1
    best_cost = np.inf

    p_start = opt_left
    if p_start < j - 1:
        p_start = j - 1
    p_end = opt_right
    if p_end > mid - 1:
        p_end = mid - 1

    for p in range(p_start, p_end + 1):
        c = cost_prev[p] + ssq(p, mid, c_x, c_xsq)
        if c < best_cost:
            best_cost = c
            best_p = p

    arg_row[mid] = best_p

    _fill_row_loglinear(
        j, left, mid - 1, opt_left, best_p, cost_prev, arg_row, c_x, c_xsq
    )
    _fill_row_loglinear(
        j, mid + 1, right, best_p, opt_right, cost_prev, arg_row, c_x, c_xsq
    )


@nb.njit(cache=True)
def ckmeans_1d_dp_loglinear(x_sorted: np.ndarray, K: int) -> np.ndarray:
    """
    Log-linear (O(k * n log n)) 1D contiguous clustering on pre-sorted data.
    Returns breakpoints (length = k_eff + 1). Interior breakpoints are midpoints
    between adjacent unique values. k_eff = min(k, number of unique values).
    """
    N = x_sorted.size
    dtype = x_sorted.dtype
    if N == 0:
        return np.empty(0, dtype=dtype)
    if K >= N:
        return breaks_if_k_is_greater_than_n(x_sorted)

    # cumulative sums (size num_unique + 1), 0-based half-open.
    c_x = np.empty(N + 1, dtype=dtype)
    c_x_sq = np.empty(N + 1, dtype=dtype)
    calculate_cumulative_ssq(x_sorted, c_x, c_x_sq)

    # DP arrays:
    # S_prev[i, j] := optimal cost for partitioning [0, i) into j clusters.
    S_prev = np.empty(N + 1, dtype=dtype)
    S_prev[0] = 0.0
    for i in range(1, N + 1):
        S_prev[i] = ssq(0, i, c_x, c_x_sq)

    # J[j, i] = argmin p giving optimal split for j clusters covering [0, i).
    J = np.full((K + 1, N + 1), -1, dtype=np.int32)
    J[1, :] = 0  # first cluster always starts at 0

    for k in range(2, K + 1):
        S_curr = np.empty(N + 1, dtype=dtype)
        # valid i: at least current_k elements ⇒ i >= current_k
        _fill_row_loglinear(
            k,
            k,
            N,
            k - 1,
            N - 1,
            S_prev,
            J[k],
            c_x,
            c_x_sq,
        )
        # compute costs using chosen splits.
        for i in range(0, k):
            S_curr[i] = np.nan  # invalid
        for i in range(k, N + 1):
            p = J[k, i]
            S_curr[i] = S_prev[p] + ssq(p, i, c_x, c_x_sq)
        S_prev = S_curr

    # backtrack: cluster boundaries (exclusive ends). boundaries[j] = end index of cluster j (0-based half-open).
    bounds = np.empty(K + 1, dtype=np.int32)
    bounds[K] = N
    end = N
    for k in range(K, 0, -1):
        p = J[k, end]
        bounds[k - 1] = p
        end = p

    # construct numeric breakpoints: midpoint between last of cluster j and first of cluster j+1.
    breaks = np.empty(K + 1, dtype=dtype)
    breaks[0] = wiggle_left(x_sorted[0])
    for j in range(1, K):
        end = bounds[j]
        breaks[j] = (x_sorted[end - 1] + x_sorted[end]) * 0.5
    breaks[K] = wiggle_right(x_sorted[-1])
    return breaks
