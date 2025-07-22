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
def _fill_row_quadratic(
    c: int,
    S: np.ndarray,
    J: np.ndarray,
    c_x: np.ndarray,
    c_x_sq: np.ndarray,
) -> None:
    """
    Fill DP row c for i in [left, right) (half-open),

    S[c, i] := optimal cost of partitioning x[0..i] into (c+1) clusters.
    J[c, i] := start index (p) of the last cluster (p in [0..i]).

    Recurrence for c > 0:
        S[c, i] = min_{p in [c .. i]} S[c-1, p-1] + ssq(p, i)
    """
    _, N = S.shape
    for i in range(c, N):
        best_cost = np.inf
        best_p = -1
        # p = start index of last cluster
        for p in range(c, i + 1):
            # cost up to element p-1 plus cost of cluster [p, i]
            prev = 0.0 if p == 0 else S[c - 1, p - 1]
            cost = prev + ssq(p, i + 1, c_x, c_x_sq)  # end-exclusive
            if cost < best_cost:
                best_cost = cost
                best_p = p
        S[c, i] = best_cost
        J[c, i] = best_p


@nb.njit(cache=True)
def ckmeans_1d_dp_quadratic(x_sorted: np.ndarray, K: int) -> np.ndarray:
    """
    Returns breakpoints (length K+1): [min, ..., max]; interior = midpoints.
    Uses 0-based DP tables S,J of shape (K, N).
    """
    N = x_sorted.size
    dtype = x_sorted.dtype
    if N == 0:
        return np.empty(0, dtype=dtype)
    if K >= N:
        return breaks_if_k_is_greater_than_n(x_sorted)

    # prefix sums length N+1
    c_x = np.empty(N + 1, dtype=dtype)
    c_x_sq = np.empty(N + 1, dtype=dtype)
    calculate_cumulative_ssq(x_sorted, c_x, c_x_sq)

    # as above:
    # S[c, i] := optimal cost of partitioning x[0..i] into (c+1) clusters.
    # J[c, i] := start index (p) of the last cluster (p in [0..i]).
    S = np.full((K, N), np.inf, dtype=dtype)
    J = np.full((K, N), -1, dtype=np.int32)

    # base row c = 0 (one cluster)
    for i in range(N):
        S[0, i] = ssq(0, i + 1, c_x, c_x_sq)
        J[0, i] = 0

    # remaining rows
    for c in range(1, K):
        _fill_row_quadratic(c, S, J, c_x, c_x_sq)

    # backtrack to get cluster start indices
    bounds = np.empty(K + 1, dtype=np.int32)
    bounds[K] = N
    end = N  # end is element count
    for c in range(K - 1, -1, -1):
        start = J[c, end - 1]
        bounds[c] = start
        end = start

    # build breakpoints
    breaks = np.empty(K + 1, dtype=dtype)
    breaks[0] = wiggle_left(x_sorted[0])
    for c in range(1, K):
        end = bounds[c]
        breaks[c] = 0.5 * (x_sorted[end - 1] + x_sorted[end])
    breaks[K] = wiggle_right(x_sorted[-1])
    return breaks
