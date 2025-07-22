import numba as nb
import numpy as np
from numba.typed import List

from optimal_bins_1d.utils import (
    breaks_if_k_is_greater_than_n,
    calculate_cumulative_ssq,
    ssq,
    wiggle_left,
    wiggle_right,
)


@nb.njit(cache=True, inline="always")
def A(m: int, j: int, S_prev: np.ndarray, c_x: np.ndarray, c_x_sq: np.ndarray) -> float:
    """
    This is implicitly the call to C_q[m, j] := the cost of clustering the first m elements using q
    clusters where the last cluster starts at j.
    """
    if j == 0 or j > m:
        return np.inf
    return S_prev[j - 1] + ssq(j, m + 1, c_x, c_x_sq)


@nb.njit(cache=True)
def reduce_columns(
    rows: np.ndarray,
    cols: np.ndarray,
    S_prev: np.ndarray,
    c_x: np.ndarray,
    c_x_sq: np.ndarray,
) -> np.ndarray:
    """
    REDUCE phase.
    Keeps a stack of 'alive' columns. For each new column c:
      - While stack nonempty and A(test_row, c) < A(test_row, top),
        pop top (it is strictly dominated from its feasible interval onward).
      - If stack size < number of rows, push c.
    Using *strict* '<' ensures leftmost tie-breaking downstream.
    """
    alive = List.empty_list(nb.int32)
    for c in cols:
        while alive:
            # test row
            tr = rows[len(alive) - 1]
            prev = alive[-1]
            # replace this with ssq calls
            if A(tr, c, S_prev, c_x, c_x_sq) < A(tr, prev, S_prev, c_x, c_x_sq):
                alive.pop()
            else:
                break
        if len(alive) < len(rows):
            alive.append(c)
        # else discard c (too many columns; c cannot add new feasible interval)

    n_alive = len(alive)
    alive_array = np.empty(n_alive, dtype=np.int32)
    for i in range(n_alive):
        alive_array[i] = alive[i]
    return alive_array


@nb.njit(cache=True)
def interpolate(
    min_col: np.ndarray,
    min_val: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    C_red: np.ndarray,
    S_prev: np.ndarray,
    c_x: np.ndarray,
    c_x_sq: np.ndarray,
) -> None:
    # 3. INTERPOLATE even-position rows
    # we will scan disjoint intervals of C exactly once.
    # precompute a mapping from column value to its position for fast advancement
    # (not strictly necessary if columns are dense integers).
    # instead, we just maintain a forward pointer because intervals are disjoint.
    m = len(C_red)

    # for convenience, keep C as array; pointer advancement only forward.
    # iterate even rows in increasing order / intervals are disjoint.
    prev_idx = 0
    rlen = len(R)
    for even_pos in range(0, rlen, 2):
        r = R[even_pos]

        # determine [L, Rb] bounds for this even row's argmin
        if even_pos == 0:
            Lcol = C[0]
        else:
            Lcol = min_col[R[even_pos - 1]]
        if even_pos + 1 < rlen:
            Rcol = min_col[R[even_pos + 1]]
        else:
            Rcol = C[-1]
        # defensive (should not happen with leftmost ties)
        if Lcol > Rcol:
            Lcol, Rcol = Rcol, Lcol

        # advance prev_idx until C[prev_idx] >= Lcol
        while prev_idx < m and C_red[prev_idx] < Lcol:
            prev_idx += 1

        scan_idx = prev_idx
        best_c = -1
        best_v = np.inf

        # scan forward while within right boundary
        while scan_idx < m and C_red[scan_idx] <= Rcol:
            c_val = C_red[scan_idx]
            v = A(r, c_val, S_prev, c_x, c_x_sq)
            if v < best_v:
                best_v = v
                best_c = c_val
            scan_idx += 1

        # Record
        min_col[r] = best_c
        min_val[r] = best_v


@nb.njit(cache=True)
def base_case(
    min_col: np.ndarray,
    min_val: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    S_prev: np.ndarray,
    c_x: np.ndarray,
    c_x_sq: np.ndarray,
) -> None:
    r = R[0]
    best_c = C[0]
    best_v = A(r, best_c, S_prev, c_x, c_x_sq)
    for c in C[1:]:
        v = A(r, c, S_prev, c_x, c_x_sq)
        # strict => leftmost tie rule
        if v < best_v:
            best_v = v
            best_c = c
    min_col[r] = best_c
    min_val[r] = best_v


def recurse(
    min_col: np.ndarray,
    min_val: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    S_prev: np.ndarray,
    c_x: np.ndarray,
    c_x_sq: np.ndarray,
) -> None:
    rlen = len(R)
    if rlen == 0:
        return
    if rlen == 1:
        base_case(min_col, min_val, R, C, S_prev, c_x, c_x_sq)
        return

    # 1. REDUCE
    C_red = reduce_columns(R, C, S_prev, c_x, c_x_sq)

    # 2. RECURSE on odd-position rows (indices 1,3,5,...)
    R_odd = R[1::2]
    recurse(min_col, min_val, R_odd, C_red, S_prev, c_x, c_x_sq)

    # 3. INTERPOLATE even-position rows (indices 0,2,4,...)
    interpolate(min_col, min_val, R, C, C_red, S_prev, c_x, c_x_sq)


def smawk(
    q: int,
    imax: int,
    S_prev: np.ndarray,
    c_x: np.ndarray,
    c_x_sq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear-time SMAWK for a totally monotone matrix with row index list `rows`
    and column index list `cols`. Returns:
        min_col[row] = leftmost minimizing column
        min_val[row] = minimal value
    assume:
      - rows, cols strictly increasing
      - leftmost minima desired (handled via strict comparisons)
    """
    # set up possible row indices: q, q+1, ..., imax
    rows = np.arange(q, imax + 1, dtype=np.int32)
    # possible column indices are also the same
    cols = np.arange(q, imax + 1, dtype=np.int32)
    min_col = np.zeros(np.max(rows) + 1)
    min_val = np.zeros(np.max(rows) + 1)
    recurse(min_col, min_val, rows, cols, S_prev, c_x, c_x_sq)
    return min_col[q:], min_val[q:]


def ckmeans_1d_dp_smawk(x_sorted: np.ndarray, K: int) -> np.ndarray:
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

    c_x = np.empty(N + 1, dtype=dtype)
    c_x_sq = np.empty(N + 1, dtype=dtype)
    calculate_cumulative_ssq(x_sorted, c_x, c_x_sq)

    # as above:
    # S[c, i] := optimal cost of partitioning x[0..i] into (c+1) clusters.
    # J[c, i] := start index (p) of the last cluster (p in [0..i]).
    # S = np.full((K, N), np.inf)
    S_prev = np.empty(N, dtype=dtype)
    J = np.zeros((K, N), dtype=np.int32)

    # base case: first cluster (q=0, but use 0-based j indexing)
    for i in range(N):
        S_prev[i] = ssq(0, i + 1, c_x, c_x_sq)
        J[0, i] = 0

    # fill subsequent rows using SMAWK (q=1,2,...,k-1)
    for q in range(1, K):
        S_curr = np.empty(N, dtype=dtype)
        J_row, S_row = smawk(q, N - 1, S_prev, c_x, c_x_sq)
        S_curr[q:N] = S_row
        S_prev = S_curr
        J[q, q:N] = J_row

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
