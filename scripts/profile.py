#!/usr/bin/env python3
"""
Performance profiling script for 1D k-means algorithms.

Usage: uv run scripts/profile.py A B S T K

Where:
- A: Minimum dataset size
- B: Maximum dataset size
- S: Number of size points to test
- T: Number of trials per size
- K: Number of clusters/bins

Outputs CSV format: N,quadratic_mean,loglinear_mean,linear_mean,quadratic_std,loglinear_std,linear_std
"""

import sys
import numpy as np
import time
import optimal_bins_1d


def generate_test_data(n: int, seed: int | None = None) -> np.ndarray:
    """Generate test data for profiling."""
    if seed is not None:
        np.random.seed(seed)

    # Generate mixed distribution similar to real-world data
    data = np.concatenate(
        [
            np.random.normal(0, 1, n // 3),
            np.random.normal(10, 2, n // 3),
            np.random.normal(20, 1, n - 2 * (n // 3)),  # Handle remainder
        ]
    )

    return data


def time_algorithm(algorithm_name: str, data: np.ndarray, k: int) -> float:
    """Time a single algorithm run."""
    if algorithm_name == "quadratic":
        start = time.time()
        optimal_bins_1d.optimal_bins_quadratic(data, k)
        return time.time() - start
    elif algorithm_name == "loglinear":
        start = time.time()
        optimal_bins_1d.optimal_bins_loglinear(data, k)
        return time.time() - start
    elif algorithm_name == "linear":
        start = time.time()
        optimal_bins_1d.optimal_bins_linear(data, k)
        return time.time() - start
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def profile_size(n: int, k: int, trials: int) -> dict:
    """Profile all algorithms for a given dataset size."""
    algorithms = ["quadratic", "loglinear", "linear"]
    results = {alg: [] for alg in algorithms}

    for trial in range(trials):
        # Generate fresh data for each trial
        data = generate_test_data(n, seed=42 + trial)

        for alg in algorithms:
            try:
                elapsed = time_algorithm(alg, data, k)
                results[alg].append(elapsed)
            except Exception:
                # If algorithm fails, record NaN
                results[alg].append(float("nan"))

    # Compute statistics
    stats = {}
    for alg in algorithms:
        times = np.array(results[alg])
        # Filter out NaN values for statistics
        valid_times = times[~np.isnan(times)]
        if len(valid_times) > 0:
            stats[f"{alg}_mean"] = np.mean(valid_times)
            stats[f"{alg}_std"] = np.std(valid_times)
        else:
            stats[f"{alg}_mean"] = float("nan")
            stats[f"{alg}_std"] = float("nan")

    return stats


def main():
    if len(sys.argv) != 6:
        print("Usage: profile.py A B S T K", file=sys.stderr)
        print("  A: Minimum dataset size", file=sys.stderr)
        print("  B: Maximum dataset size", file=sys.stderr)
        print("  S: Number of size points to test", file=sys.stderr)
        print("  T: Number of trials per size", file=sys.stderr)
        print("  K: Number of clusters/bins", file=sys.stderr)
        sys.exit(1)

    try:
        A = int(sys.argv[1])
        B = int(sys.argv[2])
        S = int(sys.argv[3])
        T = int(sys.argv[4])
        K = int(sys.argv[5])
    except ValueError:
        print("Error: All arguments must be integers", file=sys.stderr)
        sys.exit(1)

    if A <= 0 or B <= A or S <= 0 or T <= 0 or K <= 0:
        print("Error: Invalid argument values", file=sys.stderr)
        sys.exit(1)

    # Generate geometric progression of sizes
    log_sizes = np.linspace(np.log(A), np.log(B), S)
    sizes = np.round(np.exp(log_sizes)).astype(int)
    # Remove duplicates and sort
    sizes = np.unique(sizes)

    # Print CSV header
    print(
        "N,quadratic_mean,loglinear_mean,linear_mean,quadratic_std,loglinear_std,linear_std"
    )

    # Profile each size
    for n in sizes:
        stats = profile_size(n, K, T)

        # Format output as CSV
        print(
            f"{n},{stats['quadratic_mean']:.6f},{stats['loglinear_mean']:.6f},{stats['linear_mean']:.6f},"
            f"{stats['quadratic_std']:.6f},{stats['loglinear_std']:.6f},{stats['linear_std']:.6f}"
        )


if __name__ == "__main__":
    main()
