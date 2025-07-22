import warnings

import numpy as np
import pytest

import optimal_bins_1d
import optimal_bins_1d.linear
from optimal_bins_1d import (
    optimal_bins_linear,
    optimal_bins_loglinear,
    optimal_bins_quadratic,
)


# Data generation functions for test reuse
def generate_bimodal_low_middle_case(n: int = 1005, seed: int = 42) -> np.ndarray:
    """Generate bimodal distribution with very low density in middle"""
    # np.random.seed(seed)
    left_peak = np.random.normal(-5, 1, 500)
    right_peak = np.random.normal(5, 1, 500)
    middle_sparse = np.random.uniform(-1, 1, 5)  # Very sparse middle
    return np.concatenate([left_peak, middle_sparse, right_peak])


def generate_heavily_right_skewed_case(n: int = 1010, seed: int = 42) -> np.ndarray:
    """Generate heavily right skewed distribution with low density at end"""
    # np.random.seed(seed)
    main_data = np.random.uniform(-10, 10, 1000)
    tail_data = np.random.uniform(100, 200, 10)
    return np.concatenate([main_data, tail_data])


def generate_dense_right_outlier_case(n: int = 1000, seed: int = 42) -> np.ndarray:
    """Generate 95% from [-10,10] plus 5% extreme outliers at 9000"""
    # np.random.seed(seed)
    main_data = np.random.uniform(-10, 10, 950)
    outliers = np.random.normal(9000, 100, 50)
    return np.concatenate([main_data, outliers])


def generate_spike_distribution_case(n: int = 1000, seed: int = 42) -> np.ndarray:
    """Generate 99% exact zeros, 1% uniform noise up to 100"""
    # np.random.seed(seed)
    zeros = np.zeros(990)
    noise = np.random.uniform(0, 100, 10)
    return np.concatenate([zeros, noise])


def generate_random_cdf_case(
    case_id: int = 0, n: int = 200, seed: int = 42
) -> np.ndarray:
    """Generate randomly sampled distribution using random CDF"""
    # np.random.seed(seed + case_id)
    n_cdf_points = 20
    cdf_values = np.sort(np.random.uniform(0, 1, n_cdf_points))
    x_values = np.linspace(0, 10, n_cdf_points)
    uniform_samples = np.random.uniform(0, 1, n)
    return np.interp(uniform_samples, cdf_values, x_values)


def compute_total_within_ss(data: np.ndarray, edges: np.ndarray) -> float:
    """Compute total within-bin sum of squares for given data and edges"""
    n_bins = len(edges) - 1
    total_ss = 0.0

    for i in range(n_bins):
        left_edge = edges[i]
        right_edge = edges[i + 1]

        # Standard binning convention:
        # Bin 0: data <= right_edge
        # Bin i (i > 0): left_edge < data <= right_edge
        if i == 0:
            mask = data <= right_edge
        else:
            mask = (data > left_edge) & (data <= right_edge)

        if np.any(mask):
            bin_data = data[mask]
            bin_mean = np.mean(bin_data)
            total_ss += np.sum((bin_data - bin_mean) ** 2)

    return total_ss


def flip_one_bit_test(data: np.ndarray, edges: np.ndarray) -> bool:
    """
    Test optimality by flipping one bin edge and checking if it's worse or violates constraints.

    Returns True if the current solution passes the flip test (is locally optimal).
    """
    if len(edges) <= 2:  # Can't flip edges for single bin
        return True

    original_ss = compute_total_within_ss(data, edges)
    sorted_data = np.sort(data)

    # Test flipping each interior edge
    for i in range(1, len(edges) - 1):
        # Find next smaller and larger data values
        current_edge = edges[i]

        # Find data points around current edge
        smaller_values = sorted_data[sorted_data < current_edge]
        larger_values = sorted_data[sorted_data > current_edge]

        test_positions = []

        # add neighboring data values
        if len(smaller_values) > 0:
            test_positions.append(smaller_values[-1])
        if len(larger_values) > 0:
            test_positions.append(larger_values[0])

        # add extreme positions
        test_positions.extend([np.min(data) - 1e6, np.max(data) + 1e6])

        for new_edge in test_positions:
            new_edges = edges.copy()
            new_edges[i] = new_edge

            # ensure monotonicity of edges
            new_edges = np.sort(new_edges)

            # new solution should be worse
            new_ss = compute_total_within_ss(data, new_edges)
            if new_ss < original_ss:
                return False  # Found better solution - original was not optimal

    return True


# Test algorithms
algorithms = ["quadratic", "loglinear", "linear"]
algorithm_functions = {
    "quadratic": optimal_bins_quadratic,
    "loglinear": optimal_bins_loglinear,
    "linear": optimal_bins_linear,
}


class TestOptimalBinsAlgorithms:
    """Test all 4 algorithms on various data distributions"""

    @pytest.mark.parametrize("algorithm", algorithms)
    def test_basic_functionality(self, algorithm: str):
        """Test basic functionality across all algorithms"""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        func = algorithm_functions[algorithm]
        edges = func(x, n_bins=3)

        assert len(edges) == 4  # n_bins + 1 edges
        assert edges[0] < np.min(x)
        assert edges[-1] > np.max(x)
        assert np.all(edges[:-1] <= edges[1:])  # Monotonic

        # Test flip_one_bit optimality
        assert flip_one_bit_test(x, edges)

    def test_basic_functionality_cost_consistency(self):
        """Test that all algorithms produce the same cost for basic functionality"""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

        quadratic_edges = optimal_bins_quadratic(x, n_bins=3)
        loglinear_edges = optimal_bins_loglinear(x, n_bins=3)
        linear_edges = optimal_bins_linear(x, n_bins=3)

        quadratic_cost = compute_total_within_ss(x, quadratic_edges)
        loglinear_cost = compute_total_within_ss(x, loglinear_edges)
        linear_cost = compute_total_within_ss(x, linear_edges)

        tol = 1e-10
        assert abs(quadratic_cost - loglinear_cost) < tol
        assert abs(quadratic_cost - linear_cost) < tol

    @pytest.mark.parametrize("algorithm", algorithms)
    def test_bimodal_distribution(self, algorithm: str):
        """Test on bimodal distribution with low middle density"""
        x = generate_bimodal_low_middle_case()
        func = algorithm_functions[algorithm]
        edges = func(x, n_bins=3)

        assert len(edges) == 4
        assert edges[0] < np.min(x)
        assert edges[-1] > np.max(x)
        assert flip_one_bit_test(x, edges)

    def test_bimodal_distribution_cost_consistency(self):
        """Test that all algorithms produce the same cost for bimodal distribution"""
        x = generate_bimodal_low_middle_case()

        quadratic_edges = optimal_bins_quadratic(x, n_bins=3)
        loglinear_edges = optimal_bins_loglinear(x, n_bins=3)
        linear_edges = optimal_bins_linear(x, n_bins=3)

        quadratic_cost = compute_total_within_ss(x, quadratic_edges)
        loglinear_cost = compute_total_within_ss(x, loglinear_edges)
        linear_cost = compute_total_within_ss(x, linear_edges)

        tol = 1e-10
        assert abs(quadratic_cost - loglinear_cost) < tol
        assert abs(quadratic_cost - linear_cost) < tol

    @pytest.mark.parametrize("algorithm", algorithms)
    def test_right_skewed_distribution(self, algorithm: str):
        """Test on heavily right skewed distribution"""
        x = generate_heavily_right_skewed_case()
        func = algorithm_functions[algorithm]
        edges = func(x, n_bins=5)

        assert len(edges) == 6
        assert flip_one_bit_test(x, edges)

    def test_right_skewed_distribution_cost_consistency(self):
        """Test that all algorithms produce the same cost for right skewed distribution"""
        x = generate_heavily_right_skewed_case()

        quadratic_edges = optimal_bins_quadratic(x, n_bins=5)
        loglinear_edges = optimal_bins_loglinear(x, n_bins=5)
        linear_edges = optimal_bins_linear(x, n_bins=5)

        quadratic_cost = compute_total_within_ss(x, quadratic_edges)
        loglinear_cost = compute_total_within_ss(x, loglinear_edges)
        linear_cost = compute_total_within_ss(x, linear_edges)

        tol = 1e-10
        assert abs(quadratic_cost - loglinear_cost) < tol
        assert abs(quadratic_cost - linear_cost) < tol

    @pytest.mark.parametrize("algorithm", algorithms)
    def test_outlier_distribution(self, algorithm: str):
        """Test on distribution with dense right outliers"""
        x = generate_dense_right_outlier_case()
        func = algorithm_functions[algorithm]
        edges = func(x, n_bins=4)

        assert len(edges) == 5
        assert flip_one_bit_test(x, edges)

    def test_outlier_distribution_cost_consistency(self):
        """Test that all algorithms produce the same cost for outlier distribution"""
        x = generate_dense_right_outlier_case()

        quadratic_edges = optimal_bins_quadratic(x, n_bins=4)
        loglinear_edges = optimal_bins_loglinear(x, n_bins=4)
        linear_edges = optimal_bins_linear(x, n_bins=4)

        quadratic_cost = compute_total_within_ss(x, quadratic_edges)
        loglinear_cost = compute_total_within_ss(x, loglinear_edges)
        linear_cost = compute_total_within_ss(x, linear_edges)

        tol = 1e-10
        assert abs(quadratic_cost - loglinear_cost) < tol
        assert abs(quadratic_cost - linear_cost) < tol

    @pytest.mark.parametrize("algorithm", algorithms)
    def test_spike_distribution(self, algorithm: str):
        """Test on spike distribution"""
        x = generate_spike_distribution_case()
        func = algorithm_functions[algorithm]
        edges = func(x, n_bins=3)

        assert len(edges) == 4
        assert flip_one_bit_test(x, edges)

    def test_spike_distribution_cost_consistency(self):
        """Test that all algorithms produce the same cost for spike distribution"""
        x = generate_spike_distribution_case()

        quadratic_edges = optimal_bins_quadratic(x, n_bins=3)
        loglinear_edges = optimal_bins_loglinear(x, n_bins=3)
        linear_edges = optimal_bins_linear(x, n_bins=3)

        quadratic_cost = compute_total_within_ss(x, quadratic_edges)
        loglinear_cost = compute_total_within_ss(x, loglinear_edges)
        linear_cost = compute_total_within_ss(x, linear_edges)

        tol = 1e-10
        assert abs(quadratic_cost - loglinear_cost) < tol
        assert abs(quadratic_cost - linear_cost) < tol

    @pytest.mark.parametrize("algorithm", algorithms)
    def test_random_distributions(self, algorithm: str):
        """Test on randomly generated distributions"""
        func = algorithm_functions[algorithm]

        for i in range(5):
            x = generate_random_cdf_case(case_id=i)
            edges = func(x, n_bins=4)

            assert len(edges) == 5
            assert flip_one_bit_test(x, edges)

    def test_random_distributions_cost_consistency(self):
        """Test that all algorithms produce the same cost for random distributions"""
        for i in range(5):
            x = generate_random_cdf_case(case_id=i)

            quadratic_edges = optimal_bins_quadratic(x, n_bins=4)
            loglinear_edges = optimal_bins_loglinear(x, n_bins=4)
            linear_edges = optimal_bins_linear(x, n_bins=4)

            quadratic_cost = compute_total_within_ss(x, quadratic_edges)
            loglinear_cost = compute_total_within_ss(x, loglinear_edges)
            linear_cost = compute_total_within_ss(x, linear_edges)

            tol = 1e-10
            assert abs(quadratic_cost - loglinear_cost) < tol, (
                f"Case {i}: quadratic vs loglinear"
            )
            assert abs(quadratic_cost - linear_cost) < tol, (
                f"Case {i}: quadratic vs linear"
            )


class TestEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.parametrize("algorithm", algorithms)
    def test_empty_data(self, algorithm: str):
        """Test behavior with empty data"""
        func = algorithm_functions[algorithm]
        edges = func(np.array([]), n_bins=1)
        assert len(edges) == 2

    @pytest.mark.parametrize("algorithm", algorithms)
    def test_single_point(self, algorithm: str):
        """Test with single data point"""
        func = algorithm_functions[algorithm]
        x = np.array([5.0])
        edges = func(x, n_bins=1)
        assert len(edges) == 2

    @pytest.mark.parametrize("algorithm", algorithms)
    def test_identical_points(self, algorithm: str):
        """Test with all identical points"""
        func = algorithm_functions[algorithm]
        x = np.array([5.0, 5.0, 5.0, 5.0])

        # With identical points, should create 1 bin (number of unique values)
        # regardless of requested n_bins
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress expected warning
            edges = func(x, n_bins=2)

        assert len(edges) == 2  # 1 bin = 2 edges
        # Should create artificial edges around the value
        assert edges[0] < 5.0 < edges[1]

    @pytest.mark.filterwarnings("ignore:.*:UserWarning:optimal_bins_1d.utils")
    @pytest.mark.parametrize("algorithm", algorithms)
    def test_n_bins_larger_than_data(self, algorithm: str):
        """Test when n_bins >= len(data)"""
        func = algorithm_functions[algorithm]
        x = np.array([1, 3, 5], dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress expected warning
            edges = func(x, n_bins=5)

        # Should handle gracefully
        assert len(edges) >= 2
        assert edges[0] < np.min(x)
        assert edges[-1] > np.max(x)


class TestDPMatrixCorrectness:
    """Test that DP matrices computed by each algorithm are correct"""

    def test_dp_matrix_comparison_vs_brute_force(self):
        """Compare DP matrices from all algorithms against brute force ground truth"""

        dtype = np.float64
        test_cases = [
            {"data": np.array([1, 2, 3, 10, 11, 12], dtype=dtype), "k": 2},
            {"data": np.array([1, 2, 8, 9], dtype=dtype), "k": 2},
            {"data": np.array([1, 5, 6, 10], dtype=dtype), "k": 2},
            {"data": np.array([1, 2, 3, 4, 5], dtype=dtype), "k": 2},
            {"data": np.array([1, 1, 1, 5, 5, 5], dtype=dtype), "k": 2},
            {"data": np.array([1, 3, 5, 7, 9], dtype=dtype), "k": 3},
            {"data": np.arange(12, 25).astype(dtype), "k": 4},
        ]

        for case in test_cases:
            data = case["data"]
            k = case["k"]

            # Get DP matrices from all algorithms
            breaks_quadratic = optimal_bins_1d.quadratic.ckmeans_1d_dp_quadratic(
                data, k
            )
            breaks_loglinear = optimal_bins_1d.loglinear.ckmeans_1d_dp_loglinear(
                data, k
            )
            breaks_linear = optimal_bins_1d.linear.ckmeans_1d_dp_smawk(data, k)

            # compare final costs (S[k-1, n-1])
            cost_quadratic = compute_total_within_ss(data, breaks_quadratic)
            cost_loglinear = compute_total_within_ss(data, breaks_loglinear)
            cost_linear = compute_total_within_ss(data, breaks_linear)

            # Check correctness (tolerance for floating point)
            tol = 1e-10
            assert abs(cost_loglinear - cost_quadratic) < tol
            assert abs(cost_linear - cost_quadratic) < tol
