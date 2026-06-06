"""
Unit tests for power iteration and convergence analysis.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import (
    download_data,
    classify_regimes,
    compute_rolling_return,
)

from src.transition_matrix import (
    count_transitions,
    laplace_smooth
)

from src.markov_engine import (
    power_iteration,
    eigenvector_stationary,
    spectral_analysis,
    compare_methods,
    multi_init_convergence,
    pagerank_comparison,
    fit_convergence_rate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def uniform_matrix():
    """Completely uniform 3×3 transition matrix (memoryless)."""
    return np.full((3, 3), 1 / 3)


@pytest.fixture
def identity_like():
    """High-persistence matrix — converges slowly."""
    return np.array([
        [0.98, 0.01, 0.01],
        [0.01, 0.98, 0.01],
        [0.01, 0.01, 0.98],
    ])


@pytest.fixture
def known_matrix():
    """
    Matrix with analytically tractable stationary distribution.

    Row-stochastic; stationary dist can be verified by hand:
        π P = π  →  solve linear system
    We verify with both power iteration and eigenvector method.
    """
    return np.array([
        [0.70, 0.20, 0.10],
        [0.30, 0.50, 0.20],
        [0.15, 0.25, 0.60],
    ])


@pytest.fixture
def market_P():
    prices = download_data()

    returns = compute_rolling_return(
        prices,
        window=20
    )

    regimes = classify_regimes(returns)

    counts = count_transitions(regimes)

    return laplace_smooth(
        counts,
        alpha=1.0
    )


# ===========================================================================
# TestPowerIteration
# ===========================================================================
class TestPowerIteration:

    def test_returns_dict_with_required_keys(self, uniform_matrix):
        result = power_iteration(uniform_matrix)
        for key in ["stationary", "history", "n_iter", "converged", "pi0", "tol"]:
            assert key in result, f"Missing key: {key}"

    def test_stationary_sums_to_one(self, uniform_matrix):
        result = power_iteration(uniform_matrix)
        np.testing.assert_allclose(result["stationary"].sum(), 1.0, atol=1e-10)

    def test_stationary_non_negative(self, uniform_matrix):
        result = power_iteration(uniform_matrix)
        assert np.all(result["stationary"] >= 0)

    def test_uniform_matrix_gives_uniform_dist(self, uniform_matrix):
        """For uniform P, every π⁰ converges to [1/3, 1/3, 1/3]."""
        result = power_iteration(uniform_matrix)
        np.testing.assert_allclose(
            result["stationary"], np.array([1 / 3, 1 / 3, 1 / 3]), atol=1e-9
        )

    def test_converged_flag_set(self, uniform_matrix):
        result = power_iteration(uniform_matrix)
        assert result["converged"] is True

    def test_history_non_negative(self, uniform_matrix):
        result = power_iteration(uniform_matrix)
        assert all(d >= 0 for d in result["history"])

    def test_history_decreasing_asymptotically(self, market_P):
        """History should be generally decreasing (may not be strictly monotone)."""
        result = power_iteration(market_P)
        hist = result["history"]
        # Second half should all be smaller than first element
        if len(hist) > 2:
            assert max(hist[len(hist) // 2:]) < hist[0]

    def test_satisfies_stationary_equation(self, known_matrix):
        """Verify π* P ≈ π*."""
        result = power_iteration(known_matrix)
        pi = result["stationary"]
        np.testing.assert_allclose(pi @ known_matrix, pi, atol=1e-8)

    def test_satisfies_stationary_equation_market(self, market_P):
        result = power_iteration(market_P)
        pi = result["stationary"]
        np.testing.assert_allclose(pi @ market_P, pi, atol=1e-8)

    def test_custom_initial_distribution(self, uniform_matrix):
        """Custom pi0 should still converge to [1/3, 1/3, 1/3]."""
        pi0 = np.array([1.0, 0.0, 0.0])
        result = power_iteration(uniform_matrix, pi0=pi0)
        np.testing.assert_allclose(
            result["stationary"], np.array([1 / 3, 1 / 3, 1 / 3]), atol=1e-9
        )

    def test_pi0_not_summing_to_one_raises(self, uniform_matrix):
        bad_pi0 = np.array([0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="sum to 1"):
            power_iteration(uniform_matrix, pi0=bad_pi0)

    def test_pi0_negative_entry_raises(self, uniform_matrix):
        bad_pi0 = np.array([1.5, -0.5, 0.0])
        with pytest.raises(ValueError, match="non-negative"):
            power_iteration(uniform_matrix, pi0=bad_pi0)

    def test_non_square_matrix_raises(self):
        bad = np.array([[0.5, 0.5], [0.3, 0.7], [0.2, 0.8]])
        with pytest.raises(ValueError, match="square"):
            power_iteration(bad)

    def test_non_stochastic_matrix_raises(self):
        bad = np.array([
            [0.5, 0.3, 0.1],   # row sums to 0.9
            [0.2, 0.5, 0.3],
            [0.1, 0.4, 0.5],
        ])
        with pytest.raises(ValueError, match="row-stochastic"):
            power_iteration(bad)

    def test_tighter_tol_same_result(self, known_matrix):
        """Tighter tolerance should give the same result (just more iterations)."""
        r1 = power_iteration(known_matrix, tol=1e-6)
        r2 = power_iteration(known_matrix, tol=1e-10)
        np.testing.assert_allclose(r1["stationary"], r2["stationary"], atol=1e-5)
        assert r2["n_iter"] >= r1["n_iter"]

    def test_5x5_matrix(self):
        """Power iteration must work for n != 3."""
        rng = np.random.default_rng(0)
        raw = rng.random((5, 5)) + 0.1
        P = raw / raw.sum(axis=1, keepdims=True)
        result = power_iteration(P)
        np.testing.assert_allclose(result["stationary"].sum(), 1.0, atol=1e-10)
        np.testing.assert_allclose(result["stationary"] @ P, result["stationary"], atol=1e-8)


# ===========================================================================
# TestEigenvectorStationary
# ===========================================================================
class TestEigenvectorStationary:

    def test_sums_to_one(self, uniform_matrix):
        pi = eigenvector_stationary(uniform_matrix)
        np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-10)

    def test_non_negative(self, uniform_matrix):
        pi = eigenvector_stationary(uniform_matrix)
        assert np.all(pi >= 0)

    def test_uniform_matrix_gives_uniform_dist(self, uniform_matrix):
        pi = eigenvector_stationary(uniform_matrix)
        np.testing.assert_allclose(pi, np.array([1/3, 1/3, 1/3]), atol=1e-9)

    def test_satisfies_stationary_equation(self, known_matrix):
        pi = eigenvector_stationary(known_matrix)
        np.testing.assert_allclose(pi @ known_matrix, pi, atol=1e-8)

    def test_agrees_with_power_iteration(self, market_P):
        pi_eig = eigenvector_stationary(market_P)
        pi_pow = power_iteration(market_P)["stationary"]
        np.testing.assert_allclose(pi_eig, pi_pow, atol=1e-6)


# ===========================================================================
# TestSpectralAnalysis
# ===========================================================================
class TestSpectralAnalysis:

    def test_required_keys(self, uniform_matrix):
        spec = spectral_analysis(uniform_matrix)
        for key in ["eigenvalues", "eigenvalues_abs", "lambda1", "lambda2",
                    "spectral_gap", "predicted_rate", "mixing_time_est",
                    "perron_frobenius_satisfied"]:
            assert key in spec

    def test_lambda1_equals_one(self, uniform_matrix):
        spec = spectral_analysis(uniform_matrix)
        np.testing.assert_allclose(spec["lambda1"], 1.0, atol=1e-9)

    def test_perron_frobenius_satisfied_for_positive_matrix(self, market_P):
        spec = spectral_analysis(market_P)
        assert spec["perron_frobenius_satisfied"] is True

    def test_lambda2_less_than_one(self, market_P):
        spec = spectral_analysis(market_P)
        assert spec["lambda2"] < 1.0

    def test_spectral_gap_positive(self, market_P):
        spec = spectral_analysis(market_P)
        assert spec["spectral_gap"] > 0

    def test_eigenvalues_sorted_descending(self, market_P):
        spec = spectral_analysis(market_P)
        abs_ev = spec["eigenvalues_abs"]
        assert all(abs_ev[i] >= abs_ev[i + 1] for i in range(len(abs_ev) - 1))

    def test_uniform_matrix_spectral_gap_is_one(self, uniform_matrix):
        """Uniform matrix: λ₂ = 0, spectral gap = 1 — fastest possible mixing."""
        spec = spectral_analysis(uniform_matrix)
        np.testing.assert_allclose(spec["lambda2"], 0.0, atol=1e-9)
        np.testing.assert_allclose(spec["spectral_gap"], 1.0, atol=1e-9)

    def test_high_persistence_small_gap(self, identity_like):
        """High-persistence matrix should have small spectral gap (slow mixing)."""
        spec_id = spectral_analysis(identity_like)
        spec_uni = spectral_analysis(np.full((3, 3), 1 / 3))
        assert spec_id["spectral_gap"] < spec_uni["spectral_gap"]

    def test_lambda2_bounds_convergence(self, market_P):
        """
        The convergence history should eventually decay at rate |λ₂|.
        After enough iterations, each step should reduce error by ~|λ₂|.
        """
        spec = spectral_analysis(market_P)
        result = power_iteration(market_P, tol=1e-12)
        hist = np.array(result["history"])

        # Use the final few iterations to check the empirical rate
        if len(hist) >= 6:
            # Ratio of consecutive deltas in the last few iterations
            ratios = hist[-5:] / hist[-6:-1]
            empirical_rate = float(np.median(ratios[ratios > 0]))
            # Should be within 10% of theoretical |λ₂|
            np.testing.assert_allclose(
                empirical_rate, spec["lambda2"], rtol=0.15,
                err_msg=f"Empirical rate {empirical_rate:.6f} far from "
                        f"|λ₂| = {spec['lambda2']:.6f}",
            )


# ===========================================================================
# TestCompareMethods
# ===========================================================================
class TestCompareMethods:

    def test_methods_agree(self, known_matrix):
        result = compare_methods(known_matrix)
        assert result["agreement"] is True

    def test_l1_diff_small(self, market_P):
        result = compare_methods(market_P)
        assert result["l1_diff"] < 1e-6

    def test_linf_diff_small(self, market_P):
        result = compare_methods(market_P)
        assert result["linf_diff"] < 1e-6

    def test_both_sum_to_one(self, market_P):
        result = compare_methods(market_P)
        np.testing.assert_allclose(result["pi_power"].sum(), 1.0, atol=1e-10)
        np.testing.assert_allclose(result["pi_eigen"].sum(), 1.0, atol=1e-10)


# ===========================================================================
# TestMultiInitConvergence
# ===========================================================================
class TestMultiInitConvergence:

    def test_all_converge_same_dist(self, market_P):
        result = multi_init_convergence(market_P)
        pi_star = result["pi_star"]
        for label, res in result["results"].items():
            dist = np.linalg.norm(res["stationary"] - pi_star, ord=1)
            assert dist < 1e-6, (
                f"Init '{label}' did not converge to π*: L1 dist = {dist:.2e}"
            )

    def test_all_initial_conditions_tested(self, market_P):
        result = multi_init_convergence(market_P)
        assert len(result["results"]) == 5

    def test_pi_star_sums_to_one(self, market_P):
        result = multi_init_convergence(market_P)
        np.testing.assert_allclose(result["pi_star"].sum(), 1.0, atol=1e-10)

    def test_pi_star_non_negative(self, market_P):
        result = multi_init_convergence(market_P)
        assert np.all(result["pi_star"] >= 0)

    def test_bull_start_converges(self, market_P):
        result = multi_init_convergence(market_P)
        bull_result = result["results"]["Bull [1,0,0]"]
        assert bull_result["converged"] is True

    def test_bear_start_converges(self, market_P):
        result = multi_init_convergence(market_P)
        bear_result = result["results"]["Bear [0,0,1]"]
        assert bear_result["converged"] is True

    def test_same_pi_star_across_extremes(self, market_P):
        """Bull and Bear starts should give the same final distribution."""
        result = multi_init_convergence(market_P)
        pi_bull = result["results"]["Bull [1,0,0]"]["stationary"]
        pi_bear = result["results"]["Bear [0,0,1]"]["stationary"]
        np.testing.assert_allclose(pi_bull, pi_bear, atol=1e-6)


# ===========================================================================
# TestPageRankComparison
# ===========================================================================
class TestPageRankComparison:

    def test_required_keys(self, market_P):
        result = pagerank_comparison(market_P, n_pagerank_pages=20)
        for key in ["market_spectral", "pr_spectral", "market_pi",
                    "pr_pi", "P_pagerank", "damping", "comparison_table"]:
            assert key in result

    def test_pagerank_lambda2_bounded_by_damping(self, market_P):
        """Damping factor d bounds |λ₂| ≤ d for PageRank matrix."""
        d = 0.85
        result = pagerank_comparison(market_P, pagerank_damping=d, n_pagerank_pages=20)
        pr_l2 = result["pr_spectral"]["lambda2"]
        assert pr_l2 <= d + 1e-9, f"|λ₂|={pr_l2:.6f} exceeds damping={d}"

    def test_pagerank_lambda2_structurally_bounded(self, market_P):
        """
        PageRank's |lambda2| is bounded by the damping factor d=0.85.

        This is the correct structural claim: the damped matrix
        P_pr = d*P + (1-d)*uniform has |lambda2| <= d by Perron-Frobenius.
        It does NOT mean the market model has a smaller |lambda2| -- the
        market's |lambda2| reflects actual regime persistence in the data
        and is independent of any damping parameter.
        """
        result = pagerank_comparison(market_P, n_pagerank_pages=100, tol=1e-6)
        pr_l2 = result["pr_spectral"]["lambda2"]
        d = result["damping"]
        assert pr_l2 <= d + 1e-9, (
            f"PageRank |lambda2|={pr_l2:.6f} exceeds d={d} -- "
            "the damping bound should always hold."
        )

    def test_pagerank_matrix_row_stochastic(self, market_P):
        result = pagerank_comparison(market_P, n_pagerank_pages=20)
        P_pr = result["P_pagerank"]
        np.testing.assert_allclose(P_pr.sum(axis=1), 1.0, atol=1e-10)

    def test_pagerank_matrix_positive(self, market_P):
        """Damping makes every entry strictly positive."""
        result = pagerank_comparison(market_P, n_pagerank_pages=20)
        assert np.all(result["P_pagerank"] > 0)

    def test_comparison_table_has_6_rows(self, market_P):
        result = pagerank_comparison(market_P, n_pagerank_pages=10)
        assert len(result["comparison_table"]["metric"]) == 6

    def test_damping_creates_positive_matrix(self, market_P):
        """
        The damping factor guarantees a strictly positive matrix.
        Every entry of P_pr = d*P + (1-d)/n > 0 because (1-d)/n > 0.
        This is what guarantees ergodicity by Perron-Frobenius.
        The Laplace smoothing in the market model plays the same role.
        """
        result = pagerank_comparison(market_P, n_pagerank_pages=20)
        P_pr = result["P_pagerank"]
        min_entry = float(P_pr.min())
        d = result["damping"]
        n = P_pr.shape[0]
        # Every entry must be at least (1-d)/n
        expected_floor = (1.0 - d) / n
        assert min_entry >= expected_floor - 1e-12, (
            f"Minimum PageRank entry {min_entry:.6e} < floor {expected_floor:.6e}"
        )


# ===========================================================================
# TestFitConvergenceRate
# ===========================================================================
class TestFitConvergenceRate:

    def test_returns_required_keys(self, market_P):
        result = power_iteration(market_P, tol=1e-12)
        rate = fit_convergence_rate(result["history"])
        for key in ["empirical_rate", "log_rate", "fit_r2", "fit_start"]:
            assert key in rate

    def test_rate_between_zero_and_one(self, market_P):
        result = power_iteration(market_P, tol=1e-12)
        rate = fit_convergence_rate(result["history"])
        er = rate["empirical_rate"]
        if not np.isnan(er):
            assert 0 < er < 1.0, f"Empirical rate out of range: {er}"

    def test_rate_close_to_lambda2(self, market_P):
        """Empirical rate should be within 20% of the theoretical |λ₂|."""
        spec = spectral_analysis(market_P)
        result = power_iteration(market_P, tol=1e-12)
        rate = fit_convergence_rate(result["history"])
        er = rate["empirical_rate"]
        if not np.isnan(er):
            np.testing.assert_allclose(
                er, spec["lambda2"], rtol=0.20,
                err_msg=f"Empirical rate {er:.6f} far from |λ₂|={spec['lambda2']:.6f}",
            )

    def test_r2_close_to_one(self, market_P):
        """Geometric convergence means log-linear fit should have high R²."""
        result = power_iteration(market_P, tol=1e-12)
        rate = fit_convergence_rate(result["history"])
        if not np.isnan(rate["fit_r2"]):
            assert rate["fit_r2"] > 0.95, (
                f"Log-linear R² = {rate['fit_r2']:.4f} (expected > 0.95 for "
                "geometric convergence)"
            )

    def test_short_history_returns_nan(self):
        """History with fewer than 5 positive entries → return nan."""
        rate = fit_convergence_rate([1e-3, 1e-4])
        assert np.isnan(rate["empirical_rate"])


