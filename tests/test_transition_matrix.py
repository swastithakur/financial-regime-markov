"""
Unit tests for transition matrix estimation.
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
    STATES,
    N_STATES,
    count_transitions,
    mle_transition_matrix,
    laplace_smooth,
    bootstrap_ci,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_regimes() -> pd.Series:
    """Short, hand-crafted sequence for exact count verification."""
    labels = ["Bull", "Bull", "Neutral", "Bear", "Bear", "Bull", "Neutral"]
    dates = pd.bdate_range("2020-01-02", periods=len(labels))
    return pd.Series(
        labels,
        index=dates,
        dtype=pd.CategoricalDtype(categories=STATES, ordered=True),
        name="regime",
    )


@pytest.fixture
def market_regimes() -> pd.Series:
    prices = download_data()
    returns = compute_rolling_return(prices, window=20)
    return classify_regimes(returns)


@pytest.fixture
def identity_like_matrix() -> np.ndarray:
    """Near-identity transition matrix (very persistent regimes)."""
    return np.array([
        [0.98, 0.01, 0.01],
        [0.01, 0.98, 0.01],
        [0.01, 0.01, 0.98],
    ])


@pytest.fixture
def uniform_matrix() -> np.ndarray:
    """Completely uniform transition matrix (memoryless)."""
    return np.full((3, 3), 1 / 3)


# ===========================================================================
# TestCountTransitions
# ===========================================================================
class TestCountTransitions:
    def test_shape(self, simple_regimes):
        counts = count_transitions(simple_regimes)
        assert counts.shape == (N_STATES, N_STATES)

    def test_dtype_int(self, simple_regimes):
        counts = count_transitions(simple_regimes)
        assert np.issubdtype(counts.dtype, np.integer)

    def test_all_non_negative(self, simple_regimes):
        counts = count_transitions(simple_regimes)
        assert np.all(counts >= 0)

    def test_total_count(self, simple_regimes):
        """Total transitions = len(sequence) - 1."""
        counts = count_transitions(simple_regimes)
        assert counts.sum() == len(simple_regimes) - 1

    def test_known_counts(self):
        """Hand-verify exact counts for a short sequence."""
        # Sequence: Bull→Bull, Bull→Neutral, Neutral→Bear, Bear→Bear, Bear→Bull
        labels = ["Bull", "Bull", "Neutral", "Bear", "Bear", "Bull"]
        dates = pd.bdate_range("2020-01-02", periods=len(labels))
        regimes = pd.Series(labels, index=dates)
        counts = count_transitions(regimes)

        # Bull→Bull: 1, Bull→Neutral: 1, Neutral→Bear: 1,
        # Bear→Bear: 1, Bear→Bull: 1
        assert counts[0, 0] == 1  # Bull→Bull
        assert counts[0, 1] == 1  # Bull→Neutral
        assert counts[0, 2] == 0  # Bull→Bear
        assert counts[1, 2] == 1  # Neutral→Bear
        assert counts[2, 2] == 1  # Bear→Bear
        assert counts[2, 0] == 1  # Bear→Bull

    def test_row_totals_match_state_visits(self, simple_regimes):
        """
        Row total for state i = number of times i is the *current* state
        (i.e., appears in positions 0..T-2 of the sequence).
        """
        counts = count_transitions(simple_regimes)
        seq = simple_regimes.astype(str).to_numpy()
        for idx, state in enumerate(STATES):
            expected = np.sum(seq[:-1] == state)
            assert counts[idx].sum() == expected


# ===========================================================================
# TestMLETransitionMatrix
# ===========================================================================
class TestMLETransitionMatrix:
    def test_row_stochastic(self, market_regimes):
        counts = count_transitions(market_regimes)
        P = mle_transition_matrix(counts)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)

    def test_all_non_negative(self, market_regimes):
        counts = count_transitions(market_regimes)
        P = mle_transition_matrix(counts)
        assert np.all(P >= 0)

    def test_shape(self, market_regimes):
        counts = count_transitions(market_regimes)
        P = mle_transition_matrix(counts)
        assert P.shape == (N_STATES, N_STATES)

    def test_zero_row_raises(self):
        """A row of all zeros (state never observed) should raise ValueError."""
        bad_counts = np.array([
            [5, 3, 2],
            [0, 0, 0],  # Neutral never observed as current state
            [1, 2, 7],
        ])
        with pytest.raises(ValueError, match="zero observed transitions"):
            mle_transition_matrix(bad_counts)

    def test_known_values(self):
        """Verify MLE values match hand-computation."""
        counts = np.array([
            [80, 10, 10],
            [20, 60, 20],
            [5,  10, 85],
        ], dtype=float)
        P = mle_transition_matrix(counts)
        np.testing.assert_allclose(P[0], [0.80, 0.10, 0.10], atol=1e-12)
        np.testing.assert_allclose(P[1], [0.20, 0.60, 0.20], atol=1e-12)
        np.testing.assert_allclose(P[2], [0.05, 0.10, 0.85], atol=1e-12)


# ===========================================================================
# TestLaplaceSmooth
# ===========================================================================
class TestLaplaceSmooth:
    def test_row_stochastic(self, market_regimes):
        counts = count_transitions(market_regimes)
        P = laplace_smooth(counts, alpha=1.0)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)

    def test_all_strictly_positive(self, market_regimes):
        counts = count_transitions(market_regimes)
        P = laplace_smooth(counts, alpha=1.0)
        assert np.all(P > 0)

    def test_zero_count_becomes_positive(self):
        """A zero count should become positive after smoothing."""
        counts = np.array([
            [100, 0, 0],  # only self-transitions observed
            [10, 80, 10],
            [5,  10, 85],
        ], dtype=float)
        P = laplace_smooth(counts, alpha=1.0)
        assert P[0, 1] > 0   # was 0, should now be > 0
        assert P[0, 2] > 0

    def test_large_alpha_approaches_uniform(self):
        """With very large α, matrix approaches 1/K for all entries."""
        counts = np.array([
            [1000, 1, 1],
            [1, 1000, 1],
            [1, 1, 1000],
        ], dtype=float)
        P = laplace_smooth(counts, alpha=1e9)
        np.testing.assert_allclose(P, np.full((3, 3), 1 / 3), atol=1e-3)

    def test_small_alpha_approaches_mle(self):
        """With very small α, smoothed matrix approaches MLE."""
        counts = np.array([
            [80, 10, 10],
            [20, 60, 20],
            [5,  10, 85],
        ], dtype=float)
        P_mle = mle_transition_matrix(counts)
        P_smooth = laplace_smooth(counts, alpha=1e-9)
        np.testing.assert_allclose(P_smooth, P_mle, atol=1e-4)

    def test_negative_alpha_raises(self):
        counts = np.ones((3, 3))
        with pytest.raises(ValueError, match="alpha must be positive"):
            laplace_smooth(counts, alpha=-0.1)

    def test_alpha_invariant_shape(self):
        counts = np.ones((3, 3))
        for alpha in [0.01, 1.0, 10.0, 100.0]:
            P = laplace_smooth(counts, alpha=alpha)
            assert P.shape == (3, 3)
            np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)


# ===========================================================================
# TestBootstrapCI
# ===========================================================================
class TestBootstrapCI:
    def test_output_shape(self, market_regimes):
        result = bootstrap_ci(market_regimes, n_bootstrap=100, seed=0)
        for key in ["point_estimate", "ci_lower", "ci_upper", "ci_width"]:
            assert result[key].shape == (N_STATES, N_STATES), f"{key} wrong shape"

    def test_ci_ordered(self, market_regimes):
        """Lower CI ≤ point estimate ≤ upper CI for all cells."""
        result = bootstrap_ci(market_regimes, n_bootstrap=100, seed=0)
        assert np.all(result["ci_lower"] <= result["point_estimate"] + 1e-10)
        assert np.all(result["ci_upper"] >= result["point_estimate"] - 1e-10)

    def test_ci_width_non_negative(self, market_regimes):
        result = bootstrap_ci(market_regimes, n_bootstrap=100, seed=0)
        assert np.all(result["ci_width"] >= 0)

    def test_bootstrap_samples_shape(self, market_regimes):
        B = 50
        result = bootstrap_ci(market_regimes, n_bootstrap=B, seed=0)
        assert result["bootstrap_samples"].shape == (B, N_STATES, N_STATES)

    def test_point_estimate_row_stochastic(self, market_regimes):
        result = bootstrap_ci(market_regimes, n_bootstrap=100, seed=0)
        np.testing.assert_allclose(
            result["point_estimate"].sum(axis=1), 1.0, atol=1e-12
        )

    def test_bootstrap_samples_row_stochastic(self, market_regimes):
        result = bootstrap_ci(market_regimes, n_bootstrap=50, seed=0)
        row_sums = result["bootstrap_samples"].sum(axis=2)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_ci_level_stored(self, market_regimes):
        result = bootstrap_ci(market_regimes, n_bootstrap=50, ci_level=0.90, seed=0)
        assert result["ci_level"] == 0.90

    


