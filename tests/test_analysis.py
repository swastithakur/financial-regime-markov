"""
Unit tests for analysis and interpretation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import (
    sub_period_analysis,
    expected_durations,
    mean_first_passage_times,
    mfpt_summary,
    markov_order_test,
    sensitivity_analysis,
    conditional_exit_analysis,
)
from src.transition_matrix import STATES, N_STATES, laplace_smooth, count_transitions
from src.markov_engine import eigenvector_stationary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def uniform_P():
    return np.full((3, 3), 1 / 3)


@pytest.fixture
def known_P():
    """Matrix with easily-verified stationary distribution."""
    return np.array([
        [0.70, 0.20, 0.10],
        [0.30, 0.50, 0.20],
        [0.15, 0.25, 0.60],
    ])


@pytest.fixture
def market_P():
    from src.data_pipeline import download_data
    from src.data_pipeline import compute_rolling_return, classify_regimes
    prices = download_data()
    returns = compute_rolling_return(prices, window=20)
    regimes = classify_regimes(returns)
    return laplace_smooth(count_transitions(regimes), alpha=1.0)


@pytest.fixture
def market_regimes():
    from src.data_pipeline import download_data
    from src.data_pipeline import compute_rolling_return, classify_regimes
    prices = download_data()
    returns = compute_rolling_return(prices, window=20)
    return classify_regimes(returns)


@pytest.fixture
def market_regimes_with_dates() -> pd.Series:
    from src.data_pipeline import download_data
    from src.data_pipeline import compute_rolling_return, classify_regimes
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
def market_prices():
    from src.data_pipeline import download_data
    prices = download_data()
    return prices


# ===========================================================================
# TestMFPT
# ===========================================================================
class TestMFPT:

    def test_shape(self, known_P):
        M = mean_first_passage_times(known_P)
        assert M.shape == (N_STATES, N_STATES)

    def test_all_positive(self, known_P):
        M = mean_first_passage_times(known_P)
        assert np.all(M > 0)

    def test_diagonal_equals_reciprocal_of_pi(self, known_P):
        """
        Exact result from Markov theory: M[i,i] = 1/π_i.
        """
        M = mean_first_passage_times(known_P)
        pi = eigenvector_stationary(known_P)
        for i in range(N_STATES):
            np.testing.assert_allclose(
                M[i, i], 1.0 / pi[i], rtol=1e-6,
                err_msg=f"M[{i},{i}] = {M[i,i]:.4f} ≠ 1/π_{i} = {1/pi[i]:.4f}",
            )

    def test_diagonal_equals_reciprocal_uniform(self, uniform_P):
        """For uniform matrix, π_i = 1/3 so M[i,i] = 3 for all i."""
        M = mean_first_passage_times(uniform_P)
        np.testing.assert_allclose(np.diag(M), 3.0, atol=1e-9)

    def test_off_diagonal_satisfies_linear_system(self, known_P):
        """
        Verify: M[i,j] = 1 + Σ_{k≠j} P[i,k] * M[k,j]
        This is the defining equation for MFPT.
        """
        M = mean_first_passage_times(known_P)
        for i in range(N_STATES):
            for j in range(N_STATES):
                if i == j:
                    continue
                rhs = 1.0 + sum(
                    known_P[i, k] * M[k, j]
                    for k in range(N_STATES) if k != j
                )
                np.testing.assert_allclose(
                    M[i, j], rhs, atol=1e-6,
                    err_msg=f"MFPT linear system violated at M[{i},{j}]",
                )

    def test_off_diagonal_satisfies_linear_system_market(self, market_P):
        M = mean_first_passage_times(market_P)
        for i in range(N_STATES):
            for j in range(N_STATES):
                if i == j:
                    continue
                rhs = 1.0 + sum(
                    market_P[i, k] * M[k, j]
                    for k in range(N_STATES) if k != j
                )
                np.testing.assert_allclose(M[i, j], rhs, atol=1e-5)

    def test_uniform_off_diagonal_all_equal(self, uniform_P):
        """
        For a uniform matrix all off-diagonal MFPTs should be equal by symmetry.
        Each is 3 (= 1/π for uniform) since from any state you expect 3 steps
        to return, but hitting a *specific other* state first is 3 steps too
        (by symmetry π_j = 1/3, so M[i,j] = 1/π_j = 3 for any i ≠ j? 
        Actually M[i,j] = (n-1)/π_j * something — let's just verify the
        linear system is satisfied rather than the exact value.)
        """
        M = mean_first_passage_times(uniform_P)
        # For uniform: all off-diagonal elements should be equal by symmetry
        off_diag = [M[i, j] for i in range(3) for j in range(3) if i != j]
        np.testing.assert_allclose(
            off_diag, off_diag[0] * np.ones(len(off_diag)), atol=1e-8
        )

    def test_mfpt_summary_keys(self, market_P):
        M = mean_first_passage_times(market_P)
        summary = mfpt_summary(M)
        for key in ["matrix", "bear_to_bull", "bull_to_bear",
                    "mean_return_times", "most_accessible"]:
            assert key in summary

    def test_mfpt_summary_bear_to_bull_positive(self, market_P):
        M = mean_first_passage_times(market_P)
        summary = mfpt_summary(M)
        assert summary["bear_to_bull"] > 0

    def test_mfpt_summary_matches_matrix(self, market_P):
        """mfpt_summary bear_to_bull should match M[Bear, Bull] exactly."""
        M = mean_first_passage_times(market_P)
        summary = mfpt_summary(M)
        bear_idx = STATES.index("Bear")
        bull_idx = STATES.index("Bull")
        np.testing.assert_allclose(
            summary["bear_to_bull"], M[bear_idx, bull_idx], rtol=1e-10
        )

    def test_mean_return_times_in_summary(self, known_P):
        M = mean_first_passage_times(known_P)
        summary = mfpt_summary(M)
        pi = eigenvector_stationary(known_P)
        for i, state in enumerate(STATES):
            np.testing.assert_allclose(
                summary["mean_return_times"][state], 1.0 / pi[i], rtol=1e-6
            )


# ===========================================================================
# TestMarkovOrderTest
# ===========================================================================
class TestMarkovOrderTest:

    def test_required_keys(self, market_regimes):
        result = markov_order_test(market_regimes)
        for key in ["ll_first_order", "ll_second_order", "test_statistic",
                    "df", "p_value", "reject_h0", "conclusion",
                    "second_order_counts", "second_order_P",
                    "second_order_states"]:
            assert key in result

    def test_second_order_ll_geq_first_order(self, market_regimes):
        """
        Second-order log-likelihood must be ≥ first-order.
        The second-order model is strictly more expressive, so its MLE
        can never fit *worse* in-sample.
        """
        result = markov_order_test(market_regimes)
        assert result["ll_second_order"] >= result["ll_first_order"] - 1e-6

    def test_test_statistic_non_negative(self, market_regimes):
        result = markov_order_test(market_regimes)
        assert result["test_statistic"] >= 0

    def test_df_correct(self, market_regimes):
        """df = N_STATES * (N_STATES - 1)^2 = 3 * 4 = 12"""
        result = markov_order_test(market_regimes)
        assert result["df"] == N_STATES * (N_STATES - 1) ** 2

    def test_p_value_in_range(self, market_regimes):
        result = markov_order_test(market_regimes)
        assert 0 <= result["p_value"] <= 1

    def test_reject_h0_consistent_with_p_value(self, market_regimes):
        result = markov_order_test(market_regimes)
        if result["p_value"] < 0.05:
            assert result["reject_h0"] is True
        else:
            assert result["reject_h0"] is False

    def test_conclusion_is_string(self, market_regimes):
        result = markov_order_test(market_regimes)
        assert isinstance(result["conclusion"], str)
        assert len(result["conclusion"]) > 20

    def test_second_order_counts_shape(self, market_regimes):
        result = markov_order_test(market_regimes)
        assert result["second_order_counts"].shape == (N_STATES ** 2, N_STATES)

    def test_second_order_states_count(self, market_regimes):
        result = markov_order_test(market_regimes)
        assert len(result["second_order_states"]) == N_STATES ** 2

    def test_n_transitions_correct(self, market_regimes):
        result = markov_order_test(market_regimes)
        assert result["n_transitions"] == len(market_regimes) - 2

    def test_iid_sequence_fails_to_reject(self):
        """
        A truly i.i.d. sequence (memoryless, hence trivially first-order Markov)
        should fail to reject H0.  We create one by sampling independently
        from the stationary distribution.
        """
        rng = np.random.default_rng(99)
        # i.i.d. draws — every lag is irrelevant
        labels = rng.choice(STATES, size=5000, p=[0.45, 0.30, 0.25])
        dates = pd.bdate_range("2000-01-03", periods=len(labels))
        regimes = pd.Series(
            labels, index=dates,
            dtype=pd.CategoricalDtype(categories=STATES, ordered=True),
        )
        result = markov_order_test(regimes)
        # i.i.d. data should fail to reject first-order (p should be large)
        assert not result["reject_h0"], (
            f"Expected i.i.d. sequence to fail to reject H0, "
            f"but p = {result['p_value']:.4f}"
        )

    def test_second_order_chain_rejects(self):
        """
        A genuinely second-order chain should cause rejection of H0.
        We construct one: if the last TWO states are both Bear, tomorrow is
        very likely Bear; otherwise transitions are more uniform.
        """
        rng = np.random.default_rng(7)
        n = 8000

        # Transition depends on both X_{t-1} and X_t
        def next_state(prev, curr):
            if prev == "Bear" and curr == "Bear":
                probs = [0.05, 0.05, 0.90]   # very persistent Bear
            elif curr == "Bull":
                probs = [0.70, 0.20, 0.10]
            else:
                probs = [0.33, 0.34, 0.33]
            return rng.choice(STATES, p=probs)

        seq = [rng.choice(STATES), rng.choice(STATES)]
        for _ in range(n - 2):
            seq.append(next_state(seq[-2], seq[-1]))

        dates = pd.bdate_range("2000-01-03", periods=n)
        regimes = pd.Series(
            seq, index=dates,
            dtype=pd.CategoricalDtype(categories=STATES, ordered=True),
        )
        result = markov_order_test(regimes)
        assert result["reject_h0"], (
            f"Expected second-order chain to reject H0, "
            f"but p = {result['p_value']:.4f}"
        )


# ===========================================================================
# TestSensitivityAnalysis
# ===========================================================================
class TestSensitivityAnalysis:

    def test_required_keys(self, market_prices):
        result = sensitivity_analysis(
            market_prices,
            windows=[10, 20],
            thresholds=[0.01, 0.02],
        )
        for key in ["results", "baseline", "pi_bear_range",
                    "pi_bull_range", "frobenius_max", "is_robust"]:
            assert key in result

    def test_result_dataframe_shape(self, market_prices):
        windows = [10, 20]
        thresholds = [0.01, 0.02]
        result = sensitivity_analysis(market_prices, windows=windows, thresholds=thresholds)
        assert len(result["results"]) == len(windows) * len(thresholds)

    def test_all_pi_sum_to_one(self, market_prices):
        result = sensitivity_analysis(
            market_prices, windows=[10, 20], thresholds=[0.01, 0.02]
        )
        df = result["results"]
        pi_sums = df["pi_Bull"] + df["pi_Neutral"] + df["pi_Bear"]
        np.testing.assert_allclose(pi_sums.values, 1.0, atol=1e-8)

    def test_all_pi_positive(self, market_prices):
        result = sensitivity_analysis(
            market_prices, windows=[10, 20], thresholds=[0.01, 0.02]
        )
        df = result["results"]
        assert (df["pi_Bull"] > 0).all()
        assert (df["pi_Bear"] > 0).all()
        assert (df["pi_Neutral"] > 0).all()

    def test_baseline_row_present(self, market_prices):
        result = sensitivity_analysis(
            market_prices, windows=[10, 20], thresholds=[0.01, 0.02]
        )
        # Baseline (20, 0.02) is only present if in both lists
        df = result["results"]
        baseline_rows = df[df["is_baseline"]]
        # Our lists include 20 and 0.02 so baseline should appear
        assert len(baseline_rows) == 1

    def test_baseline_frobenius_zero(self, market_prices):
        """Frobenius distance from baseline to itself must be zero."""
        result = sensitivity_analysis(
            market_prices, windows=[10, 20], thresholds=[0.01, 0.02]
        )
        df = result["results"]
        baseline_frob = float(df[df["is_baseline"]]["frobenius_from_baseline"].iloc[0])
        np.testing.assert_allclose(baseline_frob, 0.0, atol=1e-10)

    def test_pi_bear_range_ordered(self, market_prices):
        result = sensitivity_analysis(
            market_prices, windows=[10, 20], thresholds=[0.01, 0.02]
        )
        lo, hi = result["pi_bear_range"]
        assert lo <= hi

    def test_is_robust_is_bool(self, market_prices):
        result = sensitivity_analysis(
            market_prices, windows=[10, 20], thresholds=[0.01, 0.02]
        )
        assert isinstance(result["is_robust"], bool)

    def test_larger_threshold_gives_more_neutral(self, market_prices):
        """
        Wider thresholds classify more observations as Neutral
        (the middle band is larger).  So π_Neutral should be larger for
        thresh=0.03 than for thresh=0.01 at the same window.
        """
        result = sensitivity_analysis(
            market_prices, windows=[20], thresholds=[0.01, 0.03]
        )
        df = result["results"]
        pi_n_low = float(df[df["threshold"] == 0.01]["pi_Neutral"].iloc[0])
        pi_n_high = float(df[df["threshold"] == 0.03]["pi_Neutral"].iloc[0])
        assert pi_n_high > pi_n_low, (
            f"Expected π_Neutral to increase with threshold: "
            f"0.01→{pi_n_low:.3f}, 0.03→{pi_n_high:.3f}"
        )


# ===========================================================================
# TestConditionalExit
# ===========================================================================
class TestConditionalExit:

    def test_required_keys(self, market_P):
        M = mean_first_passage_times(market_P)
        result = conditional_exit_analysis(market_P, M)
        for key in ["exit_destinations", "most_likely_after_bear",
                    "bear_exit_probs", "duration_in_bear",
                    "expected_bear_to_bull"]:
            assert key in result

    def test_exit_probs_sum_to_one(self, market_P):
        """Exit destination probabilities from any state must sum to 1."""
        M = mean_first_passage_times(market_P)
        result = conditional_exit_analysis(market_P, M)
        for state, probs in result["exit_destinations"].items():
            total = sum(probs.values())
            np.testing.assert_allclose(
                total, 1.0, atol=1e-9,
                err_msg=f"Exit probs from {state} don't sum to 1: {total}"
            )

    def test_exit_probs_positive(self, market_P):
        M = mean_first_passage_times(market_P)
        result = conditional_exit_analysis(market_P, M)
        for state, probs in result["exit_destinations"].items():
            assert all(p >= 0 for p in probs.values())

    def test_duration_in_bear_positive(self, market_P):
        M = mean_first_passage_times(market_P)
        result = conditional_exit_analysis(market_P, M)
        assert result["duration_in_bear"] > 0

    def test_bear_to_bull_greater_than_duration_in_bear(self, market_P):
        """
        Time to go Bear → Bull includes first getting out of Bear,
        so it must be ≥ duration in Bear.
        """
        M = mean_first_passage_times(market_P)
        result = conditional_exit_analysis(market_P, M)
        assert result["expected_bear_to_bull"] >= result["duration_in_bear"] - 1e-6

    def test_most_likely_after_bear_is_valid_state(self, market_P):
        M = mean_first_passage_times(market_P)
        result = conditional_exit_analysis(market_P, M)
        assert result["most_likely_after_bear"] in STATES

    def test_exit_formula_correct(self, known_P):
        """
        exit_prob[i][j] = P[i,j] / (1 - P[i,i]) for i ≠ j.
        Verify against hand calculation.
        """
        M = mean_first_passage_times(known_P)
        result = conditional_exit_analysis(known_P, M)
        for i, state in enumerate(STATES):
            p_exit = 1.0 - known_P[i, i]
            for j, dest in enumerate(STATES):
                if i == j:
                    continue
                expected = known_P[i, j] / p_exit
                actual = result["exit_destinations"][state].get(dest, 0.0)
                np.testing.assert_allclose(
                    actual, expected, rtol=1e-9,
                    err_msg=f"Exit prob {state}→{dest}: expected {expected:.4f}, got {actual:.4f}"
                )


# ===========================================================================
# TestSubPeriodAnalysis
# ===========================================================================
class TestSubPeriodAnalysis:
    def test_output_keys(self, market_regimes_with_dates):
        result = sub_period_analysis(market_regimes_with_dates)
        for key in ["P_pre", "P_post", "P_full", "diff", "frobenius"]:
            assert key in result
 
    def test_sub_matrices_row_stochastic(self, market_regimes_with_dates):
        result = sub_period_analysis(market_regimes_with_dates)
        for key in ["P_pre", "P_post", "P_full"]:
            np.testing.assert_allclose(
                result[key].sum(axis=1), 1.0, atol=1e-12,
                err_msg=f"{key} is not row-stochastic",
            )
 
    def test_diff_is_post_minus_pre(self, market_regimes_with_dates):
        result = sub_period_analysis(market_regimes_with_dates)
        np.testing.assert_allclose(
            result["diff"], result["P_post"] - result["P_pre"], atol=1e-12
        )
 
    def test_frobenius_non_negative(self, market_regimes_with_dates):
        result = sub_period_analysis(market_regimes_with_dates)
        assert result["frobenius"] >= 0
 
    def test_split_date_respected(self, market_regimes_with_dates):
        split = str(market_regimes_with_dates.index[500].date())
        result = sub_period_analysis(market_regimes_with_dates, split_date=split)
        assert result["n_pre"] + result["n_post"] == len(market_regimes_with_dates)
 
    def test_identical_halves_give_zero_diff(self):
        """If pre = post, difference matrix should be all zeros."""
        # Construct a regime series that is the same sequence repeated
        labels = (["Bull"] * 50 + ["Bear"] * 30 + ["Neutral"] * 20) * 2
        dates = pd.bdate_range("2010-01-04", periods=len(labels))
        regimes = pd.Series(
            labels, index=dates,
            dtype=pd.CategoricalDtype(categories=STATES, ordered=True),
        )
        midpoint = str(dates[len(labels) // 2].date())
        result = sub_period_analysis(regimes, split_date=midpoint)
        np.testing.assert_allclose(result["diff"], np.zeros((3, 3)), atol=0.05)
 
    def test_no_datetime_index_raises(self):
        """Integer-indexed series should raise ValueError."""
        regimes = pd.Series(["Bull", "Bear", "Neutral"] * 20)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            sub_period_analysis(regimes)
 
 
# ===========================================================================
# TestExpectedDurations
# ===========================================================================
class TestExpectedDurations:
    def test_formula_exact(self):
        """E[d_i] = 1/(1 - P_ii) checked against hand calculation."""
        P = np.array([
            [0.80, 0.10, 0.10],
            [0.20, 0.60, 0.20],
            [0.05, 0.10, 0.85],
        ])
        durations = expected_durations(P)
        np.testing.assert_allclose(durations["Bull"],    1 / (1 - 0.80), rtol=1e-10)
        np.testing.assert_allclose(durations["Neutral"], 1 / (1 - 0.60), rtol=1e-10)
        np.testing.assert_allclose(durations["Bear"],    1 / (1 - 0.85), rtol=1e-10)
 
    def test_all_positive(self, identity_like_matrix):
        durations = expected_durations(identity_like_matrix)
        for state in STATES:
            assert durations[state] > 0
 
    def test_high_persistence_means_long_duration(self):
        """Higher P_ii → longer expected duration."""
        P_low = np.array([[0.5, 0.3, 0.2],
                          [0.2, 0.5, 0.3],
                          [0.2, 0.3, 0.5]])
        P_high = np.array([[0.9, 0.05, 0.05],
                           [0.05, 0.9, 0.05],
                           [0.05, 0.05, 0.9]])
        d_low = expected_durations(P_low)
        d_high = expected_durations(P_high)
        for state in STATES:
            assert d_high[state] > d_low[state]
 
    def test_absorbing_state_infinite_duration(self):
        P = np.array([[1.0, 0.0, 0.0],
                      [0.1, 0.8, 0.1],
                      [0.1, 0.1, 0.8]])
        durations = expected_durations(P)
        assert durations["Bull"] == float("inf")