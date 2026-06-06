"""
Unit tests for data pipeline and regime classification.

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Adjust import path when running from project root
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import (
    download_data,
    classify_regimes,
    compute_rolling_return,
    regime_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_prices() -> pd.Series:
    """500 days of monotonically increasing prices (trivially bull)."""
    dates = pd.bdate_range("2010-01-04", periods=500)
    prices = pd.Series(
        100.0 * (1.001 ** np.arange(500)),  # +0.1% per day
        index=dates,
        name="SPY",
    )
    return prices


@pytest.fixture
def flat_prices() -> pd.Series:
    """500 days of flat prices (always Neutral)."""
    dates = pd.bdate_range("2010-01-04", periods=500)
    return pd.Series(100.0, index=dates, name="SPY")


@pytest.fixture
def crash_prices() -> pd.Series:
    """100 normal days then 100 crash days (-0.5% per day)."""
    dates = pd.bdate_range("2010-01-04", periods=200)
    p1 = 100.0 * (1.001 ** np.arange(100))
    start_crash = p1[-1]
    p2 = start_crash * (0.995 ** np.arange(100))
    prices = np.concatenate([p1, p2])
    return pd.Series(prices, index=dates, name="SPY")


# ---------------------------------------------------------------------------
# compute_rolling_return
# ---------------------------------------------------------------------------
class TestComputeRollingReturn:
    def test_length(self, sample_prices):
        returns = compute_rolling_return(sample_prices, window=20)
        assert len(returns) == len(sample_prices) - 20

    def test_positive_for_rising_prices(self, sample_prices):
        returns = compute_rolling_return(sample_prices, window=20)
        assert (returns > 0).all()

    def test_zero_for_flat_prices(self, flat_prices):
        returns = compute_rolling_return(flat_prices, window=20)
        np.testing.assert_allclose(returns.values, 0.0, atol=1e-10)

    def test_no_nans(self, sample_prices):
        returns = compute_rolling_return(sample_prices, window=20)
        assert not returns.isna().any()

    def test_different_windows(self, sample_prices):
        r5 = compute_rolling_return(sample_prices, window=5)
        r20 = compute_rolling_return(sample_prices, window=20)
        assert len(r5) == len(sample_prices) - 5
        assert len(r20) == len(sample_prices) - 20


# ---------------------------------------------------------------------------
# classify_regimes
# ---------------------------------------------------------------------------
class TestClassifyRegimes:
    def test_output_dtype(self, sample_prices):
        returns = compute_rolling_return(sample_prices, window=20)
        regimes = classify_regimes(returns)
        assert isinstance(regimes.dtype, pd.CategoricalDtype)

    def test_all_bull_for_rising(self, sample_prices):
        returns = compute_rolling_return(sample_prices, window=20)
        regimes = classify_regimes(returns, bull_thresh=0.02, bear_thresh=-0.02)
        assert (regimes == "Bull").all()

    def test_all_neutral_for_flat(self, flat_prices):
        returns = compute_rolling_return(flat_prices, window=20)
        regimes = classify_regimes(returns, bull_thresh=0.02, bear_thresh=-0.02)
        assert (regimes == "Neutral").all()

    def test_bear_appears_in_crash(self, crash_prices):
        returns = compute_rolling_return(crash_prices, window=20)
        regimes = classify_regimes(returns, bull_thresh=0.02, bear_thresh=-0.02)
        assert "Bear" in regimes.values

    def test_categories_exhaustive(self, sample_prices):
        returns = compute_rolling_return(sample_prices, window=20)
        regimes = classify_regimes(returns)
        assert set(regimes.cat.categories) == {"Bull", "Neutral", "Bear"}

    def test_invalid_thresholds_raises(self, sample_prices):
        returns = compute_rolling_return(sample_prices, window=20)
        with pytest.raises(ValueError):
            classify_regimes(returns, bull_thresh=0.01, bear_thresh=0.02)

    def test_no_nans(self, sample_prices):
        returns = compute_rolling_return(sample_prices, window=20)
        regimes = classify_regimes(returns)
        assert not regimes.isna().any()

    def test_threshold_boundary(self):
        """
        Return exactly equal to bull_thresh is Neutral (not Bull) because
        classify_regimes uses strict > for Bull, strict < for Bear.
        """
        dates = pd.bdate_range("2020-01-02", periods=50)
        # Flat for 20 days at 100, then jumps to exactly 102 on day 21.
        # The 20-day return on day 21 is exactly (102-100)/100 = +2.0% = bull_thresh.
        # Because the condition is r > bull_thresh (strict), this should be Neutral.
        prices = pd.Series(
            [100.0] * 21 + [102.0] * 29,
            index=dates,
        )
        returns = compute_rolling_return(prices, window=20)
        regimes = classify_regimes(returns, bull_thresh=0.02, bear_thresh=-0.02)
        # The first rolling return (day index 20) is (102-100)/100 = 0.02 exactly
        assert regimes.iloc[0] == "Neutral", (
            f"Expected Neutral for return == bull_thresh, got {regimes.iloc[0]}"
        )


# ---------------------------------------------------------------------------
# regime_summary
# ---------------------------------------------------------------------------
class TestRegimeSummary:
    def get_regimes(self, prices):
        returns = compute_rolling_return(prices, window=20)
        return classify_regimes(returns)

    def test_counts_sum_to_total(self, sample_prices):
        regimes = self.get_regimes(sample_prices)
        summary = regime_summary(regimes)
        assert summary["counts"].sum() == len(regimes)

    def test_frequencies_sum_to_one(self, sample_prices):
        regimes = self.get_regimes(sample_prices)
        summary = regime_summary(regimes)
        np.testing.assert_allclose(summary["frequencies"].sum(), 1.0, atol=1e-10)

    def test_run_lengths_cover_all_obs(self, sample_prices):
        regimes = self.get_regimes(sample_prices)
        summary = regime_summary(regimes)
        total_run_length = summary["run_lengths"]["length"].sum()
        assert total_run_length == len(regimes)

    def test_mean_run_positive(self, sample_prices):
        regimes = self.get_regimes(sample_prices)
        summary = regime_summary(regimes)
        assert (summary["mean_run"].dropna() > 0).all()

    def test_nonoverlapping_subset(self, sample_prices):
        regimes = self.get_regimes(sample_prices)
        summary = regime_summary(regimes, window=20)
        non_ol = summary["nonoverlapping"]
        assert len(non_ol) <= len(regimes)
        # Check it's a proper subset
        assert set(non_ol.index).issubset(set(regimes.index))

    def test_all_regimes_in_summary_keys(self, sample_prices):
        regimes = self.get_regimes(sample_prices)
        summary = regime_summary(regimes)
        for key in ["counts", "frequencies", "mean_run", "median_run", "max_run"]:
            assert key in summary


# ---------------------------------------------------------------------------
# Integration test with real data
# ---------------------------------------------------------------------------
class TestDataIntegration:

    def test_full_pipeline(self):
        prices = download_data()
        returns = compute_rolling_return(prices, window=20)
        regimes = classify_regimes(returns)
        summary = regime_summary(regimes, window=20)

        # All three regimes should appear in 1000 days of regime-switching data
        counts = summary["counts"]
        assert (counts > 0).all(), f"Not all regimes observed: {counts}"

        # Frequencies sum to 1
        np.testing.assert_allclose(summary["frequencies"].sum(), 1.0, atol=1e-10)

    def test_bull_dominated_in_strong_uptrend(self):
        """In a prolonged bull market, Bull regime should dominate."""
        dates = pd.bdate_range("2010-01-04", periods=500)
        # Strong uptrend: +0.2% per day = +40% per year
        prices = pd.Series(
            100.0 * (1.002 ** np.arange(500)),
            index=dates,
        )
        returns = compute_rolling_return(prices, window=20)
        regimes = classify_regimes(returns)
        summary = regime_summary(regimes)
        assert summary["frequencies"]["Bull"] > 0.5

    def test_bear_dominated_in_crash(self):
        """In a sustained crash, Bear regime should dominate."""
        dates = pd.bdate_range("2010-01-04", periods=500)
        prices = pd.Series(
            100.0 * (0.997 ** np.arange(500)),
            index=dates,
        )
        returns = compute_rolling_return(prices, window=20)
        regimes = classify_regimes(returns)
        summary = regime_summary(regimes)
        assert summary["frequencies"]["Bear"] > 0.5
