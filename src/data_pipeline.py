from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np

#Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_CACHE = DATA_DIR / "spy_prices.csv"


def download_data(
        ticker = "SPY", 
        start = "2000-01-01", 
        end = "2024-12-31",
        cache_path = DEFAULT_CACHE,
        force_download = False
):
    
    if cache_path is not None and cache_path.exists() and not force_download:
        print(f"[data_pipeline] Loading cached prices from {cache_path}")

        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze()
        prices.name = ticker

        return prices.sort_index()
    
    print(f"[data_pipeline] Downloading {ticker} from Yahoo Finance ({start} → {end}) …")
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    
    prices = raw["Close"].squeeze().dropna()
    prices.name = ticker
    prices = prices.sort_index()

    if cache_path is not None:
        prices.to_csv(cache_path)
        print(f"[data_pipeline] Saved {len(prices):,} rows to {cache_path}")

    return prices


def compute_rolling_return(prices, window = 20):
    """
    Parameters
    ----------
    prices : pd.Series
    window : int
    """
    rolling_return = prices.pct_change(periods = window).dropna()
    rolling_return.name = f"rolling_return_{window}d"

    return rolling_return


def classify_regimes(returns, bull_thresh = 0.02, bear_thresh = -0.02):
    """
    Parameters
    ----------
    returns : pd.Series
        Output of compute_rolling_return().
    bull_thresh : float
    bear_thresh : float
    """
    if (bear_thresh >= bull_thresh):
        raise ValueError("bear_thresh must be strictly less than bull_thresh.")
    
    conditions = [returns > bull_thresh, returns < bear_thresh]
    choices = ["Bull", "Bear"]
    labels = np.select(conditions, choices, default = "Neutral")

    regimes = pd.Series(labels, index=returns.index, name="regime")
    
    # Store the ordered category so plots sort naturally
    regimes = regimes.astype(
        pd.CategoricalDtype(categories=["Bull", "Neutral", "Bear"], ordered=True)
    )
    return regimes


def run_lengths(regimes):
    records = []
    current = regimes.iloc[0]
    start_idx = 0

    for i in range(1, len(regimes)):
        if regimes.iloc[i] != current:
            records.append(
                {
                    "regime" : str(current),
                    "start" : regimes.index[start_idx],
                    "end" : regimes.index[i-1],
                    "length" : i-start_idx
                }
            )
            current = regimes.iloc[i]
            start_idx = i

    records.append(
        {
            "regime" : str(current),
            "start" : regimes.index[start_idx],
            "end" : regimes.index[-1],
            "length" : len(regimes)-start_idx
        }
    )
    return pd.DataFrame(records)


def regime_summary(regimes, window = 20):
    run_df = run_lengths(regimes)

    counts = regimes.value_counts().reindex(["Bull", "Neutral", "Bear"])
    frequencies = counts / counts.sum()

    mean_run = run_df.groupby("regime")["length"].mean().reindex(["Bull", "Neutral", "Bear"])
    median_run = run_df.groupby("regime")["length"].median().reindex(["Bull", "Neutral", "Bear"])
    max_run = run_df.groupby("regime")["length"].max().reindex(["Bull", "Neutral", "Bear"])

    n_transitions = (run_df.shape[0] - 1)

    # Non-overlapping subsample: every `window`-th observation
    nonoverlapping = regimes.iloc[::window]

    return {
        "counts": counts,
        "frequencies": frequencies,
        "run_lengths": run_df,
        "mean_run": mean_run,
        "median_run": median_run,
        "max_run": max_run,
        "n_transitions": n_transitions,
        "nonoverlapping": nonoverlapping,
    }


def print_summary(summary, window = 20, thresholds = (0.02, -0.02)):
    #Pretty-print the regime summary to stdout.

    counts = summary["counts"]
    freqs = summary["frequencies"]
    total = counts.sum()

    bull_t, bear_t = thresholds
    print("=" * 60)
    print("REGIME CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Rolling window : {window} trading days")
    print(f"  Bull threshold : > {bull_t:+.0%}")
    print(f"  Bear threshold : < {bear_t:+.0%}")
    print(f"  Total obs.     : {total:,}")
    print()

    print(f"  {'Regime':<10} {'Count':>8}  {'Frequency':>10}  {'Mean run':>10}  {'Max run':>8}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")
    for regime in ["Bull", "Neutral", "Bear"]:
        print(
            f"  {regime:<10} {counts[regime]:>8,}  "
            f"{freqs[regime]:>10.1%}  "
            f"{summary['mean_run'][regime]:>10.1f}  "
            f"{int(summary['max_run'][regime]):>8}"
        )

    print()
    print(f"  Total regime changes     : {summary['n_transitions']:,}")
    print(
        f"  Non-overlapping obs.     : {len(summary['nonoverlapping']):,}"
        f"  (every {window}th day)"
    )
    print()
    print("  NOTE: Consecutive rolling-window observations overlap by")
    print(f"  {window - 1} days, biasing transition counts toward persistence.")
    print(f"  The non-overlapping subsample corrects for this.")
    print("=" * 60)


def build_pipeline(
    ticker = "SPY",
    start = "2000-01-01",
    end = "2024-12-31",
    window = 20,
    bull_thresh = 0.02,
    bear_thresh = -0.02,
    cache_path = DEFAULT_CACHE,
    force_download = False,
    verbose = True,
):
    
    prices = download_data(ticker, start, end, cache_path, force_download)
    returns = compute_rolling_return(prices, window)
    regimes = classify_regimes(returns, bull_thresh, bear_thresh)
    summary = regime_summary(regimes, window)

    if verbose:
        print_summary(summary, window, (bull_thresh, bear_thresh))

    return prices, returns, regimes, summary
