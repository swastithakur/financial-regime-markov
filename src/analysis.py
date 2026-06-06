from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

from src.transition_matrix import (
    STATES,
    N_STATES,
    STATE_IDX,
    count_transitions,
    laplace_smooth,
    mle_transition_matrix,
)
from src.markov_engine import eigenvector_stationary

def expected_durations(P):
    """
    Parameters
    ----------
    P : np.ndarray, shape (3, 3)
        Row-stochastic transition matrix.
    """
    durations = {}
    for i, state in enumerate(STATES):
        p_self = P[i, i]
        if p_self >= 1.0:
            durations[state] = float("inf")
        else:
            durations[state] = 1.0 / (1.0 - p_self)
    return durations


def sub_period_analysis(regimes, split_date = None, smooth_alpha = 1.0):
    """
    Parameters
    ----------
    regimes : pd.Series
    split_date : str or None
    smooth_alpha : float
    """
    if not isinstance(regimes.index, pd.DatetimeIndex):
        raise ValueError(
            "regimes must have a DatetimeIndex for sub-period analysis."
        )
    
    if split_date is None:
        mid_idx = len(regimes) // 2
        split_date = str(regimes.index[mid_idx].date())

    pre = regimes[regimes.index < split_date]
    post = regimes[regimes.index >= split_date]

    if len(pre) < 30 or len(post) < 30:
        raise ValueError(
            f"Sub-periods are too short (pre={len(pre)}, post={len(post)}). "
            "Choose a different split_date."
        )
    
    P_pre = laplace_smooth(count_transitions(pre), alpha=smooth_alpha)
    P_post = laplace_smooth(count_transitions(post), alpha=smooth_alpha)
    P_full = laplace_smooth(count_transitions(regimes), alpha=smooth_alpha)

    diff = P_post - P_pre
    abs_diff = np.abs(diff)

    return {
        "P_pre": P_pre,
        "P_post": P_post,
        "P_full": P_full,
        "diff": diff,
        "abs_diff": abs_diff,
        "frobenius": float(np.linalg.norm(diff, "fro")),
        "max_abs_diff": float(abs_diff.max()),
        "split_date": split_date,
        "n_pre": int(len(pre)),
        "n_post": int(len(post)),
    }


def mean_first_passage_times(P):
    """
    Parameters
    ----------
    P : np.ndarray, shape (n, n)
        Row-stochastic transition matrix.
    """
    n = P.shape[0]
    pi = eigenvector_stationary(P)
    M = np.zeros((n, n))

    for j in range(n):
        # Mean return time: exact formula from ergodic chain theory
        M[j, j] = 1.0 / pi[j]

        # Build the (n-1) × (n-1) sub-system by removing row j and col j
        idx_not_j = [k for k in range(n) if k != j]
        P_sub = P[np.ix_(idx_not_j, idx_not_j)]   # (n-1) × (n-1)
        A = np.eye(n - 1) - P_sub
        b = np.ones(n - 1)

        m_sub = np.linalg.solve(A, b)              # (n-1,)

        # Place back into the full matrix
        for local_i, global_i in enumerate(idx_not_j):
            M[global_i, j] = m_sub[local_i]

    return M


def mfpt_summary(M):
    bear_idx = STATE_IDX["Bear"]
    bull_idx = STATE_IDX["Bull"]
    neutral_idx = STATE_IDX["Neutral"]

    mean_return = {STATES[i]: M[i, i] for i in range(N_STATES)}

    # Average MFPT to each state across all non-self starting states
    avg_mfpt_to = {}
    for j in range(N_STATES):
        off_diag = [M[i, j] for i in range(N_STATES) if i != j]
        avg_mfpt_to[STATES[j]] = float(np.mean(off_diag))

    most_accessible = min(avg_mfpt_to, key=avg_mfpt_to.get)

    return {
        "matrix": M,
        "bear_to_bull": float(M[bear_idx, bull_idx]),
        "bull_to_bear": float(M[bull_idx, bear_idx]),
        "bear_recovery": float(M[bear_idx, bull_idx]),
        "bull_drawdown": float(M[bull_idx, bear_idx]),
        "neutral_to_bull": float(M[neutral_idx, bull_idx]),
        "neutral_to_bear": float(M[neutral_idx, bear_idx]),
        "mean_return_times": mean_return,
        "avg_mfpt_to": avg_mfpt_to,
        "most_accessible": most_accessible,
    }


def markov_order_test(regimes):
    """
    Parameters
    ----------
    regimes : pd.Series
    """
    vals = regimes.astype(str).to_numpy()
    n = len(vals)

    # First-order log-likelihood 
    counts1 = count_transitions(regimes)
    P1 = mle_transition_matrix(counts1)

    ll1 = 0.0
    for t in range(n - 1):
        i = STATE_IDX.get(vals[t])
        j = STATE_IDX.get(vals[t + 1])
        if i is not None and j is not None and P1[i, j] > 0:
            ll1 += np.log(P1[i, j])

    # Second-order transition counts
    second_order_states = [
        f"{s1}|{s2}" for s1 in STATES for s2 in STATES
    ]
    pair_to_idx = {pair: idx for idx, pair in enumerate(second_order_states)}

    counts2 = np.zeros((N_STATES ** 2, N_STATES), dtype=np.int64)

    for t in range(n - 2):
        s_prev = vals[t]
        s_curr = vals[t + 1]
        s_next = vals[t + 2]
        pair = f"{s_prev}|{s_curr}"
        if pair in pair_to_idx and s_next in STATE_IDX:
            row = pair_to_idx[pair]
            col = STATE_IDX[s_next]
            counts2[row, col] += 1

    # Second-order MLE (only rows with at least one observation)
    row_totals2 = counts2.sum(axis=1)
    P2 = np.zeros_like(counts2, dtype=float)
    for i in range(N_STATES ** 2):
        if row_totals2[i] > 0:
            P2[i] = counts2[i] / row_totals2[i]
        else:
            P2[i] = 1.0 / N_STATES   # uniform fallback for unobserved pairs
    
    # Second-order log-likelihood 
    ll2 = 0.0
    for t in range(n - 2):
        s_prev = vals[t]
        s_curr = vals[t + 1]
        s_next = vals[t + 2]
        pair = f"{s_prev}|{s_curr}"
        if pair in pair_to_idx and s_next in STATE_IDX:
            row = pair_to_idx[pair]
            col = STATE_IDX[s_next]
            if P2[row, col] > 0:
                ll2 += np.log(P2[row, col])

    # Test statistic and p-value 
    df = N_STATES * (N_STATES - 1) ** 2   
    Lambda = 2.0 * (ll2 - ll1)
    p_value = float(1.0 - stats.chi2.cdf(Lambda, df=df))
    reject_h0 = p_value < 0.05

    if reject_h0:
        conclusion = (
            f"REJECT H0 (p = {p_value:.4f} < 0.05): The data provide significant "
            "evidence that the second-order Markov model fits better. "
            "Markets carry memory beyond a single lag — the first-order "
            "Markov assumption is a simplification."
        )
    else:
        conclusion = (
            f"FAIL TO REJECT H0 (p = {p_value:.4f} ≥ 0.05): The data are "
            "consistent with a first-order Markov chain. The additional "
            "complexity of a second-order model is not statistically justified."
        )

    return {
        "ll_first_order": float(ll1),
        "ll_second_order": float(ll2),
        "test_statistic": float(Lambda),
        "df": df,
        "p_value": p_value,
        "reject_h0": reject_h0,
        "conclusion": conclusion,
        "n_transitions": n - 2,
        "second_order_counts": counts2,
        "second_order_P": P2,
        "second_order_states": second_order_states,
    }


def sensitivity_analysis(prices, windows = None, thresholds = None, smooth_alpha = 1.0,):
    """
    Baseline: window=20, threshold=0.02 (the project defaults).

    Parameters
    ----------
    prices : pd.Series
    windows : list[int]
    thresholds : list[float]
    """
    from src.data_pipeline import compute_rolling_return, classify_regimes

    if windows is None:
        windows = [5, 10, 20, 60]
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.03]

    # Baseline
    baseline_returns = compute_rolling_return(prices, window=20)
    baseline_regimes = classify_regimes(baseline_returns, 0.02, -0.02)
    baseline_counts = count_transitions(baseline_regimes)
    P_baseline = laplace_smooth(baseline_counts, alpha=smooth_alpha)
    pi_baseline = eigenvector_stationary(P_baseline)

    rows = []
    for window, thresh in product(windows, thresholds):
        returns = compute_rolling_return(prices, window=window)
        regimes = classify_regimes(returns, thresh, -thresh)
        counts = count_transitions(regimes)
        P = laplace_smooth(counts, alpha=smooth_alpha)
        pi = eigenvector_stationary(P)
        frob = float(np.linalg.norm(P - P_baseline, "fro"))

        rows.append({
            "window": window,
            "threshold": thresh,
            "pi_Bull": float(pi[0]),
            "pi_Neutral": float(pi[1]),
            "pi_Bear": float(pi[2]),
            "n_obs": len(regimes),
            "frobenius_from_baseline": frob,
            "is_baseline": (window == 20 and thresh == 0.02),
        })

    df = pd.DataFrame(rows)

    pi_bear_vals = df["pi_Bear"].values
    pi_bull_vals = df["pi_Bull"].values

    return {
        "results": df,
        "baseline": {
            "pi": pi_baseline,
            "P": P_baseline,
            "window": 20,
            "threshold": 0.02,
        },
        "pi_bear_range": (float(pi_bear_vals.min()), float(pi_bear_vals.max())),
        "pi_bull_range": (float(pi_bull_vals.min()), float(pi_bull_vals.max())),
        "frobenius_max": float(df["frobenius_from_baseline"].max()),
        "is_robust": (float(pi_bear_vals.max()) - float(pi_bear_vals.min())) < 0.15,
        "windows": windows,
        "thresholds": thresholds,
    }


def conditional_exit_analysis(P, M):
    """
    Parameters
    ----------
    P : np.ndarray (3,3)
    M : np.ndarray (3,3)  - MFPT matrix
    """
    n = P.shape[0]

    # Exit-destination probabilities: given that we leave state i, which state do we go to?
    exit_dest = {}
    for i, state in enumerate(STATES):
        p_exit = 1.0 - P[i, i]
        if p_exit > 0:
            probs = {
                STATES[j]: P[i, j] / p_exit
                for j in range(n) if j != i
            }
        else:
            probs = {}
        exit_dest[state] = probs

    # From Bear: where are you most likely to go first?
    bear_exit = exit_dest.get("Bear", {})
    if bear_exit:
        most_likely_after_bear = max(bear_exit, key=bear_exit.get)
    else:
        most_likely_after_bear = "Unknown"

    # Expected time to fully cycle Bear → exit → Bull
    bear_idx = STATE_IDX["Bear"]
    bull_idx = STATE_IDX["Bull"]
    neutral_idx = STATE_IDX["Neutral"]

    duration_in_bear = 1.0 / (1.0 - P[bear_idx, bear_idx])
    steps_bear_to_bull = M[bear_idx, bull_idx]  # includes time spent in Bear

    return {
        "exit_destinations": exit_dest,
        "most_likely_after_bear": most_likely_after_bear,
        "bear_exit_probs": bear_exit,
        "duration_in_bear": duration_in_bear,
        "expected_bear_to_bull": steps_bear_to_bull,
        "expected_bull_to_bear": M[bull_idx, bear_idx],
        "expected_neutral_to_bear": M[neutral_idx, bear_idx],
        "expected_neutral_to_bull": M[neutral_idx, bull_idx],
    }


def compile_findings(prices, regimes, P, split_date = None, smooth_alpha = 1.0, verbose = True):
    """
    Parameters
    ----------
    prices : pd.Series      
    regimes : pd.Series     
    P : np.ndarray (3,3)    
    pi : np.ndarray (3,)
    split_date : str or None
    smooth_alpha : float           
    """
    sub = sub_period_analysis(regimes, split_date=split_date, smooth_alpha=smooth_alpha)
    durations = expected_durations(P)

    M = mean_first_passage_times(P)
    mfpt = mfpt_summary(M)

    order_test = markov_order_test(regimes)

    sensitivity = sensitivity_analysis(prices)

    exit_analysis = conditional_exit_analysis(P, M)

    result = {
        "sub_period": sub,
        "durations": durations,
        "mfpt": mfpt,
        "order_test": order_test,
        "sensitivity": sensitivity,
        "exit": exit_analysis,
    }

    if verbose:
        print_findings(result)

    return result


def print_findings(results):
    sep = "=" * 65
    mfpt = results["mfpt"]
    ot = results["order_test"]
    sens = results["sensitivity"]
    ex = results["exit"]
    sub = results["sub_period"]
    durations = results["durations"]

    print(sep)
    print("ANALYSIS AND INTERPRETATION")
    print(sep)

    # Expected durations
    print("\nEXPECTED REGIME DURATIONS  E[d_i] = 1 / (1 - P[i,i])")
    for state in STATES:
        d = durations[state]

    # Sub-period stability --------------------------------------------
    print(f"\nSUB-PERIOD STABILITY  (split: {sub['split_date']})")
    print(f"  Pre-split  : {sub['n_pre']:,} observations")
    print(f"  Post-split : {sub['n_post']:,} observations")
    print(f"  Frobenius norm of (P_post - P_pre) : {sub['frobenius']:.4f}")
    print(f"  Max absolute element change        : {sub['max_abs_diff']:.4f}")
    print()
    print("  Signed difference matrix  (P_post - P_pre)")
    from src.transition_matrix import print_matrix
    print_matrix(sub["diff"], fmt="{:>+10.4f}")
    print()
    print("  Interpretation: values near 0 → stationarity; large values →")
    print("  transition dynamics shifted between sub-periods (non-stationarity).")
    print(sep)

    # MFPT 
    M = mfpt["matrix"]
    print("\nMEAN FIRST PASSAGE TIMES (trading days)")
    print("  M[i,j] = expected days to reach state j from state i")
    print(f"\n  {'':10}", end="")
    for s in STATES:
        print(f"  {'→ '+s:>12}", end="")
    print()
    for i, si in enumerate(STATES):
        print(f"  {si:<10}", end="")
        for j in range(N_STATES):
            marker = " *" if i == j else "  "
            print(f"  {M[i,j]:>10.1f}{marker}", end="")
        print()
    print("  (* diagonal = mean return time = 1/π_i)")

    print(f"\n  Key financial quantities:")
    print(f"    Bear → Bull (recovery time) : {mfpt['bear_to_bull']:>7.1f} days")
    print(f"    Bull → Bear (drawdown onset) : {mfpt['bull_to_bear']:>7.1f} days")
    print(f"    Most accessible state        : {mfpt['most_accessible']}")

    # Markov order test 
    print(f"\nMARKOV ORDER TEST  (H0: first-order Markov sufficient)")
    print(f"  Log-likelihood (1st order) : {ot['ll_first_order']:>12.2f}")
    print(f"  Log-likelihood (2nd order) : {ot['ll_second_order']:>12.2f}")
    print(f"  Λ = 2·(ℓ₂ − ℓ₁)          : {ot['test_statistic']:>12.4f}")
    print(f"  Degrees of freedom         : {ot['df']}")
    print(f"  p-value                    : {ot['p_value']:>12.6f}")
    print(f"  Reject H0 (α=0.05)        : {'YES' if ot['reject_h0'] else 'NO'}")
    print(f"\n  {ot['conclusion']}")

    # Sensitivity 
    df = sens["results"]
    print(f"\nSENSITIVITY ANALYSIS  (window × threshold grid)")
    print(f"  π_Bear range: [{sens['pi_bear_range'][0]:.3f}, {sens['pi_bear_range'][1]:.3f}]")
    print(f"  π_Bull range: [{sens['pi_bull_range'][0]:.3f}, {sens['pi_bull_range'][1]:.3f}]")
    print(f"  Max Frobenius deviation from baseline: {sens['frobenius_max']:.4f}")
    robust = "YES — qualitative conclusions hold across parameter choices" \
        if sens["is_robust"] else \
        "NO  — π_Bear changes substantially (>0.15) across parameters"
    print(f"  Robust: {robust}")

    print(f"\n  {'Window':>8}  {'Thresh':>8}  {'π_Bull':>8}  {'π_Neutral':>10}  {'π_Bear':>8}")
    print(f"  {'------':>8}  {'------':>8}  {'------':>8}  {'----------':>10}  {'------':>8}")
    for _, row in df.iterrows():
        marker = " ← baseline" if row["is_baseline"] else ""
        print(
            f"  {int(row['window']):>8}  {row['threshold']:>8.2f}  "
            f"{row['pi_Bull']:>8.3f}  {row['pi_Neutral']:>10.3f}  "
            f"{row['pi_Bear']:>8.3f}{marker}"
        )

    # Exit analysis 
    print(f"\nCONDITIONAL EXIT ANALYSIS")
    print(f"  Given exit from Bear regime:")
    for dest, prob in ex["bear_exit_probs"].items():
        print(f"    → {dest:<10}  {prob:.3f}")
    print(f"  Most likely next regime after Bear: {ex['most_likely_after_bear']}")
    print(f"  Expected duration in Bear: {ex['duration_in_bear']:.1f} days")
    print(f"  Expected days Bear → Bull: {ex['expected_bear_to_bull']:.1f}")
    print(f"  Expected days Bull → Bear: {ex['expected_bull_to_bear']:.1f}")
    print(sep)
