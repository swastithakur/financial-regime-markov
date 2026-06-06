import numpy as np
import pandas as pd

STATES = ["Bull", "Neutral", "Bear"]
N_STATES = len(STATES)
STATE_IDX = {s : i for i, s in enumerate(STATES)}

def count_transitions(regimes):
    """
    Parameters
    ----------
    regimes : pd.Series
        Typically the output of data_pipeline.classify_regimes().
    """
    counts = np.zeros((N_STATES, N_STATES), dtype = np.int64)

    regime_vals = regimes.astype(str).to_numpy()
    for t in range(len(regime_vals)-1):
        i = STATE_IDX.get(regime_vals[t])
        j = STATE_IDX.get(regime_vals[t+1])

        if i is not None and j is not None:
            counts[i, j] += 1
    
    return counts


def mle_transition_matrix(counts):
    """
    Parameters
    ----------
    counts : np.ndarray, shape (3, 3)
        Raw transition counts from count_transitions().
    """
    row_totals = counts.sum(axis = 1)

    if np.any(row_totals == 0):
        zero_states = [STATES[i] for i in np.where(row_totals == 0)[0]]
        raise ValueError(
            f"States {zero_states} have zero observed transitions. "
            "Use laplace_smooth() before calling mle_transition_matrix(), "
            "or ensure the regime sequence contains all three states."
        )
    
    P = counts.astype(float) / row_totals[:, np.newaxis]
    return P


def laplace_smooth(counts, alpha = 1.0):
    """
    Parameters
    ----------
    counts : np.ndarray, shape (3, 3)
    alpha : float
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive; got {alpha}.")
    
    smoothed = counts.astype(float) + alpha
    row_totals = smoothed.sum(axis = 1, keepdims = True)
    return smoothed / row_totals


def simulate_chain(P, T, rng):
    state = rng.integers(0, N_STATES)
    seq = np.empty(T, dtype = np.int8)
    seq[0] = state

    cum_P = P.cumsum(axis = 1)
    u = rng.random(T-1)

    for t in range(1, T):
        row = cum_P[seq[t-1]]
        seq[t] = np.searchsorted(row, u[t-1])

    labels = pd.Series(STATES[s] for s in seq)
    return labels


def bootstrap_ci(regimes, n_bootstrap = 2000, ci_level = 0.95, seed = 0, smooth_alpha = 1.0):
    """
    Parameters
    ----------
    regimes : pd.Series
    n_bootstrap : int
    ci_level : float
    seed : int
    smooth_alpha : float
    """
    rng = np.random.default_rng(seed)
    T = len(regimes)

    #Point estimate
    counts_obs = count_transitions(regimes)
    P_hat = laplace_smooth(counts_obs, alpha=smooth_alpha)

    #Bootstrap loop
    boot_samples = np.zeros((n_bootstrap, N_STATES, N_STATES))

    for b in range(n_bootstrap):
        # Simulate a Markov chain of length T from P_hat
        sim_seq = simulate_chain(P_hat, T, rng)

        #Re-estimate
        counts_b = count_transitions(sim_seq)
        boot_samples[b] = laplace_smooth(counts_b, alpha=smooth_alpha)
    
    #Percentile CIs
    alpha_tail = (1.0 - ci_level) / 2
    ci_lower = np.percentile(boot_samples, 100 * alpha_tail, axis = 0)
    ci_upper = np.percentile(boot_samples, 100 * (1-alpha_tail), axis = 0)

    return {
        "point_estimate": P_hat,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
        "bootstrap_samples": boot_samples,
    }


def matrix_summary(
    regimes,
    smooth_alpha = 1.0,
    n_bootstrap = 2000,
    ci_level = 0.95,
    bootstrap_seed = 0,
    verbose = True,
):
    counts = count_transitions(regimes)
    P_mle = mle_transition_matrix(counts)
    P_smooth = laplace_smooth(counts, alpha=smooth_alpha)
    boot = bootstrap_ci(
        regimes,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=bootstrap_seed,
        smooth_alpha=smooth_alpha,
    )

    result = {
        "counts": counts,
        "P_mle": P_mle,
        "P_smooth": P_smooth,
        "bootstrap": boot,
    }

    if verbose:
        print_matrix_summary(result)

    return result



def print_matrix_summary(results):
    P = results["P_smooth"]
    counts = results["counts"]
    boot = results["bootstrap"]

    _sep = "=" * 65

    #counts
    print(_sep)
    print("TRANSITION MATRIX ESTIMATION")
    print(_sep)

    print("\nRAW TRANSITION COUNTS  (rows = current state, cols = next state)")
    print_matrix(counts.astype(int), fmt="{:>10,d}")

    #MLE 
    print("\nMLE TRANSITION MATRIX  P̂ᵢⱼ = nᵢⱼ / nᵢ")
    print_matrix(results["P_mle"])

    #Laplace-smoothed 
    print("\nLAPLACE-SMOOTHED MATRIX  (α=1 Dirichlet prior)")
    print_matrix(P)

    #bootstrap CIs
    print(f"\n{boot['ci_level']:.0%} BOOTSTRAP CONFIDENCE INTERVALS  "
          f"(B={boot['n_bootstrap']:,})")
    print(f"  {'':10}", end="")
    for s in STATES:
        print(f"  {'→ '+s:>16}", end="")
    print()
    for i, si in enumerate(STATES):
        print(f"  {si:<10}", end="")
        for j in range(N_STATES):
            lo = boot["ci_lower"][i, j]
            hi = boot["ci_upper"][i, j]
            pt = P[i, j]
            print(f"  {pt:.3f} [{lo:.3f},{hi:.3f}]", end="")
        print()


def print_matrix(M, fmt = "{:>10.4f}"):
    """Print a matrix with state labels."""

    header = f"  {'':10}" + "".join(f"  {s:>10}" for s in STATES)
    print(header)
    for i, si in enumerate(STATES):
        row = f"  {si:<10}" + "".join(fmt.format(M[i, j]) for j in range(M.shape[1]))
        print(row)
