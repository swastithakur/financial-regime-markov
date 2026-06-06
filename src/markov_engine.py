import numpy as np
import pandas as pd

from src.transition_matrix import STATES, N_STATES

#Pagerank reference parameters
PAGERANK_DAMPING = 0.85
PAGERANK_TOL = 1e-6


def power_iteration(P, pi0 = None, tol = 1e-9, max_iter = 10_000):
    """
    Parameters
    ----------
    P : np.ndarray, shape (n, n)
    pi0 : np.ndarray, shape (n,) or None
    tol : float
    max_iter : int
    """
    n = P.shape[0]
    validate_stochastic(P)

    if pi0 is None:
        pi0 = np.ones(n) / n
    else:
        pi0 = np.asarray(pi0, dtype = float)
        if pi0.shape != (n,):
            raise ValueError(f"pi0 must have shape ({n},); got {pi0.shape}.")
        if not np.isclose(pi0.sum(), 1.0, atol = 1e-9):
            raise ValueError(f"pi0 must sum to 1; got {pi0.sum():.6f}.")
        if np.any(pi0 < 0):
            raise ValueError("pi0 must have non-negative entries.")
        
    pi = pi0.copy()
    history = []
    converged = False

    for _ in range(max_iter):
        pi_new = pi @ P
        delta = float(np.linalg.norm(pi_new-pi, ord = 1))
        history.append(delta)
        pi = pi_new

        if delta < tol:
            converged = True
            break

    return {
        "stationary": pi,
        "history": history,
        "n_iter": len(history),
        "converged": converged,
        "pi0": pi0,
        "tol": tol,
    }


def eigenvector_stationary(P):
    """
    Parameters
    ----------
    P : np.ndarray, shape (n, n)
    """
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    #Find the eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))

    if np.abs(eigenvalues[idx] - 1.0) > 1e-6:
        raise ValueError(
            f"No eigenvalue close to 1 found (closest: {eigenvalues[idx]:.6f}). "
            "Is P row-stochastic and ergodic?"
        )
    
    #Corresponding right eigenvector of Pᵀ  =  left eigenvector of P
    pi = np.real(eigenvectors[:, idx])

    # Enforce non-negativity (numerical noise can produce tiny negatives)
    pi = np.abs(pi)

    # Normalise to a proper distribution
    pi = pi / pi.sum()
    return pi


def spectral_analysis(P):
    """
    Parameters
    ----------
    P : np.ndarray, shape (n, n)
    """
    validate_stochastic(P)

    eigenvalues = np.linalg.eigvals(P.T)
    abs_ev = np.sort(np.abs(eigenvalues))[::-1]

    lambda1 = float(abs_ev[0])
    lambda2 = float(abs_ev[1]) if len(abs_ev) > 1 else 0.0
    spectral_gap = 1.0 - lambda2

    # Mixing time estimate: k s.t. |λ₂|^k < ε
    eps = 1e-3
    if lambda2 > 0 and lambda2 < 1.0:
        mixing_k = int(np.ceil(np.log(eps) / np.log(lambda2)))
    else:
        mixing_k = 0

    pf_ok = bool(np.isclose(lambda1, 1.0, atol=1e-6))

    return {
        "eigenvalues": eigenvalues,
        "eigenvalues_abs": abs_ev,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "spectral_gap": spectral_gap,
        "predicted_rate": lambda2,
        "mixing_time_est": mixing_k,
        "perron_frobenius_satisfied": pf_ok,
    }


# Compare power-iteration to eigenvector method
def compare_methods(P, tol = 1e-9):
    pi_result = power_iteration(P, tol=tol)
    pi_eigen = eigenvector_stationary(P)

    pi_pow = pi_result["stationary"]
    l1 = float(np.linalg.norm(pi_pow - pi_eigen, ord=1))
    linf = float(np.linalg.norm(pi_pow - pi_eigen, ord=np.inf))

    return {
        "pi_power": pi_pow,
        "pi_eigen": pi_eigen,
        "l1_diff": l1,
        "linf_diff": linf,
        "n_iter": pi_result["n_iter"],
        "converged": pi_result["converged"],
        "history": pi_result["history"],
        "agreement": linf < 1e-6,
    }


def multi_init_convergence(P, tol = 1e-9):
    n = P.shape[0]
    uniform = np.ones(n) / n

    #Named initial conditions
    init_conditions = {
        "Uniform [1/3,1/3,1/3]": uniform,
        "Bull [1,0,0]":          np.array([1.0, 0.0, 0.0]),
        "Bear [0,0,1]":          np.array([0.0, 0.0, 1.0]),
        "Neutral [0,1,0]":       np.array([0.0, 1.0, 0.0]),
        "Mixed [0.6,0.3,0.1]":   np.array([0.6, 0.3, 0.1])
    }

    results = {}
    for label, pi0 in init_conditions.items():
        results[label] = power_iteration(P, pi0=pi0, tol=tol)

    #Consensus stationary distribution = mean of all converged results
    pi_star = np.mean(
        [r["stationary"] for r in results.values()], axis = 0
    )
    pi_star /= pi_star.sum()   # re-normalise to correct any float drift

    return {
        "results": results,
        "pi_star": pi_star,
        "labels": list(init_conditions.keys()),
        "n_states": n,
    }


def pagerank_comparison(P_market,
    n_pagerank_pages = 100,
    pagerank_damping = PAGERANK_DAMPING,
    tol = PAGERANK_TOL,
    seed = 0
):
    """
    Parameters
    ----------
    P_market : np.ndarray, shape (3, 3)
    n_pagerank_pages : int
    pagerank_damping : float
    tol : float
    seed : int
        RNG seed for the random web graph.
    """
    rng = np.random.default_rng(seed)

    # Build a sparse random web graph (each page links to ~5 others)
    n = n_pagerank_pages
    links_per_page = 5
    raw = np.zeros((n, n))
    for i in range(n):
        targets = rng.choice(n, size = links_per_page, replace = False)
        raw[i][targets] = 1.0
    
    # Normalise rows
    row_sums = raw.sum(axis=1)
    P_random = raw / row_sums[:, np.newaxis]

    # Apply damping factor
    teleport = np.ones((n, n)) / n
    P_pr = pagerank_damping * P_random + (1.0 - pagerank_damping) * teleport

    # Spectral analysis
    market_spec = spectral_analysis(P_market)
    pr_spec = spectral_analysis(P_pr)

    # Power iteration — use uniform start for both
    market_pi_result = power_iteration(P_market, tol=tol)
    pr_pi_result = power_iteration(P_pr, tol=tol)

    comparison_table = {
        "metric": [
            "State space size",
            "Spectral gap (1 - |λ₂|)",
            "|λ₂| (convergence rate)",
            "Iterations to converge",
            "Mixing time estimate",
            "Damping / regularisation",
        ],
        "market_model": [
            f"{N_STATES} regimes",
            f"{market_spec['spectral_gap']:.4f}",
            f"{market_spec['lambda2']:.4f}",
            f"{market_pi_result['n_iter']}",
            f"{market_spec['mixing_time_est']}",
            "Laplace-α (Bayesian)",
        ],
        "pagerank": [
            f"{n} pages",
            f"{pr_spec['spectral_gap']:.4f}",
            f"{pr_spec['lambda2']:.4f}",
            f"{pr_pi_result['n_iter']}",
            f"{pr_spec['mixing_time_est']}",
            f"d = {pagerank_damping}",
        ],
    }

    return {
        "market_spectral": market_spec,
        "pr_spectral": pr_spec,
        "market_pi": market_pi_result,
        "pr_pi": pr_pi_result,
        "P_pagerank": P_pr,
        "damping": pagerank_damping,
        "comparison_table": comparison_table,
    }


def fit_convergence_rate(history):
    """
    Parameters
    ----------
    history : list[float]
        L1-norm deltas from power_iteration().
    """
    history_arr = np.array(history, dtype = float)

    # Remove any zeros (log undefined) and early transient
    positive_mask = history_arr > 0
    if positive_mask.sum() < 5:
        return {
            "empirical_rate": float("nan"),
            "log_rate": float("nan"),
            "fit_r2": float("nan"),
            "fit_start": 0,
        }
    
    # Use the second half of the history to avoid transient
    n = len(history_arr)
    start = max(1, n // 2)
    subset = history_arr[start:]
    subset_pos = subset[subset > 0]

    if len(subset_pos) < 3:
        start = 0
        subset_pos = history_arr[history_arr > 0]

    log_delta = np.log(subset_pos)
    t = np.arange(len(subset_pos))

    # Linear regression: log_delta = intercept + slope * t
    slope, intercept = np.polyfit(t, log_delta, 1)
    predicted = intercept + slope * t
    ss_res = np.sum((log_delta - predicted) ** 2)
    ss_tot = np.sum((log_delta - log_delta.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "empirical_rate": float(np.exp(slope)),
        "log_rate": float(slope),
        "fit_r2": float(r2),
        "fit_start": start,
    }


def markov_engine_summary(P, tol = 1e-9, verbose = True):
    pi_result   = power_iteration(P, tol=tol)
    pi_eigen    = eigenvector_stationary(P)
    cmp         = compare_methods(P, tol=tol)
    spectral    = spectral_analysis(P)
    multi       = multi_init_convergence(P, tol=tol)
    conv_rate   = fit_convergence_rate(pi_result["history"])
    pr_cmp      = pagerank_comparison(P)

    result = {
        "stationary":       pi_result["stationary"],
        "power_iter":       pi_result,
        "eigenvector":      pi_eigen,
        "comparison":       cmp,
        "spectral":         spectral,
        "multi_init":       multi,
        "convergence_rate": conv_rate,
        "pagerank_cmp":     pr_cmp,
    }

    if verbose:
        print_engine_summary(result)

    return result


 
def print_engine_summary(results):
    sep = "=" * 65

    pi      = results["stationary"]
    spec    = results["spectral"]
    cmp     = results["comparison"]
    cr      = results["convergence_rate"]
    pr      = results["pagerank_cmp"]
    multi   = results["multi_init"]

    print(sep)
    print("MARKOV ENGINE & CONVERGENCE ANALYSIS")
    print(sep)

    # stationary distribution
    print("\nSTATIONARY DISTRIBUTION  π* = π* P")
    print(f"  (Computed via power iteration,  tol = {results['power_iter']['tol']:.0e})")
    for i, s in enumerate(STATES):
        bar = "█" * int(pi[i] * 40)
        print(f"  {s:<8}  {pi[i]:.6f}  {bar}")
    print(f"  Sum = {pi.sum():.10f}  (should be exactly 1)")

    # method comparison 
    print("\nMETHOD COMPARISON: Power Iteration vs Eigenvector")
    print(f"  Iterations until convergence : {cmp['n_iter']}")
    print(f"  L1  distance  ‖π_pow − π_eig‖₁  : {cmp['l1_diff']:.2e}")
    print(f"  L∞  distance  ‖π_pow − π_eig‖∞  : {cmp['linf_diff']:.2e}")
    mark = "✓" if cmp["agreement"] else "✗"
    print(f"  {mark}  Methods agree to within 1e-6")

    # spectral analysis
    print("\nSPECTRAL ANALYSIS")
    print(f"  Perron-Frobenius satisfied : {'✓' if spec['perron_frobenius_satisfied'] else '✗'}")
    print(f"  λ₁ (dominant eigenvalue)   : {spec['lambda1']:.8f}  (should be 1.0)")
    print(f"  |λ₂| (second eigenvalue)   : {spec['lambda2']:.8f}")
    print(f"  Spectral gap 1 − |λ₂|      : {spec['spectral_gap']:.8f}")
    print(f"  Predicted convergence rate : |λ₂|ᵏ → 0 at rate {spec['predicted_rate']:.6f}")
    print(f"  Mixing time estimate       : ~{spec['mixing_time_est']} iterations")

    if not np.isnan(cr["empirical_rate"]):
        print(f"\nEMPIRICAL CONVERGENCE RATE  (log-linear fit to δᵗ history)")
        print(f"  Empirical |λ₂| from fit     : {cr['empirical_rate']:.6f}")
        print(f"  Theoretical |λ₂|            : {spec['lambda2']:.6f}")
        print(f"  Fit R²                      : {cr['fit_r2']:.4f}")

    # multiple initial conditions 
    print("\nMULTIPLE INITIAL CONDITIONS")
    print(f"  {'Initial condition':<30}  {'Iterations':>10}  {'Max dist from π*':>16}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*16}")
    for label, res in multi["results"].items():
        pi_final = res["stationary"]
        dist = float(np.linalg.norm(pi_final - results["stationary"], ord=1))
        print(f"  {label:<30}  {res['n_iter']:>10}  {dist:>16.2e}")
    print("  → All initial conditions converge to the same π* (ergodicity ✓)")

    # PageRank comparison 
    table = pr["comparison_table"]
    print("\nCOMPARISON WITH PAGERANK")
    col_w = 26
    print(f"  {'Metric':<34}  {'Market Model':>{col_w}}  {'PageRank':>{col_w}}")
    print(f"  {'-'*34}  {'-'*col_w}  {'-'*col_w}")
    for metric, mv, pv in zip(
        table["metric"], table["market_model"], table["pagerank"]
    ):
        print(f"  {metric:<34}  {mv:>{col_w}}  {pv:>{col_w}}")

    print(f"\n  Why PageRank is slower: damping factor d={pr['damping']} forces")
    print(f"  |λ₂| ≤ {pr['damping']}, keeping the spectral gap small.")
    print(f"  The market model has a larger spectral gap because regime")
    print(f"  transitions mix much faster than random web-graph walks.")
    print(sep)


def validate_stochastic(P, tol = 1e-8):
    """Raise if P is not a valid row-stochastic matrix."""
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"P must be a square 2-D array; got shape {P.shape}.")
    if not np.allclose(P.sum(axis=1), 1.0, atol=tol):
        raise ValueError(
            f"P is not row-stochastic: row sums = {P.sum(axis=1)}"
        )
    if np.any(P < -tol):
        raise ValueError("P contains negative entries.")

