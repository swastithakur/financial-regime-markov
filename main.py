"""
Single entry point: runs the entire project

Usage
-----
    python main.py                 # real SPY data (requires yfinance)
    python main.py --fast          # skip bootstrap (quick demo)

Output
------
    visualizations/          all individual figures
    data/spy_prices.csv      cached price data
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))

# project imports 
from src.data_pipeline import build_pipeline
from src.transition_matrix import (
    count_transitions, laplace_smooth,
     mle_transition_matrix, bootstrap_ci,
)
from src.markov_engine import markov_engine_summary
from src.analysis import (
    compile_findings, mean_first_passage_times, mfpt_summary, 
    sub_period_analysis, expected_durations,
)
from src.visualize import *

VIZ_DIR = Path("visualizations")
VIZ_DIR.mkdir(exist_ok=True)


def main(fast = False):
    t0 = time.time()
    sep = "=" * 65

    print(sep)
    print("FINANCIAL REGIME TRANSITION ANALYSIS — FULL PIPELINE")
    print(sep)

    # Data pipeline
    print("\nData pipeline …")

    prices, returns, regimes, data_summary = build_pipeline(verbose=True)

    # plot_regime_timeline(prices, regimes, save_path="visualizations/regime_timeline.png")
    # plot_return_distribution(returns, regimes, save_path="visualizations/return_distribution.png")
    # plot_regime_frequencies(data_summary, save_path="visualizations/regime_frequencies.png")

    # Transition matrix
    print("\n Estimating transition matrix …")
    counts  = count_transitions(regimes)
    P_mle   = mle_transition_matrix(counts)
    P       = laplace_smooth(counts, alpha=1.0)

    #plot_transition_heatmap(P_mle, P, counts, save_path="visualizations/transition_heatmap.png") 

    n_boot = 200 if fast else 2000
    print(f"     Bootstrap CIs (B={n_boot}) …")
    boot = bootstrap_ci(regimes, n_bootstrap=n_boot, seed=0)

    #plot_bootstrap_ci(boot, save_path="visualizations/bootstrap_ci.png")

    # Markov engine
    print("\nRunning Markov engine …")
    engine = markov_engine_summary(P, verbose=True)
    pi = engine["stationary"]
    multi_init_results = engine["multi_init"]

    # plot_convergence(multi_init_results, engine["spectral"], engine["convergence_rate"], save_path="visualizations/convergence.png")
    # plot_stationary_distribution(pi, save_path="visualizations/stationary_distribution.png")
    # plot_eigenvalue_spectrum(engine["spectral"], save_path="visualizations/eigenvalue_spectrum.png")
    # plot_pagerank_comparison(engine["pagerank_cmp"], save_path="visualizations/pagerank_comparison.png")
    # plot_state_trajectories(multi_init_results, save_path="visualizations/state_trajectories.png")

    # Analysis
    print("\nRunning analysis …")
    durations = expected_durations(P)
    sub  = sub_period_analysis(regimes)
    findings = compile_findings(prices, regimes, P, verbose=True)

    # plot_sub_period_comparison(sub, save_path="visualizations/sub_period_comparison.png")
    # plot_expected_durations(durations, boot, save_path="visualizations/expected_durations.png")
    # plot_mfpt_heatmap(findings["mfpt"], save_path="visualizations/mfpt_heatmap.png")
    # plot_sensitivity_grid(findings["sensitivity"], save_path="visualizations/sensitivity_grid.png")
    # plot_order_test(findings["order_test"], save_path="visualizations/order_tes.png")
    # plot_regime_cycle(P, pi, durations, save_path="visualizations/regime_cycle.png")
    # plot_mfpt_financial(findings["mfpt"], save_path="visualizations/mfpt_financial.png")

    # Composite figures + README
    print(f"\n{sep}")
    print("Generating composite figures …")

    plot_overview_dashboard(
        prices, regimes, P, pi,
        engine["multi_init"],
        engine["spectral"],
        data_summary["frequencies"],
        save_path=VIZ_DIR / "dashboard_overview.png",
        show=False,
    )

    plot_mathematics_panel(
        engine["spectral"],
        findings["mfpt"],
        engine["pagerank_cmp"],
        save_path=VIZ_DIR / "dashboard_mathematics.png",
        show=False,
    )

    plot_critique_panel(
        findings["order_test"],
        findings["sensitivity"],
        sub,
        save_path=VIZ_DIR / "dashboard_critique.png",
        show=False,
    )
    
    print("\nWriting final README.md …")
    _write_final_readme(findings, engine, P, pi, durations, data_summary, boot, sub)

    # Summary
    elapsed = time.time() - t0
    print(f"\n{sep}")
    print(f"COMPLETE  ({elapsed:.1f}s)")
    print(f"  Figures → {VIZ_DIR.resolve()}/")
    print(sep)


def _write_final_readme(
    findings, engine, P, pi, durations, data_summary, boot, sub) -> None:
    from src.transition_matrix import STATES

    spec  = engine["spectral"]
    cmp   = engine["comparison"]
    mfpt  = findings["mfpt"]
    ot    = findings["order_test"]
    sens  = findings["sensitivity"]
    ex    = findings["exit"]
    pr    = engine["pagerank_cmp"]

    data_src = "SPY ETF adjusted close, Yahoo Finance, 2000–2024"

    # Format bootstrap CIs
    def ci_str(i, j):
        lo = boot["ci_lower"][i, j]
        hi = boot["ci_upper"][i, j]
        return f"[{lo:.3f}, {hi:.3f}]"

    readme = f"""# Financial Regime Transition Analysis

> **PageRank is a Markov chain. So is a market.**

This project demonstrates that the PageRank algorithm and financial regime modelling share identical mathematical foundations, both compute the **stationary distribution of a Markov chain via power iteration**, then apply that machinery to model S&P 500 market regime dynamics.

---

## Project Architecture

```
financial-regime-markov/
├── data/
├── notebooks/
│  └── exploration.ipynb
├── src/
│   ├── data_pipeline.py      # SPY data → Bull/Neutral/Bear regimes
│   ├── transition_matrix.py  # MLE estimation, Laplace smoothing, bootstrap CIs
│   ├── markov_engine.py      # power iteration, spectral analysis, PageRank comparison
│   ├── analysis.py           # MFPT, Markov order test, sensitivity analysis
│   └── visualize.py          # Publication-quality figures for each milestone
├── tests/                    # 150 unit tests across 4 modules
├── visualizations/           # All generated figures
├── main.py                   # ← Run everything with one command
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Real SPY data:
python main.py
```

---

## Mathematical Foundation

### PageRank = Stationary Distribution of a Markov Chain

PageRank and this project solve the **identical equation**:

$$\\pi^{{(t+1)}} = \\pi^{{(t)}} P$$

| Concept | PageRank | Market Regime Model |
|---------|----------|---------------------|
| States | Web pages | Bull / Neutral / Bear |
| Transitions | Hyperlink clicks | Empirical regime transitions |
| Transition matrix P | Web graph (column-stochastic) | Estimated from SPY data |
| Damping / regularisation | d = 0.85 (ergodicity guarantee) | Laplace-α smoothing |
| Stationary distribution | PageRank vector | Long-run regime probabilities |
| Algorithm | Power iteration | **Same algorithm** |

**Why the PageRank engine works on market data:** The Perron-Frobenius theorem guarantees that any positive, irreducible, row-stochastic matrix has a unique stationary distribution, and power iteration converges to it geometrically at rate $|\\lambda_2|$. Laplace smoothing plays the role of the damping factor, both ensure strict positivity, which guarantees ergodicity.

---

## Data and Regime Definition

**Source:** {data_src}

**Regime classification:**

$$\\text{{regime}}_t = \\begin{{cases}} \\text{{Bull}} & r_t^{{20}} > +2\\% \\\\ \\text{{Bear}} & r_t^{{20}} < -2\\% \\\\ \\text{{Neutral}} & \\text{{otherwise}} \\end{{cases}}$$

where $r_t^{{20}} = (P_t - P_{{t-20}}) / P_{{t-20}}$ is the trailing 20-day return.

**Note on overlapping observations:** consecutive rolling-window observations share 19 days of data, inflating apparent transition persistence. We address this by reporting a non-overlapping subsample and testing robustness explicitly.

---

## Results

### Transition Matrix

Estimated via maximum likelihood: $\\hat{{P}}_{{ij}} = n_{{ij}} / n_i$, then Laplace-smoothed with $\\alpha = 1$ (uniform Dirichlet prior).

|  | → Bull | → Neutral | → Bear |
|---|---|---|---|
| **Bull** | {P[0,0]:.4f}  {ci_str(0,0)} | {P[0,1]:.4f}  {ci_str(0,1)} | {P[0,2]:.4f}  {ci_str(0,2)} |
| **Neutral** | {P[1,0]:.4f}  {ci_str(1,0)} | {P[1,1]:.4f}  {ci_str(1,1)} | {P[1,2]:.4f}  {ci_str(1,2)} |
| **Bear** | {P[2,0]:.4f}  {ci_str(2,0)} | {P[2,1]:.4f}  {ci_str(2,1)} | {P[2,2]:.4f}  {ci_str(2,2)} |

*Values: point estimate  95% bootstrap CI (B=2,000)*

### Stationary Distribution

| Regime | $\\pi^*$ (model) | Empirical freq. | $E[\\text{{duration}}]$ |
|--------|----------------|----------------|----------------------|
| Bull    | {pi[0]:.4f} | {data_summary['frequencies'].get('Bull',0):.4f} | {durations['Bull']:.1f} days |
| Neutral | {pi[1]:.4f} | {data_summary['frequencies'].get('Neutral',0):.4f} | {durations['Neutral']:.1f} days |
| Bear    | {pi[2]:.4f} | {data_summary['frequencies'].get('Bear',0):.4f} | {durations['Bear']:.1f} days |

$\\pi^*$ and empirical frequencies agree closely — an internal consistency check on the estimation.

Expected duration formula: $E[d_i] = 1/(1 - P_{{ii}})$ — exact consequence of the geometric sojourn time distribution implied by the Markov property.

### Spectral Analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| $\\lambda_1$ | {spec['lambda1']:.8f} | Must equal 1.0 (Perron-Frobenius ✓) |
| $\\lambda_2$ | {spec['lambda2']:.6f} | Convergence rate per iteration |
| Spectral gap $1-\\lambda_2$ | {spec['spectral_gap']:.6f} | Larger = faster mixing |
| Iterations to converge | {cmp['n_iter']} | At tolerance $10^{{-9}}$ |
| $\\|\\pi_{{\\text{{power}}}} - \\pi_{{\\text{{eigen}}}}\\|_\\infty$ | {cmp['linf_diff']:.2e} | Both methods agree |

**PageRank comparison:** The PageRank damping factor $d=0.85$ forces $|\\lambda_2| \\leq 0.85$ by construction — a structural guarantee that the market model does not have. The market model's $|\\lambda_2| = {spec['lambda2']:.4f}$ reflects actual regime persistence in the data.

### Mean First Passage Times

$M_{{ij}}$ = expected trading days to reach regime $j$ from regime $i$ for the first time.

| From \\ To | Bull | Neutral | Bear |
|---|---|---|---|
| **Bull** | {mfpt['matrix'][0,0]:.1f} | {mfpt['matrix'][0,1]:.1f} | {mfpt['matrix'][0,2]:.1f} |
| **Neutral** | {mfpt['matrix'][1,0]:.1f} | {mfpt['matrix'][1,1]:.1f} | {mfpt['matrix'][1,2]:.1f} |
| **Bear** | {mfpt['matrix'][2,0]:.1f} | {mfpt['matrix'][2,1]:.1f} | {mfpt['matrix'][2,2]:.1f} |

Diagonal = mean return time $= 1/\\pi_i$ (exact theoretical result from ergodic Markov theory).

**Computation:** Each column solves the $(n-1) \\times (n-1)$ linear system $(I - P_{{-j,-j}}) \\mathbf{{m}} = \\mathbf{{1}}$.

**Bear recovery path:** Given a Bear exit, $P(\\text{{next}} = \\text{{Neutral}}) = {ex['bear_exit_probs'].get('Neutral', 0):.3f}$ — Bear markets almost always transition through Neutral before reaching Bull. The MFPT Bear → Bull ({mfpt['bear_to_bull']:.1f} days) includes this intermediate stop.

---

## Model Limitations

### 1. Markov Order Test

**H₀:** First-order Markov is sufficient.

$$\\Lambda = 2(\\ell_2 - \\ell_1) = {ot['test_statistic']:.2f} \\sim \\chi^2({ot['df']}) \\text{{ under H₀}}$$

**Result:** p = {ot['p_value']:.6f} → **{'REJECT H₀' if ot['reject_h0'] else 'FAIL TO REJECT H₀'}**

{ot['conclusion']}

The first-order model is used because it is the minimal structure to demonstrate the core mathematics. The natural extension — Hidden Markov Models — treats the regime as latent and can implicitly encode richer temporal dependence.

### 2. Sensitivity to Parameter Choices

| Parameter | Range tested | $\\pi_{{\\text{{Bear}}}}$ range |
|-----------|-------------|--------------------------|
| Window | 5, 10, 20, 60 days | [{sens['pi_bear_range'][0]:.3f}, {sens['pi_bear_range'][1]:.3f}] |
| Threshold | ±1%, ±2%, ±3% | (combined above) |

**Qualitative robustness:** {'YES — π_Bear ordering and regime structure are stable' if sens['is_robust'] else 'The absolute level of π_Bear is sensitive to parameter choice. The qualitative structure (Bull and Bear are persistent; Neutral is transient) holds across all combinations.'}

### 3. Non-Stationarity

Sub-period analysis (split: {sub['split_date'][:7]}) shows transition dynamics are not constant:
- Frobenius norm of $(P_{{\\text{{post}}}} - P_{{\\text{{pre}}}})$: **{sub['frobenius']:.4f}**
- Maximum element change: **{sub['max_abs_diff']:.4f}**

The model estimates average dynamics over the full period. Time-varying transition matrices (rolling re-estimation) would address this.

---

## Figures

### Overview Dashboard
![Overview Dashboard](visualizations/dashboard_overview.png)

### Mathematical Foundations
![Mathematics Panel](visualizations/dashboard_mathematics.png)

### Model Critique
![Critique Panel](visualizations/dashboard_critique.png)

---

## Testing

```bash
# Run all 150 tests:
python -m pytest tests/ -v

# Individual milestones:
python -m pytest tests/test_data_pipeline.py    # 22 tests
python -m pytest tests/test_transition_matrix.py # 25 tests
python -m pytest tests/test_markov_engine.py     # 53 tests
python -m pytest tests/test_analysis.py          # 50 tests
```

Key test categories:
- **Exact formulas:** MFPT diagonal $= 1/\\pi_i$, MFPT linear system, geometric duration formula
- **Statistical properties:** bootstrap CIs widen with shorter series; Laplace → uniform as $\\alpha \\to \\infty$
- **Edge cases:** absorbing states, i.i.d. sequences, zero-count rows, non-square matrices

---

## Extensions

1. **Hidden Markov Model** — treat regime as latent; infer via Baum-Welch (EM). Addresses the Markov order test rejection and overlapping-observation bias simultaneously.
2. **Time-varying $P$** — rolling-window re-estimation with exponential weighting of recent observations.
3. **Regime-conditioned factor analysis** — given the current regime posterior, compute expected return and volatility; directly applicable to risk overlay strategies.
4. **Continuous-state extension** — Ornstein-Uhlenbeck process for log-volatility; the stationary distribution becomes Gaussian rather than discrete.
"""

    Path("README.md").write_text(readme)
    print(f"  README.md written ({len(readme):,} characters)")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full project pipeline")
    parser.add_argument("--fast", action="store_true",
                        help="Use B=200 bootstrap (faster, less precise)")
    args = parser.parse_args()
    main(fast=args.fast)
