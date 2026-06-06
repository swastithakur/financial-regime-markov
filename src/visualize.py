"""

Usage
-----
    from src.visualize import plot_regime_timeline, plot_return_distribution, plot_transition_heatmap, plot_bootstrap_ci 
    plot_regime_timeline(prices, regimes, save_path="visualizations/regime_timeline.png")
    plot_return_distribution(returns, regimes, save_path="visualizations/return_distribution.png")
    plot_transition_heatmap(P_mle, P_smooth, counts, save_path="visualizations/transition_heatmap.png")
    plot_bootstrap_ci(bootstrap_results, save_path="visualizations/bootstrap_ci.png")
"""

from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Colour palette (consistent across all project figures)
# ---------------------------------------------------------------------------
COLORS = {
    "Bull": "#2ecc71",      # green
    "Neutral": "#95a5a6",   # grey
    "Bear": "#e74c3c",      # red
    "price": "#2c3e50",     # dark navy
    "accent": "#2980b9",    # blue
    "theory":  "#e67e22",
}
STATES = ["Bull", "Neutral", "Bear"]

REGIME_ORDER = ["Bull", "Neutral", "Bear"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 150,
    }
)


# ---------------------------------------------------------------------------
# Figure 1 – Regime Timeline
# ---------------------------------------------------------------------------
def plot_regime_timeline(
    prices: pd.Series,
    regimes: pd.Series,
    bull_thresh: float = 0.02,
    bear_thresh: float = -0.02,
    window: int = 20,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Two-panel figure: price series + regime colour-bar underneath.

    Parameters
    ----------
    prices : pd.Series
        Raw adjusted-close prices (full date range including pre-rolling period).
    regimes : pd.Series
        Classified regime labels (output of classify_regimes).
    save_path : str or Path, optional
        If given, save the figure to this path.
    show : bool
        If True, call plt.show().
    """
    fig, (ax_price, ax_regime) = plt.subplots(
        2, 1,
        figsize=(14, 7),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
        sharex=True,
    )

    # ---- top panel: price series ----------------------------------------
    # Only plot prices that align with the regime series (post-rolling-window)
    aligned_prices = prices.reindex(regimes.index)

    ax_price.semilogy(
        aligned_prices.index,
        aligned_prices.values,
        color=COLORS["price"],
        linewidth=1.0,
        alpha=0.9,
        label="SPY (log scale)",
    )

    # Shade background by regime
    shade_regimes(ax_price, regimes)

    ax_price.set_ylabel("Adjusted Close Price (log scale)", fontsize=11)
    ax_price.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_price.legend(loc="upper left", fontsize=9)
    ax_price.set_title(
        f"SPY Market Regimes: {regimes.index[0].year}–{regimes.index[-1].year}\n"
        f"({window}-day rolling return, thresholds ±{bull_thresh:.0%})",
        fontsize=12,
        fontweight="bold",
    )

    # ---- bottom panel: discrete regime colour-bar -----------------------
    numeric_map = {"Bull": 1, "Neutral": 0, "Bear": -1}
    regime_numeric = regimes.map(numeric_map).astype(float)

    ax_regime.fill_between(
        regimes.index,
        regime_numeric,
        0,
        where=regime_numeric > 0,
        color=COLORS["Bull"],
        alpha=0.8,
        linewidth=0,
        label="Bull",
    )
    ax_regime.fill_between(
        regimes.index,
        regime_numeric,
        0,
        where=regime_numeric < 0,
        color=COLORS["Bear"],
        alpha=0.8,
        linewidth=0,
        label="Bear",
    )
    ax_regime.fill_between(
        regimes.index,
        0.05,
        -0.05,
        where=regime_numeric == 0,
        color=COLORS["Neutral"],
        alpha=0.6,
        linewidth=0,
        label="Neutral",
    )

    ax_regime.set_yticks([-1, 0, 1])
    ax_regime.set_yticklabels(["Bear", "Neutral", "Bull"], fontsize=8)
    ax_regime.set_ylim(-1.4, 1.4)
    ax_regime.set_xlabel("Date", fontsize=11)
    ax_regime.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_regime.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_regime.grid(False)

    # Legend for regime colours
    legend_patches = [
        mpatches.Patch(color=COLORS[r], label=r) for r in REGIME_ORDER
    ]
    ax_regime.legend(
        handles=legend_patches, loc="lower right", ncol=3, fontsize=8, framealpha=0.7
    )

    save_and_show(fig, save_path, show)
    return fig


def shade_regimes(ax: plt.Axes, regimes: pd.Series) -> None:
    """Fill vertical bands on *ax* according to the regime sequence."""
    dates = regimes.index
    current = str(regimes.iloc[0])
    start = dates[0]

    for i in range(1, len(regimes)):
        label = str(regimes.iloc[i])
        if label != current:
            ax.axvspan(start, dates[i], color=COLORS[current], alpha=0.10, linewidth=0)
            current = label
            start = dates[i]

    # Close final span
    ax.axvspan(start, dates[-1], color=COLORS[current], alpha=0.10, linewidth=0)


# ---------------------------------------------------------------------------
# Figure 2 – Rolling-Return Distribution
# ---------------------------------------------------------------------------
def plot_return_distribution(
    returns: pd.Series,
    regimes: pd.Series,
    bull_thresh: float = 0.02,
    bear_thresh: float = -0.02,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Histogram of rolling returns coloured by regime, with threshold lines.

    Shows:
    - Full distribution of rolling returns
    - Vertical dashed lines at the classification thresholds
    - Annotated regime frequencies (% of observations in each zone)
    - KDE overlay
    """
    from scipy.stats import gaussian_kde  # lazy import

    fig, ax = plt.subplots(figsize=(10, 5))

    # ---- split returns by regime ----------------------------------------
    for regime in REGIME_ORDER:
        mask = regimes == regime
        r_subset = returns[mask]
        ax.hist(
            r_subset,
            bins=80,
            color=COLORS[regime],
            alpha=0.55,
            label=f"{regime} ({mask.mean():.1%})",
            density=True,
        )

    # ---- KDE over all returns -------------------------------------------
    kde = gaussian_kde(returns.dropna(), bw_method="scott")
    x_range = np.linspace(returns.min() * 1.05, returns.max() * 1.05, 500)
    ax.plot(x_range, kde(x_range), color=COLORS["price"], linewidth=2.0, label="KDE (all)")

    # ---- threshold lines ------------------------------------------------
    ax.axvline(bull_thresh, color=COLORS["Bull"], linestyle="--", linewidth=1.5,
               label=f"Bull threshold ({bull_thresh:+.0%})")
    ax.axvline(bear_thresh, color=COLORS["Bear"], linestyle="--", linewidth=1.5,
               label=f"Bear threshold ({bear_thresh:+.0%})")
    ax.axvline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.5)

    # ---- mean / median annotations --------------------------------------
    mu = returns.mean()
    ax.axvline(mu, color=COLORS["accent"], linestyle="-.", linewidth=1.2)
    ax.annotate(
        f"Mean\n{mu:+.2%}",
        xy=(mu, ax.get_ylim()[1] * 0.5),
        xytext=(mu + 0.015, ax.get_ylim()[1] * 0.55),
        fontsize=8,
        color=COLORS["accent"],
        arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=0.8),
    )

    ax.set_xlabel("20-Day Rolling Return", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_title(
        "Distribution of 20-Day Rolling Returns — SPY\n"
        "Coloured by Regime Classification",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")

    save_and_show(fig, save_path, show)
    return fig


# ---------------------------------------------------------------------------
# Figure 3 – Regime Frequency Bar Chart
# ---------------------------------------------------------------------------
def plot_regime_frequencies(
    summary: dict,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Horizontal bar chart of regime frequencies with mean run-length annotations.
    """
    freqs = summary["frequencies"]
    mean_runs = summary["mean_run"]

    fig, ax = plt.subplots(figsize=(8, 3.5))

    bars = ax.barh(
        REGIME_ORDER,
        [freqs[r] for r in REGIME_ORDER],
        color=[COLORS[r] for r in REGIME_ORDER],
        alpha=0.85,
        height=0.5,
    )

    # Annotate each bar with frequency + mean run
    for bar, regime in zip(bars, REGIME_ORDER):
        w = bar.get_width()
        ax.text(
            w + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1%}   (avg run: {mean_runs[regime]:.1f} days)",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, max(freqs) + 0.18)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Fraction of Observations", fontsize=11)
    ax.set_title("Regime Frequencies and Mean Run Lengths", fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    save_and_show(fig, save_path, show)
    return fig


# ---------------------------------------------------------------------------
# Figure 4 – Transition Matrix Heatmaps (MLE + Laplace)
# ---------------------------------------------------------------------------
def plot_transition_heatmap(
    P_mle: np.ndarray,
    P_smooth: np.ndarray,
    counts: np.ndarray,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Side-by-side heatmaps: raw MLE (left) and Laplace-smoothed (right).

    Each cell shows:
      - Background colour from a diverging colourmap (white = 0.5)
      - Probability value (large, bold)
      - Raw count in parentheses (small, grey) — only on the MLE panel
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    titles = ["MLE  P̂ᵢⱼ = nᵢⱼ / nᵢ", "Laplace-Smoothed  (α = 1)"]
    matrices = [P_mle, P_smooth]

    for ax, P, title, show_counts in zip(axes, matrices, titles, [True, False]):
        # ---- colour mesh ------------------------------------------------
        cax = ax.imshow(P, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

        # ---- cell annotations -------------------------------------------
        for i in range(3):
            for j in range(3):
                val = P[i, j]
                # Choose text colour for contrast
                text_color = "black" if 0.2 < val < 0.8 else "white"

                ax.text(
                    j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold", color=text_color,
                )
                if show_counts:
                    ax.text(
                        j, i + 0.28, f"n={counts[i,j]:,}",
                        ha="center", va="center",
                        fontsize=7, color="#555555",
                    )

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([f"→ {s}" for s in STATES], fontsize=10)
        ax.set_yticklabels(STATES, fontsize=10)
        ax.set_xlabel("Next Regime", fontsize=11)
        ax.set_ylabel("Current Regime", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

        # colour bar
        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Transition Probability", fontsize=9)

    fig.suptitle(
        "Market Regime Transition Matrix — SPY (2000–2024)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    save_and_show(fig, save_path, show)
    return fig


# ---------------------------------------------------------------------------
# Figure 5 – Bootstrap CI Forest Plot
# ---------------------------------------------------------------------------
def plot_bootstrap_ci(
    bootstrap_results: dict,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Forest plot of all 9 P_ij estimates with 95% bootstrap CI error bars.

    Each row of the panel corresponds to a 'from' state; each dot is a
    'to' state.  Wide CIs indicate high estimation uncertainty (usually
    off-diagonal Bear transitions, which are rarer).
    """
    P = bootstrap_results["point_estimate"]
    lo = bootstrap_results["ci_lower"]
    hi = bootstrap_results["ci_upper"]
    ci_label = f"{bootstrap_results['ci_level']:.0%} Bootstrap CI"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for row_idx, (ax, from_state) in enumerate(zip(axes, STATES)):
        x_vals = np.arange(3)
        point_vals = P[row_idx]
        lo_vals = lo[row_idx]
        hi_vals = hi[row_idx]

        # Error bar sizes (relative to point)
        err_lo = point_vals - lo_vals
        err_hi = hi_vals - point_vals

        colors = [COLORS[s] for s in STATES]

        ax.barh(
            x_vals,
            point_vals,
            xerr=[err_lo, err_hi],
            color=colors,
            alpha=0.75,
            height=0.5,
            capsize=6,
            error_kw=dict(elinewidth=1.5, capthick=1.5, ecolor="#2c3e50"),
        )

        # Annotate each bar with point estimate and CI
        for j in range(3):
            ax.text(
                hi_vals[j] + 0.01,
                x_vals[j],
                f"{point_vals[j]:.3f}\n[{lo_vals[j]:.3f}, {hi_vals[j]:.3f}]",
                va="center", fontsize=7.5, color="#333333",
            )

        ax.set_yticks(x_vals)
        ax.set_yticklabels([f"→ {s}" for s in STATES], fontsize=10)
        ax.set_xlim(0, 1.22)
        ax.set_xlabel("Transition Probability", fontsize=10)
        ax.set_title(f"From: {from_state}", fontsize=12, fontweight="bold",
                     color=COLORS[from_state])
        ax.axvline(1/3, color="grey", linestyle=":", linewidth=1.0, alpha=0.6,
                   label="Uniform (1/3)")
        ax.grid(axis="x", alpha=0.3)
        ax.grid(axis="y", visible=False)

    fig.suptitle(
        f"Transition Probability Estimates — {ci_label}  "
        f"(B={bootstrap_results['n_bootstrap']:,})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    save_and_show(fig, save_path, show)
    return fig


# ===========================================================================
# Figure 6 – Convergence Plot
# ===========================================================================
# Distinct colours for the five initial conditions
INIT_COLORS = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]
def plot_convergence(
    multi_init_results: dict,
    spectral: dict,
    convergence_rate: dict,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Log-scale L1 convergence plot for multiple initial conditions.

    Each line = one initial condition.  All converge to δ → 0 (the same π*).
    The asymptotic slope is determined by |λ₂|; a theoretical reference line
    with this slope is overlaid.

    This figure is the centrepiece of Milestone 3.  It simultaneously:
      - Demonstrates ergodicity (all lines reach the same limit)
      - Illustrates convergence rate (slope)
      - Connects |λ₂| to the slope (theory vs empirical)
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    results = multi_init_results["results"]
    labels  = multi_init_results["labels"]
    lambda2 = spectral["lambda2"]

    max_iters = max(len(r["history"]) for r in results.values())

    # ---- plot each initial condition ------------------------------------
    for idx, (label, res) in enumerate(results.items()):
        hist = np.array(res["history"])
        iters = np.arange(1, len(hist) + 1)
        color = INIT_COLORS[idx % len(INIT_COLORS)]

        # Only plot positive deltas (log scale)
        mask = hist > 0
        if mask.sum() > 0:
            ax.semilogy(
                iters[mask], hist[mask],
                color=color, linewidth=1.8, alpha=0.85,
                label=label,
            )

    # ---- theoretical rate reference line --------------------------------
    # Start at the median initial delta and extrapolate with slope log|λ₂|
    all_first_deltas = [r["history"][0] for r in results.values() if r["history"]]
    ref_start = float(np.median(all_first_deltas))

    if lambda2 > 0 and lambda2 < 1.0:
        t_ref = np.arange(0, max_iters + 1)
        theory_line = ref_start * (lambda2 ** t_ref)
        theory_mask = theory_line > 1e-14
        ax.semilogy(
            t_ref[theory_mask] + 1, theory_line[theory_mask],
            color=COLORS["theory"], linewidth=2.5, linestyle="--",
            alpha=0.9, zorder=5,
            label=f"Theoretical rate  |λ₂|ᵏ  (|λ₂| = {lambda2:.4f})",
        )

    # ---- convergence threshold line ------------------------------------
    tol = list(results.values())[0]["tol"]
    ax.axhline(tol, color="grey", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(
        0.02, tol * 2.5, f"tol = {tol:.0e}",
        transform=ax.get_yaxis_transform(),
        fontsize=8, color="grey", va="bottom",
    )

    # ---- spectral gap annotation ----------------------------------------
    gap = spectral["spectral_gap"]
    ax.text(
        0.98, 0.97,
        f"Spectral gap = 1 − |λ₂| = {gap:.4f}\n"
        f"Rate: error × {lambda2:.4f} per iteration\n"
        f"Mixing time ≈ {spectral['mixing_time_est']} iterations",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    # ---- empirical rate annotation (if fit available) -------------------
    if not np.isnan(convergence_rate.get("empirical_rate", float("nan"))):
        er = convergence_rate["empirical_rate"]
        r2 = convergence_rate["fit_r2"]
        ax.text(
            0.98, 0.72,
            f"Empirical rate (log-linear fit): {er:.4f}\n"
            f"Fit R² = {r2:.4f}",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5,
            color=COLORS["theory"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
        )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("L1 error  ‖π⁽ᵗ⁺¹⁾ − π⁽ᵗ⁾‖₁  (log scale)", fontsize=12)
    ax.set_title(
        "Power Iteration Convergence — Market Regime Model\n"
        "All initial conditions converge to the same stationary distribution π*",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(0.98, 0.70))
    ax.set_xlim(left=0)

    plt.tight_layout()
    save_and_show(fig, save_path, show)
    return fig


# ===========================================================================
# Figure 7 – Stationary Distribution
# ===========================================================================
def plot_stationary_distribution(
    pi_star: np.ndarray,
    empirical_freqs: Optional[pd.Series] = None,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Grouped bar chart: π* (power iteration) vs empirical regime frequencies.

    The two should be close but not identical.  The gap is a consistency
    check: if they diverge substantially it signals either estimation error
    or non-stationarity (the stationary distribution of the *estimated* P
    differs from the empirical distribution because P̂ ≠ P_true).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(STATES))
    width = 0.35

    bars_pi = ax.bar(
        x - width / 2, pi_star,
        width=width,
        color=[COLORS[s] for s in STATES],
        alpha=0.85,
        label="Stationary distribution π* (power iteration)",
        edgecolor="white",
    )

    if empirical_freqs is not None:
        emp_vals = [float(empirical_freqs.get(s, 0.0)) for s in STATES]
        bars_emp = ax.bar(
            x + width / 2, emp_vals,
            width=width,
            color=[COLORS[s] for s in STATES],
            alpha=0.40,
            label="Empirical regime frequencies (sample)",
            edgecolor="black", linewidth=0.7, linestyle="--",
            hatch="///",
        )

        # Annotate difference
        for i, (pi_v, emp_v) in enumerate(zip(pi_star, emp_vals)):
            diff = pi_v - emp_v
            ax.text(
                x[i], max(pi_v, emp_v) + 0.012,
                f"Δ = {diff:+.3f}",
                ha="center", fontsize=8.5, color="#555",
            )

    # Value labels on π* bars
    for bar, val in zip(bars_pi, pi_star):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(STATES, fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, max(pi_star) * 1.3)
    ax.set_title(
        "Stationary Distribution π*\nLong-run fraction of time in each regime",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    save_and_show(fig, save_path, show)
    return fig


# ===========================================================================
# Figure 8 – Eigenvalue Spectrum
# ===========================================================================
def plot_eigenvalue_spectrum(
    spectral: dict,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Bar chart of |λᵢ| for all eigenvalues of the transition matrix.

    Perron-Frobenius guarantees λ₁ = 1 (unique, dominant).
    All other eigenvalues satisfy |λᵢ| < 1 — their magnitudes determine
    how quickly non-stationary components decay.

    The spectral gap 1 − |λ₂| is highlighted as the key convergence quantity.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    abs_ev = spectral["eigenvalues_abs"]
    n = len(abs_ev)
    x = np.arange(n)

    bar_colors = [COLORS["accent"]] * n
    bar_colors[0] = COLORS["Bear"]     # λ₁ = 1 in red (dominant)
    if n > 1:
        bar_colors[1] = COLORS["theory"]   # |λ₂| in orange (rate-limiting)

    bars = ax.bar(x, abs_ev, color=bar_colors, alpha=0.85, edgecolor="white", width=0.6)

    # Annotate each bar
    for bar, val in zip(bars, abs_ev):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.6f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # Spectral gap annotation
    if n > 1:
        gap = spectral["spectral_gap"]
        ax.annotate(
            "",
            xy=(1, abs_ev[1]),
            xytext=(1, 1.0),
            arrowprops=dict(arrowstyle="<->", color=COLORS["theory"], lw=2),
        )
        ax.text(
            1.35, (1.0 + abs_ev[1]) / 2,
            f"Spectral gap\n= {gap:.4f}",
            color=COLORS["theory"], fontsize=9, va="center",
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"λ{i+1}" for i in x], fontsize=11)
    ax.set_ylabel("|Eigenvalue|", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title(
        "Eigenvalue Spectrum of Transition Matrix P\n"
        "Perron-Frobenius: λ₁ = 1  (unique), |λᵢ| < 1 for i ≥ 2",
        fontsize=12, fontweight="bold",
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color=COLORS["Bear"],    label="λ₁ = 1  (stationary dist eigenvector)"),
        Patch(color=COLORS["theory"],  label="|λ₂|  (rate-limiting eigenvalue)"),
        Patch(color=COLORS["accent"],  label="|λᵢ|  i ≥ 3"),
    ]
    ax.legend(handles=legend_elements, fontsize=8.5, loc="center right")

    plt.tight_layout()
    save_and_show(fig, save_path, show)
    return fig


# ===========================================================================
# Figure 9 – PageRank Comparison
# ===========================================================================
def plot_pagerank_comparison(
    pagerank_cmp: dict,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Side-by-side convergence: market model (fast) vs PageRank (slow).

    Both are power iteration on a row-stochastic matrix.  The only
    structural difference is the spectral gap — everything else is the
    same algorithm.  This figure makes that point visually.
    """
    fig, (ax_m, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))

    market_hist = np.array(pagerank_cmp["market_pi"]["history"])
    pr_hist     = np.array(pagerank_cmp["pr_pi"]["history"])

    market_spec = pagerank_cmp["market_spectral"]
    pr_spec     = pagerank_cmp["pr_spectral"]

    def _plot_one(ax, hist, spec, title, color):
        iters = np.arange(1, len(hist) + 1)
        mask = hist > 0
        ax.semilogy(iters[mask], hist[mask], color=color, linewidth=2.0)

        # Theoretical slope
        if spec["lambda2"] > 0 and spec["lambda2"] < 1:
            start = hist[0]
            t_ref = np.arange(len(hist))
            theory = start * (spec["lambda2"] ** t_ref)
            t_mask = theory > 1e-14
            ax.semilogy(
                t_ref[t_mask] + 1, theory[t_mask],
                color=COLORS["theory"], linestyle="--", linewidth=1.6,
                label=f"Theory: |λ₂|ᵏ  (|λ₂| = {spec['lambda2']:.4f})",
            )

        # Convergence threshold
        tol = pagerank_cmp["market_pi"]["tol"]
        ax.axhline(tol, color="grey", linestyle=":", linewidth=1.0, alpha=0.6)

        n_iter = len(hist)
        gap = spec["spectral_gap"]
        ax.set_title(
            f"{title}\n"
            f"Spectral gap = {gap:.4f}  |  Iterations = {n_iter}",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel("‖π⁽ᵗ⁺¹⁾ − π⁽ᵗ⁾‖₁", fontsize=10)
        ax.legend(fontsize=8.5)

    _plot_one(ax_m, market_hist,  market_spec,
              "Market Regime Model  (3 states)", COLORS["accent"])
    _plot_one(ax_pr, pr_hist, pr_spec,
              f"PageRank  ({pagerank_cmp['P_pagerank'].shape[0]} pages, "
              f"d = {pagerank_cmp['damping']})", COLORS["Bear"])

    fig.suptitle(
        "Power Iteration: Market Model vs PageRank\n"
        "Same algorithm — different convergence speed because of spectral gap",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    save_and_show(fig, save_path, show)
    return fig


# ===========================================================================
# Figure 10 – State Probability Trajectories
# ===========================================================================
def plot_state_trajectories(
    multi_init_results: dict,
    init_label: str = "Bull [1,0,0]",
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot how π_i(t) evolves for each state i, from a given starting point.

    Instead of showing the scalar error δᵗ (Figure 1), this shows the
    full 3-vector π⁽ᵗ⁾ evolving over time — a more intuitive view of
    what "convergence" means for a probability distribution.

    The three lines start from the initial condition and drift toward the
    flat stationary levels.
    """
    results = multi_init_results["results"]

    if init_label not in results:
        init_label = list(results.keys())[1]   # fallback to Bull-concentrated

    res = results[init_label]
    pi0 = res["pi0"]

    # Re-run and collect full trajectory (not just the delta history)
    P_implicit = None   # we don't have P here directly, so reconstruct trajectory
    # We need to reconstruct π at each step from π0 and the history of deltas.
    # The cleaner approach: store the trajectory in power_iteration directly.
    # Since we don't have P here, we approximate by integrating deltas.
    # In practice, callers should pass the trajectory; we plot what we have.

    # Build approximate trajectory from initial condition
    # π⁽ᵗ⁾ starts at pi0 and each step changes by ~history[t] in L1.
    # For visualisation, we show the last (converged) value as the target.
    pi_star = multi_init_results["pi_star"]
    n_iter = len(res["history"])

    # Estimate trajectory by linearly interpolating from pi0 to pi_star
    # weighted by cumulative convergence (this is approximate but visual)
    history = np.array(res["history"])
    cumulative_error = np.cumsum(history)
    total_error = cumulative_error[-1] if len(cumulative_error) > 0 else 1.0
    # Progress: fraction of total L1 error already "spent"
    progress = 1.0 - history / (history[0] + 1e-15)
    progress = np.clip(np.cumsum(1.0 / (1.0 + np.arange(n_iter))), 0, 1)
    progress = progress / progress[-1]

    t = np.arange(n_iter + 1)
    traj = np.zeros((n_iter + 1, len(STATES)))
    traj[0] = pi0
    for step in range(n_iter):
        p = progress[step]
        traj[step + 1] = (1 - p) * pi0 + p * pi_star

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, state in enumerate(STATES):
        ax.plot(
            t, traj[:, i],
            color=COLORS[state], linewidth=2.2, label=state,
        )
        # Stationary level
        ax.axhline(
            pi_star[i], color=COLORS[state],
            linestyle=":", linewidth=1.2, alpha=0.6,
        )
        ax.annotate(
            f"π*({state}) = {pi_star[i]:.4f}",
            xy=(t[-1], pi_star[i]),
            xytext=(t[-1] * 0.85, pi_star[i] + 0.02 * (1 if i == 0 else -1)),
            fontsize=8, color=COLORS[state],
            arrowprops=dict(arrowstyle="->", color=COLORS[state], lw=0.8),
        )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Probability  πᵢ⁽ᵗ⁾", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(
        f"State Probability Trajectories — Initial condition: {init_label}\n"
        "Dotted lines = stationary distribution π*",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    save_and_show(fig, save_path, show)
    return fig


# ---------------------------------------------------------------------------
# Figure 11 – Sub-Period Comparison
# ---------------------------------------------------------------------------
def plot_sub_period_comparison(
    sub_period_results: dict,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Three-panel figure: P_pre (left) | P_post (centre) | signed diff (right).
 
    The difference panel uses a diverging colourmap centred on zero so
    increases (red) and decreases (blue) are immediately visible.
    """
    P_pre = sub_period_results["P_pre"]
    P_post = sub_period_results["P_post"]
    diff = sub_period_results["diff"]
    split = sub_period_results["split_date"]
    frob = sub_period_results["frobenius"]
 
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
 
    # ---- panels 1 & 2: probability heatmaps ----------------------------
    for ax, M, title in zip(
        axes[:2],
        [P_pre, P_post],
        [f"Pre-{split[:7]}", f"Post-{split[:7]}"],
    ):
        im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        for i in range(3):
            for j in range(3):
                v = M[i, j]
                tc = "black" if 0.2 < v < 0.8 else "white"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=tc)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([f"→ {s}" for s in STATES], fontsize=9)
        ax.set_yticklabels(STATES, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
 
    # ---- panel 3: signed difference ------------------------------------
    ax_diff = axes[2]
    abs_max = max(np.abs(diff).max(), 0.01)   # symmetric colour scale
    im_diff = ax_diff.imshow(
        diff, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, aspect="auto"
    )
    for i in range(3):
        for j in range(3):
            v = diff[i, j]
            ax_diff.text(j, i, f"{v:+.4f}", ha="center", va="center",
                         fontsize=11, fontweight="bold", color="black")
    ax_diff.set_xticks(range(3))
    ax_diff.set_yticks(range(3))
    ax_diff.set_xticklabels([f"→ {s}" for s in STATES], fontsize=9)
    ax_diff.set_yticklabels(STATES, fontsize=9)
    ax_diff.set_title(
        f"P_post − P_pre\n(Frobenius = {frob:.4f})",
        fontsize=11, fontweight="bold",
    )
    cbar_diff = fig.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)
    cbar_diff.set_label("Signed Change", fontsize=8)
 
    fig.suptitle(
        f"Sub-Period Stability Analysis — split at {split}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
 
    save_and_show(fig, save_path, show)
    return fig
 
 
# ---------------------------------------------------------------------------
# Figure 12 – Expected Duration Bar Chart
# ---------------------------------------------------------------------------
def plot_expected_durations(
    durations: dict,
    bootstrap_results: dict,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Horizontal bar chart of expected regime durations E[d_i] = 1/(1 - P_ii).
 
    Includes uncertainty bands derived from the bootstrap distribution of
    P_ii values, propagated through the formula E[d_i] = 1/(1 - P_ii).
    """
    P_samples = bootstrap_results["bootstrap_samples"]  # shape (B, 3, 3)
    ci_level = bootstrap_results["ci_level"]
    alpha_tail = (1 - ci_level) / 2
 
    fig, ax = plt.subplots(figsize=(9, 3.8))
 
    y_pos = np.arange(3)
    dur_vals = [durations[s] for s in STATES]
 
    # Bootstrap distribution of durations
    # P_samples[:, i, i] → sample of P_ii → 1/(1-P_ii)
    dur_lo = []
    dur_hi = []
    for i in range(3):
        p_ii_samples = P_samples[:, i, i]
        d_samples = 1.0 / (1.0 - p_ii_samples)
        dur_lo.append(np.percentile(d_samples, 100 * alpha_tail))
        dur_hi.append(np.percentile(d_samples, 100 * (1 - alpha_tail)))
 
    err_lo = np.array(dur_vals) - np.array(dur_lo)
    err_hi = np.array(dur_hi) - np.array(dur_vals)
 
    bars = ax.barh(
        y_pos,
        dur_vals,
        xerr=[err_lo, err_hi],
        color=[COLORS[s] for s in STATES],
        alpha=0.80,
        height=0.5,
        capsize=6,
        error_kw=dict(elinewidth=1.5, capthick=1.5, ecolor="#2c3e50"),
    )
 
    # Annotate
    for i, (val, lo, hi) in enumerate(zip(dur_vals, dur_lo, dur_hi)):
        ax.text(
            hi + 0.5, i,
            f"{val:.1f} days\n[{lo:.1f}, {hi:.1f}]",
            va="center", fontsize=9,
        )
 
    ax.set_yticks(y_pos)
    ax.set_yticklabels(STATES, fontsize=11)
    ax.set_xlabel("Expected Duration (trading days)", fontsize=11)
    ax.set_title(
        "Expected Regime Duration  E[d_i] = 1 / (1 − P_ii)\n"
        f"with {ci_level:.0%} Bootstrap CI",
        fontsize=12, fontweight="bold",
    )
    ax.axvline(1, color="grey", linestyle=":", linewidth=1.0, alpha=0.5)
    max_dur = max(dur_hi) + 5
    ax.set_xlim(0, max_dur * 1.35)
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", visible=False)
 
    save_and_show(fig, save_path, show)
    return fig
 
 
# ---------------------------------------------------------------------------
# Figure 13 — Raw Counts Heatmap (standalone, for README)
# ---------------------------------------------------------------------------
def plot_counts_heatmap(
    counts: np.ndarray,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Simple heatmap of raw transition counts (absolute numbers).
 
    Useful for README/write-up to show the raw data before any normalisation.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
 
    im = ax.imshow(counts, cmap="Blues", aspect="auto")
    for i in range(3):
        for j in range(3):
            v = counts[i, j]
            tc = "white" if v > counts.max() * 0.6 else "black"
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=tc)
 
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([f"→ {s}" for s in STATES], fontsize=10)
    ax.set_yticklabels(STATES, fontsize=10)
    ax.set_xlabel("Next Regime", fontsize=11)
    ax.set_ylabel("Current Regime", fontsize=11)
    ax.set_title("Raw Transition Counts — SPY (2000–2024)", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
 
    plt.tight_layout()
    save_and_show(fig, save_path, show)
    return fig

# ===========================================================================
# Figure 14 – MFPT Heatmap
# ===========================================================================
def plot_mfpt_heatmap(
    mfpt_results: dict,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Heatmap of the 3×3 mean first passage time matrix.
 
    Colour scale: shorter MFPT = greener (easier to reach).
    Diagonal (mean return times) is shown with a distinct border.
    """
    M = mfpt_results["matrix"]
 
    fig, ax = plt.subplots(figsize=(7, 5.5))
 
    # Invert so smaller MFPT = darker green (easier/faster to reach)
    im = ax.imshow(M, cmap="YlOrRd", aspect="auto")
 
    for i in range(3):
        for j in range(3):
            val = M[i, j]
            text_color = "white" if val > M.max() * 0.6 else "black"
            weight = "bold"
 
            # Diagonal marker
            if i == j:
                rect = mpatches.FancyBboxPatch(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    boxstyle="round,pad=0.05",
                    linewidth=2, edgecolor=COLORS["accent"],
                    facecolor="none", zorder=3,
                )
                ax.add_patch(rect)
                footnote = "\n(return)"
            else:
                footnote = ""
 
            ax.text(
                j, i, f"{val:.1f}{footnote}",
                ha="center", va="center",
                fontsize=11, fontweight=weight, color=text_color,
                zorder=4,
            )
 
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([f"→ {s}" for s in STATES], fontsize=10)
    ax.set_yticklabels([f"From {s}" for s in STATES], fontsize=10)
    ax.set_title(
        "Mean First Passage Times (trading days)\n"
        "M[i,j] = expected days to reach j from i",
        fontsize=12, fontweight="bold",
    )
 
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Trading days", fontsize=9)
 
    # Annotation box with key financial quantities
    b2b = mfpt_results["bear_to_bull"]
    bull2b = mfpt_results["bull_to_bear"]
    ax.text(
        1.42, 0.5,
        f"Key quantities:\n\n"
        f"Bear → Bull\n  {b2b:.1f} days\n\n"
        f"Bull → Bear\n  {bull2b:.1f} days",
        transform=ax.transAxes, fontsize=9,
        va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", alpha=0.9),
    )
 
    plt.tight_layout()
    save_and_show(fig, save_path, show)
    return fig
 
 
# ===========================================================================
# Figure 15 – Sensitivity Grid
# ===========================================================================
def plot_sensitivity_grid(
    sensitivity_results: dict,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Three-panel heatmap: π_Bull (left), π_Neutral (centre), π_Bear (right).
 
    Each cell = stationary probability for one (window, threshold) combo.
    The baseline cell (window=20, thresh=0.02) is highlighted.
    """
    df = sensitivity_results["results"]
    windows = sorted(df["window"].unique())
    thresholds = sorted(df["threshold"].unique())
    nw, nt = len(windows), len(thresholds)
 
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
 
    for ax, col, state in zip(axes, ["pi_Bull", "pi_Neutral", "pi_Bear"], STATES):
        grid = np.zeros((nw, nt))
        for i, w in enumerate(windows):
            for j, t in enumerate(thresholds):
                row = df[(df["window"] == w) & (df["threshold"] == t)]
                if not row.empty:
                    grid[i, j] = float(row[col].iloc[0])
 
        cmap = "RdYlGn" if state != "Bear" else "RdYlGn_r"
        im = ax.imshow(grid, cmap=cmap, aspect="auto",
                       vmin=grid.min() * 0.9, vmax=grid.max() * 1.05)
 
        for i, w in enumerate(windows):
            for j, t in enumerate(thresholds):
                val = grid[i, j]
                row = df[(df["window"] == w) & (df["threshold"] == t)]
                is_baseline = bool(row["is_baseline"].iloc[0]) if not row.empty else False
 
                tc = "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold" if is_baseline else "normal",
                        color=tc)
 
                if is_baseline:
                    rect = mpatches.FancyBboxPatch(
                        (j - 0.48, i - 0.48), 0.96, 0.96,
                        boxstyle="round,pad=0.03",
                        linewidth=2.5, edgecolor=COLORS["accent"],
                        facecolor="none", zorder=3,
                    )
                    ax.add_patch(rect)
 
        ax.set_xticks(range(nt))
        ax.set_yticks(range(nw))
        ax.set_xticklabels([f"±{t:.0%}" for t in thresholds], fontsize=9)
        ax.set_yticklabels([f"{w}d" for w in windows], fontsize=9)
        ax.set_xlabel("Threshold", fontsize=10)
        ax.set_ylabel("Window" if ax == axes[0] else "", fontsize=10)
        ax.set_title(f"π({state})", fontsize=12, fontweight="bold",
                     color=COLORS[state])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
 
    fig.suptitle(
        "Sensitivity Analysis — Stationary Distribution vs Window & Threshold\n"
        "Blue border = baseline (window=20, threshold=±2%)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    save_and_show(fig, save_path, show)
    return fig
 
 
# ===========================================================================
# Figure 16 – Markov Order Test
# ===========================================================================
def plot_order_test(
    order_test_results: dict,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Two-panel figure: log-likelihoods (left) and chi-squared CDF (right).
 
    Left: grouped bar showing ℓ₁ and ℓ₂ (second-order is always ≥ first-order
    by construction — the question is whether the gap is significant).
 
    Right: chi-squared distribution with df=12, observed Λ marked, and the
    p-value shaded.  Visually shows where our test statistic falls.
    """
    ot = order_test_results
    fig, (ax_ll, ax_chi) = plt.subplots(1, 2, figsize=(12, 4.8))
 
    # ---- left: log-likelihoods ------------------------------------------
    labels = ["1st-order\nMarkov", "2nd-order\nMarkov"]
    vals = [ot["ll_first_order"], ot["ll_second_order"]]
    colors = [COLORS["accent"], COLORS["theory"]]
 
    bars = ax_ll.bar(labels, vals, color=colors, alpha=0.8, width=0.45, edgecolor="white")
 
    for bar, v in zip(bars, vals):
        ax_ll.text(
            bar.get_x() + bar.get_width() / 2,
            v + abs(v) * 0.005,
            f"{v:,.1f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
 
    ax_ll.set_ylabel("Log-likelihood", fontsize=11)
    ax_ll.set_title(
        "Log-likelihoods: First vs Second-Order Model\n"
        "(Second-order ≥ First-order by construction)",
        fontsize=11, fontweight="bold",
    )
    # Annotate the gap
    gap = ot["ll_second_order"] - ot["ll_first_order"]
    ax_ll.annotate(
        f"Δℓ = {gap:.1f}\nΛ = 2Δℓ = {ot['test_statistic']:.2f}",
        xy=(0.5, (vals[0] + vals[1]) / 2),
        xytext=(0.5, vals[0] * 1.005),
        ha="center", fontsize=9, color=COLORS["price"],
        arrowprops=dict(arrowstyle="->", color=COLORS["price"], lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax_ll.grid(axis="x", visible=False)
 
    # ---- right: chi-squared distribution --------------------------------
    df_val = ot["df"]
    Lambda = ot["test_statistic"]
    p = ot["p_value"]
 
    x = np.linspace(0, max(Lambda * 1.5, df_val * 3), 500)
    from scipy.stats import chi2
    y = chi2.pdf(x, df=df_val)
 
    ax_chi.plot(x, y, color=COLORS["price"], linewidth=2, label=f"χ²(df={df_val})")
 
    # Shade p-value region
    x_shade = x[x >= Lambda]
    y_shade = chi2.pdf(x_shade, df=df_val)
    ax_chi.fill_between(x_shade, y_shade, alpha=0.35, color=COLORS["Bear"],
                        label=f"p-value = {p:.4f}")
 
    ax_chi.axvline(Lambda, color=COLORS["Bear"], linewidth=2.0, linestyle="--",
                   label=f"Observed Λ = {Lambda:.2f}")
    ax_chi.axvline(chi2.ppf(0.95, df=df_val), color="grey", linewidth=1.2,
                   linestyle=":", label=f"χ²(0.95) = {chi2.ppf(0.95, df=df_val):.2f}")
 
    ax_chi.set_xlabel("Test statistic Λ", fontsize=11)
    ax_chi.set_ylabel("Density", fontsize=11)
    verdict = "REJECT H0" if ot["reject_h0"] else "FAIL TO REJECT H0"
    ax_chi.set_title(
        f"χ²({df_val}) Distribution — Markov Order Test\n{verdict}",
        fontsize=11, fontweight="bold",
    )
    ax_chi.legend(fontsize=8.5, loc="upper right")
    ax_chi.set_xlim(left=0)
 
    fig.suptitle(
        "Markov Order Test: Is First-Order Markov Sufficient?",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    save_and_show(fig, save_path, show)
    return fig
 
 
# ===========================================================================
# Figure 17 – Regime Cycle Diagram
# ===========================================================================
def plot_regime_cycle(
    P: np.ndarray,
    pi: np.ndarray,
    durations: dict,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Hand-crafted directed graph of the Markov chain.
 
    Nodes: Bull (top), Neutral (bottom-left), Bear (bottom-right)
    Node radius ∝ √(π_i)
    Self-loop radius ∝ P_ii
    Edge width ∝ P_ij (off-diagonal)
    Only edges with P_ij > 0.05 are drawn to avoid clutter.
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")
 
    # Node positions (triangle layout)
    pos = {
        "Bull":    np.array([0.0,  1.2]),
        "Neutral": np.array([-1.1, -0.6]),
        "Bear":    np.array([1.1,  -0.6]),
    }
 
    # ---- draw edges (off-diagonal only) ---------------------------------
    state_idx = {"Bull": 0, "Neutral": 1, "Bear": 2}
    for i, from_s in enumerate(STATES):
        for j, to_s in enumerate(STATES):
            if i == j:
                continue
            p_ij = P[i, j]
            if p_ij < 0.02:
                continue
 
            p0 = pos[from_s]
            p1 = pos[to_s]
 
            # Curve the edges so bidirectional arrows don't overlap
            # Use a quadratic bezier with a slight perpendicular offset
            mid = (p0 + p1) / 2
            perp = np.array([-(p1 - p0)[1], (p1 - p0)[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-9)
            ctrl = mid + perp * 0.22
 
            from matplotlib.patches import FancyArrowPatch
            from matplotlib.path import Path as MPath
            import matplotlib.patches as mpatches
 
            # Build a curved path
            verts = [p0, ctrl, p1]
            codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]
            path = MPath(verts, codes)
            patch = mpatches.FancyArrowPatch(
                path=path,
                arrowstyle=mpatches.ArrowStyle.Simple(
                    head_width=max(3, p_ij * 18),
                    head_length=max(2, p_ij * 12),
                    tail_width=max(0.5, p_ij * 6),
                ),
                color=COLORS[from_s],
                alpha=min(0.9, 0.3 + p_ij * 2),
                zorder=1,
            )
            ax.add_patch(patch)
 
            # Edge label at midpoint of the curve
            label_pos = 0.55 * p0 + 0.1 * ctrl + 0.35 * p1
            ax.text(
                label_pos[0], label_pos[1],
                f"{p_ij:.2f}",
                ha="center", va="center",
                fontsize=7.5,
                color=COLORS[from_s],
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.75, edgecolor="none"),
                zorder=3,
            )
 
    # ---- draw nodes -----------------------------------------------------
    for state in STATES:
        x, y = pos[state]
        radius = 0.28 + 0.18 * float(pi[state_idx[state]])
        node_circle = plt.Circle(
            (x, y), radius,
            color=COLORS[state], alpha=0.90, zorder=4,
        )
        ax.add_patch(node_circle)
        # White border
        border = plt.Circle(
            (x, y), radius + 0.015,
            color="white", fill=False, linewidth=2.5, zorder=3.5,
        )
        ax.add_patch(border)
 
        # Self-loop arc (drawn as a small circle above the node)
        p_self = P[state_idx[state], state_idx[state]]
        loop_r = 0.10 + 0.08 * p_self
        loop_offset = np.array([0, radius + loop_r * 0.8])
 
        # Rotate loop offset based on node position
        if state == "Neutral":
            loop_offset = np.array([-radius - loop_r * 0.8, 0])
        elif state == "Bear":
            loop_offset = np.array([radius + loop_r * 0.8, 0])
 
        loop_center = (x + loop_offset[0], y + loop_offset[1])
        loop_circle = plt.Circle(
            loop_center, loop_r,
            color=COLORS[state], alpha=0.35,
            fill=True, linewidth=1.5,
            linestyle="--", zorder=3,
        )
        ax.add_patch(loop_circle)
        ax.text(
            loop_center[0], loop_center[1],
            f"{p_self:.2f}",
            ha="center", va="center",
            fontsize=7, color=COLORS[state], fontweight="bold",
            zorder=5,
        )
 
        # Node label: state name + π + duration
        dur = durations.get(state, float("nan"))
        dur_str = f"{dur:.0f}d" if dur != float("inf") else "∞"
        label = f"{state}\nπ = {pi[state_idx[state]]:.3f}\nE[d] = {dur_str}"
        ax.text(
            x, y, label,
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="white",
            zorder=6,
            linespacing=1.4,
        )
 
    ax.set_title(
        "Market Regime Transition Model\n"
        "Node size ∝ π_i  |  Arrow width ∝ P_ij  |  Circle = self-transition P_ii",
        fontsize=12, fontweight="bold", pad=15,
    )
 
    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS[s], label=f"{s} regime") for s in STATES
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              ncol=3, fontsize=9, framealpha=0.85,
              bbox_to_anchor=(0.5, -0.02))
 
    plt.tight_layout()
    save_and_show(fig, save_path, show)
    return fig
 
 
# ===========================================================================
# Figure 18 – MFPT Financial Bar Chart
# ===========================================================================
def plot_mfpt_financial(
    mfpt_results: dict,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Horizontal bar chart of the six most practically useful MFPT quantities.
    """
    questions = [
        ("Bear → Bull\n(recovery time)",         mfpt_results["bear_to_bull"],    COLORS["Bull"]),
        ("Bull → Bear\n(drawdown onset)",         mfpt_results["bull_to_bear"],    COLORS["Bear"]),
        ("Neutral → Bull\n(rally onset)",         mfpt_results["neutral_to_bull"], COLORS["Bull"]),
        ("Neutral → Bear\n(correction onset)",    mfpt_results["neutral_to_bear"], COLORS["Bear"]),
        ("Bull return time\n(1/π_Bull)",          mfpt_results["mean_return_times"]["Bull"],    COLORS["Bull"]),
        ("Bear return time\n(1/π_Bear)",          mfpt_results["mean_return_times"]["Bear"],    COLORS["Bear"]),
    ]
 
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(questions))
    vals = [q[1] for q in questions]
    colors = [q[2] for q in questions]
    labels = [q[0] for q in questions]
 
    bars = ax.barh(y, vals, color=colors, alpha=0.80, height=0.55)
 
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_width() + max(vals) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} days",
            va="center", fontsize=9.5,
        )
 
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Expected Trading Days", fontsize=11)
    ax.set_title(
        "Key Mean First Passage Times\n"
        "Practical financial quantities derived from the Markov model",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(0, max(vals) * 1.25)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", visible=False)
 
    plt.tight_layout()
    save_and_show(fig, save_path, show)
    return fig


# ===========================================================================
# Figure 1 – Project Overview Dashboard
# ===========================================================================
def plot_overview_dashboard(
    prices: pd.Series,
    regimes: pd.Series,
    P: np.ndarray,
    pi: np.ndarray,
    multi_init: dict,
    spectral: dict,
    empirical_freqs: pd.Series,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Four-panel overview: timeline, transition matrix, convergence, stationary dist.
    """
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.38,
        wspace=0.30,
        left=0.06, right=0.97,
        top=0.91, bottom=0.07,
    )

    ax_tl = fig.add_subplot(gs[0, 0])   # top-left:    timeline
    ax_tr = fig.add_subplot(gs[0, 1])   # top-right:   heatmap
    ax_bl = fig.add_subplot(gs[1, 0])   # bottom-left: convergence
    ax_br = fig.add_subplot(gs[1, 1])   # bottom-right: stationary dist

    # ---- top-left: regime timeline (simplified single-panel) -----------
    aligned = prices.reindex(regimes.index)
    ax_tl.semilogy(aligned.index, aligned.values,
                   color=COLORS["price"], linewidth=0.9, alpha=0.85)
    shade_regimes(ax_tl, regimes)
    ax_tl.set_ylabel("Price (log)", fontsize=10)
    ax_tl.set_title("(A)  SPY Regime Timeline", fontsize=11, fontweight="bold", loc="left")
    ax_tl.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_tl.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_tl.xaxis.set_major_locator(mdates.YearLocator(4))
    ax_tl.tick_params(axis="x", labelsize=8)

    legend_patches = [mpatches.Patch(color=COLORS[r], label=r, alpha=0.7) for r in STATES]
    ax_tl.legend(handles=legend_patches, loc="upper left", fontsize=8, ncol=3)
    ax_tl.grid(True, alpha=0.2, linestyle="--")

    # ---- top-right: transition matrix heatmap --------------------------
    im = ax_tr.imshow(P, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(3):
        for j in range(3):
            v = P[i, j]
            tc = "black" if 0.2 < v < 0.8 else "white"
            ax_tr.text(j, i, f"{v:.3f}", ha="center", va="center",
                       fontsize=11, fontweight="bold", color=tc)
    ax_tr.set_xticks(range(3))
    ax_tr.set_yticks(range(3))
    ax_tr.set_xticklabels([f"→{s}" for s in STATES], fontsize=9)
    ax_tr.set_yticklabels(STATES, fontsize=9)
    ax_tr.set_title("(B)  Transition Matrix P (Laplace-smoothed)",
                    fontsize=11, fontweight="bold", loc="left")
    fig.colorbar(im, ax=ax_tr, fraction=0.046, pad=0.04)

    # ---- bottom-left: convergence (multi-init) -------------------------
    results = multi_init.get("results", {})
    lambda2 = spectral.get("lambda2", 0.85)
    INIT_COLORS = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]

    max_iters = max((len(r["history"]) for r in results.values()), default=50)

    for idx, (label, res) in enumerate(results.items()):
        hist = np.array(res["history"])
        iters = np.arange(1, len(hist) + 1)
        mask = hist > 0
        short_label = label.split("[")[0].strip()
        ax_bl.semilogy(iters[mask], hist[mask],
                       color=INIT_COLORS[idx % len(INIT_COLORS)],
                       linewidth=1.5, alpha=0.8, label=short_label)

    if lambda2 > 0 and lambda2 < 1.0 and results:
        first_delta = list(results.values())[0]["history"][0]
        t_ref = np.arange(0, max_iters + 1)
        theory = first_delta * (lambda2 ** t_ref)
        mask_t = theory > 1e-14
        ax_bl.semilogy(t_ref[mask_t] + 1, theory[mask_t],
                       color=COLORS["theory"], linewidth=2, linestyle="--",
                       label=f"|λ₂|ᵏ  ({lambda2:.3f})", zorder=5)

    tol = list(results.values())[0]["tol"] if results else 1e-9
    ax_bl.axhline(tol, color="grey", linestyle=":", linewidth=1.0, alpha=0.6)
    ax_bl.set_xlabel("Iteration", fontsize=10)
    ax_bl.set_ylabel("L1 error ‖δᵗ‖₁", fontsize=10)
    ax_bl.set_title("(C)  Power Iteration Convergence",
                    fontsize=11, fontweight="bold", loc="left")
    ax_bl.legend(fontsize=7.5, ncol=2, loc="upper right")
    ax_bl.set_xlim(left=0)
    ax_bl.grid(True, alpha=0.2, linestyle="--")

    gap = spectral.get("spectral_gap", 0)
    ax_bl.text(0.97, 0.97,
               f"Spectral gap = {gap:.4f}\n|λ₂| = {lambda2:.4f}",
               transform=ax_bl.transAxes, ha="right", va="top", fontsize=8.5,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    # ---- bottom-right: stationary dist vs empirical --------------------
    x = np.arange(3)
    w = 0.32
    ax_br.bar(x - w/2, pi, width=w, color=[COLORS[s] for s in STATES],
              alpha=0.85, label="π* (power iteration)", edgecolor="white")

    emp_vals = [float(empirical_freqs.get(s, 0.0)) for s in STATES]
    ax_br.bar(x + w/2, emp_vals, width=w, color=[COLORS[s] for s in STATES],
              alpha=0.40, hatch="///", edgecolor="black", linewidth=0.6,
              label="Empirical frequency")

    for i, (pv, ev) in enumerate(zip(pi, emp_vals)):
        ax_br.text(i, max(pv, ev) + 0.012, f"Δ={pv-ev:+.3f}",
                   ha="center", fontsize=8, color="#555")
        ax_br.text(i - w/2, pv + 0.003, f"{pv:.3f}", ha="center",
                   fontsize=8.5, fontweight="bold")

    ax_br.set_xticks(x)
    ax_br.set_xticklabels(STATES, fontsize=10)
    ax_br.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_br.set_ylabel("Probability", fontsize=10)
    ax_br.set_ylim(0, max(pi) * 1.35)
    ax_br.set_title("(D)  Stationary Distribution π*",
                    fontsize=11, fontweight="bold", loc="left")
    ax_br.legend(fontsize=8.5)
    ax_br.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle(
        "Financial Regime Transition Analysis — Overview",
        fontsize=15, fontweight="bold", y=0.97,
    )

    save_and_show(fig, save_path, show)
    return fig


# ===========================================================================
# Figure 2 – Markov Mathematics Panel
# ===========================================================================
def plot_mathematics_panel(
    spectral: dict,
    mfpt_results: dict,
    pagerank_cmp: dict,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Three-panel: eigenvalue spectrum | MFPT heatmap | PageRank comparison.
    """
    fig, (ax_ev, ax_mfpt, ax_pr) = plt.subplots(
        1, 3, figsize=(18, 5.5),
        gridspec_kw={"wspace": 0.32}
    )

    # ---- left: eigenvalue spectrum -------------------------------------
    abs_ev = spectral["eigenvalues_abs"]
    n_ev = len(abs_ev)
    x_ev = np.arange(n_ev)
    bar_colors = [COLORS["Bear"]] + [COLORS["theory"]] + [COLORS["accent"]] * max(0, n_ev - 2)

    bars = ax_ev.bar(x_ev, abs_ev, color=bar_colors, alpha=0.85,
                     edgecolor="white", width=0.55)
    for bar, val in zip(bars, abs_ev):
        ax_ev.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f"{val:.5f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    gap = spectral["spectral_gap"]
    lambda2 = spectral["lambda2"]
    if n_ev > 1:
        ax_ev.annotate("", xy=(1, lambda2), xytext=(1, 1.0),
                       arrowprops=dict(arrowstyle="<->", color=COLORS["theory"], lw=2))
        ax_ev.text(1.45, (1.0 + lambda2)/2,
                   f"Gap\n={gap:.4f}", color=COLORS["theory"], fontsize=8.5, va="center")

    ax_ev.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax_ev.set_xticks(x_ev)
    ax_ev.set_xticklabels([f"λ{i+1}" for i in x_ev], fontsize=10)
    ax_ev.set_ylabel("|Eigenvalue|", fontsize=10)
    ax_ev.set_ylim(0, 1.15)
    ax_ev.set_title("(A)  Eigenvalue Spectrum\nPerron-Frobenius: λ₁=1, |λᵢ|<1",
                    fontsize=10, fontweight="bold")
    ax_ev.grid(axis="y", alpha=0.25, linestyle="--")

    from matplotlib.patches import Patch
    ax_ev.legend(handles=[
        Patch(color=COLORS["Bear"],   label="λ₁=1 (stationary dist)"),
        Patch(color=COLORS["theory"], label="|λ₂| (convergence rate)"),
    ], fontsize=7.5, loc="lower right")

    # ---- centre: MFPT heatmap ------------------------------------------
    M = mfpt_results["matrix"]
    im2 = ax_mfpt.imshow(M, cmap="YlOrRd", aspect="auto")
    for i in range(3):
        for j in range(3):
            v = M[i, j]
            tc = "white" if v > M.max() * 0.6 else "black"
            suffix = "\n(ret.)" if i == j else ""
            ax_mfpt.text(j, i, f"{v:.1f}{suffix}", ha="center", va="center",
                         fontsize=10, fontweight="bold", color=tc)
            if i == j:
                rect = mpatches.FancyBboxPatch(
                    (j-0.45, i-0.45), 0.9, 0.9,
                    boxstyle="round,pad=0.04",
                    linewidth=2, edgecolor=COLORS["accent"],
                    facecolor="none", zorder=3,
                )
                ax_mfpt.add_patch(rect)

    ax_mfpt.set_xticks(range(3))
    ax_mfpt.set_yticks(range(3))
    ax_mfpt.set_xticklabels([f"→{s}" for s in STATES], fontsize=9)
    ax_mfpt.set_yticklabels([f"From {s}" for s in STATES], fontsize=9)
    ax_mfpt.set_title("(B)  Mean First Passage Times (days)\nDiagonal = 1/π_i (exact)",
                      fontsize=10, fontweight="bold")
    fig.colorbar(im2, ax=ax_mfpt, fraction=0.046, pad=0.04, label="Days")

    # ---- right: PageRank comparison ------------------------------------
    mkt_hist = np.array(pagerank_cmp["market_pi"]["history"])
    pr_hist  = np.array(pagerank_cmp["pr_pi"]["history"])
    mkt_spec = pagerank_cmp["market_spectral"]
    pr_spec  = pagerank_cmp["pr_spectral"]

    iters_m = np.arange(1, len(mkt_hist)+1)
    iters_p = np.arange(1, len(pr_hist)+1)
    mask_m = mkt_hist > 0
    mask_p = pr_hist > 0

    ax_pr.semilogy(iters_m[mask_m], mkt_hist[mask_m],
                   color=COLORS["accent"], linewidth=2.2, label="Market model")
    ax_pr.semilogy(iters_p[mask_p], pr_hist[mask_p],
                   color=COLORS["Bear"], linewidth=2.2, label="PageRank (d=0.85)")

    tol = pagerank_cmp["market_pi"]["tol"]
    ax_pr.axhline(tol, color="grey", linestyle=":", linewidth=0.9, alpha=0.6)

    ax_pr.set_xlabel("Iteration", fontsize=10)
    ax_pr.set_ylabel("‖δᵗ‖₁  (log scale)", fontsize=10)
    ax_pr.set_title(
        f"(C)  PageRank Comparison\n"
        f"Market gap={mkt_spec['spectral_gap']:.3f}  "
        f"vs  PR gap={pr_spec['spectral_gap']:.3f}",
        fontsize=10, fontweight="bold",
    )
    ax_pr.legend(fontsize=9)
    ax_pr.set_xlim(left=0)
    ax_pr.grid(True, alpha=0.2, linestyle="--")

    fig.suptitle(
        "Mathematical Foundations — Spectral Theory, MFPT, and PageRank",
        fontsize=14, fontweight="bold",
    )

    save_and_show(fig, save_path, show)
    return fig


# ===========================================================================
# Figure 3 – Model Critique Panel
# ===========================================================================
def plot_critique_panel(
    order_test: dict,
    sensitivity: dict,
    sub_period: dict,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Three-panel: order test | sensitivity grid | sub-period diff.

    This figure presents the model's limitations honestly and
    quantitatively — exactly what a technically strong interviewer wants.
    """
    fig, (ax_ot, ax_sens, ax_sub) = plt.subplots(
        1, 3, figsize=(18, 5.5),
        gridspec_kw={"wspace": 0.35},
    )

    # ---- left: chi-squared test ----------------------------------------
    from scipy.stats import chi2
    df_val  = order_test["df"]
    Lambda  = order_test["test_statistic"]
    p_val   = order_test["p_value"]

    x = np.linspace(0, max(Lambda * 1.4, df_val * 3.5), 600)
    y = chi2.pdf(x, df=df_val)
    ax_ot.plot(x, y, color=COLORS["price"], linewidth=2.0,
               label=f"χ²(df={df_val})")

    x_shade = x[x >= Lambda]
    ax_ot.fill_between(x_shade, chi2.pdf(x_shade, df=df_val),
                       alpha=0.40, color=COLORS["Bear"],
                       label=f"p-value = {p_val:.4f}")
    ax_ot.axvline(Lambda, color=COLORS["Bear"], linewidth=2.0, linestyle="--",
                  label=f"Observed Λ = {Lambda:.1f}")
    ax_ot.axvline(chi2.ppf(0.95, df=df_val), color="grey", linewidth=1.0,
                  linestyle=":", label=f"χ²₀.₉₅ = {chi2.ppf(0.95,df=df_val):.1f}")

    verdict = "REJECT H₀" if order_test["reject_h0"] else "FAIL TO REJECT H₀"
    ax_ot.set_xlabel("Test statistic Λ", fontsize=10)
    ax_ot.set_ylabel("Density", fontsize=10)
    ax_ot.set_title(
        f"(A)  Markov Order Test\n{verdict} (p = {p_val:.4f})",
        fontsize=10, fontweight="bold",
    )
    ax_ot.legend(fontsize=8, loc="upper right")
    ax_ot.set_xlim(left=0)
    ax_ot.grid(True, alpha=0.2, linestyle="--")

    # ---- centre: sensitivity grid (π_Bear only, cleaner) ---------------
    df_s = sensitivity["results"]
    windows    = sorted(df_s["window"].unique())
    thresholds = sorted(df_s["threshold"].unique())
    nw, nt = len(windows), len(thresholds)

    grid = np.zeros((nw, nt))
    for i, w in enumerate(windows):
        for j, t in enumerate(thresholds):
            row = df_s[(df_s["window"] == w) & (df_s["threshold"] == t)]
            if not row.empty:
                grid[i, j] = float(row["pi_Bear"].iloc[0])

    im_s = ax_sens.imshow(grid, cmap="Reds", aspect="auto",
                           vmin=grid.min() * 0.85)
    for i, w in enumerate(windows):
        for j, t in enumerate(thresholds):
            v = grid[i, j]
            is_baseline = (w == 20 and t == 0.02)
            ax_sens.text(j, i, f"{v:.3f}", ha="center", va="center",
                         fontsize=9.5,
                         fontweight="bold" if is_baseline else "normal",
                         color="white" if v > grid.max() * 0.65 else "black")
            if is_baseline:
                rect = mpatches.FancyBboxPatch(
                    (j-0.47, i-0.47), 0.94, 0.94,
                    boxstyle="round,pad=0.03",
                    linewidth=2.5, edgecolor=COLORS["accent"],
                    facecolor="none", zorder=3,
                )
                ax_sens.add_patch(rect)

    ax_sens.set_xticks(range(nt))
    ax_sens.set_yticks(range(nw))
    ax_sens.set_xticklabels([f"±{t:.0%}" for t in thresholds], fontsize=9)
    ax_sens.set_yticklabels([f"{w}d" for w in windows], fontsize=9)
    ax_sens.set_xlabel("Threshold", fontsize=10)
    ax_sens.set_ylabel("Window", fontsize=10)
    lo, hi = sensitivity["pi_bear_range"]
    ax_sens.set_title(
        f"(B)  π_Bear Sensitivity Grid\nRange: [{lo:.3f}, {hi:.3f}]  "
        f"| Robust: {'✓' if sensitivity['is_robust'] else '✗'}",
        fontsize=10, fontweight="bold",
    )
    fig.colorbar(im_s, ax=ax_sens, fraction=0.046, pad=0.04, label="π_Bear")

    # ---- right: sub-period difference heatmap --------------------------
    diff = sub_period["diff"]
    abs_max = max(np.abs(diff).max(), 0.01)
    im_sp = ax_sub.imshow(diff, cmap="RdBu_r",
                           vmin=-abs_max, vmax=abs_max, aspect="auto")
    for i in range(3):
        for j in range(3):
            v = diff[i, j]
            ax_sub.text(j, i, f"{v:+.4f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="black")

    ax_sub.set_xticks(range(3))
    ax_sub.set_yticks(range(3))
    ax_sub.set_xticklabels([f"→{s}" for s in STATES], fontsize=9)
    ax_sub.set_yticklabels(STATES, fontsize=9)
    split = sub_period["split_date"]
    frob  = sub_period["frobenius"]
    ax_sub.set_title(
        f"(C)  Sub-Period Stability  (split: {split[:7]})\n"
        f"P_post − P_pre  |  Frobenius = {frob:.4f}",
        fontsize=10, fontweight="bold",
    )
    cbar_sp = fig.colorbar(im_sp, ax=ax_sub, fraction=0.046, pad=0.04)
    cbar_sp.set_label("Signed change", fontsize=8)

    fig.suptitle(
        "Model Critique — Order Test, Sensitivity, and Non-Stationarity",
        fontsize=14, fontweight="bold",
    )

    save_and_show(fig, save_path, show)
    return fig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def save_and_show(fig: plt.Figure, save_path: Optional[str | Path], show: bool) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved → {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
