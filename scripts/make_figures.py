#!/usr/bin/env python3
"""Regenerate the figures embedded in README.md.

Writes a light and a dark variant of each figure to ``docs/figures/``. The
README pairs them with ``<picture>`` so GitHub serves the one matching the
reader's theme.

Colors come from a validated categorical palette; only the first three slots
are used, which are documented to clear the color-vision-deficiency gates for
every pair in both modes. The benchmark series is deliberately neutral gray
rather than a fourth hue: it is context, not a peer.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Render headlessly. This must be set before pyplot is imported.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from portfolio_bl.config import load_config
from portfolio_bl.data.disclosures import (
    latest_portfolio_for_aliases,
    load_disclosures_csv,
)
from portfolio_bl.data.prices import load_prices_csv, to_return_matrix
from portfolio_bl.models.black_litterman import (
    black_litterman_posterior,
    diagonal_omega_from_confidence,
    implied_equilibrium_returns,
)
from portfolio_bl.models.mean_variance import estimate_mean_cov
from portfolio_bl.pipeline import run_case_study

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

# --- theme -------------------------------------------------------------------

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "grid": "#e4e3df",
        "muted": "#6f6e6a",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "ramp": ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
        "suffix": "",
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "grid": "#383835",
        "muted": "#9a9a92",
        "series": ["#3987e5", "#d95926", "#199e70"],
        # On a dark surface magnitude reads as luminance the other way round,
        # so the ramp runs dark -> light as volatility rises.
        "ramp": ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"],
        "suffix": "-dark",
    },
}

STRATEGY_LABEL = {
    "disclosed": "Disclosed",
    "mean_variance": "Mean-variance",
    "black_litterman": "Black-Litterman",
}


def style(theme: dict) -> None:
    """Apply recessive chrome: hairline grid, no top/right spines."""
    plt.rcParams.update({
        "figure.facecolor": theme["surface"],
        "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
        "text.color": theme["primary"],
        "axes.labelcolor": theme["secondary"],
        "axes.edgecolor": theme["grid"],
        "xtick.color": theme["secondary"],
        "ytick.color": theme["secondary"],
        "grid.color": theme["grid"],
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "font.size": 9,
        "legend.frameon": False,
    })


def save(fig, name: str, theme: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}{theme['suffix']}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# --- figures -----------------------------------------------------------------

def fig_equity_curves(results, spy, theme):
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.8))
    handles = None
    for ax, (key, res) in zip(axes, results.items()):
        idx = res.strategy_results["black_litterman"].returns.index
        for i, (name, sr) in enumerate(res.strategy_results.items()):
            ax.plot(sr.nav.index, sr.nav.values, color=theme["series"][i],
                    linewidth=2.0, label=STRATEGY_LABEL[name], zorder=3)
        bench = (1.0 + spy.reindex(idx).fillna(0.0)).cumprod()
        ax.plot(bench.index, bench.values, color=theme["muted"], linewidth=1.5,
                linestyle=(0, (5, 3)), label="SPY (benchmark)", zorder=2)
        ax.set_title(f"{res.person_label}", loc="left", pad=10)
        # Log scale: equal vertical distance is equal percentage change, which
        # keeps every strategy legible beside the DJT-driven spike.
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="all", numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel("Growth of 1.0")
        ax.margins(x=0.01)
        if handles is None:
            handles = ax.get_legend_handles_labels()
    fig.legend(*handles, loc="upper left", bbox_to_anchor=(0.008, 0.955),
               ncols=4, fontsize=9)
    fig.suptitle("Equity curves by strategy (log scale)", x=0.005, ha="left",
                 fontsize=13, fontweight="bold", color=theme["primary"])
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    save(fig, "equity-curves", theme)


def fig_drawdowns(results, theme):
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.4))
    handles = None
    for ax, (key, res) in zip(axes, results.items()):
        for i, (name, sr) in enumerate(res.strategy_results.items()):
            nav = (1.0 + sr.returns).cumprod()
            dd = (nav / nav.cummax() - 1.0) * 100.0
            ax.plot(dd.index, dd.values, color=theme["series"][i],
                    linewidth=1.8, label=STRATEGY_LABEL[name])
        ax.axhline(0, color=theme["grid"], linewidth=1.0)
        ax.set_title(f"{res.person_label}", loc="left", pad=10)
        ax.set_ylabel("Drawdown (%)")
        ax.margins(x=0.01)
        if handles is None:
            handles = ax.get_legend_handles_labels()
    fig.legend(*handles, loc="upper left", bbox_to_anchor=(0.008, 0.952),
               ncols=3, fontsize=9)
    fig.suptitle("Peak-to-trough drawdown", x=0.005, ha="left",
                 fontsize=13, fontweight="bold", color=theme["primary"])
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    save(fig, "drawdowns", theme)


ROLLING_WINDOW = 252


def rolling_beta(returns: pd.Series, market: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """Rolling OLS beta of a return series against the market."""
    paired = pd.concat([returns.rename("y"), market.rename("x")], axis=1).dropna()
    covariance = paired["y"].rolling(window).cov(paired["x"])
    variance = paired["x"].rolling(window).var()
    return (covariance / variance).dropna()


def fig_rolling_beta(results, spy, theme):
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.4), sharey=False)
    handles = None
    for ax, (key, res) in zip(axes, results.items()):
        for i, (name, sr) in enumerate(res.strategy_results.items()):
            beta = rolling_beta(sr.returns, spy)
            ax.plot(beta.index, beta.values, color=theme["series"][i],
                    linewidth=1.8, label=STRATEGY_LABEL[name], zorder=3)
        ax.axhline(1.0, color=theme["muted"], linewidth=1.2,
                   linestyle=(0, (4, 3)), zorder=2)
        ax.annotate("market", xy=(0.995, 1.0), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                    fontsize=8, color=theme["muted"])
        ax.set_title(res.person_label, loc="left", pad=10)
        ax.set_ylabel("Beta vs SPY")
        ax.margins(x=0.01)
        if handles is None:
            handles = ax.get_legend_handles_labels()
    fig.legend(*handles, loc="upper left", bbox_to_anchor=(0.008, 0.952),
               ncols=3, fontsize=9)
    fig.suptitle("Rolling market exposure (252-day beta)", x=0.005, ha="left",
                 fontsize=13, fontweight="bold", color=theme["primary"])
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    save(fig, "rolling-beta", theme)


def fig_confidence_sweep(sweep, theme):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    c = sweep["confidence"]
    axes[0].plot(c, sweep["to_disclosed"], color=theme["series"][0], linewidth=2.0,
                 marker="o", markersize=5, label="Distance to disclosed")
    axes[0].plot(c, sweep["to_mvo"], color=theme["series"][1], linewidth=2.0,
                 marker="o", markersize=5, label="Distance to mean-variance")
    axes[0].set_title("The posterior slides between its two inputs", loc="left", pad=10)
    axes[0].set_xlabel("View confidence")
    axes[0].set_ylabel("Mean distance between weight vectors")
    axes[0].legend(loc="center right")
    axes[1].plot(c, sweep["sharpe"], color=theme["series"][2], linewidth=2.0,
                 marker="o", markersize=5)
    axes[1].set_title("Trusting the trailing mean more costs Sharpe", loc="left", pad=10)
    axes[1].set_xlabel("View confidence")
    axes[1].set_ylabel("Sharpe ratio")
    for ax in axes:
        ax.margins(x=0.03)
    fig.suptitle("View confidence sweep  —  Warren Buffett", x=0.005, ha="left",
                 fontsize=13, fontweight="bold", color=theme["primary"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "confidence-sweep", theme)


def fig_calibration(calib, theme):
    grid, labels, before, after = calib
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, data, title in ((axes[0], before, "Before: an absolute ridge"),
                            (axes[1], after, "After: a ridge scaled to the matrix")):
        ax.plot([0, 1], [0, 1], color=theme["muted"], linewidth=1.2,
                linestyle=(0, (4, 3)), zorder=1, label="Perfect calibration")
        for i, lab in enumerate(labels):
            ax.plot(grid, data[i], color=theme["ramp"][i], linewidth=2.0,
                    marker="o", markersize=4, zorder=3, label=lab)
        ax.set_title(title, loc="left", pad=10)
        ax.set_xlabel("Configured confidence")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Realized confidence")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].annotate("all five assets coincide\non the identity line",
                     xy=(0.44, 0.20), xycoords="axes fraction",
                     fontsize=9, color=theme["secondary"], ha="left")
    fig.suptitle(
        "Does a view move the posterior as far as you asked?",
        x=0.005, ha="left", fontsize=12, fontweight="bold", color=theme["primary"])
    fig.text(0.005, 0.895,
             "One view at a time, five assets spanning 3% to 49% annual volatility. "
             "Per-asset calibration is a single-view property; see the README.",
             ha="left", fontsize=8.5, color=theme["secondary"])
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    save(fig, "confidence-calibration", theme)


def fig_concentration(books, theme):
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 4.2))
    for ax, (label, port) in zip(axes, books):
        port = port.sort_values("weight", ascending=True)
        y = np.arange(len(port))
        ax.barh(y, port["weight"].values * 100.0, color=theme["series"][0],
                height=0.72, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(port["ticker"].values, fontsize=7)
        top = port.iloc[-1]
        ax.annotate(f"{top['ticker']}  {top['weight']:.0%}",
                    xy=(top["weight"] * 100.0, len(port) - 1),
                    xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5, fontweight="bold",
                    color=theme["surface"], zorder=4)
        hhi = float((port["weight"] ** 2).sum())
        ax.set_title(f"{label}\n{len(port)} holdings, HHI {hhi:.2f}", loc="left", pad=10)
        ax.set_xlabel("Weight (%)")
        ax.grid(axis="x", zorder=0)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Disclosed portfolio concentration", x=0.005, ha="left",
                 fontsize=13, fontweight="bold", color=theme["primary"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "disclosed-concentration", theme)


# --- data --------------------------------------------------------------------

def _absolute_ridge_posterior(pi, sigma, p, q, tau, omega, ridge=1e-6):
    """The pre-fix behavior, reproduced here only to draw the 'before' panel."""
    sigma = np.asarray(sigma, dtype=float) + np.eye(sigma.shape[0]) * ridge
    omega = np.asarray(omega, dtype=float) + np.eye(omega.shape[0]) * ridge
    tsi = np.linalg.pinv(tau * sigma)
    oi = np.linalg.pinv(omega)
    middle_inv = np.linalg.pinv(tsi + p.T @ oi @ p)
    return middle_inv @ (tsi @ pi + p.T @ oi @ q)


def build_calibration(cfg, rets):
    """Realized confidence per asset, before and after the fix."""
    universe = ["MUB", "BND", "LQD", "SPY", "UNG"]      # low -> high volatility
    window = rets[universe].dropna().tail(126)
    _, cov = estimate_mean_cov(window)
    sigma = cov.to_numpy()
    pi = implied_equilibrium_returns(
        cov, pd.Series(1.0 / len(universe), index=universe), cfg.backtest.risk_aversion)
    grid = np.round(np.linspace(0.05, 1.0, 20), 4)
    tau = cfg.backtest.tau
    before, after, labels = [], [], []
    for i, ticker in enumerate(universe):
        vol = float(np.sqrt(sigma[i, i] * 252))
        labels.append(f"{ticker}  {vol:.0%}")
        p = np.zeros((1, len(universe)))
        p[0, i] = 1.0
        b_row, a_row = [], []
        for c in grid:
            q = np.array([pi[i] + 0.05 / 252])
            omega = diagonal_omega_from_confidence(sigma, p, tau, float(c))
            mu_old = _absolute_ridge_posterior(pi, sigma, p, q, tau, omega)
            mu_new, _ = black_litterman_posterior(pi, sigma, p, q, tau, omega)
            denom = q[0] - pi[i]
            b_row.append((mu_old[i] - pi[i]) / denom)
            a_row.append((mu_new[i] - pi[i]) / denom)
        before.append(b_row)
        after.append(a_row)
    return grid, labels, np.array(before), np.array(after)


def main() -> None:
    logging.disable(logging.CRITICAL)
    cfg = load_config(ROOT / "configs" / "case_studies.yaml")
    rets = to_return_matrix(load_prices_csv(cfg.prices_path))
    disclosures = load_disclosures_csv(cfg.disclosures_path)

    print("running case studies ...")
    results = {k: run_case_study(cfg, k) for k in cfg.case_studies}
    spy = rets["SPY"]

    print("sweeping view confidence ...")
    base = results["buffett"]
    disc = base.strategy_results["disclosed"].weight_history
    mvo = base.strategy_results["mean_variance"].weight_history
    grid = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.99]
    sweep = {"confidence": grid, "to_disclosed": [], "to_mvo": [], "sharpe": []}
    for c in grid:
        r = run_case_study(cfg, "buffett", view_confidence=c)
        bl = r.strategy_results["black_litterman"].weight_history
        sweep["to_disclosed"].append(float(np.linalg.norm(bl.values - disc.values, axis=1).mean()))
        sweep["to_mvo"].append(float(np.linalg.norm(bl.values - mvo.values, axis=1).mean()))
        sweep["sharpe"].append(float(r.summary.loc["black_litterman", "sharpe"]))

    print("building calibration panel ...")
    calib = build_calibration(cfg, rets)

    books = []
    for case in cfg.case_studies.values():
        port, _ = latest_portfolio_for_aliases(disclosures, case.disclosure_aliases)
        books.append((case.person_label, port))

    for name, theme in THEMES.items():
        print(f"rendering {name} theme ...")
        style(theme)
        fig_equity_curves(results, spy, theme)
        fig_drawdowns(results, theme)
        fig_rolling_beta(results, spy, theme)
        fig_confidence_sweep(sweep, theme)
        fig_calibration(calib, theme)
        fig_concentration(books, theme)
    print("done")


if __name__ == "__main__":
    main()
