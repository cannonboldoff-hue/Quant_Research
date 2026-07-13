"""Visualization utilities.

Consolidates ``plot_pnl_distribution`` / ``plot_trade_summary`` /
``save_plot_as_image`` plus the per-dimension PnL breakdowns
(``plot_pnl_by_exit_reason``, ``plot_pnl_by_weekday``, ``plot_win_rate_by_direction``,
``plot_monthly_profit_heatmap``, ``plot_yearly_returns``, ``plot_drawdown``)
scattered across 02_data_cleaning / 19_visualization. All re-targeted at the
package's own ``trades`` schema (columns: entry_time, exit_time, direction,
pnl, ret, reason) rather than each notebook's ad-hoc tradebook column names —
one plotting convention instead of the notebooks' many. Matplotlib-only
(the notebooks' plotly variants are dropped: same charts, one less plotting
library to maintain). Headless-safe.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(trades, ax=None, title="Equity Curve"):
    ax = ax or plt.subplots(figsize=(9, 4))[1]
    eq = (1 + trades["ret"]).cumprod()
    ax.plot(eq.values)
    ax.set_title(title); ax.set_xlabel("Trade #"); ax.set_ylabel("Equity (x)")
    ax.grid(alpha=0.3)
    return ax


def plot_pnl_distribution(trades, ax=None, bins=40):
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    ax.hist(trades["ret"], bins=bins)
    ax.axvline(0, color="k", lw=1)
    ax.set_title("Per-trade Return Distribution"); ax.set_xlabel("return"); ax.set_ylabel("count")
    return ax


def plot_trade_summary(trades, metrics: dict, save_path: str | Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    plot_equity_curve(trades, ax=axes[0])
    plot_pnl_distribution(trades, ax=axes[1])
    txt = "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in metrics.items())
    fig.suptitle(txt, fontsize=9)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_drawdown(trades, ax=None):
    """Running drawdown of the trades' equity curve — ``plot_drawdown``."""
    ax = ax or plt.subplots(figsize=(9, 4))[1]
    equity = (1 + trades["ret"]).cumprod()
    dd = equity / equity.cummax() - 1
    ax.fill_between(range(len(dd)), dd.values, 0, color="crimson", alpha=0.4)
    ax.set_title("Drawdown"); ax.set_xlabel("Trade #"); ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    return ax


def plot_pnl_by_exit_reason(trades, ax=None):
    """Total PnL grouped by exit reason (stop/target/eod) — ``plot_pnl_by_exit_reason``."""
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    trades.groupby("reason")["pnl"].sum().plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("PnL by Exit Reason"); ax.set_ylabel("Total PnL")
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_pnl_by_weekday(trades, ax=None):
    """Total PnL grouped by exit weekday — ``plot_pnl_by_weekday``."""
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = pd.to_datetime(trades["exit_time"]).dt.day_name()
    by_day = trades.groupby(weekday)["pnl"].sum().reindex(weekday_order).dropna()
    colors = ["green" if v > 0 else "red" for v in by_day]
    by_day.plot(kind="bar", ax=ax, color=colors, edgecolor="black")
    ax.set_title("PnL by Weekday"); ax.set_ylabel("Total PnL")
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_win_rate_by_direction(trades, ax=None):
    """Win rate split by trade direction (long/short) — ``plot_win_rate_by_direction``."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    win = trades["pnl"] > 0
    rate = win.groupby(trades["direction"]).mean() * 100
    rate.index = ["short" if d < 0 else "long" for d in rate.index]
    rate.plot(kind="bar", ax=ax, color=["green", "red"][: len(rate)], edgecolor="black")
    ax.set_title("Win Rate by Direction"); ax.set_ylabel("Win Rate (%)"); ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_yearly_returns(trades, ax=None):
    """Total PnL grouped by exit year — ``plot_yearly_returns``."""
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    year = pd.to_datetime(trades["exit_time"]).dt.year
    trades.groupby(year)["pnl"].sum().plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
    ax.set_title("Yearly Returns"); ax.set_ylabel("Total PnL")
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_monthly_profit_heatmap(trades, ax=None):
    """Year x month total-PnL heatmap — ``plot_monthly_profit_heatmap``."""
    import numpy as np
    exit_time = pd.to_datetime(trades["exit_time"])
    monthly = trades.groupby([exit_time.dt.year, exit_time.dt.month])["pnl"].sum()
    table = monthly.unstack(fill_value=0.0)
    table = table.reindex(columns=range(1, 13), fill_value=0.0)

    ax = ax or plt.subplots(figsize=(11, 4))[1]
    im = ax.imshow(table.values, cmap="RdYlGn", aspect="auto",
                   vmin=-np.abs(table.values).max() or -1, vmax=np.abs(table.values).max() or 1)
    ax.set_xticks(range(12)); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(table.index))); ax.set_yticklabels(table.index)
    ax.set_title("Monthly Profit Heatmap")
    plt.colorbar(im, ax=ax, label="PnL")
    return ax
