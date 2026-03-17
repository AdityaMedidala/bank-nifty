import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_data, add_atr, add_daily_atr
from strategy    import compute_signals, VWAP_STD_WINDOW
from backtester  import run_backtest, LOT_SIZE

try:
    from statsmodels.tsa.stattools import adfuller, acf as sm_acf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from scipy.stats import norm as sp_norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "results")
TRADING_DAYS   = 252
RISK_FREE_RATE = 0.06


# ── Entry point ───────────────────────────────────────────────────────────────

def run_analysis(results: dict = None) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if results is None:
        df      = load_data()
        df      = add_atr(df)
        df      = add_daily_atr(df)
        df      = compute_signals(df)
        results = run_backtest(df)

    metrics = compute_metrics(results)
    _print_metrics(metrics)

    run_relationship_discovery(results["df"])
    _plot_relationship_discovery(results["df"])

    _plot_price_and_signals(results["df"])
    _plot_equity_curve(results["equity"])
    _plot_drawdown(results["equity"])
    _plot_trade_distribution(results["trades"])

    run_train_test_split(results["df"])

    print(f"\n[analysis] Charts saved to: {os.path.abspath(RESULTS_DIR)}/")
    return metrics


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: dict) -> dict:
    equity, trades, daily_pnl = results["equity"], results["trades"], results["daily_pnl"]
    init   = results["df"]["Close"].iloc[0]
    n_days = equity.index.normalize().nunique()
    years  = n_days / TRADING_DAYS

    total_pts = equity.iloc[-1] - equity.iloc[0]
    total_pct = total_pts / (init * LOT_SIZE) * 100
    base      = 1 + total_pct / 100
    ann_pct   = (abs(base) ** (1 / max(years, 0.01)) * (1 if base >= 0 else -1) - 1) * 100

    excess = daily_pnl / (init * LOT_SIZE) - RISK_FREE_RATE / TRADING_DAYS
    sharpe = excess.mean() / excess.std() * np.sqrt(TRADING_DAYS) if excess.std() > 0 else 0.0

    dd_pts = (equity - equity.cummax()).min()
    dd_pct = dd_pts / (init * LOT_SIZE) * 100

    return {
        "total_return_pts" : total_pts,
        "total_return_pct" : total_pct,
        "ann_return_pct"   : ann_pct,
        "sharpe_ratio"     : sharpe,
        "max_drawdown_pts" : dd_pts,
        "max_drawdown_pct" : dd_pct,
        "win_rate_pct"     : (trades["net_pnl"] > 0).mean() * 100 if not trades.empty else 0.0,
        "avg_duration_min" : trades["duration_min"].mean() if not trades.empty else 0.0,
        "n_trades"         : len(trades),
        "n_days"           : n_days,
    }


def _print_metrics(m: dict) -> None:
    print("\n══ Performance Metrics ═══════════════════════════════")
    print(f"  Total Return       : {m['total_return_pts']:+,.1f} pts  ({m['total_return_pct']:+.2f}%)")
    print(f"  Annualized Return  : {m['ann_return_pct']:+.2f}%")
    print(f"  Sharpe Ratio       : {m['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown       : {m['max_drawdown_pts']:,.1f} pts  ({m['max_drawdown_pct']:.2f}%)")
    print(f"  Win Rate           : {m['win_rate_pct']:.1f}%")
    print(f"  Avg Trade Duration : {m['avg_duration_min']:.1f} min")
    print(f"  Total Trades       : {m['n_trades']}")
    print(f"  Trading Days       : {m['n_days']}")
    print("══════════════════════════════════════════════════════\n")


# ── Relationship Discovery ────────────────────────────────────────────────────

def run_relationship_discovery(df: pd.DataFrame) -> dict:
    z_series, sp_series = _compute_spread_series(df)

    stats = _run_stationarity_tests(z_series)
    stats.update({
        "spread_mean"  : float(sp_series.mean()),
        "spread_std"   : float(sp_series.std()),
        "z_score_mean" : float(z_series.mean()),
        "z_score_std"  : float(z_series.std()),
        "n_obs"        : len(z_series),
    })

    print("\n══ Relationship Discovery ════════════════════════════")
    print(f"  Instrument   : BankNifty (single asset)")
    print(f"  Spread type  : Intraday Close − TWAP anchor  (β = 1.0)")
    print(f"  Spread mean  : {stats['spread_mean']:+.2f} pts  (≈ 0 → no drift)")
    print(f"  Spread std   : {stats['spread_std']:.2f} pts")
    print(f"  Z-score mean : {stats['z_score_mean']:+.4f}  Z-score std : {stats['z_score_std']:.4f}")
    print(f"  ADF stat     : {stats['adf_stat']:.4f}  p-value : {stats['adf_pvalue']:.4f}  "
          f"{'✓ STATIONARY' if stats['adf_pvalue'] < 0.05 else '✗ non-stationary'}")
    print(f"  ACF lag-1    : {stats['acf_lag1']:+.4f}  "
          f"({'mean-reverting' if stats['acf_lag1'] < 0 else 'trending'})")
    print(f"  Observations : {stats['n_obs']:,}")
    print("══════════════════════════════════════════════════════\n")
    return stats


def _compute_spread_series(df: pd.DataFrame):
    z_list, sp_list = [], []
    for _, day in df.groupby(df.index.date):
        anchor  = day["Close"].expanding().mean()
        dev     = day["Close"] - anchor
        std     = dev.rolling(VWAP_STD_WINDOW, min_periods=VWAP_STD_WINDOW).std()
        z_list.append(dev / std.replace(0, np.nan))
        sp_list.append(dev)
    return pd.concat(z_list).dropna(), pd.concat(sp_list).dropna()


def _run_stationarity_tests(z_series: pd.Series) -> dict:
    sample = z_series.sample(min(20_000, len(z_series)), random_state=42).sort_index()
    if HAS_STATSMODELS:
        adf   = adfuller(sample, autolag="AIC")
        acf_v = sm_acf(sample, nlags=5, fft=True)
        return {"adf_stat": adf[0], "adf_pvalue": adf[1], "acf_lag1": acf_v[1]}
    # Fallback: manual unit-root t-stat
    delta  = z_series.diff().dropna()
    lagged = z_series.shift(1).dropna().iloc[:len(delta)]
    cov    = np.cov(delta.values, lagged.values)
    beta   = cov[0, 1] / cov[1, 1]
    se     = np.std(delta.values - beta * lagged.values) / (np.std(lagged.values) * np.sqrt(len(lagged)))
    t      = beta / se if se > 0 else 0
    return {"adf_stat": t, "adf_pvalue": 0.01 if t < -3.0 else 0.10,
            "acf_lag1": float(np.corrcoef(z_series[:-1], z_series[1:])[0, 1])}


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#f8f8f8",
        "axes.grid": True, "grid.color": "white", "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    })

def _save(name: str):
    plt.savefig(os.path.join(RESULTS_DIR, name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] Saved {name}")

def _fmt_xaxis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)


def _plot_price_and_signals(df: pd.DataFrame) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(14, 5))
    s = df.iloc[::5]
    ax.plot(s.index, s["Close"], color="#4a90d9", linewidth=0.8, label="Close")
    longs  = df[df["signal"].diff() ==  1]
    shorts = df[df["signal"].diff() == -1]
    exits  = df[(df["signal"] == 0) & (df["signal"].shift(1) != 0)]
    ax.scatter(longs.index,  longs["Close"],  marker="^", color="#2ecc71", s=60, zorder=5, label="Long entry")
    ax.scatter(shorts.index, shorts["Close"], marker="v", color="#e74c3c", s=60, zorder=5, label="Short entry")
    ax.scatter(exits.index,  exits["Close"],  marker="x", color="#888",    s=40, zorder=5, label="Exit")
    ax.set_title("BankNifty — Close price with trade signals", fontweight="bold")
    ax.set_ylabel("Index points")
    _fmt_xaxis(ax)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    _save("price_and_signals.png")


def _plot_equity_curve(equity: pd.Series) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(14, 4))
    color = "#2ecc71" if equity.iloc[-1] >= 0 else "#e74c3c"
    ax.plot(equity.index, equity.values, color=color, linewidth=1.2)
    ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
    ax.fill_between(equity.index, equity.values, 0, where=(equity.values >= 0), alpha=0.15, color="#2ecc71")
    ax.fill_between(equity.index, equity.values, 0, where=(equity.values < 0),  alpha=0.15, color="#e74c3c")
    ax.set_title("Equity curve — Cumulative P&L (index pts × lot size)", fontweight="bold")
    ax.set_ylabel("Cumulative P&L (pts)")
    _fmt_xaxis(ax)
    plt.tight_layout()
    _save("equity_curve.png")


def _plot_drawdown(equity: pd.Series) -> None:
    _style()
    dd = equity - equity.cummax()
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.fill_between(dd.index, dd.values, 0, color="#e74c3c", alpha=0.5)
    ax.plot(dd.index, dd.values, color="#c0392b", linewidth=0.8)
    ax.axhline(0, color="#aaa", linewidth=0.8)
    ax.set_title("Drawdown curve — Distance from equity peak", fontweight="bold")
    ax.set_ylabel("Drawdown (pts)")
    _fmt_xaxis(ax)
    plt.tight_layout()
    _save("drawdown_curve.png")


def _plot_trade_distribution(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("[analysis] No trades — skipping distribution chart.")
        return
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    wins   = trades.loc[trades["net_pnl"] > 0,  "net_pnl"]
    losses = trades.loc[trades["net_pnl"] <= 0, "net_pnl"]
    ax.hist(wins,   bins=40, color="#2ecc71", alpha=0.7, label="Wins")
    ax.hist(losses, bins=40, color="#e74c3c", alpha=0.7, label="Losses")
    ax.axvline(trades["net_pnl"].mean(), color="#333", linestyle="--", linewidth=1.2,
               label=f"Mean {trades['net_pnl'].mean():+.1f} pts")
    ax.set_title("Trade P&L distribution", fontweight="bold")
    ax.set_xlabel("Net P&L per trade (pts)")
    ax.set_ylabel("Number of trades")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save("trade_distribution.png")


def _plot_relationship_discovery(df: pd.DataFrame) -> None:
    """Chart 5: Z-score stationarity evidence and autocorrelation."""
    z_series, _ = _compute_spread_series(df)
    sample = z_series.sample(min(5_000, len(z_series)), random_state=42).sort_index()

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Relationship Discovery — Intraday Z-Score Analysis", fontweight="bold", fontsize=13)

    # Panel 1: time series
    ax = axes[0]
    ax.plot(sample.index, sample.values, color="#4a90d9", linewidth=0.4, alpha=0.7)
    ax.axhline(0,    color="#aaa", linewidth=0.8, linestyle="--")
    ax.axhline( 2.5, color="#e74c3c", linewidth=0.8, linestyle=":", label="±2.5σ entry")
    ax.axhline(-2.5, color="#2ecc71", linewidth=0.8, linestyle=":")
    ax.set_title("Z-Score Series (sample 5k bars)")
    ax.set_ylabel("z-score")
    ax.legend(fontsize=8)

    # Panel 2: distribution
    ax = axes[1]
    ax.hist(z_series.clip(-6, 6), bins=80, color="#4a90d9", alpha=0.7, density=True, label="Observed")
    if HAS_SCIPY:
        xr = np.linspace(-6, 6, 200)
        ax.plot(xr, sp_norm.pdf(xr), color="#e74c3c", linewidth=1.5, linestyle="--", label="N(0,1)")
    ax.set_title("Z-Score Distribution")
    ax.set_xlabel("z-score")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

    # Panel 3: ACF
    ax = axes[2]
    lags = range(1, 21)
    s20  = z_series.sample(min(20_000, len(z_series)), random_state=42)
    acf_v = sm_acf(s20, nlags=20, fft=True)[1:] if HAS_STATSMODELS else \
            [float(np.corrcoef(z_series[:-l], z_series[l:])[0, 1]) for l in lags]
    ax.bar(list(lags), acf_v, color=["#e74c3c" if v < 0 else "#2ecc71" for v in acf_v], alpha=0.8)
    ax.axhline(0, color="#aaa", linewidth=0.8)
    ci = 1.96 / np.sqrt(len(z_series))
    ax.axhline( ci, color="#aaa", linewidth=0.8, linestyle="--", label="95% CI")
    ax.axhline(-ci, color="#aaa", linewidth=0.8, linestyle="--")
    ax.set_title("ACF of Z-Score (neg lag-1 → mean-reversion)")
    ax.set_xlabel("lag (5-min bars)")
    ax.set_ylabel("autocorrelation")
    ax.legend(fontsize=8)

    plt.tight_layout()
    _save("relationship_discovery.png")

# ── Train / Test Split ────────────────────────────────────────────────────────

SPLIT_DATE = "2020-01-01"


def run_train_test_split(df: pd.DataFrame = None) -> None:
    """Split at SPLIT_DATE, backtest each half, plot equity curves + metrics table."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if df is None:
        df = load_data()
        df = add_atr(df)
        df = add_daily_atr(df)
        df = compute_signals(df)

    train_df = df[df.index < SPLIT_DATE]
    test_df  = df[df.index >= SPLIT_DATE]

    print(f"[analysis] Train: {train_df.index[0].date()} → {train_df.index[-1].date()}  "
          f"({train_df.index.normalize().nunique()} days)")
    print(f"[analysis] Test : {test_df.index[0].date()} → {test_df.index[-1].date()}  "
          f"({test_df.index.normalize().nunique()} days)")

    r_train = run_backtest(train_df)
    r_test  = run_backtest(test_df)

    m_train = compute_metrics(r_train)
    m_test  = compute_metrics(r_test)

    _print_split_metrics(m_train, m_test)
    _plot_train_test(r_train["equity"], r_test["equity"], m_train, m_test)


def _print_split_metrics(m_train: dict, m_test: dict) -> None:
    print("\n══ Train / Test Metrics ══════════════════════════════")
    rows = [
        ("Annualized return", "ann_return_pct",   "{:+.2f}%"),
        ("Sharpe ratio",      "sharpe_ratio",      "{:.3f}"),
        ("Max drawdown",      "max_drawdown_pct",  "{:.2f}%"),
        ("Win rate",          "win_rate_pct",       "{:.1f}%"),
        ("Total trades",      "n_trades",           "{:,}"),
    ]
    print(f"  {'Metric':<22}  {'Train (2015–2019)':>18}  {'Test (2020–2024)':>18}")
    print(f"  {'-'*22}  {'-'*18}  {'-'*18}")
    for label, key, fmt in rows:
        tv = fmt.format(m_train[key])
        ev = fmt.format(m_test[key])
        print(f"  {label:<22}  {tv:>18}  {ev:>18}")
    print("══════════════════════════════════════════════════════\n")


def _plot_train_test(eq_train: pd.Series, eq_test: pd.Series,
                     m_train: dict, m_test: dict) -> None:
    _style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    fig.suptitle("Out-of-Sample Validation — Train vs Test Equity Curves",
                 fontweight="bold", fontsize=13)

    # ── Train panel ───────────────────────────────────────────────────────────
    ax1.plot(eq_train.index, eq_train.values, color="#4a90d9", linewidth=1.1)
    ax1.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
    ax1.fill_between(eq_train.index, eq_train.values, 0,
                     where=(eq_train.values >= 0), alpha=0.15, color="#4a90d9")
    ax1.fill_between(eq_train.index, eq_train.values, 0,
                     where=(eq_train.values < 0),  alpha=0.15, color="#e74c3c")
    ax1.set_title(
        f"Train  2015–2019  |  Ann. return {m_train['ann_return_pct']:+.2f}%  "
        f"|  Sharpe {m_train['sharpe_ratio']:.3f}  "
        f"|  Max DD {m_train['max_drawdown_pct']:.2f}%  "
        f"|  Win rate {m_train['win_rate_pct']:.1f}%",
        fontsize=10,
    )
    ax1.set_ylabel("Cumulative P&L (pts)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20)

    # ── Test panel ────────────────────────────────────────────────────────────
    ax2.plot(eq_test.index, eq_test.values, color="#2ecc71", linewidth=1.1)
    ax2.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
    ax2.fill_between(eq_test.index, eq_test.values, 0,
                     where=(eq_test.values >= 0), alpha=0.15, color="#2ecc71")
    ax2.fill_between(eq_test.index, eq_test.values, 0,
                     where=(eq_test.values < 0),  alpha=0.15, color="#e74c3c")
    ax2.set_title(
        f"Test  2020–2024  |  Ann. return {m_test['ann_return_pct']:+.2f}%  "
        f"|  Sharpe {m_test['sharpe_ratio']:.3f}  "
        f"|  Max DD {m_test['max_drawdown_pct']:.2f}%  "
        f"|  Win rate {m_test['win_rate_pct']:.1f}%",
        fontsize=10,
    )
    ax2.set_ylabel("Cumulative P&L (pts)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=20)

    plt.tight_layout()
    _save("equity_train_test.png")


if __name__ == "__main__":
    run_analysis()