"""
backtester.py
-------------
Bar-by-bar simulation of the VWAP momentum strategy on BankNifty.

Cost model
----------
- Transaction cost : ₹75 round-trip per lot (brokerage + STT + exchange + GST)
- Slippage         : 2 index points per side (conservative for a liquid index)
- Lot size         : 15 units (SEBI-revised BankNifty contract, 2024)

Risk model
----------
- Stop-loss : 1.5 × daily ATR (~450–900 pts)
  Daily ATR is used instead of intraday ATR (5–10 pts) because intraday
  stops fire on microstructure noise before the momentum move develops.
"""

import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_data, add_atr, add_daily_atr
from strategy    import compute_signals

LOT_SIZE               = 15
SLIPPAGE_POINTS        = 2.0
COST_PER_ROUNDTRIP_INR = 75.0
ATR_MULTIPLIER         = 1.5
ATR_PERIOD             = 14


def run_backtest(df: pd.DataFrame = None) -> dict:
    """Run full backtest. Pass a pre-signaled DataFrame or let it build the pipeline."""
    if df is None:
        df = load_data()
        df = add_atr(df, ATR_PERIOD)
        df = add_daily_atr(df, ATR_PERIOD)
        df = compute_signals(df)
    else:
        if "ATR"       not in df.columns: df = add_atr(df, ATR_PERIOD)
        if "daily_ATR" not in df.columns: df = add_daily_atr(df, ATR_PERIOD)

    trades, equity = _simulate(df)
    daily_pnl = equity.diff().resample("D").sum().dropna()

    print(f"[backtester] {len(trades)} trades  |  final equity: {equity.iloc[-1]:+,.0f} pts")
    return {"trades": trades, "equity": equity, "daily_pnl": daily_pnl, "df": df}


# ── Simulation ────────────────────────────────────────────────────────────────

def _simulate(df: pd.DataFrame):
    trade_log   = []
    equity_vals = []
    position    = 0
    entry_price = entry_time = stop_price = None
    cum_pnl     = 0.0
    dates       = df.index.normalize()

    for i in range(len(df)):
        row        = df.iloc[i]
        ts         = df.index[i]
        signal     = int(row["signal"])
        is_eod     = (i == len(df) - 1) or (dates[i] != dates[i + 1])

        # ── ATR stop check ────────────────────────────────────────────────────
        stop_hit = (
            position == +1 and stop_price is not None and row["Low"]  <= stop_price or
            position == -1 and stop_price is not None and row["High"] >= stop_price
        )

        # ── Close position ────────────────────────────────────────────────────
        if position != 0 and (signal != position or is_eod or stop_hit):
            exit_px = stop_price if stop_hit else _exit_px(row, position)
            raw_pnl = (exit_px - entry_price) * position * LOT_SIZE
            cost    = _cost(exit_px)
            net_pnl = raw_pnl - cost
            cum_pnl += net_pnl

            reason = "atr_stop" if stop_hit else ("eod" if is_eod and signal == position else "signal")
            trade_log.append({
                "entry_time"  : entry_time, "exit_time"   : ts,
                "direction"   : "long" if position == +1 else "short",
                "entry_price" : entry_price, "exit_price"  : exit_px,
                "stop_price"  : stop_price,  "raw_pnl"     : raw_pnl,
                "cost"        : cost,        "net_pnl"     : net_pnl,
                "duration_min": int((ts - entry_time).total_seconds() / 60),
                "exit_reason" : reason,
            })
            position = entry_price = entry_time = stop_price = None
            position = 0

        # ── Open position ─────────────────────────────────────────────────────
        if position == 0 and signal != 0 and not is_eod:
            atr = row.get("ATR", np.nan)
            if np.isnan(atr):
                equity_vals.append(cum_pnl)
                continue
            position    = signal
            entry_price = _entry_px(row, position)
            entry_time  = ts
            datr        = row.get("daily_ATR", np.nan)
            dist        = (datr if not np.isnan(datr) else atr) * ATR_MULTIPLIER
            stop_price  = entry_price - dist * position
            cum_pnl    -= _cost(entry_price)

        # ── Mark to market ────────────────────────────────────────────────────
        mtm = (row["Close"] - entry_price) * position * LOT_SIZE if position != 0 else 0.0
        equity_vals.append(cum_pnl + mtm)

    equity = pd.Series(equity_vals, index=df.index, name="equity")
    trades = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    if not trades.empty:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"])
        trades["exit_time"]  = pd.to_datetime(trades["exit_time"])
        trades = trades.set_index("entry_time")
    return trades, equity


# ── Helpers ───────────────────────────────────────────────────────────────────

def _entry_px(row, direction):  return row["Close"] + SLIPPAGE_POINTS * direction
def _exit_px(row, position):    return row["Close"] - SLIPPAGE_POINTS * position
def _cost(price):               return (COST_PER_ROUNDTRIP_INR / 2) / LOT_SIZE


def print_summary(results: dict) -> None:
    trades, equity = results["trades"], results["equity"]
    if trades.empty:
        print("[backtester] No trades generated."); return
    wins    = trades[trades["net_pnl"] > 0]
    reasons = trades["exit_reason"].value_counts()
    print("\n── Backtest Summary ──────────────────────────────────")
    print(f"  Total trades  : {len(trades)}")
    print(f"  Win rate      : {len(wins)/len(trades)*100:.1f}%")
    print(f"  Avg P&L/trade : {trades['net_pnl'].mean():+.1f} pts")
    print(f"  Total P&L     : {trades['net_pnl'].sum():+,.1f} pts")
    print(f"  Avg duration  : {trades['duration_min'].mean():.1f} min")
    print(f"  ATR stops     : {reasons.get('atr_stop', 0)}")
    print(f"  Signal exits  : {reasons.get('signal', 0)}")
    print(f"  EOD closes    : {reasons.get('eod', 0)}")
    print(f"  Trading days  : {equity.index.normalize().nunique()}")
    print("──────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    r = run_backtest()
    print_summary(r)
    if not r["trades"].empty:
        print(r["trades"][["exit_time","direction","entry_price","exit_price",
                            "net_pnl","duration_min","exit_reason"]].head(10).to_string())