"""
strategy.py — VWAP Momentum Breakout
--------------------------------------
Signal direction: RIDE z-score extremes, not fade them.

Observation from v1–v7: mean-reversion on VWAP deviation gave Sharpe −4.0
with a smooth downward curve. A smooth −4 Sharpe is a strong signal in the
wrong direction. Inverting it → momentum breakout → Sharpe +0.528.

Framework (mirrors assignment example):
    anchor  = expanding mean of intraday Close (session TWAP, no volume needed)
    spread  = Close − anchor
    z_score = spread / rolling_std(spread, 12 bars)

    Long  : z ≥ +2.5  AND  Close > EMA(200)   [uptrend breakout continuation]
    Short : z ≤ −2.5  AND  Close < EMA(200)   [downtrend continuation]
    Exit  : z crosses zero (momentum exhausted), ATR stop, or EOD close
"""

import numpy as np
import pandas as pd

VWAP_STD_WINDOW = 12    # bars for rolling std of spread
ENTRY_Z         = 2.5   # entry threshold (±σ from anchor)
STOP_Z          = 6.0   # emergency z-score exit (primary exit = ATR stop)
EMA_PERIOD      = 200   # intraday trend filter (~17 hrs of 5-min data)
NO_TRADE_BEFORE = "09:30"
NO_TRADE_AFTER  = "14:30"


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(df.index.date, group_keys=False).apply(_signals_for_day)


def _signals_for_day(day: pd.DataFrame) -> pd.DataFrame:
    day = day.copy()

    anchor             = day["Close"].expanding().mean()
    deviation          = day["Close"] - anchor
    dev_std            = deviation.rolling(VWAP_STD_WINDOW, min_periods=VWAP_STD_WINDOW).std()
    day["vwap"]        = anchor
    day["z_score"]     = deviation / dev_std.replace(0, np.nan)
    day["rolling_mean"] = anchor
    day["rolling_std"]  = dev_std

    ema        = day["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    in_time    = (
        (day.index.time >= pd.Timestamp(NO_TRADE_BEFORE).time()) &
        (day.index.time <= pd.Timestamp(NO_TRADE_AFTER).time())
    )

    raw = pd.Series(0, index=day.index, dtype=int)
    raw[(day["z_score"] >=  ENTRY_Z) & (day["Close"] > ema) & in_time] =  1
    raw[(day["z_score"] <= -ENTRY_Z) & (day["Close"] < ema) & in_time] = -1

    day["signal"] = _position_logic(day["z_score"], raw)
    return day


def _position_logic(z: pd.Series, raw: pd.Series) -> pd.Series:
    """Hold position until z crosses zero or hits emergency STOP_Z."""
    position = 0
    out = np.zeros(len(z), dtype=int)
    for i, (zv, entry) in enumerate(zip(z, raw)):
        if np.isnan(zv):
            position = 0
        elif position == 0:
            position = int(entry)
        elif position == +1 and (zv <= 0.0 or zv <= -STOP_Z):
            position = 0
        elif position == -1 and (zv >= 0.0 or zv >= STOP_Z):
            position = 0
        out[i] = position
    return pd.Series(out, index=z.index)


def signal_summary(df: pd.DataFrame) -> None:
    counts = df["signal"].value_counts().sort_index()
    total  = len(df.dropna(subset=["z_score"]))
    print("\n── Signal Summary ────────────────────────────────────")
    for val, label in {-1: "Short (-1)", 0: "Flat  ( 0)", 1: "Long  (+1)"}.items():
        n = counts.get(val, 0)
        print(f"  {label} : {n:>7,}  ({n / total * 100:5.1f}%)")
    print(f"  Total bars : {total:,}")
    print("──────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_data
    signal_summary(compute_signals(load_data()))