

import os
import pandas as pd
import numpy as np

CSV_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "banknifty_candlestick_data.csv")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_banknifty.parquet")
MARKET_OPEN  = "09:15"
MARKET_CLOSE = "15:30"
OUTLIER_THRESHOLD = 0.10


# ── Parquet preparation (run once from raw CSV) ───────────────────────────────

def prepare_parquet(csv_path: str = CSV_PATH, out_path: str = DATA_PATH) -> None:
    if os.path.exists(out_path):
        return  # already prepared

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Raw CSV not found at {csv_path}.\n"
            "Download banknifty_candlestick_data.csv and place it in data/."
        )

    print("[data_loader] Building parquet from CSV (one-time step)…")
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d-%m-%Y %H:%M:%S", errors="coerce"
    )
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    df[["Open", "High", "Low", "Close"]].to_parquet(out_path)
    print(f"[data_loader] Parquet written → {out_path}")


# ── Main public function ──────────────────────────────────────────────────────

def load_data(path: str = DATA_PATH, resample: str = "5min") -> pd.DataFrame:
    """Load, clean and return BankNifty OHLC at the requested bar size."""
    prepare_parquet()          # no-op if parquet already exists
    df = _load_parquet(path)
    df = _filter_market_hours(df)
    df = _handle_missing(df)
    df = _remove_outliers(df)
    if resample:
        df = resample_ohlc(df, resample)
    print(f"[data_loader] Loaded {len(df):,} rows ({resample or '1min'} bars)  "
          f"|  {df.index.date[0]} → {df.index.date[-1]}")
    return df


# ── Private helpers ───────────────────────────────────────────────────────────

def _load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Parquet missing columns: {missing}")
    return df[required]


def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (df.index.time >= pd.Timestamp(MARKET_OPEN).time()) &
        (df.index.time <= pd.Timestamp(MARKET_CLOSE).time())
    )
    filtered = df[mask]
    dropped = len(df) - len(filtered)
    if dropped:
        print(f"[data_loader] Dropped {dropped:,} rows outside market hours.")
    return filtered


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = df.isnull().sum().sum()
    df = df.groupby(df.index.date, group_keys=False).apply(lambda d: d.ffill())
    filled = before - df.isnull().sum().sum()
    if filled:
        print(f"[data_loader] Forward-filled {filled:,} missing values.")
    return df.dropna()


def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    bad = df["Close"].pct_change().abs() > OUTLIER_THRESHOLD
    if bad.sum():
        print(f"[data_loader] Replaced {bad.sum():,} outlier bars (|return| > {OUTLIER_THRESHOLD:.0%}).")
        df.loc[bad, ["Open", "High", "Low", "Close"]] = np.nan
        df = df.ffill()
    return df


# ── Resampling ────────────────────────────────────────────────────────────────

def resample_ohlc(df: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    """Resample to larger bars within each session (never across overnight)."""
    def _agg(day):
        return day.resample(rule, closed="left", label="left").agg(
            Open=("Open", "first"), High=("High", "max"),
            Low=("Low", "min"),   Close=("Close", "last"),
        ).dropna()
    return df.groupby(df.index.date, group_keys=False).apply(_agg)


# ── ATR helpers ───────────────────────────────────────────────────────────────

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add rolling intraday ATR column (True Range average over `period` bars)."""
    df = df.copy()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period, min_periods=period).mean()
    return df


def add_daily_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Broadcast daily ATR to every intraday bar of that day.

    Uses daily OHLC so stops are sized to the day's expected range, not the
    noisy 5-minute ATR (~10 pts). Daily ATR on BankNifty is typically 300–600 pts.
    """
    daily = df.resample("D").agg(
        Open=("Open","first"), High=("High","max"),
        Low=("Low","min"),    Close=("Close","last"),
    ).dropna()
    tr = pd.concat([
        daily["High"] - daily["Low"],
        (daily["High"] - daily["Close"].shift(1)).abs(),
        (daily["Low"]  - daily["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    daily_atr = tr.rolling(period, min_periods=period).mean()
    df = df.copy()
    df["daily_ATR"] = df.index.normalize().map(daily_atr.to_dict())
    return df


def summarize(df: pd.DataFrame) -> None:
    print(f"\n── Data Summary ──────────────────────────────────────")
    print(f"  Rows        : {len(df):,}")
    print(f"  Date range  : {df.index[0]}  →  {df.index[-1]}")
    print(f"  Trading days: {df.index.normalize().nunique()}")
    print(f"  Close range : {df['Close'].min():,.2f}  –  {df['Close'].max():,.2f}")
    print(f"  Missing vals: {df.isnull().sum().sum()}")
    print("──────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    summarize(load_data())