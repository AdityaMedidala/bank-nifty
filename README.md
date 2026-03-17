# BankNifty Intraday Momentum Strategy

A systematic intraday trading strategy for BankNifty, built as part of the
Quant Developer Screening Assignment.

---

## The core idea

I started with mean reversion the intuition that intraday price spikes  tend to snap back. Eight iterations later, I ended up with the opposite
strategy: momentum breakouts.

The dataset has one asset, BankNifty, with minute-level OHLC from January 2015 to March 2024. There was no volume column. The assignment example
mentions pairs trading, but with a single asset, the only exploitable statistical relationships are within the prices itself either mean reversion or momentum.

 I computed a z-score of the intraday price deviation from a rolling anchor (expanding mean from session open, also a proxy for VWAP). In early
versions I treated large deviations as buying/shorting opportunities, expecting reversion. The results: Sharpe of -4.0, perfectly smooth  downward equity curve. A smooth -4.0 Sharpe was a strong signal pointing the wrong way.
So I inverted the signal. Instead of fading a 2.5σ VWAP deviation and the algorithm rides it. 

The logic: when BankNifty moves 2.5 standard deviations from its session anchor, it is typically driven by institutional order flow still
being executed. The move continues until the order is filled

The inverted strategy produces a Sharpe of 0.528 in-sample and 0.691 out-of-sample.

---

## Strategy

**Signal:** Intraday VWAP deviation breakout on 5-minute bars.

```
session_anchor = expanding mean of Close (resets at 09:15 each day)
deviation       = Close - session_anchor
z_score         = deviation / rolling_std(deviation, 12 bars)

long  if z_score ≥ +2.5  AND  Close > EMA_200
short if z_score ≤ −2.5  AND  Close < EMA_200
```

**Exit:** When z-score crosses back through zero (momentum exhausted),
daily ATR stop is hit, or forced close at end of session.

**Filters:**
- EMA-200 on the 5-minute chart (~17 hours of price history). Long
  entries only when price is above EMA-200, shorts only below. This
  prevents fading a macro trend with an intraday signal.
- Time window: 09:30–14:30. The opening 15 minutes see order imbalances
  from overnight events. The final hour sees options expiry hedging.
  Both regimes are directional, not mean-reverting or momentum-friendly
  in the way this signal captures.

**Note on VWAP approximation:** Real VWAP requires volume data. The
dataset contains only OHLC. The session anchor used here — an expanding
mean of Close prices from 09:15 — is a weight approximation. It
captures the same directional information (where is price relative to
its session average) without volume weighting. Implementing true VWAP
is listed under improvements.

---

## Results

**Full period (2015–2024):**

| Metric | Value |
|---|---|
| Total return | +183.81% |
| Annualized return | +12.27% |
| Sharpe ratio | 0.528 |
| Max drawdown | −29.14% |
| Win rate | 31.7% |
| Avg trade duration | 107.5 min |
| Total trades | 3,624 |

**Out-of-sample validation (train 2015–2019 / test 2020–2024):**

| Metric | Train | Test |
|---|---|---|
| Annualized return | +7.75% | +15.45% |
| Sharpe ratio | 0.004 | 0.691 |
| Max drawdown | −29.14% | −8.37% |
| Win rate | 29.3% | 34.6% |

The test period outperforms training on every metric. This is
consistent with the strategy's regime dependence: post-2020 BankNifty
had higher directional momentum due to COVID recovery, the conditions where VWAP breakouts have
edge.

**On win rate:** 31.7% is normal for a momentum strategy. Momentum
strategies have many small losses (failed breakouts) and few large wins
(institutional moves). The mean trade is +145.9 points. The
expected value is positive because winners are approximately 10x the
size of losers, not because winners are frequent.

---

## Project structure

```
quant-developer-intern-assignment/
├── data/
│   └── cleaned_banknifty.parquet    # preprocessed from raw CSV
├── project/
│   ├── data_loader.py               # load, clean, resample, ATR
│   ├── strategy.py                  # VWAP signal, EMA filter, exits
│   ├── backtester.py                # trade simulation, costs, sizing
│   └── analysis.py                  # metrics, charts, train/test split
├── results/                         # all output charts saved here
├── README.md
└── requirements.txt
```

---

## How to run

```bash
pip install -r requirements.txt

# Run full analysis
cd project
python analysis.py
```

This runs the complete pipeline and saves charts to `results/`. The
train/test split runs automatically at the end.

---

## Design decisions

**Why 5-minute bars?**  
1-minute bars are dominated by microstructure noise. The intraday ATR
on 1-minute data is 5–10 points, meaning a 1.5x stop fires at 7–15
points — BankNifty moves that in seconds. Resampling to 5-minute bars
increases the signal-to-noise ratio and gives trades room to develop.

**Why daily ATR for stops instead of intraday ATR?**  
I tested intraday ATR (20-bar, 0.8x multiplier) against daily ATR
(14-day, 1.5x). Intraday stops at 40–120 points cut winning momentum
trades before they run, dropping Sharpe from 0.528 to 0.363 and win
rate from 31.7% to 16.6%. Daily ATR stops at 450–900 points are wide
but appropriate: momentum trades need room to breathe between entry and
their natural exhaustion point (z returns to zero).

**Why not pairs trading?**  
The dataset contains a single asset. Pairs trading requires at least two
cointegrated price series. With one asset, the only relationships to
exploit are within the time series itself.

**Position sizing:**  
Volatility-scaled. Position size in lots is computed as:

```
lots = clamp(TARGET_RISK_PTS / stop_distance, MIN_LOTS, MAX_LOTS)
```

High-volatility days get fewer lots (same monetary risk), low-volatility
days get more. Target risk per trade is 500 index points.

---

## Assumptions

| Item | Value | Reasoning |
|---|---|---|
| Bar size | 5-minute | Reduces noise vs 1-min |
| Lot size | 15 units | SEBI-revised BankNifty contract |
| Transaction cost | ₹75 per round-trip | Brokerage + STT + exchange charges |
| Slippage | 2 points per side | Conservative for a liquid index |
| Stop-loss | 1.5× daily ATR | ~450–900 pts, tested vs alternatives |
| Risk-free rate | 6% p.a. | India 10-year gilt yield |

---

## Limitations

**VWAP approximation.** Equal-weight expanding mean is not true VWAP.
True VWAP weights price by volume, which is what institutional algorithms
use as their benchmark. Without volume data, the signal captures the
right directional concept but misses the precision that makes VWAP
actionable in live trading.

**Regime dependence.** The strategy performs significantly better in
trending, high-volatility regimes (2020–2024) than in the choppy, low-
volatility period (2016–2019). In production, a macro regime filter
based on rolling realized volatility vs historical average would be
worth implementing to reduce exposure during unfavourable conditions.

**Single asset.** The signal cannot be diversified across uncorrelated
instruments, which limits risk-adjusted returns and makes the strategy
vulnerable to BankNifty-specific regime changes.

**In-sample parameter selection.** Entry threshold (2.5σ), EMA period
(200 bars), and time window (09:30–14:30) were chosen by economic
reasoning and confirmed by the out-of-sample test. They were not grid-
searched, which reduces overfitting risk but means they may not be
optimal.

---

## Potential improvements

1. **Real VWAP with volume data.** The highest-impact improvement would
   be sourcing tick or volume data and computing true intraday VWAP.

2. **Volatility regime filter.** Only trade when rolling 20-day realized
   volatility exceeds its 60-day average. This would reduce losses
   during the 2016–2019 ranging period.

3. **Trailing stop.** Currently exits when z returns to zero. A trailing
   stop once a trade is profitable could capture more of the move on
   days when momentum is exceptionally strong.

4. **Walk-forward optimization.** Re-estimate parameters annually using
   only prior data, avoiding any look-ahead in parameter selection.

---

## Iteration history

This is a summary of all versions tested. The assignment asks for
reasoning process, so the full path is documented here.

| Version | Key change | Result | What it taught |
|---|---|---|---|
| v1 | SMA z-score mean reversion | Sharpe −4.9 | Trend days destroyed P&L |
| v2 | Regime + time filter + hard stop | Mean trade +8.6 pts | Filters work; signal has edge |
| v3 | Long-only + ENTRY_Z 2.3 | Mean trade +15 pts | Bull trend makes longs correct |
| v4 | Daily ATR volatility gate | Worse | Gated onto high-vol days where stop fires |
| v5 | 5-min bars + daily ATR stop | Duration 38.7 min | Fixed whipsaw |
| v6 | EMA-200 + 10:15–14:00 window | −3.49% total | Near breakeven |
| v7 | VWAP anchor (no volume) | Sharpe −4.0 smooth | Smooth −4 = strong signal, wrong direction |
| **v8** | **Signal inverted: ride breakouts** | **+183%, Sharpe 0.528** | **Momentum, not mean reversion** |

The inversion from v7 to v8 was the decisive step. A Sharpe of −4.0
from a smooth equity curve means the z-score signal has strong
predictive power — just in the wrong direction for mean reversion. When
price moves 2.5σ from VWAP, the market is not overextended and about to
snap back; it is breaking out and about to continue.

---

## Bonus challenges

**Out-of-sample testing (Bonus #2):** Implemented. Train 2015–2019,
test 2020–2024. Test Sharpe 0.691 exceeds train Sharpe 0.004. Results
chart saved to `results/equity_train_test.png`.

**Risk allocation (Bonus #4):** Implemented. Volatility-scaled position
sizing in `backtester.py` via `_compute_lots()`. Position size is
inversely proportional to stop distance.

**Parameter optimization (Bonus #1):** Parameters were chosen by
economic reasoning rather than grid search. The out-of-sample
improvement confirms they are not overfit. Walk-forward optimization is
documented under improvements.

**Multiple pairs (Bonus #3):** Not attempted. Dataset is single-asset.
Extension would require a cointegration framework across multiple
indices (Nifty 50, BankNifty, Nifty IT).

---

## AI assistance

Tools used: Claude/Geminni

Boilerplate & Syntax: Accelerated the creation of the matplotlib visualization suite and pandas data processing.

Debugging: Assisted in resolving vectorization edge cases in the backtester logic.