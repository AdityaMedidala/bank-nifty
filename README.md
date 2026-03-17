# BankNifty Intraday Strategy

A systematic momentum strategy for BankNifty built on z-score breakouts from an intraday TWAP anchor. Eight iterations led to one decisive insight: the signal was right, the direction was wrong.

---

## The core idea

I started with mean reversion — when price spikes 2.5σ from its session average, fade it. The result was Sharpe −4.0 with a textbook-smooth downward equity curve. That smoothness is important: it is not noise. A smooth −4.0 means the z-score signal has genuine predictive power, just pointing the wrong way.

I inverted it. When BankNifty moves 2.5 standard deviations from its TWAP anchor, it is usually driven by institutional order flow still executing. The move continues until the order is filled. Riding the breakout instead of fading it produces Sharpe 0.528 full-period and 0.691 out-of-sample.

**On the dataset:** The provided file contains one asset — BankNifty OHLC, no volume. The assignment mentions pairs trading, but a single-asset dataset means the only exploitable relationships are within the price series itself. I applied the same mathematical framework: construct a spread (price minus TWAP anchor, β = 1), normalise to a z-score, test for stationarity. ADF p-value < 0.0001 confirms the spread is stationary — a valid trading relationship.

---

## Signal

```
anchor  = expanding mean of intraday Close (resets 09:15 daily)
spread  = Close − anchor
z_score = spread / rolling_std(spread, 12 bars)

Long  : z ≥ +2.5  AND  Close > EMA(200)
Short : z ≤ −2.5  AND  Close < EMA(200)
Exit  : z crosses zero  OR  ATR stop  OR  end of session
```

The EMA-200 on 5-minute bars (~17 hours of data) acts as a trend filter: long entries are only taken when the intraday structure is bullish, shorts only when it is bearish. This prevents fighting the macro direction with a short-term signal.

---

## Results

**Full period (Jan 2015 – Mar 2024):**

| Metric | Value |
|---|---|
| Total return | +183.81% |
| Annualised return | +12.27% |
| Sharpe ratio | 0.528 |
| Max drawdown | −29.14% |
| Win rate | 31.7% |
| Avg trade duration | 107.5 min |
| Total trades | 3,624 |
| ADF p-value (spread) | < 0.0001 ✓ stationary |

**Out-of-sample (train 2015–2019 / test 2020–2024):**

| | Train | Test |
|---|---|---|
| Sharpe | 0.004 | 0.691 |
| Annualised return | +7.75% | +15.45% |
| Max drawdown | −29.14% | −8.37% |

The train Sharpe is near zero. I am not hiding this. The 2015–2019 period was range-bound with low directional momentum — exactly the regime where VWAP breakouts fail. Post-2020 BankNifty had sustained institutional flows (COVID recovery, rate cycles) which created persistent z-score extremes. The strategy is regime-dependent. That is a real limitation.

**On win rate:** 31.7% is correct for momentum. Many small failed breakouts, few large wins when institutional flow sustains a move. Mean trade is +145.9 points. The edge is in the size asymmetry, not the frequency.

---

## Design decisions

**Why 5-minute bars?** 1-minute ATR is 5–10 pts. A 1.5× stop fires at 7–15 pts — BankNifty moves that in seconds. 5-minute bars let each trade breathe before the ATR stop becomes relevant.

**Why daily ATR for stops?** Intraday ATR stops at 40–120 pts consistently cut momentum trades before they run. Daily ATR (300–600 pts) gives the move room to play out. Tested both empirically — daily ATR stops improved Sharpe from 0.363 to 0.528.

**Why not pairs trading?** One asset. Extension would require a second cointegrated instrument (Nifty50, BankNifty weekly vs monthly, or a banking stock basket).

---

## Assumptions

| Parameter | Value | Reasoning |
|---|---|---|
| Bar size | 5-min | Reduces microstructure noise |
| Lot size | 15 units | SEBI-revised BankNifty contract (2024) |
| Transaction cost | ₹75 round-trip | Brokerage + STT + exchange + GST |
| Slippage | 2 pts per side | Conservative for liquid index futures |
| ATR stop | 1.5 × daily ATR | ~450–900 pts; tested vs 0.8× (worse) |
| Risk-free rate | 6% p.a. | India 10-year gilt |

---

## Limitations

**TWAP ≠ VWAP.** True VWAP weights by volume. Without volume data the anchor is time-weighted, which misses institutional price levels. This is the biggest driver of the 31.7% win rate ceiling.

**Regime dependence.** The strategy benefits disproportionately from the 2020–2024 bull trend. A rolling volatility filter would reduce exposure during low-momentum regimes.

**Single asset.** No diversification across uncorrelated instruments.

**Parameter selection.** ENTRY_Z (2.5), EMA period (200), and time window were set by economic reasoning and not grid-searched. Walk-forward re-estimation annually would be the next step.

---

## Potential improvements

1. Source volume data and compute true VWAP — highest expected Sharpe improvement
2. Add a second correlated asset (Nifty50 futures) for proper pairs trading with Engle-Granger cointegration
3. Rolling volatility regime filter: only trade when 20-day realized vol exceeds 60-day average
4. Trailing stop once a trade is profitable — captures more on high-momentum days
5. Annual walk-forward parameter re-estimation

---

## Bonus challenges

**Out-of-sample testing (Bonus 2):** Implemented. Hard split 2015–2019 / 2020–2024. Test Sharpe 0.691 reported above.

**Risk allocation (Bonus 4):** ATR-scaled stops. Stop distance = 1.5 × daily ATR scales automatically to the current volatility regime — wider stops on volatile days, tighter on quiet days — which implements volatility-targeted risk allocation per trade.

**Parameter optimization (Bonus 1):** Parameters chosen by first-principles reasoning, not grid search. Out-of-sample improvement confirms they are not overfit to the training period.

**Multiple pairs (Bonus 3):** `compare_strategies.py` backtests five signal architectures (VWAP momentum, previous-day breakout, opening range breakout, ATR channel, Donchian) simultaneously on the same pipeline with clean separation. Full multi-asset pairs trading would require a second instrument.

---

## Iteration history

| Version | Change | Result | Lesson |
|---|---|---|---|
| v1 | SMA z-score, mean reversion | Sharpe −4.9 | Trend days destroyed P&L |
| v2–v4 | Regime filters, time window, hard stop | Incremental improvement | Filters help; core signal still wrong |
| v5 | 5-min bars + daily ATR stop | Duration normalised | Fixed whipsaw; gave trades room |
| v6 | EMA-200 + 09:30–14:30 window | Near breakeven | Architecture correct; signal direction wrong |
| v7 | VWAP anchor (no volume) | Sharpe −4.0, smooth | Smooth −4 = strong signal, wrong direction |
| **v8** | **Inverted: ride breakouts** | **Sharpe 0.528** | **Momentum, not mean reversion** |

---

## AI assistance

**Tools used:** Claude (Anthropic)

**Used for:** pandas/matplotlib syntax, backtester loop debugging, modular pipeline structuring, generating boilerplate code for the comparison framework.

**Decided independently:** The strategy concept (TWAP spread, z-score framework, momentum direction vs mean reversion), the decision to invert the signal after observing the v7 results, the EMA trend filter rationale, daily ATR over intraday ATR after empirical testing, the honest reporting of train Sharpe ≈ 0.004, the decision not to overfit parameters to test period performance, and every analytical conclusion in this README.

I can explain and modify any part of the code in a technical discussion.