## 3. Data

The benchmark universe holds {{n:runs.benchmark.meta.n_instruments}} instruments with daily
bars, {{n:runs.benchmark.meta.n_rows:,}} instrument-days in total. It spans equity indices
(34 indices across five regions), continuous futures (equity, rates, energy,
metals, agriculture, softs, livestock, currencies), FX majors, crosses and emerging-market
pairs, large-cap cryptocurrencies, and US-listed ETFs (equity, country, sector, rates,
credit, commodity, real estate, currency). Two extra panels are kept out of the
benchmark:

- Binance spot crypto at 1-hour and 4-hour bars (frequency robustness).
- Single stocks from the *current* S&P 100 and NIFTY 50 (survivorship-biased by
  construction; used only as a labelled robustness run).

Over all frequencies we use {{n:data.n_series_ok}} cleaned series
({{n:data.n_bars_total:,}} bars). Table T1 lists coverage per group. Sources, cleaning
rules and limitations are in `docs/TFML_DATA.md`. Cleaning removed
{{n:data.spikes_removed}} isolated bad prints, widened {{n:data.hl_fixed:,}} inconsistent
high/low values (mostly Yahoo FX), and trimmed series with holes longer than one month to
the segment after the last hole.

{{table:T1_data_coverage|group,freq,instruments,first_start,end,total_bars,spikes_removed,rows_trimmed_gap,source,survivorship,benchmark}}

## 4. Strategies and indicators

We implement {{n:n_strategies}} published or widely used trend-following rules (Table T2).
They span eight families: moving-average crossovers, price-vs-MA filters, adaptive moving
averages, channel breakouts, volatility stops, time-series momentum, oscillator-based
trend rules and directional-movement systems. Each rule maps OHLC history up to the close
of bar *t* to a target position *p_t* ∈ [−1, 1], with parameters set to the originator's
defaults. Rules are long/short (stop-and-reverse or with a flat state where the original
has one). The two long-only exceptions follow their sources: Faber's 10-month MA and
Antonacci's absolute momentum. For parameter adaptation (stage S5) each rule also has a
*slow* (lookbacks ×2) and a *fast* (×0.5) variant.

The ML models see {{n:n_features}} features drawn from {{n:n_indicator_families}}
indicator families (Table T3): momentum at 5–252 bars; five volatility estimators
(close-to-close, Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang); ADX/DMI; Aroon;
RSI; MACD; stochastics; CCI; TRIX; Vortex; Kaufman efficiency ratio; choppiness;
regression slope t-statistics and R²; distances to SMA/KAMA/PSAR/SuperTrend/Ichimoku
levels; Bollinger, Keltner and Donchian channel positions; variance ratio and
autocorrelation; higher moments; drawdown state; and four volume features. Every feature
is dimensionless (ratio, oscillator or volatility-scaled distance), so one pooled model
serves all instruments.

Two causality checks are unit-tested for all 56 features and 39 rules: the value at bar *t*
is identical whether computed on the full history or on history truncated at *t*. That
test caught one subtle look-ahead during development. A month-end rebalancing flag set on
the last available bar made a truncated series rebalance on a day the full series would
not.

## 5. ML enhancement stages and models

ML is inserted at five clearly separated points. Each stage is evaluated **on its own
against the same rule-based baseline**, so its incremental effect is identified:

| Stage | Pipeline role | Model target | Decision rule |
|---|---|---|---|
| S1 filter | signal filtering / trade selection (meta-labelling) | P(trade's net open-to-open return > 0), features at entry | skip a new trade if its probability falls below a validation-quantile threshold; stay flat until the next signal |
| S2 regime | trend / regime detection (strategy-agnostic) | P(forward 20-bar Kaufman efficiency ratio > training median) | hold the rule's position only while the trending-regime probability exceeds the threshold |
| S3 sizing | position sizing | P(direction of current position is right over the next 10 bars) | scale the position by 2·F(q), where F is the empirical CDF of validation predictions (mean multiplier ≈ 1, range 0–2) |
| S4 exit | entry/exit decisions | same as S3 | exit an open trade when the probability drops below the threshold; re-enter only on the next rule signal |
| S5 params | parameter adaptation | which of the slow/base/fast variants has the best net return over the next 21 bars (3-class) | each month, trade the variant with the highest predicted probability |

Bar-level model outputs (S2–S4) go through a causal 5-bar EMA before any decision, a
setting fixed before the study. Without it, day-to-day prediction noise churned positions
and multiplied turnover. The intervention intensity (S1/S2: 20/35/50% quantile; S4:
10/20/30%) is the only decision parameter tuned, and only on validation data.

Every stage trains **five models** with hyper-parameters fixed a priori:

- L2-regularised logistic regression.
- A random forest (100 bagged trees, depth 7, 15% features per tree).
- Extremely randomised trees.
- LightGBM gradient boosting (200 trees, 15 leaves, learning rate 0.03).
- A two-layer MLP (32-16) with early stopping.

Minimum leaf sizes scale with the training-set size. Models are pooled across all
instruments of the panel (one model per strategy, stage and fold). Bar-level training
rows are sub-sampled at the label horizon, so labels from one instrument do not overlap.

## 6. Validation and backtesting protocol (benchmark configuration)

**Walk-forward with purging.** Test years run from {{n:runs.benchmark.meta.first_test_year}}
to 2026, one fold per calendar year. For test year *Y*:

- Models are trained on all data before *Y*−2 (expanding window).
- Model and intensity are selected on the validation block [*Y*−2, *Y*).
- The frozen choice is applied to *Y*.

A training row is admitted only if its label is fully realised (horizon plus a 5-bar
embargo) before the validation block starts. Validation labels must be realised before
the test year. The *selected* configuration is the (model, intensity) pair with the
highest validation-period portfolio Sharpe ratio. Each alternative model keeps its own
validation-chosen intensity, so all five models are also reported out of sample, but
nothing out of sample feeds back into any choice. The out-of-sample (OOS) record is the
concatenation of the test years, {{n:runs.benchmark.meta.oos_period}}.

**Execution and costs.** A target decided at the close of bar *t* is executed at the
**open of bar t+1**. The overnight gap belongs to the previous position, so a signal never
earns the jump that created it. Each unit of position change pays a per-side cost plus
slippage by asset class: 2 bps for indices and ETFs, 3 bps for futures, 1–6 bps for FX
majors/crosses/EM, and 12.5 bps for crypto (taker fee plus slippage).

**Sizing and aggregation.** At the instrument level, rules trade unit notional, as in
their original specifications. At the portfolio level, each instrument's position is
scaled to 10% annualised volatility (EWMA, causal, leverage cap 3×). Instruments get
equal capital, and the portfolio return on a date is the sum of instrument returns
divided by the number of instruments live at that date.

**Alternative mechanisms**, reported for the baseline and every selected ML stage:

- costs at 0× and 3×
- next-close execution (one bar of extra delay)
- same-close execution (the optimistic textbook convention)
- unit sizing

Alternative validation designs are separate runs:

- a rolling 10-year training window
- a single 2013 hold-out split with no retraining

## 7. Statistical inference

For each strategy and stage we test H₀: SR(ML) = SR(rule-based) on the paired daily OOS
portfolio returns:

- **Portfolio level:** a paired circular block bootstrap (block 20 days, 2,000 resamples).
- **Instrument level:** the Ledoit–Wolf (2008) HAC delta-method test.

The headline family is the 39 strategies × 5 stages = 195 portfolio-level tests, reported
raw and with Holm (FWER) and Benjamini–Hochberg (FDR) adjustment. Within each strategy,
Hansen's SPA test and White's Reality Check ask whether *any* of the 25 stage × model
configurations beats the baseline. We also report Romano–Wolf stepdown p-values. To
diagnose selection overfitting we compute the probability of backtest overfitting (CSCV,
16 blocks) across each strategy's 26 configurations and the deflated Sharpe ratio of the
ex-post best configuration. Finally, an OLS regression with instrument-clustered standard
errors relates instrument-level ΔSharpe to stage, model, strategy family, asset class and
the baseline's own Sharpe: the "under what conditions" question.
