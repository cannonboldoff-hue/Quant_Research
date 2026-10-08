---
title: "Does Machine Learning Improve Trend Following? A Stage-by-Stage, Multi-Asset Walk-Forward Evaluation of 39 Rule-Based Strategies"
author:
  - "[Author Name]^1^ — Corresponding author: kirannandi2412@gmail.com"
  - "^1^ [Affiliation, City, Country]"
date: "Manuscript version: October 2026"
link-citations: true
nocite: |
  @*
abstract: |
  We quantify how much machine learning (ML) adds to 39 established trend-following rules and
  identify the conditions under which the improvement is statistically robust. ML is inserted
  at five separately evaluated stages of the trading pipeline (trade filtering, regime gating,
  position sizing, exit timing and parameter adaptation), with five model families per stage
  selected on a validation block that precedes each test year in a purged, expanding
  walk-forward design. On 174 instruments (equity indices, futures, currencies,
  cryptocurrencies and ETFs) traded out of sample from January 2006 to October 2026 with
  next-open execution, asset-class transaction costs and volatility targeting, the rule-based
  baselines earn a mean Sharpe ratio of 0.01. ML trade filtering and ML position sizing raise
  it by 0.47 and 0.44 on average, with significant improvements for 31 and 33 of the 39
  strategies after Benjamini–Hochberg correction. ML exits add 0.28, parameter adaptation 0.16
  and a strategy-agnostic regime gate nothing. The gains survive a random-prediction placebo,
  a random-walk null experiment, alternative validation designs, every five-year sub-period
  and zero to triple transaction costs. They are absent for G10 currencies and, for position
  sizing, at intraday cryptocurrency frequencies. A feature ablation shows that the
  improvement does not require technical indicators: a model that sees only volatility and the
  state of the open trade matches a model given 56 indicators. ML therefore acts as a learned
  trade-management layer rather than as an extractor of indicator information. The
  improvements are statistically robust, but most ML-enhanced rules remain only marginally
  profitable on their own.
---

**Keywords:** trend following; technical trading rules; machine learning; meta-labelling;
walk-forward validation; backtest overfitting; multiple hypothesis testing; placebo test.

**JEL classification:** G11, G14, G17, C12, C45, C53.

# 1. Introduction

Trend following is among the oldest and best-documented systematic investment styles. Its
building blocks (moving-average crossovers, channel breakouts, volatility stops and
time-series momentum) are simple, public and, in most cases, several decades old
[@donchian1960; @wilder1978; @brock1992; @moskowitz2012; @hurst2017]. Over the past decade it
has become routine to bolt machine learning onto such rules: classifiers filter signals,
regime models switch strategies on and off, and predicted probabilities set position sizes
[@lopezdeprado2018; @lim2019; @joubert2022]. The value of doing so is, however, rarely
measured against the rule itself under one controlled protocol. Academic studies typically
report the profitability of an ML strategy in isolation, and practitioner evidence usually
consists of a filter that "helped" on the instrument on which it was tuned. Neither answers
the question that matters to someone who already trades a trend rule: *what does ML add to
it, and can that addition be trusted?*

This paper asks that question directly:

> How much does ML improve established trend-following strategies over their original
> rule-based implementations, and under what market, instrument, strategy, indicator, model
> and validation conditions does the improvement remain statistically significant and robust?

We answer it with a controlled, stage-by-stage design. Thirty-nine published or widely used
trend-following rules are implemented faithfully as baselines. ML is then inserted at five
distinct points of the trading pipeline (trade selection, trend-regime detection, position
sizing, exit timing and parameter adaptation), and each insertion is evaluated *on its own*
against the same baseline on identical days. At every stage, five model families compete
(regularised logistic regression, random forest, extremely randomised trees, gradient
boosting and a neural network); the winner is chosen only on a validation block that precedes
each out-of-sample year, and every alternative is reported as well. The test bed is a public
multi-asset daily panel of 174 instruments across equity indices, futures, currencies,
cryptocurrencies and ETFs, with robustness panels at one-hour and four-hour frequency and a
deliberately survivorship-biased single-stock universe that is labelled as such.

Because the design involves many comparisons (39 strategies × 5 stages × 5 models, plus
backtest variants), the inference accounts explicitly for the size of the search: paired
block-bootstrap and HAC tests of Sharpe-ratio differences, Holm and Benjamini–Hochberg
adjustment across the 195 strategy–stage tests, Hansen's test of superior predictive ability
and White's Reality Check within each strategy, Romano–Wolf stepdown p-values, the probability
of backtest overfitting and the deflated Sharpe ratio. We add three controls that are, to our
knowledge, rarely used together in this literature: (i) a *placebo* that applies each ML
decision rule with random numbers instead of model predictions, which isolates the
information contributed by the model from the mechanical effect of intervening; (ii) a
*null experiment* that runs the entire pipeline unchanged on synthetic random-walk markets as
a screen for look-ahead leakage; and (iii) a *feature ablation* that asks which inputs carry
the improvement.

Our main findings are as follows. First, *where* ML is inserted matters more than *which*
model is used. Trade filtering (meta-labelling) and position sizing raise the out-of-sample
Sharpe ratio of the average rule by 0.47 and 0.44, from a baseline of essentially zero, and
the improvement is significant for 31 and 33 of the 39 rules after false-discovery-rate
control. ML exits help less reliably (+0.28), parameter adaptation little (+0.16), and a
strategy-agnostic trend-regime gate does not help at all. Second, the improvements are not an
artefact of trading less: the placebo is negative or null at every stage, and the random-walk
experiment produces no broad improvement. Third, the gains are robust to the validation
design, the time period and the backtest assumptions, but not to the sampling frequency or
to the most efficiently priced market (G10 currencies). Fourth, ML mainly *repairs weak
rules*: the benefit falls sharply with the rule's own Sharpe ratio. Fifth, and most
importantly for interpretation, the improvement does not depend on the technical indicators
at all. A model that sees only recent volatility and the state of the open trade performs as
well as one that sees 56 indicators. Finally, a statistically robust improvement is not the
same as a profitable strategy: the absolute Sharpe ratio of the best ML configuration survives
deflation for only 9 of the 39 rules.

The paper makes four contributions. (1) It provides a controlled, stage-by-stage measurement
of the incremental value of ML for a large, diverse set of published trend rules, under one
protocol and with honest out-of-sample model selection. (2) It reports the full search
(every model, every stage, every backtest variant) together with multiple-testing and
overfitting diagnostics, rather than only the best configuration. (3) It introduces a placebo
control and a feature ablation that separate learned information from mechanical
intervention and identify the source of the gain. (4) It releases a modular, tested framework
in which each experiment is recorded in a registry that links strategy, indicators, market,
instrument, dataset, ML stage, model, validation design and backtest configuration to its
result.

The remainder of the paper is organised as follows. Section 2 reviews the related literature.
Section 3 describes the data and Section 4 the methodology. Section 5 presents the main
results, Section 6 the controls that test whether the improvement reflects learned
information, and Section 7 the robustness experiments. Section 8 answers the research
question and discusses the implications, Section 9 states the limitations and Section 10
concludes.

# 2. Related literature

**Trend following and technical rules.** Systematic trend following has a long empirical
record. Donchian's channel-breakout rule [-@donchian1960] and the moving-average and
trading-range-break rules tested by @brock1992 are the canonical technical rules. @moskowitz2012
document time-series momentum across 58 futures and forward contracts, and @hurst2017 extend
the evidence to more than a century of data. @levine2016 show that moving-average crossovers
and time-series momentum are closely related, weighting past returns differently. Simple
tactical rules such as @faber2007's ten-month moving-average filter and @antonacci2014's
absolute momentum are widely followed by practitioners. @sullivan1999 showed that the apparent
profitability of a *universe* of technical rules must be judged against the data snooping
implied by searching over it, a point that applies with greater force when ML is layered on
top.

**Machine learning in trading.** @gu2020 document substantial gains from tree ensembles and
neural networks in predicting the cross-section of returns under strict
train/validation/test separation. @krauss2017 and @fischer2018 apply deep networks, gradient
boosting and random forests to statistical arbitrage. Closer to our setting, @lim2019 learn
trend-following positions and sizes end-to-end with deep networks. @lopezdeprado2018 proposed
*meta-labelling*: a secondary classifier that decides whether to act on a primary model's
signal and how much to bet, and @joubert2022 formalises the framework. Our stage S1 is a
meta-labelling filter and stage S3 a meta-labelling sizer; the contribution here is not the
technique but its controlled evaluation against 39 primary rules, alongside other
insertion points.

**Backtest overfitting and multiple testing.** @white2000's Reality Check and
@hansen2005's test of superior predictive ability (SPA) test whether the best of many models
beats a benchmark. @romano2005 provide stepdown family-wise error control. @benjamini1995 and
@benjamini2001 control the false discovery rate. @ledoit2008 provide robust tests of
Sharpe-ratio differences. @bailey2014 propose the deflated Sharpe ratio, and @bailey2017 the
probability of backtest overfitting (PBO) estimated by combinatorially symmetric
cross-validation. @harvey2016 argue for higher significance hurdles in factor research, and
@arnott2019 set out a backtesting protocol for the ML era. We apply this toolkit to a
search of 39 strategies × 5 stages × 5 models.

**The gap.** What is missing is a *controlled, stage-by-stage* measurement. The question is
not whether an ML trading model can be profitable, but how much ML adds to a given, published
trend rule, at which point in the pipeline, and with what statistical confidence once the
search over models and configurations is accounted for.

# 3. Data

## 3.1 Universe

The benchmark universe consists of 174 instruments with daily bars, totalling 1,100,307
instrument-days. It covers 34 equity indices across five regions; 37 continuous futures
contracts (equity, interest rates, energy, metals, agriculture, softs, livestock and
currencies); 7 G10 currency majors, 21 G10 crosses and 11 emerging-market currency pairs; 15
large-capitalisation cryptocurrencies; and 49 US-listed ETFs (equity, country, sector, rates,
credit, commodity, real estate and currency). The universe deliberately avoids constituent
selection: indices, continuous futures, currencies, ETFs and long-lived cryptocurrencies are
not chosen because they survived, which limits survivorship bias.

Two further panels are kept out of the benchmark. Binance spot cryptocurrency data at one-hour
and four-hour bars are used to test sensitivity to the sampling frequency. Single stocks from
the *current* S&P 100 and NIFTY 50 constituents are survivorship-biased by construction and
are used only as a labelled robustness run. Over all panels and frequencies we use 259 cleaned
series with 2,670,334 bars (Table 1). Data come from Yahoo Finance, the Binance public API and
FRED.

Table: Table 1. Data coverage by instrument group.

{{tbl:data}}

::: {.notes custom-style="Table Note"}
*Notes:* "From" is the earliest start date in the group; all series end in September or October
2026. Instruments enter the panel when their history begins. Daily Binance bars are used only
for cross-source validation.
:::

## 3.2 Cleaning and validation

Cleaning is mechanical and identical for all instruments. Duplicate dates and missing or
non-positive closes are dropped. Isolated bad prints, defined as log returns larger than 12
robust (MAD-based) standard deviations that are immediately reversed on the next bar, are
removed (625 cases). Inconsistent
bars in which the high or low lies inside the open–close range are widened to include the
open and close (16,962 cases, mostly in Yahoo currency data). Series with holes longer than
one month are trimmed to the segment after the last hole. Three requested series could not
be used: sugar futures (no usable history), the US dollar index future and one Indian stock
that is no longer listed on the data source.

We validate the data by comparing overlapping series from different sources (Appendix Table
C1). Daily Yahoo and Binance bitcoin returns correlate at 0.99; the S&P 500 index and the SPY
ETF at 0.985 (0.995 weekly); and domestic index–ETF pairs between 0.96 and 0.99. Currency spot rates and
currency futures correlate weakly at the daily frequency (about 0.4) because of differing
fixing times, but at 0.86 at the weekly frequency. These checks confirm that the series are
usable for a *comparative* study. Because the rule-based baseline and every ML variant are
evaluated on the same bars, data imperfections affect both sides of each comparison.

# 4. Methodology

## 4.1 Rule-based baselines

We implement 39 published or widely used trend-following rules spanning eight families:
moving-average crossovers (7 rules), price-versus-moving-average filters (2), adaptive moving
averages (6), channel breakouts (5), volatility stops (4), time-series momentum (4),
oscillator-based trend rules (6) and directional-movement systems (5). Appendix Table A1 lists
every rule with its reference and parameters. Each rule maps the OHLC history up to the close
of bar $t$ to a target position $p_t \in [-1, 1]$, using the originator's default parameters.
Rules are long/short (stop-and-reverse, or with a flat state where the original has one); the
two long-only exceptions, @faber2007's ten-month moving average and @antonacci2014's absolute
momentum, follow their sources. For parameter adaptation (stage S5) each rule also has a
*slow* variant, with all lookbacks doubled, and a *fast* variant, with lookbacks halved.

## 4.2 Features

The ML models see 56 features drawn from ten indicator families (Appendix Table A2). They
include momentum over 5 to 252 bars; five volatility estimators (close-to-close,
@parkinson1980, @garman1980, @rogers1991 and @yang2000); ADX/DMI, Aroon, RSI, MACD,
stochastics, CCI, TRIX and Vortex; Kaufman's efficiency ratio and the choppiness index;
regression-slope t-statistics and $R^2$; distances to SMA, KAMA, Parabolic SAR, SuperTrend and
Ichimoku levels; Bollinger, Keltner and Donchian channel positions; the variance ratio of
@lo1988 and the first-order autocorrelation; higher moments; drawdown state; and four volume
features. Every feature is dimensionless (a ratio, a bounded oscillator or a
volatility-scaled distance), so a single pooled model can serve all instruments.

The rule-specific stages (S1, S3, S4) additionally receive six *trade-state* features: the
current position's sign and size, the log age of the current trade, the volatility-scaled
open profit or loss, and the volatility-scaled result and log length of the previous
completed trade. These are causal functions of prices and of the rule's own signals, but they
are not technical indicators. Stage S5 receives the current positions of the three variants
instead, and the regime model (S2) receives indicators only.

**Causality.** For every one of the 56 features and 39 rules, a unit test verifies that the
value at bar $t$ is identical whether it is computed on the full history or on the history
truncated at $t$. This test caught one subtle look-ahead during development: a month-end
rebalancing flag set on the last available bar caused a truncated series to rebalance on a
day on which the full series would not. The flag is now derived from the calendar.

## 4.3 ML enhancement stages

ML is inserted at five clearly separated points. Each stage is evaluated *on its own* against
the same rule-based baseline, so that its incremental effect is identified. Table 2
summarises the stages.

Table: Table 2. ML enhancement stages, model targets and decision rules.

| Stage | Pipeline role | Model target | Decision rule |
|:---|:---|:---|:---|
| S1 Filter | Trade selection (meta-labelling) | $P$(trade's net open-to-open return $>0$), features at entry | Skip a new trade if its probability is below a validation-quantile threshold; stay flat until the next signal |
| S2 Regime | Trend-regime detection (strategy-agnostic) | $P$(forward 20-bar efficiency ratio $>$ training median) | Hold the rule's position only while the trending-regime probability exceeds the threshold |
| S3 Sizing | Position sizing | $P$(current position direction is right over the next 10 bars) | Scale the position by $2\hat F(\hat s_t)$, where $\hat F$ is the empirical CDF of validation predictions |
| S4 Exit | Exit timing | As S3 | Exit an open trade when the probability falls below the threshold; re-enter only on the next rule signal |
| S5 Params | Parameter adaptation | Which of slow/base/fast variants has the best net return over the next 21 bars (3 classes) | Each month, trade the variant with the highest predicted probability |

Bar-level predictions $\hat\pi_t$ (stages S2–S4) are smoothed by a causal five-bar
exponential moving average before any decision,
$$\hat s_t = \alpha\,\hat\pi_t + (1-\alpha)\,\hat s_{t-1}, \quad \alpha = 2/6, \qquad\qquad (1)$$
a setting fixed before the study because unsmoothed day-to-day prediction noise churned
positions and multiplied turnover. The S1 and S2 decisions compare the prediction with the
$\tau$-quantile of the validation-period predictions, $\tau \in \{0.20, 0.35, 0.50\}$; S4 uses
$\tau \in \{0.10, 0.20, 0.30\}$. For S3 the multiplier
$$m_t = 2\,\hat F_{\text{val}}(\hat s_t) \in [0, 2] \qquad\qquad (2)$$
has a mean close to one, so sizing redistributes exposure without changing its average
level. The intensity $\tau$ is the only decision parameter tuned, and only on validation data.

## 4.4 Models

Every stage trains five models whose hyper-parameters are fixed a priori:
L2-regularised logistic regression; a random forest [@breiman2001] of 100 bagged trees of
depth 7 with 15% of features per tree; extremely randomised trees [@geurts2006]; LightGBM
gradient boosting [@ke2017] with 200 trees, 15 leaves and a learning rate of 0.03; and a
two-layer perceptron (32 and 16 units) with early stopping. Minimum leaf sizes scale with the
size of the training set. Models are pooled across all instruments of a panel, with one model
per strategy, stage and fold. Training rows are sub-sampled at the label horizon so that labels
from the same instrument do not overlap.

## 4.5 Walk-forward validation with purging

Test years run from 2006 to 2026, with one fold per calendar year. For test year $Y$:

1. models are trained on all data before $Y-2$ (expanding window);
2. the model and the intervention intensity are selected on the validation block $[Y-2, Y)$;
3. the frozen choice is applied to year $Y$.

A training row is admitted only if its label is fully realised, including a five-bar embargo,
before the validation block starts, and validation labels must be realised before the test
year begins [@lopezdeprado2018]. The *selected* configuration is the (model, intensity) pair
with the highest validation-period portfolio Sharpe ratio. Each alternative model keeps its
own validation-chosen intensity, so all five models are also reported out of sample, but
nothing out of sample feeds back into any choice. The out-of-sample (OOS) record is the
concatenation of the test years, from 2 January 2006 to 6 October 2026. Two alternative
designs are run separately: a rolling ten-year training window and a single 2013 hold-out
split with no retraining.

## 4.6 Execution, costs and portfolio construction

A target position decided at the close of bar $t$ is executed at the *open* of bar $t+1$.
The overnight gap is earned by the previous position, so a signal never earns the price jump
that created it. With $h_t = q_{t-1}$ the position held during bar $t$, open $O_t$ and close
$C_t$, the net return of an instrument on bar $t$ is
$$r_t = h_{t-1}\left(\frac{O_t}{C_{t-1}} - 1\right) + h_t\left(\frac{C_t}{O_t} - 1\right) - \kappa\,\lvert h_t - h_{t-1}\rvert, \qquad\qquad (3)$$
where $\kappa$ is the per-side cost plus slippage of the asset class: 2 bp for indices and ETFs,
3 bp for futures, 1 to 6 bp for currency majors, crosses and emerging-market pairs, and 12.5 bp
for cryptocurrencies.

At the instrument level, rules trade unit notional as in their original specifications
($q_t = p_t$). At the portfolio level, each instrument's position is scaled to 10% annualised
volatility,
$$q_{i,t} = p_{i,t}\,\min\!\left(\frac{0.10}{\hat\sigma_{i,t}},\, 3\right), \qquad\qquad (4)$$
where $\hat\sigma_{i,t}$ is a causal exponentially weighted volatility estimate (60-day span).
Instruments receive equal capital, and the portfolio return on a date is the average of the
returns of the $N_t$ instruments live on that date,
$R_t = N_t^{-1}\sum_i r_{i,t}$. We additionally report costs at 0× and 3×, next-close
execution (one extra bar of delay), same-close execution (the optimistic textbook convention)
and unit sizing.

## 4.7 Performance measures

Returns are returns on notional exposure, following the excess-return convention of the
managed-futures literature, so no risk-free rate is subtracted. The annualised Sharpe ratio is
$\mathrm{SR} = \sqrt{A}\,\bar R / \hat\sigma_R$, with $A$ the number of bars per year. The
quantity of interest is the paired difference
$$\Delta \mathrm{SR} = \mathrm{SR}(\text{ML}) - \mathrm{SR}(\text{rule}), \qquad\qquad (5)$$
computed on identical days. We also report CAGR, total return, volatility, the Sortino and
Calmar ratios, maximum drawdown, trade win rate, profit factor, annual turnover, cost drag,
exposure, the share of positive calendar years and a stability measure (the signed $R^2$ of
log equity on time).

## 4.8 Statistical inference

For each strategy and stage we test $H_0\!: \mathrm{SR}(\text{ML}) = \mathrm{SR}(\text{rule})$
on the paired daily OOS returns. At the portfolio level we use a paired circular block
bootstrap [@politis1992] with 20-day blocks and 2,000 resamples, which preserves both the
serial dependence and the cross-correlation of the two return series. At the instrument level
we use the HAC test of @ledoit2008. The headline family consists of the 39 × 5 = 195
portfolio-level tests, reported raw and with Holm (family-wise) and Benjamini–Hochberg (false
discovery rate) adjustment. Within each strategy, Hansen's SPA test and White's Reality Check
ask whether *any* of the 25 stage × model configurations beats the baseline, and Romano–Wolf
stepdown p-values are computed. To diagnose selection overfitting we compute the probability
of backtest overfitting across each strategy's 26 configurations (combinatorially symmetric
cross-validation with 16 blocks) and the deflated Sharpe ratio of the ex-post best
configuration. Finally, an ordinary least squares regression with instrument-clustered
standard errors relates instrument-level $\Delta\mathrm{SR}$ to the stage, model, strategy
family, asset class and the baseline's own Sharpe ratio.

## 4.9 Controls: placebo, random-walk null and feature ablation

*Placebo.* Removing trades or resizing positions can change a Sharpe ratio even without any
information. Each decision rule is therefore re-run with the same validation-chosen intensity
but with uniform random numbers in place of the model predictions (20 draws per strategy and
stage). The difference between the ML and placebo results measures what the model *learns*.

*Random-walk null.* If any component leaked future information, ML would "improve" rules even
where there is nothing to predict. The whole pipeline is run unchanged on 60 synthetic
random-walk instruments with GARCH-$t$ returns and the same costs, for 8 representative
strategies.

*Feature ablation.* Stages S1, S3 and S4 are re-run with the feature set restricted to one
indicator family at a time (momentum, volatility, trend, oscillator/channel, other) and to no
indicator at all ("context only": `vol_20` plus the trade-state features). For computational
reasons the ablation uses a lighter protocol (two models, logistic regression and LightGBM,
retraining every three years) on 15 representative strategies covering all eight families, so
the blocks are compared with each other and with a full-feature reference run under the same
protocol.

## 4.10 Reproducibility and experiment registry

Every experiment writes rows to an SQLite registry that links strategy → indicators → market
→ instrument → dataset hash → ML stage → ML model (with per-fold selection counts) →
validation design → backtest configuration → metrics → stored return series, together with a
hash of the source code that produced it. All tables in this paper are generated from that
registry. Data downloads, cleaning, experiments, statistics and the paper itself are
reproduced by scripts, and every component (universe, data source, frequency, features,
strategies, models, stages, validation scheme, execution, costs and sizing) can be swapped
through configuration.

# 5. Results

All results in this section are out of sample, from January 2006 to October 2026, on 174
instruments with next-open execution and 1× costs, unless stated otherwise. $\Delta$SR is the
annualised Sharpe ratio of the ML-enhanced strategy minus that of its own rule-based baseline
on identical days. The *selected* model is the validation winner, never the ex-post best.

## 5.1 The rule-based baselines

After costs, the 39 published rules earn an average portfolio Sharpe ratio of 0.01 (median
−0.01) over 2006–2026, and only 18 of the 39 have a positive Sharpe ratio (Appendix Table B1).
The best rules are slow, long-only or monthly: @faber2007's ten-month moving average reaches
0.85 and absolute momentum 0.84. The worst is the fractal adaptive moving average at −0.83.
This is consistent with the well-documented weak decade for trend following after 2009. Fast
crossovers and adaptive moving averages pay substantial costs for whipsaw trades: the
fastest rules have annual portfolio turnover of 50 to 95 times their capital. Each instrument is scaled to
10% volatility, but the equal-capital portfolio of 174 weakly correlated instruments has a
realised volatility of only 2–3% a year. CAGRs are therefore small in absolute terms and scale
linearly with any leverage applied; the Sharpe, Sortino and Calmar ratios are the comparable
quantities.

## 5.2 How much does ML improve the rules?

Table 3 reports the headline result: the out-of-sample $\Delta$SR of the validation-selected
model at each stage, across the 39 strategies.

Table: Table 3. Improvement in Sharpe ratio from ML, by stage (benchmark, validation-selected model, 39 strategies).

{{tbl:headline}}

::: {.notes custom-style="Table Note"}
*Notes:* "Sig.+" counts strategies with a significant improvement (two-sided paired circular
block bootstrap, $\alpha = 5\%$), raw and after Holm and Benjamini–Hochberg (BH) adjustment over
all 195 strategy × stage tests; "Sig.− BH" counts significant deteriorations. ΔMaxDD is the
change in maximum drawdown in percentage points (positive = shallower). ΔTurnover is the
change in annual portfolio turnover. The sign test is across strategies.
:::

The answer depends sharply on *where* ML is inserted. Trade selection (S1) and position
sizing (S3) give the largest and most reliable improvements: mean $\Delta$SR of +0.47 and
+0.44, with 38 of the 39 strategies improved in both cases, and significant improvements for
31 and 33 strategies after BH adjustment (10 and 16 after the stricter Holm correction). Both
also reduce maximum drawdowns. ML exits (S4) help on average (+0.28), but less reliably
individually: 15 strategies are significant after BH and one after Holm. Parameter adaptation
(S5) adds little (+0.16; four strategies significant after BH, none after Holm).

The strategy-agnostic trend-regime gate (S2) does not help: its mean $\Delta$SR is −0.01 and
only 46% of strategies improve. Predicting *that* a market will trend is not the same as
predicting that a given rule's current position will pay; only the latter, strategy-specific
target produced gains. S2 is also the only stage at which validation-based selection actively
hurts. Every individual model except the MLP has a slightly positive mean OOS $\Delta$SR
(Table 5), but the fold-by-fold validation winner does not: the gate's effects are too small
and unstable for a two-year validation block to rank them reliably.

Across all 195 tests, 27 improvements survive the Holm correction and 84 survive BH, while
only one significant deterioration survives BH. Within each strategy, Hansen's SPA test
rejects the hypothesis that no ML configuration beats the rule for 38 of the 39 strategies
(38 after Holm adjustment across strategies); White's Reality Check rejects it for 37.
Figure 1 shows the full strategy × stage grid and Figure 2 the distribution across strategies.

![Figure 1. Out-of-sample ΔSR for every strategy × stage (validation-selected model). Rows are sorted by the baseline Sharpe ratio; an asterisk marks BH-adjusted p < 0.05.](figures/benchmark_F1_heatmap_dsharpe.png){width=88%}

![Figure 2. Distribution of ΔSR across the 39 strategies, by stage.](figures/benchmark_F2_dsharpe_distribution.png){width=80%}

## 5.3 A diversified view: the 39-strategy composite

An investor would rarely trade a single rule. Table 4 reports an equal-weight composite of
all 39 strategies. The rule-based composite has a Sharpe ratio of 0.01. With ML trade
selection it reaches 0.60 ($\Delta$SR +0.59, 95% confidence interval [0.24, 0.96],
$p = 0.003$), and with ML sizing 0.57 ($p = 0.001$). The maximum drawdown falls from −9.5% to
−4.1% with S1, and the share of positive calendar years rises from 52% to 71% (S1) and 81% (S3).
The regime gate leaves the composite unchanged ($p = 0.93$). Because the composite is highly
diversified across instruments and rules, its volatility is low (2.2% for the baseline), so
Sharpe ratios rather than raw CAGRs are the meaningful comparison. Figure 3 shows cumulative
returns.

Table: Table 4. Equal-weight composite of all 39 strategies: rule-based versus each ML stage (benchmark).

{{tbl:composite}}

::: {.notes custom-style="Table Note"}
*Notes:* ΔSR, its 95% confidence interval and p-value come from the paired circular block
bootstrap (20-day blocks, 2,000 resamples). "Pos. years" is the share of calendar years with a
positive return.
:::

![Figure 3. Cumulative out-of-sample log return of the 39-strategy composite: rule-based versus each ML stage.](figures/benchmark_F3_composite_equity.png){width=85%}

## 5.4 Portfolio versus single instruments

At the instrument level, where the rules trade unit size exactly as published, the
improvements are real but much smaller. Across the 6,786 strategy–instrument pairs, S1 raises
the Sharpe ratio by +0.12 on average and improves 77% of pairs, but only 86 pairs (1.3%) are
individually significant after BH adjustment. S3 gives +0.12 (76% improved, 264 significant),
S4 +0.08, S5 +0.04 and S2 +0.01, with 12 significant deteriorations for S2. Single-instrument
Sharpe ratios are noisy over twenty years, and the portfolio-level gains arise because small,
consistent per-instrument improvements accumulate across 174 weakly correlated instruments. A
practitioner who trades one market should expect the instrument-level effect, not the
portfolio-level one.

## 5.5 Which model, and does validation-based selection work?

Table 5 reports the OOS $\Delta$SR of every model at every stage, how often each was selected
on validation, and how the validation-selected configuration compares with the alternatives.

Table: Table 5. Mean ΔSR by model and stage; in parentheses, the share of the 819 fold × strategy choices in which validation selected the model.

{{tbl:models}}

Tree ensembles are the most useful models. Extremely randomised trees lead at S1 (+0.54),
S3 (+0.48) and S4 (+0.36), and are the most frequently selected model at these stages; random forests follow closely, and
gradient boosting is competitive (and best at S5). The MLP is the weakest model at every
stage, and logistic regression lies in between. The instrument-level regression (Section 5.7)
confirms this ordering: relative to logistic regression, extra trees add +0.025 and random
forests +0.015 to the Sharpe ratio, while the MLP subtracts 0.026.

Validation-based selection is honest but imperfect. At S1 and S3 it does slightly better than
the average model (+0.47 against +0.45; +0.44 against +0.40) and clearly worse than the best
model ex post (+0.56; +0.48). At S4 and S5 it is slightly below the average model, and at S2
it does worse than every individual model. The gap between "selected" and "best ex post"
is the price of honest model selection: reporting the ex-post best model would overstate the
benefit of ML trade selection by about 0.09 Sharpe. The models' average validation scores
rank their OOS results consistently (Spearman correlation 0.95 at S3 and 0.75 at S1). This is
partly mechanical, because the two-year validation block of one fold covers calendar years
that are test years of earlier folds; no fold ever uses its own test year.

![Figure 4. Mean out-of-sample ΔSR by model and stage; "selected" denotes the validation winner.](figures/benchmark_F4_model_comparison.png){width=80%}

## 5.6 Markets

Table 6 breaks the results down by market group. The gains are broad but not universal.
Equity indices, ETFs, emerging-market currencies and cryptocurrencies benefit from S1 and S3
for nearly every strategy, and continuous futures benefit somewhat less. G10 currencies are the
exception: currency majors show no ML benefit at all, with S3 and S4 slightly *reducing* the
Sharpe ratio (S3: −0.07), and currency crosses show only a small S1 gain (+0.10). G10
currencies are the most efficiently priced and cheapest-to-trade market in the sample, and the
rules appear to offer little conditional structure to exploit there. The regime gate hurts in
cryptocurrencies (−0.28) and futures (−0.14).

Table: Table 6. Mean ΔSR by market group and stage (39 strategies per group); in parentheses, the number of strategies significant after BH adjustment within the stage.

{{tbl:markets}}

![Figure 5. Mean ΔSR by market group and stage.](figures/benchmark_F5_market_heatmap.png){width=78%}

## 5.7 Strategies: when does ML help most?

The strongest single predictor of the ML benefit is how good the rule already is. Table 7
regresses instrument-level $\Delta$SR on design factors. The coefficient on the baseline's own
Sharpe ratio is −0.19 ($t = -11.7$): every additional unit of baseline Sharpe ratio reduces the
ML gain by about a fifth of a unit. Conditional on the other factors, the regime and
parameter stages add significantly less than S1, the tree ensembles significantly more than
logistic regression, and currencies and futures gain 0.12 and 0.09 less than equities.

Table: Table 7. Determinants of the instrument-level ML benefit: OLS regression of ΔSR on design factors (all stages × models; n = 169,650; R² = 0.190).

{{tbl:regression}}

::: {.notes custom-style="Table Note"}
*Notes:* Standard errors are clustered by instrument. Reference categories: stage S1, logistic
regression, moving-average crossover family, equity asset class.
:::

Table 8 shows the same pattern by strategy family. The slow and already profitable
price-versus-moving-average rules (mean baseline Sharpe 0.65) gain only +0.12 from S1, and
time-series momentum (baseline 0.48) gains +0.18. The fast adaptive moving averages (baseline
−0.25) gain +0.61. ML mostly *repairs weak rules*, largely by removing trades it expects to lose:
S1 reduces annual portfolio turnover by 14 units on average. It improves established, strong
rules much less. Section 6.1 tests whether this benefit survives a placebo with the same kind
of turnover reduction.

Table: Table 8. Mean ΔSR by strategy family and stage (families sorted by baseline Sharpe ratio).

{{tbl:families}}

## 5.8 Time periods

Table 9 splits the OOS period into four sub-periods. $\Delta$SR is positive for S1, S3, S4
and S5 in every sub-period. It is largest in 2021–2026 (S1: +0.84), a period in which the rules
themselves lost money (mean baseline Sharpe −0.18), and smallest in 2006–2010, the strong
trend-following years around the global financial crisis (baseline 0.32; S1 +0.33). This is
the time-series counterpart of Section 5.7: ML helps most when the rule struggles. The regime
gate changes sign across sub-periods. Sub-period tests have far fewer observations, so only
the 2021–2026 improvements are individually significant for most strategies.

Table: Table 9. Mean ΔSR by sub-period and stage; in parentheses, the number of strategies with a significant improvement (raw, 5%).

{{tbl:subperiods}}

## 5.9 Backtest mechanics: costs, execution and sizing

Table 10 repeats the evaluation under alternative backtest assumptions. S1 and S3 remain the
two most valuable stages, and S2 the least valuable, under every assumption.

Table: Table 10. Mean ΔSR under alternative backtest assumptions; in parentheses, the numbers of strategies with significant improvements / deteriorations (raw, 5%).

{{tbl:sensitivity}}

*Costs.* Without costs, the baselines look far better (mean Sharpe 0.27) and the S1 gain
shrinks to +0.41, so part of S1's value is cost avoidance, but most of it is not. At 3×
costs the S1 advantage grows to +0.58, while the regime gate becomes clearly harmful (−0.24,
with significant deteriorations for 15 strategies). *Execution.* Executing one bar later or
at the signal close leaves the gains essentially unchanged (S1: +0.44 and +0.47), so the
improvement does not depend on fragile fill assumptions. The optimistic same-close convention
flatters the baselines only mildly (mean Sharpe 0.05, against 0.01 for next-open and −0.05 for
next-close). *Sizing.* Unit sizing without volatility targeting gives the same picture
(S1 +0.45, S3 +0.51).

![Figure 6. Mean ΔSR per stage under alternative cost, execution and sizing assumptions.](figures/benchmark_F6_backtest_sensitivity.png){width=80%}

## 5.10 Overfitting diagnostics

Across each strategy's 26 OOS configurations (baseline plus 5 stages × 5 models), the
probability of backtest overfitting is low: the median PBO is 0.009 and no strategy exceeds
0.5 (Appendix Table B2). The ex-post best configuration is almost always an S3 or S1 model.
However, the *absolute* Sharpe ratio of the best configuration survives deflation for 25
trials (deflated Sharpe ratio above 0.95) for only 9 of the 39 strategies. ML makes weak rules
markedly better, but most ML-enhanced rules are still not convincingly profitable on their own
after costs over 2006–2026. The improvement is significant; the resulting stand-alone strategy
usually is not.

# 6. Is the improvement learned information?

## 6.1 Placebo control

Table 11 compares each stage with its placebo, which applies the same decision rule at the
same intensity using random numbers instead of model predictions.

Table: Table 11. ML ΔSR versus random-prediction placebo ΔSR (benchmark, selected model, 20 placebo draws per strategy and stage).

{{tbl:placebo}}

::: {.notes custom-style="Table Note"}
*Notes:* "ML > placebo 95th pct." counts strategies whose ML ΔSR exceeds the 95th percentile of
their own placebo draws; "Same, BH" applies the Benjamini–Hochberg adjustment to these
comparisons.
:::

The mechanical effect of intervening is negative or null, so the ML gains are not an artefact
of trading less. At S1 the placebo $\Delta$SR is −0.02, against +0.47 for ML, and ML beats the
95th percentile of its own placebo for 37 of the 39 strategies. At S3, random sizing is
strongly harmful (−0.56) whereas model-based sizing helps (+0.44). S4 and S5 show the same
pattern. Even at S2, where the selected $\Delta$SR is about zero, the placebo is clearly
negative (−0.66): gating on random numbers removes profitable exposure, so the regime models
do carry information, just too little to improve on the rule.

## 6.2 Random-walk null experiment

Table 12 reports the same pipeline run on synthetic random-walk markets. No stage shows broad,
significant improvement: $\Delta$SR is +0.11 (S1), −0.20 (S2), −0.33 (S3), +0.04 (S4) and +0.07
(S5), small and mixed in sign, in contrast with gains of +0.16 to +0.47 on real data. The
instrument-level false-positive rate is 2.5% at S1 and 1.0% at S3, below the nominal 5%.

Table: Table 12. Null experiment on 60 synthetic random-walk instruments (8 strategies, GARCH-t returns, same costs).

{{tbl:null}}

Two features of the null results sharpen the interpretation of the real-data results. First,
what remains at S1 on random walks is cost avoidance, not prediction: with zero costs S1's
$\Delta$SR is −0.01, and it rises to +0.35 at 3× costs, because skipping trades saves money
even when nothing is predictable. Part of S1's benefit on real data is therefore this
mechanical saving, which is why the placebo, which also skips trades at random, is the cleaner
test of what the models learn. Second, on noise, model-based sizing and gating *hurt*: S3 is
significantly negative for 7 of the 8 strategies. Models fitted to noise produce confident but
meaningless position sizes, so the positive S3 result on real markets is evidence of genuine
structure rather than of a leak. This experiment is a screen against gross leakage, not a
proof of its absence: it covers 8 strategies, and leakage that requires real market structure
to bite would not show up.

## 6.3 Which inputs carry the improvement? Feature ablation

Table 13 reports the ablation. Every block always sees `vol_20` (needed to scale the
trade-state features) and the trade-state features themselves.

Table: Table 13. Feature ablation: mean ΔSR of the selected model when ML sees one indicator family, all 56 indicators, or no indicators (15 strategies, lighter protocol); in parentheses, strategies with a significant improvement / strategies.

{{tbl:ablation}}

::: {.notes custom-style="Table Note"}
*Notes:* All blocks use the same lighter protocol (logistic regression and LightGBM,
retraining every three years), so they should be compared with each other rather than with
Table 3. Every block includes `vol_20` and the six trade-state features.
:::

The result is clear, and it is not what the indicator-heavy framing of the study would
suggest. No indicator family is necessary: each of the five single-family blocks retains the
full S1 gain (+0.46 to +0.48, against +0.43 with all 56 indicators). No indicator at all is
needed either: with only `vol_20` and the trade-state features, the gains are +0.48 (S1),
+0.36 (S3) and +0.23 (S4), against +0.43, +0.37 and +0.28 with the full library, and the number
of strategies with a significant improvement is the same (13 of 15 at S1). The broader
indicator set adds no detectable value beyond volatility and the state of the open trade
(Figure 7).

![Figure 7. Mean ΔSR (selected model) when ML sees one indicator family, all 56 indicators, or no indicators at all. All blocks use the same lighter protocol.](build/F7_ablation.png){width=80%}

The mechanism is not tested here. A plausible reading is that the models learn
state-dependent trade management: cut trades that are losing relative to their volatility,
let older winning trades run, and avoid re-entering immediately after a loss. In effect this
is a learned stop-and-trail rule. This reading is consistent with the S1, S3 and S4 results,
with the finding that weak, high-turnover rules gain most, and with the placebo evidence, but
it remains a hypothesis. A comparison against hand-written volatility-scaled trailing stops is
the natural next experiment.

# 7. Robustness: validation design, frequency and survivorship

Table 14 repeats the experiment under alternative validation designs and universes.

Table: Table 14. Mean ΔSR of the validation-selected model under alternative validation designs and universes; in parentheses, the numbers of strategies with significant improvements / deteriorations after BH adjustment within run.

{{tbl:runs}}

*Validation design.* A rolling ten-year training window gives +0.44 (S1) and +0.42 (S3), and a
single 2013 hold-out with no retraining gives +0.62 and +0.39. The conclusions do not depend
on the expanding walk-forward design. In every daily design, S1 and S3 have a mean $\Delta$SR
of at least +0.39, improve at least 94.9% of the strategies and are significant for 20 to 37
strategies after BH adjustment. S2 never helps materially, and S5 is small but positive.

*Frequency.* At intraday cryptocurrency frequencies the picture changes. At four-hour bars ML
sizing no longer helps (−0.09); S1 survives (+0.35, significant for 6 strategies) and S5 is
positive on average (+0.18) but significant for only 2. At one-hour bars every baseline loses
heavily after costs (mean Sharpe −2.52). S1 and S5 improve them (+1.15 and +0.91) while S2 and
S3 make them worse: S3 is significantly negative for 21 strategies. The S1 gain at this
frequency is mostly avoided trading cost, and almost all ML variants remain unprofitable: only
79 of the 1,209 strategy × stage × model configurations have a positive Sharpe ratio, against
4 of the 39 baselines. The sizing effect in particular appears specific to daily bars.

*Survivorship.* On the survivorship-biased single-stock universe (current S&P 100 and NIFTY
50 members), the gains are the largest of any run: S1 +0.95, S3 +0.80 and S4 +0.70, from a mean
baseline Sharpe of −0.05. Because these stocks are in the sample *because* they survived, these
larger gains are an upper bound that does not generalise, and we exclude them from the
conclusions.

# 8. Discussion

## 8.1 Answer to the research question

*How much?* It depends on the stage. Out of sample over 2006–2026, ML that selects trades
raises the Sharpe ratio of the average trend rule by 0.47, and ML position sizing by 0.44,
from a baseline of 0.01. ML exits add 0.28 and parameter adaptation 0.16, while a
strategy-agnostic regime gate adds nothing. In a diversified composite of all 39 rules the
Sharpe ratio rises from 0.01 to 0.60 (S1) or 0.57 (S3), and the maximum drawdown falls from
−9.5% to −4.1%.

*Is it statistically significant?* Yes for S1 and S3, and partly for S4. After BH adjustment
over the 195 strategy × stage tests, 31 (S1), 33 (S3) and 15 (S4) of the 39 strategies improve
significantly; after Holm adjustment, 10, 16 and 1. Hansen's SPA test rejects "no ML
configuration beats the rule" for 38 of 39 strategies, the probability of backtest overfitting
is low, the random-prediction placebo is negative or null, and the random-walk experiment shows
no broad improvement.

*Under what conditions is it robust?* The improvement is robust to the validation design
(expanding, rolling, single hold-out), to the time period (positive in every five-year
sub-period for S1, S3, S4 and S5), and to costs, execution timing and sizing. It holds for
equity indices, ETFs, emerging-market currencies, cryptocurrencies and, less strongly,
futures. It is largest for weak, high-turnover rules and smallest for slow, already-profitable
rules. Tree ensembles are the best models and the neural network the weakest.

*Where it is not robust.* The improvement does not transfer to intraday frequencies for
position sizing, it is absent for G10 currencies, and regime detection does not help in any
daily design. Part of the trade-filtering gain is cost avoidance. Most importantly, the gain
does not come from the technical indicators: a model that sees no indicator, only volatility
and the state of the open trade, obtains the same improvement as one given 56 indicators.

## 8.2 Interpretation and implications

Three implications follow. First, for practitioners, the most productive use of ML on a trend
rule is *rule-specific*: deciding whether to take a given trade and how large to make it.
Strategy-agnostic forecasts of whether a market will trend did not help, even though the
regime models carry some information (their placebo is strongly negative). The economically
relevant target is the payoff of the rule's own position.

Second, the evidence that ML "improves trend following" should be read as evidence that a
*learned trade-management layer* improves it, not as evidence that the indicator library has
predictive content. This matters both for the design of such systems (a parsimonious state
representation suffices) and for the interpretation of the many studies that attribute ML
gains to elaborate indicator sets. It also suggests that part of what the models learn might
be replicable by simpler, hand-written rules, which is the most important open question
left by this study.

Third, statistical and economic significance diverge. A robust improvement of 0.4–0.5 in
Sharpe ratio, measured against baselines that earn about zero, still leaves most individual
ML-enhanced rules short of convincing stand-alone profitability once the search is deflated.
The improvement is largest precisely where the rules are weakest. The case for ML is
strongest in diversified portfolios of rules and instruments, where small, consistent
per-instrument gains accumulate, and weakest for a single rule on a single market.

# 9. Limitations

*Data quality.* The data come from free vendors. Continuous futures are not roll-adjusted,
indices exclude dividends, currency returns exclude carry and cryptocurrency prices are
cross-venue aggregates. These affect the baseline and the ML variants identically, so the
*differences* are less exposed than the levels, but absolute Sharpe ratios should be read with
these caveats.

*Costs.* Costs are per-asset-class assumptions. Market impact is not modelled, which matters
for large capital in the less liquid futures and emerging-market currencies.

*Fixed hyper-parameters.* Each model family is represented by one a-priori configuration.
Tuning on validation data could change the model ranking, but would also enlarge the search
that the multiple-testing corrections must cover.

*Mechanism.* The ablation shows that the gains do not require technical indicators, but it
cannot establish what the models learn from the trade state and volatility. We did not compare
the models with hand-written volatility-scaled trailing stops or re-entry rules.

*Ablation scope.* The single-family and context-only blocks use a lighter protocol (S1, S3 and
S4; two models; three-year retraining; 15 representative strategies).

*Pooled models.* One model is fitted per strategy and stage across all instruments.
Per-asset-class models are a natural extension.

*History.* The OOS period starts in 2006. Longer histories (pre-1990 futures, century-long
index series) would cover more trend regimes.

*Survivorship.* The benchmark universe avoids constituent selection. The single-stock run is
survivorship-biased by construction and is reported only as such.

# 10. Conclusion

We measured how much machine learning improves 39 established trend-following rules, stage by
stage, on a multi-asset panel with purged walk-forward validation, realistic execution and
costs, and inference that accounts for the size of the search. ML that selects trades or sizes
positions improves the average rule by 0.44–0.47 in Sharpe ratio, significantly for most rules,
robustly across validation designs, periods and backtest assumptions, and beyond what a
random-prediction placebo achieves. ML exits help less, parameter adaptation little, and
strategy-agnostic regime detection not at all. The gains concentrate in weak rules, fail at
intraday frequencies for position sizing and are absent for G10 currencies. Most importantly,
they do not depend on technical indicators: volatility and the state of the open trade
suffice. The natural next step is to test whether hand-written trade-management rules can
recover the learned improvement, which would tell us whether ML adds something genuinely new
to trend following or rediscovers rules that traders could write down themselves.

# Declarations

**Data availability.** Market data were obtained from Yahoo Finance, the Binance public API
and FRED. Vendor terms may not permit redistribution of the raw data; the repository
contains scripts that download and clean the identical data set and record file hashes, so
that the data set used here can be verified.

**Code availability.** All code (data pipeline, 39 strategies, 56 features, ML stages,
backtester, statistical tests, experiment registry and the scripts that generate every table
and figure) is available at [repository URL to be inserted].

**Use of AI tools.** [Authors to complete according to the target journal's policy. For
example: "Generative AI tools (Anthropic Claude) were used to assist with software
implementation, running experiments and drafting text. The author designed the study,
reviewed all code, outputs and text, and takes full responsibility for the content."]

**Funding.** [To be completed.]

**Competing interests.** [To be completed.]

# References

::: {#refs}
:::

::: pagebreak
:::

# Appendix A. Strategy and feature definitions

::: wide
{{tbl:strategies}}

Table: Table A1. The 39 rule-based trend-following strategies.
:::

::: wide
{{tbl:features}}

Table: Table A2. The 56 indicator features (all dimensionless).
:::

# Appendix B. Per-strategy results

::: wide
{{tbl:baseline_metrics}}

Table: Table B1. Out-of-sample performance of the rule-based baselines (benchmark portfolio level, 2006–2026, after costs), sorted by Sharpe ratio.
:::

::: wide
{{tbl:strategy_ml}}

Table: Table B2. Per-strategy ΔSR of the validation-selected model at each stage, Hansen SPA p-value across the 25 ML configurations, probability of backtest overfitting (PBO) and deflated Sharpe ratio (DSR) of the ex-post best configuration. An asterisk marks BH-adjusted p < 0.05.
:::

# Appendix C. Data validation

::: keep
Table: Table C1. Cross-source validation of overlapping series (daily and weekly return correlations and annualised tracking error).

{{tbl:validation}}
:::
