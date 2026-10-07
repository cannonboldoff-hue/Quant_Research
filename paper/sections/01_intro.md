## 1. Introduction

Trend following is among the oldest and best-documented systematic trading styles. Its
rules (moving-average crossovers, channel breakouts, volatility stops, time-series
momentum) are simple, public and decades old. Machine learning is now routinely bolted
onto such rules to filter signals, detect regimes, size positions or adapt parameters.
Claims about the value of doing so are rarely measured against the rule itself under one
controlled protocol. ML papers report the profitability of an ML strategy. Practitioners
report that "the filter helped" on the instrument where it was tuned.

This paper asks a narrower and more useful question:

> **How much does ML improve established trend-following strategies over their original
> rule-based implementations, and under what market, instrument, strategy, indicator,
> model and validation conditions does the improvement remain statistically significant
> and robust?**

We answer it with a controlled, stage-by-stage design:

1. **39 published trend-following rules** are implemented faithfully as baselines. ML is
   then inserted at **five separate stages** of the trading pipeline, each evaluated
   against the same baseline:
   - signal filtering / trade selection
   - trend-regime detection
   - position sizing
   - exit decisions
   - parameter adaptation
2. At every stage **five model families** compete (logistic regression, random forest,
   extra trees, gradient boosting, neural network). The winner is chosen only on a
   validation block that precedes each out-of-sample year. Every alternative is reported
   as well.
3. The test bed is a public multi-asset daily panel: equity indices, futures, FX,
   crypto and ETFs on many exchanges and regions. Robustness panels cover 1-hour and
   4-hour crypto bars and a survivorship-biased single-stock universe, which is reported
   as such.
4. Inference accounts for the size of the search:
   - block-bootstrap and HAC tests of Sharpe differences
   - Holm and Benjamini–Hochberg adjustment across the 195 strategy × stage tests
   - Hansen's SPA and White's Reality Check within each strategy
   - probability of backtest overfitting and deflated Sharpe ratios
   - a **placebo control** that applies each ML decision rule with random instead of
     model predictions, isolating the information contributed by ML from the mechanical
     effect of intervening.

The framework is modular: data, strategies, indicators, models, stages, validation and
backtest mechanics are all swappable. Every number in the paper is generated from an
experiment registry that links each result to its strategy, indicators, market,
instrument, dataset hash, ML stage, model, validation design, backtest configuration and
code version.
