## Abstract

We measure how much machine learning improves 39 established trend-following rules
(moving-average crossovers, channel breakouts, volatility stops, time-series momentum,
oscillator and directional systems), and when the improvement is statistically robust.
ML is inserted at five separate pipeline stages (trade filtering, regime gating, position
sizing, exits, parameter adaptation), with five model families per stage. Models and
decision thresholds are chosen on a validation block that precedes each test year, in a
purged annual walk-forward design.

The test bed is {{n:runs.benchmark.meta.n_instruments}} instruments (equity indices,
futures, FX, crypto, ETFs), {{n:runs.benchmark.meta.oos_period}} out of sample, with next-open
execution, asset-class transaction costs and volatility-targeted sizing.

Rule-based baselines have a mean Sharpe ratio of {{n:runs.benchmark.baseline.mean_sharpe:.2f}}.
ML trade filtering and ML position sizing raise it by
{{n:runs.benchmark.headline[stage=S1_filter].mean_dSharpe:+.2f}} and
{{n:runs.benchmark.headline[stage=S3_sizing].mean_dSharpe:+.2f}}. The improvements are
significant for {{n:runs.benchmark.headline[stage=S1_filter].sig_pos_bh}} and
{{n:runs.benchmark.headline[stage=S3_sizing].sig_pos_bh}} of 39 strategies after
Benjamini–Hochberg correction. ML exits add {{n:runs.benchmark.headline[stage=S4_exit].mean_dSharpe:+.2f}},
parameter adaptation {{n:runs.benchmark.headline[stage=S5_params].mean_dSharpe:+.2f}}, and a
strategy-agnostic regime gate adds nothing.

The gains are not an artefact of trading less: a placebo with random predictions is
negative or null, and a random-walk null test shows no broad improvement. The S1 and S3 gains hold across a
hold-out split, a rolling window, every 5-year sub-period and 0×–3× costs (at 3× costs
the regime gate becomes harmful). They concentrate in
weak rules, and are absent for G10 FX and for intraday (1-hour, 4-hour) position sizing.

An ablation shows the benefit does **not** come from the technical indicators. A model given
only volatility and the state of the open trade matches one given 56 indicators. The
improvement is statistically robust, but most ML-enhanced rules remain only marginally
profitable on their own after costs.

Every figure is generated from a reproducible experiment registry linking strategy,
indicators, market, instrument, dataset, ML stage, model, validation and backtest to its
result. The framework is modular, and all code, data pipelines and tests are in the
repository.
