## 9. Answer to the research question

> *How much does ML improve established trend-following strategies over their original
> rule-based implementations, and under what market, instrument, strategy, indicator, model
> and validation conditions does the improvement remain statistically significant and
> robust?*

**How much.** It depends on the stage at which ML is applied. Out of sample over
{{n:runs.benchmark.meta.oos_period}}, ML that **selects trades (S1)** raises the Sharpe ratio
of the average rule by {{n:runs.benchmark.headline[stage=S1_filter].mean_dSharpe:+.2f}}, and ML
**position sizing (S3)** by {{n:runs.benchmark.headline[stage=S3_sizing].mean_dSharpe:+.2f}}.
This is large relative to the baselines, whose mean Sharpe is
{{n:runs.benchmark.baseline.mean_sharpe:.2f}}. ML **exits (S4)** add
{{n:runs.benchmark.headline[stage=S4_exit].mean_dSharpe:+.2f}} and **parameter adaptation (S5)**
{{n:runs.benchmark.headline[stage=S5_params].mean_dSharpe:+.2f}}. The strategy-agnostic **regime
gate (S2)** adds nothing ({{n:runs.benchmark.headline[stage=S2_regime].mean_dSharpe:+.3f}}). In a
diversified composite of all 39 rules, the Sharpe ratio goes from
{{n:runs.benchmark.composite[config=base].sharpe:.2f}} to
{{n:runs.benchmark.composite[config=S1_filter].sharpe:.2f}} (S1) or
{{n:runs.benchmark.composite[config=S3_sizing].sharpe:.2f}} (S3), and the maximum drawdown falls
from {{n:runs.benchmark.composite[config=base].max_drawdown:.1%}} to
{{n:runs.benchmark.composite[config=S1_filter].max_drawdown:.1%}}.

**Statistically significant?** Yes for S1 and S3, and partly for S4:

- After Benjamini–Hochberg adjustment over the 195 strategy × stage tests,
  {{n:runs.benchmark.headline[stage=S1_filter].sig_pos_bh}} (S1) and
  {{n:runs.benchmark.headline[stage=S3_sizing].sig_pos_bh}} (S3) of 39 strategies improve
  significantly, and {{n:runs.benchmark.headline[stage=S4_exit].sig_pos_bh}} (S4).
- After the stricter Holm correction the counts are
  {{n:runs.benchmark.headline[stage=S1_filter].sig_pos_holm}}, {{n:runs.benchmark.headline[stage=S3_sizing].sig_pos_holm}}
  and {{n:runs.benchmark.headline[stage=S4_exit].sig_pos_holm}}.
- Hansen's SPA test rejects "no ML configuration beats the rule" for
  {{n:runs.benchmark.spa.n_spa_sig}} of {{n:runs.benchmark.spa.n}} strategies.
- The probability of backtest overfitting is low (median
  {{n:runs.benchmark.pbo.median_pbo:.3f}}).
- A placebo that applies the same decision rules with random predictions is **negative or
  null** ({{n:runs.benchmark.placebo[stage=S1_filter].mean_d_placebo:+.2f}} at S1,
  {{n:runs.benchmark.placebo[stage=S3_sizing].mean_d_placebo:+.2f}} at S3), so the gain is not an
  artefact of merely trading less. The same pipeline run on random-walk markets shows no
  broad improvement, which argues against gross look-ahead leakage.

**Where it is robust.**

- **Validation design.** A single hold-out (no retraining) and a rolling 10-year window give
  essentially the same results as the expanding walk-forward benchmark (§8.11).
- **Time.** S1, S3, S4 and S5 are positive in every 5-year sub-period, and largest in
  2021–2026, when the rules themselves lost money.
- **Backtest mechanics.** The stage ranking is unchanged by 0×/3× costs, one bar of
  execution delay, the optimistic same-close convention, or unit instead of
  volatility-targeted sizing.
- **Markets.** Equity indices, ETFs, emerging-market FX, crypto and (less so) futures
  benefit. At the *instrument* level the effect is smaller
  (S1 {{n:runs.benchmark.instrument[ml_stage=S1_filter].mean_dSharpe:+.2f}} on average): the
  portfolio gain relies on aggregating many weakly correlated markets.
- **Strategies.** ML helps weak rules most. Fast adaptive moving averages gain
  {{n:runs.benchmark.family[strategy_family=adaptive_ma,ml_stage=S1_filter].mean_dSharpe:+.2f}} from S1, whereas the
  best slow rules (Faber, BLL: {{n:runs.benchmark.family[strategy_family=price_vs_ma,ml_stage=S1_filter].mean_dSharpe:+.2f}}) and time-series
  momentum ({{n:runs.benchmark.family[strategy_family=time_series_momentum,ml_stage=S1_filter].mean_dSharpe:+.2f}}) gain much less.
- **Models.** Extra trees and random forests are the best, gradient boosting is close, and
  the neural network is the weakest (Table 8, Table 15). Validation-based model selection
  recovers part of the gap to the ex-post best model but not all of it (Table 8c).

**Where it is not robust, and what the evidence does not support.**

- **Frequency.** On 4-hour and 1-hour crypto bars, ML sizing (S3) no longer helps and the
  regime gate hurts. Only trade filtering (S1) and, at 1 hour, parameter adaptation (S5)
  help, and at 1 hour mostly by avoiding trading costs on baselines that lose heavily.
- **G10 FX** shows no ML benefit.
- **Regime detection (S2)** does not help in any daily design.
- **The indicators are not the source of the gain.** A model that sees no technical
  indicator at all (only volatility and the state of the open trade) obtains the same
  improvement as one given 56 indicators (§8.12). The finding that the benefit comes from
  trade-state and volatility information, not from the indicator library, is the most
  important qualification of the study's framing.
- **Part of S1's benefit is cost avoidance** (§8.10), and the S1 gain on random-walk
  markets rises with transaction costs.
- **Absolute profitability remains modest.** Only
  {{n:runs.benchmark.pbo.n_dsr_above_095}} of {{n:runs.benchmark.pbo.n}} strategies' best ML
  configuration survives deflation for the 25 configurations tried, and many ML-enhanced
  rules are still not convincingly profitable on their own. The improvement is
  statistically robust, and the resulting strategies are mostly not.
- **Survivorship-biased single stocks** show the largest gains of all, which we treat as an
  upper bound that does not generalise, and exclude from conclusions.
