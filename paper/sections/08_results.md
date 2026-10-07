## 8. Results

All results in this section are out of sample:
{{n:runs.benchmark.meta.oos_period}}, {{n:runs.benchmark.meta.n_instruments}} instruments,
next-open execution, 1× costs. "ΔSharpe" is the annualised Sharpe ratio of the
ML-enhanced strategy minus that of its own rule-based baseline on identical days. The
selected model is the validation winner, never the ex-post best.

### 8.1 The rule-based baselines

Over 2006–2026 and after costs, the 39 published rules earn an average portfolio Sharpe of
{{n:runs.benchmark.baseline.mean_sharpe:.2f}} (median
{{n:runs.benchmark.baseline.median_sharpe:.2f}}). Only {{n:runs.benchmark.baseline.n_positive_sharpe}}
of {{n:runs.benchmark.baseline.n}} have a positive Sharpe. The best are the slow,
long-only or monthly rules: `{{n:runs.benchmark.baseline.best}}` reaches
{{n:runs.benchmark.baseline.best_sharpe:.2f}}. The worst is `{{n:runs.benchmark.baseline.worst}}`
at {{n:runs.benchmark.baseline.worst_sharpe:.2f}}. This matches the well-documented weak
decade for trend following after 2009. Fast crossovers and adaptive moving averages pay
substantial costs for whipsaw trades. Each instrument is scaled to 10% volatility, but the
equal-capital portfolio of 174 weakly correlated instruments has a realised volatility of
only about 2–3% a year (Table 7). CAGRs are therefore small in absolute terms and scale
linearly with any leverage applied; Sharpe, Sortino and Calmar ratios are the comparable
quantities. Full metrics (CAGR, total return, volatility,
Sharpe, Sortino, Calmar, maximum drawdown, win rate, profit factor, turnover, cost drag,
exposure, stability) are in Appendix C.

### 8.2 How much does ML improve the rules? (headline)

{{table:benchmark_T5_headline_by_stage|stage,n_strategies,mean_dSharpe,median_dSharpe,share_improved,sig_pos_raw,sig_pos_holm,sig_pos_bh,sig_neg_bh,mean_d_maxdd,mean_d_turnover,sign_test_p}}

*Table 5. ΔSharpe of the validation-selected model per stage, across 39 strategies.*
`sig_*` counts strategies with a significant improvement (two-sided paired block
bootstrap, α = 5%), raw and after Holm and Benjamini–Hochberg adjustment over all 195
strategy × stage tests. `mean_d_maxdd` > 0 means shallower drawdowns. `mean_d_turnover`
is the change in annual turnover.

The answer depends sharply on **where** ML is inserted:

- **Trade selection (S1)** and **position sizing (S3)** give the largest and most
  reliable improvements:
  - S1: mean ΔSharpe {{n:runs.benchmark.headline[stage=S1_filter].mean_dSharpe:+.2f}};
    {{pct:runs.benchmark.headline[stage=S1_filter].share_improved}} of strategies improved;
    {{n:runs.benchmark.headline[stage=S1_filter].sig_pos_holm}} significant after Holm and
    {{n:runs.benchmark.headline[stage=S1_filter].sig_pos_bh}} after BH.
  - S3: {{n:runs.benchmark.headline[stage=S3_sizing].mean_dSharpe:+.2f}};
    {{pct:runs.benchmark.headline[stage=S3_sizing].share_improved}} improved;
    {{n:runs.benchmark.headline[stage=S3_sizing].sig_pos_holm}} significant after Holm and
    {{n:runs.benchmark.headline[stage=S3_sizing].sig_pos_bh}} after BH.
- **ML exits (S4)** help on average
  ({{n:runs.benchmark.headline[stage=S4_exit].mean_dSharpe:+.2f}}), but individually
  less reliably: {{n:runs.benchmark.headline[stage=S4_exit].sig_pos_bh}} strategies are
  significant after BH and {{n:runs.benchmark.headline[stage=S4_exit].sig_pos_holm}}
  after Holm.
- **Parameter adaptation (S5)** adds little
  ({{n:runs.benchmark.headline[stage=S5_params].mean_dSharpe:+.2f}}; significant for
  {{n:runs.benchmark.headline[stage=S5_params].sig_pos_bh}} strategies after BH and
  {{n:runs.benchmark.headline[stage=S5_params].sig_pos_holm}} after Holm).
- **The strategy-agnostic trend-regime gate (S2) does not help**: mean ΔSharpe
  {{n:runs.benchmark.headline[stage=S2_regime].mean_dSharpe:+.3f}}, with improvements in
  only {{pct:runs.benchmark.headline[stage=S2_regime].share_improved}} of strategies.
  Predicting *that* a market will trend is not the same as predicting that a given rule's
  current position will pay. Only the latter, strategy-specific target produced gains.
  S2 is also the only stage where validation-based selection actively hurts. Every
  individual model except the MLP has a slightly positive mean OOS ΔSharpe (Table 8; e.g.
  extra trees
  {{n:runs.benchmark.model_comparison[ml_stage=S2_regime,ml_model=et].mean_dSharpe:+.3f}}),
  but the fold-by-fold validation winner does not. The gate's effects are too small and
  unstable for a 2-year validation block to rank reliably.

Across all 195 tests, {{n:runs.benchmark.n_sig_pos_holm_total}} improvements survive the
Holm family-wise correction and {{n:runs.benchmark.n_sig_pos_bh_total}} survive BH. Only
{{n:runs.benchmark.n_sig_neg_bh_total}} significant *deteriorations* survive BH. Within
each strategy, Hansen's SPA test rejects "no ML configuration beats the rule" for
{{n:runs.benchmark.spa.n_spa_sig}} of {{n:runs.benchmark.spa.n}} strategies
({{n:runs.benchmark.spa.n_spa_sig_holm}} after Holm across strategies; White's Reality
Check: {{n:runs.benchmark.spa.n_rc_sig}}).

{{fig:benchmark_F1_heatmap_dsharpe.png|Figure 1. OOS ΔSharpe for every strategy × stage (selected model). Rows sorted by baseline Sharpe; * marks BH-adjusted p < 0.05.}}

{{fig:benchmark_F2_dsharpe_distribution.png|Figure 2. Distribution of ΔSharpe across the 39 strategies per stage.}}

**A diversified view.** The equal-weight composite of all 39 strategies (Table 17) has a
rule-based Sharpe of {{n:runs.benchmark.composite[config=base].sharpe:.2f}}. With ML trade
selection it is {{n:runs.benchmark.composite[config=S1_filter].sharpe:.2f}}
(ΔSharpe {{n:runs.benchmark.composite[config=S1_filter].d_sharpe:+.2f}}, 95% CI
[{{n:runs.benchmark.composite[config=S1_filter].ci_lo:.2f}},
{{n:runs.benchmark.composite[config=S1_filter].ci_hi:.2f}}], p =
{{n:runs.benchmark.composite[config=S1_filter].p_boot_two:.3f}}). With ML sizing it is
{{n:runs.benchmark.composite[config=S3_sizing].sharpe:.2f}}
(p = {{n:runs.benchmark.composite[config=S3_sizing].p_boot_two:.3f}}). Maximum drawdown
falls from {{n:runs.benchmark.composite[config=base].max_drawdown:.1%}} to
{{n:runs.benchmark.composite[config=S1_filter].max_drawdown:.1%}} (S1). The composite is
very diversified across instruments and rules, so its volatility is low
({{n:runs.benchmark.composite[config=base].ann_vol:.1%}} for the baseline): Sharpe ratios,
not raw CAGRs, are the meaningful comparison.

{{table:benchmark_T17_composite_portfolio|config,cagr,total_return,ann_vol,sharpe,sortino,calmar,max_drawdown,pct_pos_years,stability_r2,d_sharpe,p_boot_two,ci_lo,ci_hi}}

{{fig:benchmark_F3_composite_equity.png|Figure 3. Cumulative OOS log return of the 39-strategy composite: rule-based vs each ML stage.}}

**Portfolio vs single instrument.** At the instrument level (unit size, the rules exactly
as published), the improvements are real but much smaller:

- S1: mean ΔSharpe {{n:runs.benchmark.instrument[ml_stage=S1_filter].mean_dSharpe:+.2f}}
  over {{n:runs.benchmark.instrument[ml_stage=S1_filter].n_tests}} strategy-instrument
  pairs, {{pct:runs.benchmark.instrument[ml_stage=S1_filter].share_improved}} improved,
  {{n:runs.benchmark.instrument[ml_stage=S1_filter].sig_pos_bh}} significant after BH.
- S3: {{n:runs.benchmark.instrument[ml_stage=S3_sizing].mean_dSharpe:+.2f}},
  {{n:runs.benchmark.instrument[ml_stage=S3_sizing].sig_pos_bh}} significant.

Single-instrument Sharpe ratios are noisy. The portfolio-level gains arise because small,
consistent per-instrument improvements add up across 174 weakly correlated instruments.
A practitioner trading one market should expect the instrument-level effect, not the
portfolio-level one.

### 8.3 Which model, and does selection work?

{{table:benchmark_T8_model_comparison|ml_stage,ml_model,mean_dSharpe,share_improved,sig_pos,sig_neg,times_selected,selection_share}}

*Table 8. OOS ΔSharpe of every model at every stage, and how often validation selected it
(over 21 folds × 39 strategies).*

Tree ensembles are the most useful models. Extra trees and random forests lead at S1, S3
and S4. Gradient boosting is competitive. The MLP is the weakest at every stage, and
logistic regression is in between. The instrument-level regression (Table 15) confirms
it: relative to logistic regression, ET adds
{{n:runs.benchmark.regression.terms[term=ml_model=et].coef:+.3f}} and RF
{{n:runs.benchmark.regression.terms[term=ml_model=rf].coef:+.3f}} Sharpe, while the MLP
subtracts {{n:runs.benchmark.regression.terms[term=ml_model=mlp].coef:+.3f}}.

Validation-based selection does slightly better than the average model and clearly
worse than the ex-post best (Table 8c):

| Stage | Selected | Average model | Best ex post |
|---|---|---|---|
| S1 | {{n:runs.benchmark.selected_vs_alternatives[ml_stage=S1_filter].selected_model:+.2f}} | {{n:runs.benchmark.selected_vs_alternatives[ml_stage=S1_filter].avg_model:+.2f}} | {{n:runs.benchmark.selected_vs_alternatives[ml_stage=S1_filter].best_ex_post:+.2f}} |
| S3 | {{n:runs.benchmark.selected_vs_alternatives[ml_stage=S3_sizing].selected_model:+.2f}} | {{n:runs.benchmark.selected_vs_alternatives[ml_stage=S3_sizing].avg_model:+.2f}} | {{n:runs.benchmark.selected_vs_alternatives[ml_stage=S3_sizing].best_ex_post:+.2f}} |

The gap between "selected" and "best ex post" is the price of honest model selection.
Reporting the ex-post best model would overstate the ML benefit by about
{{n:runs.benchmark.selected_vs_alternatives[ml_stage=S1_filter].best_ex_post:.2f}} −
{{n:runs.benchmark.selected_vs_alternatives[ml_stage=S1_filter].selected_model:.2f}} Sharpe
at S1.

Models' average validation scores rank their OOS results consistently (Spearman
{{n:runs.benchmark.val_test_spearman[stage=S3_sizing].mean:.2f}} at S3,
{{n:runs.benchmark.val_test_spearman[stage=S1_filter].mean:.2f}} at S1). This is partly
mechanical. The 2-year validation block of a fold covers calendar years that are test
years of earlier folds, so models that are better in general rank similarly in both.
No fold ever uses its own test year.

{{fig:benchmark_F4_model_comparison.png|Figure 4. Mean OOS ΔSharpe by model and stage; "selected" = validation winner.}}

### 8.4 Markets and instruments

{{table:benchmark_T9_market_groups|market,ml_stage,mean_base_sharpe,mean_dSharpe,share_improved,sig_pos_bh,sig_neg_bh}}

*Table 9. ΔSharpe by universe group (39 strategies each; BH within stage).*

The gains are broad but not universal:

- **Equity indices, ETFs, EM FX and crypto** benefit from S1 and S3 for nearly every
  strategy.
- **Continuous futures** benefit less.
- **G10 FX** is the exception:
  - FX majors show no ML benefit at all. S3 and S4 slightly *reduce* the Sharpe ratio
    (S3: {{n:runs.benchmark.market[market=fx_major,ml_stage=S3_sizing].mean_dSharpe:+.2f}}).
  - FX crosses show only a small S1 gain
    ({{n:runs.benchmark.market[market=fx_cross,ml_stage=S1_filter].mean_dSharpe:+.2f}}).

  G10 FX is the most efficiently priced, lowest-cost market in the sample, and the rules
  offer little conditional structure to exploit.

The instrument-level regression (Table 15) puts it the same way. Relative to equities, FX
instruments gain {{n:runs.benchmark.regression.terms[term=asset_class=fx].coef:+.3f}} and
futures {{n:runs.benchmark.regression.terms[term=asset_class=futures].coef:+.3f}} less
Sharpe from ML.

{{fig:benchmark_F5_market_heatmap.png|Figure 5. Mean ΔSharpe by market group and stage.}}

### 8.5 Strategies and indicators: when does ML help most?

{{table:benchmark_T15_conditions_regression}}

*Table 15. Instrument-level ΔSharpe (all stages × models, n =
{{n:runs.benchmark.regression.n_obs:,}}) regressed on design factors; standard errors
clustered by instrument; R² = {{n:runs.benchmark.regression.r2:.3f}}.*

The strongest single predictor of the ML benefit is **how good the rule already is**.
The coefficient on the baseline's own Sharpe is
{{n:runs.benchmark.regression.terms[term=base_sharpe].coef:+.3f}} (t =
{{n:runs.benchmark.regression.terms[term=base_sharpe].t:.1f}}). By family (Table 13):

- The slow, well-performing price-vs-MA rules (Faber, BLL; mean baseline Sharpe
  {{n:runs.benchmark.family[strategy_family=price_vs_ma,ml_stage=S1_filter].mean_base_sharpe:.2f}})
  gain only {{n:runs.benchmark.family[strategy_family=price_vs_ma,ml_stage=S1_filter].mean_dSharpe:+.2f}}
  from S1.
- Time-series momentum (baseline
  {{n:runs.benchmark.family[strategy_family=time_series_momentum,ml_stage=S1_filter].mean_base_sharpe:.2f}})
  gains {{n:runs.benchmark.family[strategy_family=time_series_momentum,ml_stage=S1_filter].mean_dSharpe:+.2f}}.
- Fast adaptive moving averages (baseline
  {{n:runs.benchmark.family[strategy_family=adaptive_ma,ml_stage=S1_filter].mean_base_sharpe:.2f}})
  gain {{n:runs.benchmark.family[strategy_family=adaptive_ma,ml_stage=S1_filter].mean_dSharpe:+.2f}}.

ML mostly **repairs weak rules**, largely by removing trades it expects to lose. S1
changes annual portfolio turnover by
{{n:runs.benchmark.headline[stage=S1_filter].mean_d_turnover:+.1f}} units on average
(fewer trades). It improves established strong rules much less. Section 8.9 tests
whether the benefit survives a placebo with the same turnover reduction.

{{table:benchmark_T13_strategy_family|strategy_family,ml_stage,n,mean_base_sharpe,mean_dSharpe,share_improved}}

### 8.6 Time periods

{{table:benchmark_T11_subperiods|period,ml_stage,mean_base_sharpe,mean_dSharpe,share_improved,sig_pos,sig_neg}}

ΔSharpe is positive for S1, S3, S4 and S5 in every 5-year sub-period. It is largest in
2021–2026 (S1 {{n:runs.benchmark.subperiods[period=2021-2026,ml_stage=S1_filter].mean_dSharpe:+.2f}}),
a period in which the rules themselves lost money (mean baseline Sharpe
{{n:runs.benchmark.subperiods[period=2021-2026,ml_stage=S1_filter].mean_base_sharpe:.2f}}).
It is smallest in 2006–2010, the strong trend-following years around the financial
crisis (baseline {{n:runs.benchmark.subperiods[period=2006-2010,ml_stage=S1_filter].mean_base_sharpe:.2f}};
S1 {{n:runs.benchmark.subperiods[period=2006-2010,ml_stage=S1_filter].mean_dSharpe:+.2f}}).
This is the time-series version of §8.5: ML helps most when the rule struggles.
Sub-period tests have far fewer observations, so only the 2021–2026 improvements are
individually significant for most strategies.

### 8.7 Backtest mechanics: costs, execution, sizing

{{table:benchmark_T12_backtest_sensitivity|backtest,ml_stage,mean_base_sharpe,mean_ml_sharpe,mean_dSharpe,share_improved,sig_pos,sig_neg}}

S1 and S3 stay the two most valuable stages and S2 the least valuable under every
backtest assumption. Only S4 and S5 swap places at 3× costs.

- **Costs.** Without costs the baselines look far better (mean Sharpe
  {{n:runs.benchmark.sensitivity[backtest=next_open|cost0|vol_target0.1,ml_stage=S1_filter].mean_base_sharpe:.2f}}),
  and the S1 gain shrinks to
  {{n:runs.benchmark.sensitivity[backtest=next_open|cost0|vol_target0.1,ml_stage=S1_filter].mean_dSharpe:+.2f}}.
  Part of S1's value is cost avoidance, but most of it is not. At 3× costs, S1's
  advantage grows to
  {{n:runs.benchmark.sensitivity[backtest=next_open|cost3|vol_target0.1,ml_stage=S1_filter].mean_dSharpe:+.2f}},
  while the regime gate turns clearly harmful
  ({{n:runs.benchmark.sensitivity[backtest=next_open|cost3|vol_target0.1,ml_stage=S2_regime].mean_dSharpe:+.2f}}).
- **Execution.** Executing one bar later (next close) or at the signal close leaves the
  gains essentially unchanged (S1:
  {{n:runs.benchmark.sensitivity[backtest=next_close|cost1|vol_target0.1,ml_stage=S1_filter].mean_dSharpe:+.2f}}
  and
  {{n:runs.benchmark.sensitivity[backtest=same_close|cost1|vol_target0.1,ml_stage=S1_filter].mean_dSharpe:+.2f}}).
  The improvement does not depend on fragile fill assumptions. The optimistic same-close
  convention flatters the *baselines* only mildly: mean Sharpe
  {{n:runs.benchmark.sensitivity[backtest=same_close|cost1|vol_target0.1,ml_stage=S1_filter].mean_base_sharpe:.2f}},
  vs {{n:runs.benchmark.headline[stage=S1_filter].mean_base_sharpe:.2f}} for next-open
  and {{n:runs.benchmark.sensitivity[backtest=next_close|cost1|vol_target0.1,ml_stage=S1_filter].mean_base_sharpe:.2f}}
  for next-close.
- **Sizing.** Unit sizing (no volatility targeting) gives the same picture (S1
  {{n:runs.benchmark.sensitivity[backtest=next_open|cost1|unit,ml_stage=S1_filter].mean_dSharpe:+.2f}},
  S3 {{n:runs.benchmark.sensitivity[backtest=next_open|cost1|unit,ml_stage=S3_sizing].mean_dSharpe:+.2f}}).

{{fig:benchmark_F6_backtest_sensitivity.png|Figure 6. Mean ΔSharpe per stage under alternative cost, execution and sizing assumptions.}}

### 8.8 Overfitting diagnostics

Across each strategy's 26 OOS configurations (baseline plus 5 stages × 5 models), the
probability of backtest overfitting is low: median PBO
{{n:runs.benchmark.pbo.median_pbo:.3f}}, and no strategy above 0.5. The ex-post best
configuration is almost always an S3 or S1 model (Table 14). The *absolute* Sharpe of the
best configuration survives deflation for 25 trials (DSR > 0.95) for only
{{n:runs.benchmark.pbo.n_dsr_above_095}} of {{n:runs.benchmark.pbo.n}} strategies. ML makes
weak rules markedly better, but most ML-enhanced rules are still not, on their own,
convincingly profitable after costs over 2006–2026. The improvement is significant. The
resulting standalone strategy usually is not.
