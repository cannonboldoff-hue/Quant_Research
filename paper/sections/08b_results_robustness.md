### 8.9 Is it the information in the models, or just the intervention? (placebo control)

Cutting trades or resizing positions can change a Sharpe ratio even with no information
at all. To isolate what the models *learn*, each decision rule is re-run with the same
validation-chosen intensity but with **uniform random numbers in place of model
predictions** (20 draws per strategy × stage).

{{table:benchmark_T19_placebo_by_stage}}

*Table 19. ML ΔSharpe vs random-prediction placebo ΔSharpe (benchmark, selected model).
`n_ml_above_placebo_p95` counts strategies whose ML ΔSharpe exceeds the 95th percentile of
their own placebo draws; `n_bh` the same after Benjamini–Hochberg adjustment.*

The mechanical effect of intervening is **negative or null**, so the ML gains are not an
artefact of trading less:

- At S1 the placebo ΔSharpe is {{n:runs.benchmark.placebo[stage=S1_filter].mean_d_placebo:+.2f}}
  against {{n:runs.benchmark.placebo[stage=S1_filter].mean_d_ml:+.2f}} for ML. ML beats its
  placebo's 95th percentile for {{n:runs.benchmark.placebo[stage=S1_filter].n_ml_above_placebo_p95}}
  of {{n:runs.benchmark.placebo[stage=S1_filter].n}} strategies.
- At S3 random sizing is strongly harmful
  ({{n:runs.benchmark.placebo[stage=S3_sizing].mean_d_placebo:+.2f}}), whereas model-based
  sizing helps ({{n:runs.benchmark.placebo[stage=S3_sizing].mean_d_ml:+.2f}}).
- S4 and S5 show the same pattern.

Even for S2, where the *selected* ΔSharpe is about zero, the placebo is clearly negative
({{n:runs.benchmark.placebo[stage=S2_regime].mean_d_placebo:+.2f}}). Gating on random
numbers removes profitable exposure, so the regime models carry information, but too
little to improve on the rule.

### 8.10 Leakage and false-discovery check on random-walk markets

If any component leaked future information, ML would "improve" rules even on data with
nothing to predict. We ran the whole benchmark pipeline unchanged on
{{n:null_test[stage=S1_filter].n_strategies}}-strategy panels of **synthetic random-walk
instruments** (GARCH-t returns, zero predictability, same costs).

{{table:T21_null_randomwalk}}

*Table 21. Null test: no stage produces broad improvements.*

- No stage shows broad, significant improvement: ΔSharpe is
  {{n:null_test[stage=S1_filter].mean_dSharpe_selected:+.2f}} (S1),
  {{n:null_test[stage=S2_regime].mean_dSharpe_selected:+.2f}} (S2),
  {{n:null_test[stage=S3_sizing].mean_dSharpe_selected:+.2f}} (S3),
  {{n:null_test[stage=S4_exit].mean_dSharpe_selected:+.2f}} (S4),
  {{n:null_test[stage=S5_params].mean_dSharpe_selected:+.2f}} (S5). These are small and
  mixed in sign, in contrast to the +0.16 to +0.47 gains on real data (S1–S5).
- The instrument-level false-positive rate (significantly positive at 5%) is
  {{n:null_test[stage=S1_filter].instrument_false_pos_rate:.3f}} at S1 and
  {{n:null_test[stage=S3_sizing].instrument_false_pos_rate:.3f}} at S3.
- What remains at S1 is cost avoidance, not prediction. With zero costs S1's ΔSharpe is
  {{n:null_test[stage=S1_filter].mean_dSharpe_cost0:+.2f}}, and it rises to
  {{n:null_test[stage=S1_filter].mean_dSharpe_cost3:+.2f}} at 3× costs: skipping trades
  saves money even when nothing is predictable. Part of S1's benefit on real data is
  therefore this mechanical saving, which is why §8.9's placebo (which also skips trades,
  at random) is the cleaner test of what the models learn.
- On noise, model-based **sizing and regime gating hurt**: S3
  {{n:null_test[stage=S3_sizing].mean_dSharpe_selected:+.2f}} (significantly negative for
  {{n:null_test[stage=S3_sizing].sig_neg_raw}} of {{n:null_test[stage=S3_sizing].n_strategies}}
  strategies) and S2 {{n:null_test[stage=S2_regime].mean_dSharpe_selected:+.2f}}. Models fitted
  to noise produce confident but meaningless position sizes. The positive S3 result on real
  markets is therefore evidence of genuine structure, not of a leak.

This is a screen against gross leakage, not a proof of its absence. It covers 8 strategies,
and leakage that required real market structure to bite would not show up here.

### 8.11 Validation design, frequency and survivorship

{{table:T18_cross_run_robustness|run,stage,n_strategies,mean_base_sharpe,mean_dSharpe,share_improved,sig_pos_bh,sig_neg_bh}}

*Table 18. The same experiment under alternative validation designs and universes
(selected model, BH within run).*

- **Rolling 10-year training window** (`robust_rolling10y`): the pattern is
  {{n:runs.robust_rolling10y.headline[stage=S1_filter].mean_dSharpe:+.2f}} (S1) and
  {{n:runs.robust_rolling10y.headline[stage=S3_sizing].mean_dSharpe:+.2f}} (S3).
- **Single 2013 hold-out, no retraining** (`robust_holdout2013`): S1
  {{n:runs.robust_holdout2013.headline[stage=S1_filter].mean_dSharpe:+.2f}}, S3
  {{n:runs.robust_holdout2013.headline[stage=S3_sizing].mean_dSharpe:+.2f}}.
- **Crypto, 4-hour bars**: S1 {{n:runs.crypto_intraday_4h.headline[stage=S1_filter].mean_dSharpe:+.2f}},
  S3 {{n:runs.crypto_intraday_4h.headline[stage=S3_sizing].mean_dSharpe:+.2f}}.
- **Crypto, 1-hour bars**: S1 {{n:runs.crypto_intraday_1h.headline[stage=S1_filter].mean_dSharpe:+.2f}},
  S3 {{n:runs.crypto_intraday_1h.headline[stage=S3_sizing].mean_dSharpe:+.2f}}.
- **Survivorship-biased single stocks** (current S&P 100 / NIFTY 50 members): S1
  {{n:runs.robust_single_stocks.headline[stage=S1_filter].mean_dSharpe:+.2f}}, S3
  {{n:runs.robust_single_stocks.headline[stage=S3_sizing].mean_dSharpe:+.2f}}, S4
  {{n:runs.robust_single_stocks.headline[stage=S4_exit].mean_dSharpe:+.2f}}: the largest
  gains of any run, from a mean baseline Sharpe of
  {{n:runs.robust_single_stocks.baseline.mean_sharpe:+.2f}}. This universe is biased by
  construction (the stocks are present *because* they survived), so these larger gains are
  an upper bound that does not generalise, and they are excluded from the headline
  conclusions.

**What is robust.** In every *daily* design (benchmark, 2013 hold-out, rolling window, and
the biased stock universe), S1 and S3 have a mean ΔSharpe of at least
{{n:runs.robust_holdout2013.headline[stage=S3_sizing].mean_dSharpe:+.2f}}. They improve at
least {{pct:runs.robust_holdout2013.headline[stage=S3_sizing].share_improved}} of the strategies,
and are significant for 20 to 37 strategies after BH adjustment (Table 18). S2 never helps
materially. S5 is small but positive.

**What is not robust: frequency.** At intraday crypto frequencies the picture changes:

- At 4-hour bars S3 sizing no longer helps
  ({{n:runs.crypto_intraday_4h.headline[stage=S3_sizing].mean_dSharpe:+.2f}}). Only S1
  survives ({{n:runs.crypto_intraday_4h.headline[stage=S1_filter].mean_dSharpe:+.2f}}, significant
  for {{n:runs.crypto_intraday_4h.headline[stage=S1_filter].sig_pos_bh}} strategies after BH).
- At 1-hour bars every baseline loses heavily after costs (mean Sharpe
  {{n:runs.crypto_intraday_1h.baseline.mean_sharpe:+.2f}}). S1 and S5 improve them
  ({{n:runs.crypto_intraday_1h.headline[stage=S1_filter].mean_dSharpe:+.2f}} and
  {{n:runs.crypto_intraday_1h.headline[stage=S5_params].mean_dSharpe:+.2f}}), while S2 and S3
  make them *worse*: S3 is significantly negative for
  {{n:cross_run[run=crypto_intraday_1h,stage=S3_sizing].sig_neg_bh}} strategies after BH. The S1
  gain here is mostly avoided trading cost, and no ML variant of any baseline becomes
  profitable.

The improvement is therefore robust across validation design, time period and (daily) asset
class, but **not** across sampling frequency. The sizing effect in particular appears
specific to daily bars.

### 8.12 Which indicator families carry the improvement?

{{table:T20_feature_family_ablation|feature_block,stage,model,mean_dSharpe,share_improved,sig_pos}}

{{fig:F7_feature_ablation.png|Figure 7. Mean ΔSharpe (selected model) when ML sees only one indicator family. All blocks use the same lighter protocol (2 models, retraining every 3 years), so compare blocks with each other and with "all" only, not with Table 5.}}

`vol_20` is present in every block because it scales the trade-context features. The
comparison with the "all features" reference shows how much of the gain is retained by each
family alone. The text of §9 summarises this.
