# `qresearch.tfml` — ML-enhanced vs rule-based trend-following framework

This is the code behind `paper/paper.md`. It answers one question: **how much does ML
add to established trend-following rules, and when is the improvement statistically
robust?** It extends the repository's original `qresearch` package and reuses its JMA
core, stats module and data conventions. It is a separate subpackage because the
research design (pooled walk-forward ML over a multi-asset panel) differs from the
earlier per-notebook JMA/ATR studies.

## Pipeline

```
configs/tfml/universe.yaml ──► data.py  (download → raw/, clean → processed/, manifest.json)
                                  │
                                  ▼
                               panel.py (stacked multi-instrument panel + cached features)
                                  │
         features.py (56 causal, scale-free features from indicators.py)
         strategies.py (39 rule-based baselines → target position p_t ∈ [-1, 1])
                                  │
                                  ▼
       pipeline.py  — per strategy, per walk-forward fold:
          labels (purged + embargoed) → fit 5 models per stage (ml.py)
          → pick model + intervention intensity on the validation block only
          → apply frozen choice to the test block
                                  │
                                  ▼
       evaluate.py — OOS metrics (metrics.py), ΔSharpe tests (stats.py),
          portfolio / market / instrument / sub-period / sensitivity levels
                                  │
                                  ▼
       registry.py — SQLite rows (experiments, ml_selection, runs) + return series
                                  │
                                  ▼
       scripts/tfml_report.py → paper/tables, paper/figures, paper/results_summary.json
       paper/build_paper.py    → paper/paper.md (+ .html), numbers injected from the summary
```

## Swapping components

| Component | Where | How to swap |
|---|---|---|
| Instruments / markets / costs | `configs/tfml/universe.yaml` | add a group or instrument; `benchmark: false` keeps it out of the headline universe |
| Data source | `data.py` (`fetch_yahoo`, `fetch_binance_klines`, `clean_stooq`, …) | add a `fetch_*`/`clean_*` pair and set `source:` on the group |
| Frequency | run config `frequency:` (`daily`, `1h`, `4h`, `daily_binance`) | `data.resample_bars` builds others |
| Indicators / features | `indicators.py`, `features.py` (`FEATURE_SPECS`) | add a causal function + a spec entry; the causality test covers it automatically |
| Strategies | `strategies.py` (`STRATEGIES` list) | add a `Strategy(...)` with rule function, parameters, `scale` (lookbacks for variants) |
| ML models | `ml.py` (`make_model`, `MODEL_NAMES`) | any sklearn-style classifier with `predict_proba` |
| ML stages / decision rules | `ml.py` (`apply_*`), `pipeline.py` (`STAGES`, `INTENSITY`) | add a decision rule and a branch in `run_strategy` |
| Validation | run config `validation:` (`wf_expanding`, `wf_rolling`, `holdout`), `val_years`, `retrain_every_years` | `pipeline.folds` |
| Backtest mechanics | `backtest.py` / `panel.simulate_panel` (`next_open`, `next_close`, `same_close`), `cost_mult`, `target_vol` | sensitivity variants are evaluated automatically in `evaluate.py` |

Run any configuration with:

```bash
python scripts/tfml_fetch_data.py                                   # data (idempotent)
python scripts/tfml_run.py --config configs/tfml/benchmark.yaml     # one run
bash   scripts/tfml_run_all.sh 6                                    # every run in the paper
python scripts/tfml_report.py && python paper/build_paper.py        # tables, figures, paper
```

## Leakage, look-ahead and bias controls (each enforced in code)

| Risk | Control | Where enforced / tested |
|---|---|---|
| Look-ahead in indicators/strategies | every feature and position at bar t is computed from bars ≤ t; month-end flags come from the calendar, not from the next bar | `tests/test_tfml.py::test_features_are_causal`, `::test_strategy_positions_are_causal_and_bounded` (truncated-history equality for all 56 features and 39 strategies) |
| Execution look-ahead | position decided at close t is executed at the open of t+1; the overnight gap is earned by the old position | `backtest.simulate`, `panel.simulate_panel`; `test_next_open_accounting`, `test_signal_cannot_earn_its_own_gap` |
| Label leakage | each training row's label must be fully realised (label horizon + 5-bar embargo) before the validation start; validation labels before the test start | `pipeline._train_rows`, `_label_date` |
| Overlapping labels | training rows sub-sampled at the label horizon (non-overlapping labels per instrument) | `train_stride: 0` |
| Model-selection bias | model and intervention intensity chosen on the validation block only; test results of every alternative model are reported, not just the winner | `pipeline.run_strategy`, `ml_selection` table |
| Data snooping across configurations | Hansen SPA + White RC per strategy (25 configurations); Romano–Wolf stepdown; Holm/BH/BY across the 39×5 strategy-stage family; PBO (CSCV) and deflated Sharpe | `stats.py`, `evaluate.py`, `scripts/tfml_report.py` |
| Hyper-parameter overfitting | all model hyper-parameters fixed a priori; only (model, intensity ∈ 3 values) chosen on validation | `ml.make_model` |
| Survivorship bias | headline universe uses indices, continuous futures, FX, ETFs and long-lived crypto; single stocks chosen from *current* index membership are run separately and labelled biased | `universe.yaml` (`survivorship`, `benchmark`) |
| Costs | per-asset-class per-side cost + slippage on every unit of position change; 0×/1×/3× sensitivity | `universe.yaml`, `evaluate.py` |

## Registry schema (`results/tfml/registry.sqlite`)

`experiments` — one row per evaluated configuration. Lineage columns:
`strategy, strategy_family, strategy_params, strategy_reference, indicators, feature_set,
market, instrument, asset_class, region, dataset (frequency, #instruments, hash of raw-file
hashes), ml_stage, ml_model, selected_models (fold counts when ml_model = selected),
validation, backtest, level, period, run_id, git_commit, code_hash, exp_id, result_path`.
Metrics: `cagr, total_return, ann_vol, sharpe, sortino, calmar, max_drawdown,
win_rate_days, win_rate_trades, profit_factor, profit_factor_trades, turnover, cost_drag,
exposure, n_trades, avg_hold_bars, stability_r2, pct_pos_years, skew`. Comparison with the
baseline: `base_sharpe, d_sharpe, d_cagr, d_max_drawdown, d_sortino, d_calmar, d_turnover,
p_boot_two, p_boot_one, ci_lo, ci_hi, p_hac_two, z_hac, p_romano_wolf`, and on the `spa`
rows `p_spa, p_rc, best_config, n_configs`.

`level` takes the values `portfolio` (all instruments, 10% vol-target per instrument,
equal capital), `market` (same per universe group), `instrument` (unit-size, faithful
rules), `subperiod`, `sensitivity` (alternative costs, execution, sizing) and `spa`.

`ml_selection` — for every fold × stage × model: the chosen intensity, validation Sharpe,
baseline validation Sharpe, whether the model was selected, training-set size and class
balance.

`runs` — the full run configuration (JSON), git commit, code hash and dataset hash.

`result_path` points to `results/tfml/<run>/<strategy>.parquet`. That file holds the daily
OOS portfolio return series of the baseline and every stage × model, plus the
market-group series of the baseline and the selected models.

## Experiment scripts (in the order used for the paper)

| Script | Purpose |
|---|---|
| `scripts/tfml_fetch_data.py` | download + clean the universe, write `manifest.json` |
| `scripts/tfml_run.py --config configs/tfml/<run>.yaml` | one run: baselines + 5 ML stages x 5 models, walk-forward, evaluation, registry rows |
| `scripts/tfml_run_all.sh` / `tfml_resume.sh` | benchmark + 5 robustness runs (`resume` runs only strategies missing from the registry) |
| `scripts/tfml_null_test.py` | full pipeline on synthetic random-walk markets (leakage / false-discovery screen) |
| `scripts/tfml_placebo.py` | same decision rules with random predictions (isolates what the models learn) |
| `scripts/tfml_ablation.py` / `tfml_ablation_all.sh` | feature-family ablation; `all` on 39 strategies, single-family and `context_only` blocks on 15 |
| `scripts/tfml_rerun_strategy.sh <id>` | re-run one strategy in every run (e.g. after an indicator fix) |
| `scripts/tfml_validate_data.py` | cross-source data validation (Yahoo vs Binance, index vs ETF vs futures) |
| `scripts/tfml_report.py` | all tables/figures/`results_summary.json`; `--ablation-only` refreshes just the ablation |
| `paper/build_paper.py` | renders `paper.md`/`paper.html`, every number injected from `results_summary.json` |

## Changes made after the main runs (for the record)

- **McGinley Dynamic** diverged after large price drops (`ZeroDivisionError` on WIPRO.NS,
  `inf` on crypto). The step is now capped; `mcginley_14` was re-run in every run, and its
  placebo too. No other strategy uses this indicator.
- **Row filter in the pipeline** (`_rows_with_features`) ignores fully masked feature
  columns. It is identical to the original rule for all non-ablation runs (verified on the
  full benchmark panel) and was needed because the first ablation silently trained nothing.
