# Code Map

Mapping of the most-repeated notebook functions to their new home in `src/qresearch/`.

| Legacy function (notebooks) | Occurrences | Consolidated into |
|---|---|---|
| `resample_1hr_custom` | 96 | `data.loaders.resample_ohlcv` |
| `process_ticker` | 57 | `data.loaders.process_by_ticker` |
| `backtest_trades` | 88 | `backtest.engine.backtest_trades` |
| `run_backtest` | 25 | `backtest.engine.run_backtest` |
| `cy_backtest_trades` | 17 | `backtest.engine.backtest_trades` |
| `py_backtest_trades` | 12 | `backtest.engine.backtest_trades` |
| `calculate_metrics` | 64 | `backtest.metrics.calculate_metrics` |
| `calculate_metrics_fast` | 24 | `backtest.metrics.calculate_metrics` |
| `calculate_composite_score` | 12 | `backtest.metrics.rank_and_score` |
| `rank_and_score` | 8 | `backtest.metrics.rank_and_score` |
| `normalize_metric` | 11 | `backtest.metrics._normalize` |
| `compute_jma_indicator` | 32 | `indicators.moving_averages.jma` |
| `compute_jma_and_signals` | 25 | `signals.generators.jma_signals` |
| `compute_atr` | 27 | `indicators.volatility.atr` |
| `compute_atr_max` | 22 | `indicators.volatility.atr` |
| `walk_forward_optimize` | 11 | `optimize.walk_forward.walk_forward_optimize` |
| `split_data_by_periods` | 7 | `optimize.walk_forward.split_data_by_periods` |
| `close_position` | 15 | `risk.stops.apply_stop_take` |
| `calc_pnl` | 12 | `backtest.engine.backtest_trades` |
| `map_signals_to_1min` | 7 | `signals.generators.map_signals_to_timeframe` |
| `plot_pnl_distribution` | 13 | `viz.plots.plot_pnl_distribution` |
| `plot_trade_summary` | 9 | `viz.plots.plot_trade_summary` |
| `save_plot_as_image` | 8 | `viz.plots.plot_trade_summary(save_path=...)` |
| `apply_strategy` | 44 | `backtest.engine` + `signals.generators` |

## Exhaustive-extraction pass (2026-07-12)

The table above covered only the ~24 highest-frequency functions from the
original migration. This pass extracted the remaining reusable logic still
inline in notebooks — including the functions the original pass missed
entirely (VWAP, curvature, LHP/LTP-KAMA DSL). ~45 near-duplicate
`backtest_*`/`cy_`/`py_`/`_numba` strategy-glue wrappers and 2 hardcoded
Colab-specific grid-search functions were deliberately **not** re-extracted —
they're redundant with `backtest.engine.backtest_trades`/`run_backtest` and
`optimize.walk_forward.grid_search`, which already cover the same pattern.

### Indicators (Phase 1)

| Legacy function(s) | Consolidated into |
|---|---|
| `compute_vwap`, `calculate_vwap` | `indicators.vwap.vwap` / `tick_vwap` |
| curvature calc in `calculate_strategy_signals` | `indicators.curvature.curvature` |
| `compute_lhp_dsl_indicator` (5 notebooks) | `indicators.dsl.lhp_dsl` |
| `compute_ltpkamadsl_indicator` (2 variants) | `indicators.dsl.ltp_kama_dsl` |
| `kalman_filter`/`_fast`/`_optimized`, `initialize_/apply_kalman_filter` | `indicators.kalman.kalman_filter` |
| `goertzel`, `detect_strongest_cycle`, `rainflow_hilbert`, `cfba_jdmx_histogram` | `indicators.cycle.*` |
| `calculate_supertrend` (bug: fed LTP as high/low/close, TR≡0) | `indicators.trend.supertrend` (fixed: real OHLC) |
| `halftrend`, `halftrend_with_session_reset` | `indicators.trend.halftrend` / `halftrend_session_reset` |
| `dema`, `compute_custom_dema`, `compute_normalized_dema`, `compute_norm_dema_for_group` | `indicators.trend.dema` / `normalized_dema` |
| `compute_kama_indicator`, `compute_kama_numba` | `indicators.trend.kama` |
| `zero_lag_ma`, `weighted_moving_average` | `indicators.trend.*` |
| `compute_obv`, `compute_mfi`/`calculate_mfi`, `calculate_tmf` | `indicators.volume.obv` / `mfi` / `tmf` |
| `calculate_pvs_vms`, `calculate_vms`, `calculate_samx` | `indicators.volume.pvs_vms` / `samx` |
| `compute_breadth`, `calculate_adx` (2 variants, kept the correct Wilder DM) | `indicators.volume.breadth` / `adx` |
| `atr_vec`, `rogers_satchell_vol`, `compute_rs_volatility_fast`, `compute_hv`, `rolling_zscore`, `rolling_r2`, `calculate_ohlc4` | `indicators.volatility.*` / `indicators.moving_averages.ohlc4` |
| `stochastic_oscillator`/`calculate_quad_stochastic` (core calc) | `indicators.momentum.stochastic_oscillator` |

### Signals + Backtest (Phase 2)

| Legacy function(s) | Consolidated into |
|---|---|
| `identify_fvg`, `find_order_blocks`, `identify_liquidity_sweeps`, `calculate_fibonacci_levels`, `filter_session` | `signals.smc.*` |
| `find_divergences`/`_divergences`, `add_stoch_signals`/`generate_quad_stochastic_signals` | `signals.patterns.find_divergences` / `quad_stochastic_signals` |
| `calculate_z_score` (2 variants) | `signals.patterns.zscore_spread` |
| `range_detector` | `signals.patterns.range_detector` |
| `mean_reversion_signal` | `signals.patterns.mean_reversion_signal` |
| `robustness_testing`, `test_combination` | `backtest.optimize.robustness_testing` |
| `safe_profit_factor` | `backtest.metrics.profit_factor` (bug fixed: both-zero case) |
| `calculate_win_rate` | `backtest.metrics.win_rate` |
| `calculate_quarterly_hpr` | `backtest.metrics.quarterly_hpr` |
| `max_drawdown` (was dead code — never called) | `backtest.metrics.max_drawdown` (now used by `calculate_metrics`) |

### Risk + Viz + Data (Phase 3)

| Legacy function(s) | Consolidated into |
|---|---|
| `check_margin_call` | `risk.analysis.check_margin_call` |
| `monte_carlo_simulation` | `risk.analysis.monte_carlo_simulation` |
| `calculate_expected_loss` | `risk.analysis.expected_loss` |
| `calculate_drawdown`/`compute_drawdown`/`calculate_max_drawdown` | dedup'd into existing `backtest.metrics.max_drawdown` |
| `plot_drawdown`, `plot_pnl_by_exit_reason`, `plot_pnl_by_weekday`, `plot_win_rate_by_direction`, `plot_yearly_returns`, `plot_monthly_profit_heatmap` | `viz.plots.*` (matplotlib-only; plotly variants dropped) |
| `fetch_ohlcv_okx_safe`, `fetch_funding_okx_safe`, `merge_funding` | `data.preprocess.*` (exchange now an explicit param, not a notebook global) |
| `preprocess_ohlc`, `create_quarterly_features`, `add_tradebook_features` | `data.preprocess.*` |
| `resolve_columns`/`_resolve`/`_ALIASES` | `data.preprocess.resolve_columns` |
| `save_parquet_by_ticker` | `data.loaders.save_parquet_by_ticker` |
| `remove_pps_outliers`, `is_float`, `convert` (data-cleaning_006) | **not ported** — real-estate price-per-sqft tutorial code, not trading data |
| `load_data`, `_is_colab` | **not ported** — Colab environment glue |

### ML + Execution (Phase 4)

| Legacy function(s) | Consolidated into |
|---|---|
| `train_ml_model` (2 near-identical copies) | `ml.models.train_classifier` |
| `live_trade_decision`, `predict_signals` | `ml.models.predict_signals` |
| `classify_trade` | `ml.models.label_trade_outcome` |
| `label_signals` | `ml.models.label_signals` |
| `align_signals_to_ohlcv` | `ml.models.align_signals_to_ohlcv` |
| `get_score` | `ml.models.get_score` |
| `objective` (Optuna, LightGBM path) | `ml.models.lgbm_objective` (CatBoost/SMOTE path dropped — not project dependencies) |
| `step_forward_prediction_with_top_10_tickers` | `ml.models.walk_forward_topn` |
| `create_advanced_features` | `features.pipeline.add_lag_features` / `add_cyclical_time_features` |
| `generate_features` | `features.pipeline.add_market_microstructure_features` |
| `submit_market_order`, `confirm_order_filled`, `get_position`, `get_account`, `cancel_all_orders`, `round_price`, `round_quantity` (Alpaca + Binance, duplicated per-exchange) | `execution.broker.CcxtBroker` (one adapter over ccxt, already a dependency; Alpaca-specific adapter not added — `alpaca-py` isn't a dependency) |