# Folder Descriptions

One-line purpose per top-level folder. For design principles, the `qresearch`
package entry points, and strategy sub-organization, see `ARCHITECTURE.md`
(this doc used to be a byte-identical copy of it — trimmed to just the table).

| Folder | Purpose |
|---|---|
| `src/qresearch/` | The importable package — all reusable logic. See `CODE_MAP.md` for what moved where. |
| `notebooks/01_data_collection` | Fetching OHLCV from exchanges/brokers (Binance, yfinance, NSE, Kite). |
| `notebooks/02_data_cleaning` | Resampling, NaN handling, deduplication, schema normalization. |
| `notebooks/03_feature_engineering` | Returns, rolling stats, z-scores, lags. |
| `notebooks/04_indicators` | Indicator construction (JMA, ATR, RSI, MACD, custom). |
| `notebooks/05_strategies` | Full strategies, organized `by_market/`, cross-referenced `by_style` & `by_session` in `STRATEGY_INDEX.md`. |
| `notebooks/06_signal_generation` | Entry/exit signal logic in isolation. |
| `notebooks/07_portfolio_construction` | Multi-asset allocation / weighting. |
| `notebooks/08_position_sizing` | Kelly, fixed-fractional, risk-per-trade. |
| `notebooks/09_risk_management` | Stops, drawdown control, exposure limits, margin/monte-carlo analysis. |
| `notebooks/10_backtesting` | Backtest engines and equity-curve studies. |
| `notebooks/11_walk_forward` | Out-of-sample / anchored walk-forward validation. |
| `notebooks/12_optimization` | Parameter grids and search. |
| `notebooks/13_hyperparameter_tuning` | ML/DL hyperparameter search (Optuna). |
| `notebooks/14_machine_learning` | Classical ML models (sklearn/XGBoost/LightGBM). |
| `notebooks/15_deep_learning` | Neural nets (PyTorch/Keras). |
| `notebooks/16_reinforcement_learning` | RL agents/environments for trading. |
| `notebooks/17_statistical_arbitrage` | Cointegration, pairs, GARCH, mean reversion. |
| `notebooks/18_execution` | Order placement, slippage, broker interaction. |
| `notebooks/19_visualization` | Plotting and reporting utilities/experiments. |
| `notebooks/20_research_notes` | Written notes and findings (Markdown). |
| `notebooks/21_experiments` | Early/abandoned/exploratory trading research (non-trading experiments quarantined to `99_archive/non_quant/`). |
| `notebooks/99_archive/duplicates` | Exact/near-duplicate notebooks kept for safety, not migrated. |
| `notebooks/99_archive/non_quant` | Computer-vision/virtual-try-on research (SAM2, pose estimation, DeepFashion) — not trading research. See its `README.md`. |
| `docs/` | This documentation set — architecture, strategy/research indexes, code map, dataset requirements, environment setup, roadmap. |
| `tests/` | pytest suite for `qresearch` (smoke tests + per-module unit/golden-value tests). |
| `data/` | Git-ignored raw/processed/external data directories (see `DATASET_REQUIREMENTS.md`). |
