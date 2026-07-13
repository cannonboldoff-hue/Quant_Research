# Repository Architecture

## Design principles
1. **Single source of truth for logic** — reusable code lives in `src/qresearch/`, never duplicated
   in notebooks. Notebooks import from the package.
2. **Notebooks are narratives, not libraries** — each notebook documents a hypothesis and shows
   results; heavy lifting is delegated to the package.
3. **Reproducible & Git-friendly** — outputs stripped, secrets in `.env`, deterministic seeds
   encouraged, data directories ignored.
4. **Nothing valuable lost** — uncertain/duplicate material is archived, not deleted.

## Notebook sections
| Folder | Purpose |
|---|---|
| `01_data_collection` | Fetching OHLCV from exchanges/brokers (Binance, yfinance, NSE, Kite). |
| `02_data_cleaning` | Resampling, NaN handling, deduplication, schema normalization. |
| `03_feature_engineering` | Returns, rolling stats, z-scores, lags. |
| `04_indicators` | Indicator construction (JMA, ATR, RSI, MACD, custom). |
| `05_strategies` | Full strategies, organized `by_market/`, cross-referenced `by_style` & `by_session`. |
| `06_signal_generation` | Entry/exit signal logic in isolation. |
| `07_portfolio_construction` | Multi-asset allocation / weighting. |
| `08_position_sizing` | Kelly, fixed-fractional, risk-per-trade. |
| `09_risk_management` | Stops, drawdown control, exposure limits. |
| `10_backtesting` | Backtest engines and equity-curve studies. |
| `11_walk_forward` | Out-of-sample / anchored walk-forward validation. |
| `12_optimization` | Parameter grids and search. |
| `13_hyperparameter_tuning` | ML/DL hyperparameter search. |
| `14_machine_learning` | Classical ML models (sklearn/XGBoost/LightGBM). |
| `15_deep_learning` | Neural nets (PyTorch/Keras). |
| `16_reinforcement_learning` | RL agents/environments for trading. |
| `17_statistical_arbitrage` | Cointegration, pairs, GARCH, mean reversion. |
| `18_execution` | Order placement, slippage, broker interaction. |
| `19_visualization` | Plotting and reporting utilities/experiments. |
| `20_research_notes` | Written notes and findings (Markdown). |
| `21_experiments` | Early/abandoned/exploratory work (incl. non-trading experiments). |
| `99_archive` | `duplicates/` — exact/near-duplicate notebooks kept for safety. |

## Strategy sub-organization
- **`by_market/`** — physical folders: `crypto`, `indian_equities`, `commodities`, `options`,
  `futures`, `forex`, `us_equities`, plus `_unclassified` (8 remaining — no clear ticker/data-path
  evidence found; not guessed) for notebooks whose market couldn't be inferred with confidence.
- **`by_style/`** and **`by_session/`** — reference folders; the mapping is maintained in
  `STRATEGY_INDEX.md` (style: scalping/intraday/swing/positional; session: opening/midday/closing/overnight).

## The `qresearch` package
A flat, importable library. Key entry points: `data.load_crypto_ohlcv` / `resample_ohlcv`,
`indicators.jma` / `atr`, `signals.jma_signals` / `map_signals_to_timeframe`,
`backtest.backtest_trades` / `run_backtest` / `calculate_metrics` / `rank_and_score`,
`risk.atr_stop_levels`, `optimize.walk_forward_optimize`, `viz.plot_trade_summary`,
`execution.PaperBroker`. See `CODE_MAP.md` for legacy-function correspondences.
