# Future Roadmap

## Immediate (review pass)
- [ ] Verify heuristic classifications in `STRATEGY_INDEX.md` / `RESEARCH_INDEX.md`; correct any
      mis-filed notebooks and reassign `_unclassified` strategies to their real market.
- [ ] Fill the auto-doc header in each notebook (hypothesis, alpha source, entry/exit, risk).
- [ ] Confirm the 74 archived duplicates are truly redundant before any deletion.

## Short term (consolidation)
- [ ] Migrate notebooks to import from `qresearch` instead of re-defining functions inline.
- [ ] Replace the reference `jma` implementation with your exact production JMA parameters.
- [ ] Add unit tests for indicators (golden values) and the backtest engine.
- [ ] Standardize the data layer on Parquet; add a small cached data catalog.

## Medium term (rigor)
- [ ] Add transaction-cost and slippage models per market to the backtest engine.
- [ ] Formalize walk-forward reports (per-fold OOS tables, deflated Sharpe, PBO).
- [ ] Portfolio-level backtesting (combine strategies, correlation, capital allocation).
- [ ] Experiment tracking (MLflow/Weights & Biases) for ML/DL/RL notebooks.

## Long term (production)
- [ ] Live execution adapters (Binance/ccxt, Kite) implementing `execution.BrokerBase`.
- [ ] Paper-trading harness and monitoring/alerting.
- [ ] CI matrix with data fixtures; nightly regression of headline strategies.
