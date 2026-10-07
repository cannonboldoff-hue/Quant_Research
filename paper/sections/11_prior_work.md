## Appendix A. Audit of the repository's earlier research code

This study builds on an existing research repository: 154 notebooks consolidated into the
`qresearch` package, plus a forex JMA/ATR research log. Before reusing any of it we
audited it. The study reuses the JMA numba core and the repository's statistics module.
We found and fixed these issues:

| Issue | Where | Consequence | Fix |
|---|---|---|---|
| Random `train_test_split` on time-ordered trades (14 notebooks; also `ml.models.train_classifier`); no notebook used a time-ordered split | `notebooks/14_machine_learning/*`, `src/qresearch/ml/models.py` | future trades leak into training, inflating ML accuracy | chronological split in the library; the new framework uses purged walk-forward folds |
| Per-trade returns annualised with √252 | `backtest.metrics.calculate_metrics` | Sharpe inflated by √(average holding days) for multi-day trades | annualise by actual trades per year |
| "Paired" bootstrap resampled the two return series independently and i.i.d. | `stats.risk.paired_bootstrap_ci` | ignores cross-correlation and serial dependence; overstates significance of overlapping strategies | paired circular block bootstrap |
| JMA wrapper dropped the input index | `indicators.moving_averages.jma` | silent misalignment with date-indexed data | index preserved |
| Trades entered at the close of the signal bar | `backtest.engine` | same-close execution (optimistic) | the new engine executes at the next open; same-close is reported only as a sensitivity case |
| pandas-3 copy-on-write read-only arrays | `indicators.trend.supertrend`, `indicators.curvature` | crashes (2 failing tests) | explicit copies |
| Hive-partition schema clash (string vs large_string) | `campaign.provenance` | provenance store unreadable under pandas 3 | partition keys no longer duplicated in files |
| Stale test expectation; "real-data" test not skipped when data absent | `tests/test_campaign_e2e.py` | failing suite | corrected |

The research log's headline claim (forex daily JMA crossover, Sharpe 3.75 on 4 pairs
picked from 28) combined per-pair walk-forward parameter selection, ex-post pair
selection and same-close fills. The JMA 7/21 crossover is one of our 39 rules (`jma_7_21`),
so Table T7 shows its out-of-sample performance under the stricter protocol used here.
