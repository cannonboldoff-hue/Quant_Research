## 10. Limitations

- **Data quality.** Data come from free vendors:
  - Continuous futures are not roll-adjusted.
  - Indices exclude dividends.
  - FX excludes carry.
  - Crypto prices are cross-venue aggregates.

  These affect the baseline and the ML variants identically, since both are evaluated on
  the same bars, so the *difference* is less exposed than the levels. Absolute Sharpe
  ratios should still be read with these caveats.
- **Costs are assumptions** per asset class. Market impact is not modelled, which matters
  for large capital in the less liquid futures and EM FX.
- **Fixed hyper-parameters.** Each model family is represented by one a-priori
  configuration. Tuning them (on validation data) could change the model ranking. It
  would also enlarge the search space the multiple-testing corrections must cover.
- **Pooled models.** A model is fitted per strategy and stage across all instruments.
  Per-asset-class models are a natural extension, supported by the framework through the
  universe groups.
- **History.** The OOS period starts in 2006. Longer histories (pre-1990 futures, Stooq's
  century-long index series) would cover more trend regimes. Stooq's export is
  CAPTCHA-gated and was not automated. `data.clean_stooq` ingests manually downloaded
  files.
- **Survivorship.** The benchmark universe avoids constituent selection. The single-stock
  run is survivorship-biased by construction and is reported only as such.

## 11. Reproducibility

Every number in this paper is a placeholder in `paper/paper_template.md`, filled by
`paper/build_paper.py` from `paper/results_summary.json`. That file is computed by
`scripts/tfml_report.py` from the experiment registry. Each registry row links strategy →
indicators → market → instrument → dataset hash → ML stage → ML model (and the per-fold
selection counts) → validation design → backtest configuration → metrics → stored
return series. It carries the code hash of the framework source that produced it.

To reproduce:

```bash
pip install -r requirements-tfml.txt && pip install -e . --no-deps
python scripts/tfml_fetch_data.py
bash scripts/tfml_run_all.sh 6
python scripts/tfml_report.py && python paper/build_paper.py
```

The benchmark configuration is `configs/tfml/benchmark.yaml`. Every component
(universe, data source, frequency, features, strategies, models, stages, validation
scheme, execution, costs, sizing) is swappable through configuration or a single
registry list (`docs/TFML_FRAMEWORK.md`).
