## 2. Related literature

**Trend following.** Rule-based trend following has a long empirical record:

- Moving-average and channel-breakout rules (Donchian 1960; Brock, Lakonishok & LeBaron 1992).
- Time-series momentum across futures markets (Moskowitz, Ooi & Pedersen 2012) and over a
  century of data (Hurst, Ooi & Pedersen 2017).
- Simple tactical rules such as Faber's (2007) 10-month moving average and Antonacci's
  (2014) absolute momentum.

Sullivan, Timmermann & White (1999) showed that the apparent profitability of
technical-rule *universes* must be judged against the data-snooping implied by searching
over them.

**ML on top of rules.** López de Prado (2018) proposed *meta-labelling*: a secondary
classifier decides whether to act on a primary model's signal and how much to bet.
Lim, Zohren & Roberts (2019) learn trend-following positions with deep networks. Gu,
Kelly & Xiu (2020) document gains from tree ensembles and neural networks for return
prediction, under strict train/validation/test splits.

**Inference.** We rely on:

- Ledoit & Wolf (2008) for Sharpe-ratio differences.
- White's (2000) Reality Check and Hansen's (2005) SPA test for "best of many" comparisons.
- Romano & Wolf (2005) stepdown FWER control.
- Benjamini & Hochberg (1995) and Benjamini & Yekutieli (2001) for FDR.
- Bailey & López de Prado (2014) for the deflated Sharpe ratio, and Bailey, Borwein,
  López de Prado & Zhu (2017) for the probability of backtest overfitting.

Harvey, Liu & Zhu (2016) argue for higher significance hurdles in factor research. The
same multiplicity concern applies here, with 39 strategies × 5 stages × 5 models.

What is missing from this literature is a *controlled, stage-by-stage* measurement. The
question is not whether an ML trading model can be profitable. It is how much ML adds to
a given, published trend rule, at which point in the pipeline, and with what statistical
confidence once the search over models and configurations is accounted for.
