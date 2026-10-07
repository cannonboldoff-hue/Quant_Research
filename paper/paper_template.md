# Does Machine Learning Improve Trend Following? A Stage-by-Stage, Multi-Asset, Walk-Forward Evaluation of 39 Rule-Based Strategies

*Research repository: `qresearch.tfml`. Benchmark run code hash
`{{n:runs.benchmark.meta.code_hash}}`, dataset hash `{{n:runs.benchmark.meta.dataset_hash}}`.
Every number below is generated from the experiment registry. See §11.*

{{include:sections/00_abstract.md}}

{{include:sections/01_intro.md}}

{{include:methods_draft.md}}

{{include:sections/03_methods.md}}

{{include:sections/08_results.md}}

{{include:sections/09_answer.md}}

{{include:sections/10_limitations_repro.md}}

## References

- Antonacci, G. (2014). *Dual Momentum Investing*. McGraw-Hill.
- Appel, G. (1979). *The Moving Average Convergence-Divergence Method*. Signalert.
- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5).
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4).
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS B*, 57(1).
- Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics*, 29(4).
- Brock, W., Lakonishok, J., & LeBaron, B. (1992). Simple technical trading rules and the stochastic properties of stock returns. *Journal of Finance*, 47(5).
- Chande, T. (1992). Adapting moving averages to market volatility. *Technical Analysis of Stocks & Commodities*.
- Chande, T., & Kroll, S. (1994). *The New Technical Trader*. Wiley.
- Donchian, R. (1960). High finance in copper. *Financial Analysts Journal*, 16(6).
- Ehlers, J. (2005). Fractal adaptive moving averages. *Technical Analysis of Stocks & Commodities*.
- Elder, A. (2002). *Come Into My Trading Room*. Wiley.
- Faber, M. (2007). A quantitative approach to tactical asset allocation. *Journal of Wealth Management*, 9(4).
- Faith, C. (2007). *Way of the Turtle*. McGraw-Hill.
- Garman, M., & Klass, M. (1980). On the estimation of security price volatilities from historical data. *Journal of Business*, 53(1).
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5).
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business & Economic Statistics*, 23(4).
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). … and the cross-section of expected returns. *Review of Financial Studies*, 29(1).
- Hurst, B., Ooi, Y. H., & Pedersen, L. H. (2017). A century of evidence on trend-following investing. *Journal of Portfolio Management*, 44(1).
- Kaufman, P. (1995). *Smarter Trading*. McGraw-Hill.
- Ledoit, O., & Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. *Journal of Empirical Finance*, 15(5).
- Lim, B., Zohren, S., & Roberts, S. (2019). Enhancing time-series momentum strategies using deep neural networks. *Journal of Financial Data Science*, 1(4).
- Lo, A., & MacKinlay, A. C. (1988). Stock market prices do not follow random walks. *Review of Financial Studies*, 1(1).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Moskowitz, T., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2).
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 53(1).
- Politis, D., & Romano, J. (1992). A circular block-resampling procedure for stationary data. In *Exploring the Limits of Bootstrap*. Wiley.
- Rogers, L. C. G., & Satchell, S. (1991). Estimating variance from high, low and closing prices. *Annals of Applied Probability*, 1(4).
- Romano, J., & Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica*, 73(4).
- Sullivan, R., Timmermann, A., & White, H. (1999). Data-snooping, technical trading rule performance, and the bootstrap. *Journal of Finance*, 54(5).
- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5).
- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
- Yang, D., & Zhang, Q. (2000). Drift-independent volatility estimation based on high, low, open, and close prices. *Journal of Business*, 73(3).

Strategy and indicator originators: Bollinger (2001), *Bollinger on Bollinger Bands*;
Blau (1991, TSI); Botes & Siepman (2010, Vortex); Clenow (2015), *Stocks on the Move*;
Ehlers & Way (2010, zero-lag EMA); Hutson (1983, TRIX); Keltner (1960), *How to Make Money
in Commodities*; Lambert (1980, CCI); LeBeau & Lucas (1992, Chandelier exit); Mulloy
(1994, DEMA/TEMA); Pring (1992, KST); Tillson (1998, T3). Practitioner indicators without
a single canonical publication (Hull MA, McGinley Dynamic, GMMA, SuperTrend, Heikin-Ashi,
the 4-9-18 method) are attributed to their originators in Table T2.

{{include:sections/11_prior_work.md}}

## Appendix B. Strategy, feature and stage definitions

{{table:T2_strategies|id,family,reference,params,direction}}

{{table:T3_features}}

## Appendix C. Baseline and ML-selected metrics per strategy (benchmark)

{{table:benchmark_T7_baseline_metrics}}

## Appendix D. Data validation and exclusions

{{table:T0_data_validation|pair,overlap_days,corr_daily,corr_weekly,tracking_err_ann,max_abs_diff,max_diff_date}}

{{table:T1b_data_exclusions}}
