| strategy         | strategy_family      | ml_stage   |   base_sharpe |   sharpe |   d_sharpe |     p |   p_holm |   p_bh | selected_models                                      |
|:-----------------|:---------------------|:-----------|--------------:|---------:|-----------:|------:|---------:|-------:|:-----------------------------------------------------|
| faber_10m        | price_vs_ma          | S1_filter  |         0.233 |    0.233 |      0.000 | 1.000 |    1.000 |  1.000 | {"logreg": 6}                                        |
| faber_10m        | price_vs_ma          | S2_regime  |         0.233 |   -0.592 |     -0.825 | 0.000 |    0.000 |  0.000 | {"lgbm": 1, "mlp": 1, "et": 2, "logreg": 2}          |
| faber_10m        | price_vs_ma          | S3_sizing  |         0.233 |   -0.327 |     -0.560 | 0.002 |    0.156 |  0.003 | {"logreg": 3, "lgbm": 2, "mlp": 1}                   |
| faber_10m        | price_vs_ma          | S4_exit    |         0.233 |    0.361 |      0.128 | 0.819 |    1.000 |  0.868 | {"rf": 2, "mlp": 2, "lgbm": 1, "et": 1}              |
| faber_10m        | price_vs_ma          | S5_params  |         0.233 |    0.066 |     -0.167 | 0.020 |    1.000 |  0.035 | {"rf": 1, "lgbm": 2, "logreg": 2, "mlp": 1}          |
| gmma             | ma_crossover         | S1_filter  |        -1.136 |   -0.638 |      0.497 | 0.009 |    0.819 |  0.017 | {"logreg": 2, "mlp": 1, "et": 2, "lgbm": 1}          |
| gmma             | ma_crossover         | S2_regime  |        -1.136 |   -2.021 |     -0.885 | 0.000 |    0.000 |  0.000 | {"rf": 1, "logreg": 3, "lgbm": 2}                    |
| gmma             | ma_crossover         | S3_sizing  |        -1.136 |   -1.692 |     -0.556 | 0.011 |    0.990 |  0.020 | {"lgbm": 1, "et": 2, "logreg": 3}                    |
| gmma             | ma_crossover         | S4_exit    |        -1.136 |   -1.516 |     -0.380 | 0.095 |    1.000 |  0.144 | {"lgbm": 2, "rf": 2, "logreg": 1, "mlp": 1}          |
| gmma             | ma_crossover         | S5_params  |        -1.136 |   -0.201 |      0.935 | 0.000 |    0.000 |  0.000 | {"et": 4, "lgbm": 1, "logreg": 1}                    |
| sma_50_200       | ma_crossover         | S1_filter  |         0.387 |    0.521 |      0.134 | 0.420 |    1.000 |  0.518 | {"logreg": 3, "mlp": 3}                              |
| sma_50_200       | ma_crossover         | S2_regime  |         0.387 |   -0.419 |     -0.806 | 0.000 |    0.000 |  0.000 | {"logreg": 2, "mlp": 1, "et": 2, "lgbm": 1}          |
| sma_50_200       | ma_crossover         | S3_sizing  |         0.387 |   -0.631 |     -1.018 | 0.000 |    0.000 |  0.000 | {"logreg": 4, "lgbm": 2}                             |
| sma_50_200       | ma_crossover         | S4_exit    |         0.387 |    0.147 |     -0.240 | 0.546 |    1.000 |  0.626 | {"mlp": 1, "rf": 1, "logreg": 3, "lgbm": 1}          |
| sma_50_200       | ma_crossover         | S5_params  |         0.387 |    0.215 |     -0.172 | 0.147 |    1.000 |  0.212 | {"mlp": 1, "rf": 3, "et": 1, "logreg": 1}            |
| bll_vma_200      | price_vs_ma          | S1_filter  |         0.269 |    0.399 |      0.130 | 0.475 |    1.000 |  0.565 | {"logreg": 2, "rf": 2, "et": 1, "mlp": 1}            |
| bll_vma_200      | price_vs_ma          | S2_regime  |         0.269 |   -0.635 |     -0.904 | 0.000 |    0.000 |  0.000 | {"logreg": 3, "rf": 1, "mlp": 1, "lgbm": 1}          |
| bll_vma_200      | price_vs_ma          | S3_sizing  |         0.269 |   -0.464 |     -0.733 | 0.000 |    0.000 |  0.000 | {"lgbm": 2, "logreg": 4}                             |
| bll_vma_200      | price_vs_ma          | S4_exit    |         0.269 |   -0.018 |     -0.287 | 0.455 |    1.000 |  0.548 | {"lgbm": 2, "rf": 1, "logreg": 2, "et": 1}           |
| bll_vma_200      | price_vs_ma          | S5_params  |         0.269 |    0.207 |     -0.062 | 0.495 |    1.000 |  0.586 | {"mlp": 1, "et": 1, "logreg": 3, "rf": 1}            |
| ema_12_26        | ma_crossover         | S1_filter  |        -1.003 |   -0.365 |      0.638 | 0.002 |    0.156 |  0.003 | {"lgbm": 1, "logreg": 2, "et": 2, "mlp": 1}          |
| ema_12_26        | ma_crossover         | S2_regime  |        -1.003 |   -1.751 |     -0.748 | 0.000 |    0.000 |  0.000 | {"logreg": 4, "lgbm": 2}                             |
| ema_12_26        | ma_crossover         | S3_sizing  |        -1.003 |   -1.663 |     -0.661 | 0.002 |    0.156 |  0.003 | {"lgbm": 2, "logreg": 4}                             |
| ema_12_26        | ma_crossover         | S4_exit    |        -1.003 |   -1.099 |     -0.097 | 0.724 |    1.000 |  0.780 | {"logreg": 5, "mlp": 1}                              |
| ema_12_26        | ma_crossover         | S5_params  |        -1.003 |   -0.067 |      0.935 | 0.000 |    0.000 |  0.000 | {"rf": 2, "lgbm": 3, "logreg": 1}                    |
| triple_ma_4_9_18 | ma_crossover         | S1_filter  |        -4.179 |   -2.630 |      1.549 | 0.000 |    0.000 |  0.000 | {"lgbm": 3, "et": 2, "logreg": 1}                    |
| triple_ma_4_9_18 | ma_crossover         | S2_regime  |        -4.179 |   -4.812 |     -0.633 | 0.003 |    0.297 |  0.006 | {"lgbm": 4, "logreg": 2}                             |
| triple_ma_4_9_18 | ma_crossover         | S3_sizing  |        -4.179 |   -4.042 |      0.137 | 0.517 |    1.000 |  0.604 | {"lgbm": 2, "rf": 1, "logreg": 3}                    |
| triple_ma_4_9_18 | ma_crossover         | S4_exit    |        -4.179 |   -3.942 |      0.238 | 0.348 |    1.000 |  0.444 | {"lgbm": 2, "rf": 2, "mlp": 1, "logreg": 1}          |
| triple_ma_4_9_18 | ma_crossover         | S5_params  |        -4.179 |   -1.824 |      2.355 | 0.000 |    0.000 |  0.000 | {"et": 2, "rf": 3, "logreg": 1}                      |
| dema_20_50       | ma_crossover         | S1_filter  |        -1.079 |   -0.245 |      0.834 | 0.000 |    0.000 |  0.000 | {"lgbm": 3, "logreg": 1, "mlp": 2}                   |
| dema_20_50       | ma_crossover         | S2_regime  |        -1.079 |   -1.747 |     -0.668 | 0.000 |    0.000 |  0.000 | {"logreg": 3, "lgbm": 3}                             |
| dema_20_50       | ma_crossover         | S3_sizing  |        -1.079 |   -1.672 |     -0.593 | 0.006 |    0.552 |  0.011 | {"rf": 3, "logreg": 2, "et": 1}                      |
| dema_20_50       | ma_crossover         | S4_exit    |        -1.079 |   -0.830 |      0.249 | 0.424 |    1.000 |  0.518 | {"lgbm": 1, "rf": 3, "mlp": 2}                       |
| dema_20_50       | ma_crossover         | S5_params  |        -1.079 |   -1.636 |     -0.557 | 0.197 |    1.000 |  0.266 | {"logreg": 4, "lgbm": 2}                             |
| zlema_10_30      | ma_crossover         | S1_filter  |        -4.161 |   -2.770 |      1.391 | 0.000 |    0.000 |  0.000 | {"mlp": 3, "logreg": 1, "lgbm": 1, "et": 1}          |
| zlema_10_30      | ma_crossover         | S2_regime  |        -4.161 |   -4.527 |     -0.366 | 0.074 |    1.000 |  0.116 | {"lgbm": 2, "logreg": 2, "et": 2}                    |
| zlema_10_30      | ma_crossover         | S3_sizing  |        -4.161 |   -5.333 |     -1.172 | 0.000 |    0.000 |  0.000 | {"lgbm": 3, "mlp": 1, "et": 1, "logreg": 1}          |
| zlema_10_30      | ma_crossover         | S4_exit    |        -4.161 |   -4.149 |      0.012 | 0.953 |    1.000 |  0.974 | {"lgbm": 2, "logreg": 1, "mlp": 2, "rf": 1}          |
| zlema_10_30      | ma_crossover         | S5_params  |        -4.161 |   -1.417 |      2.744 | 0.000 |    0.000 |  0.000 | {"et": 3, "rf": 3}                                   |
| jma_7_21         | ma_crossover         | S1_filter  |        -5.689 |   -3.531 |      2.158 | 0.000 |    0.000 |  0.000 | {"lgbm": 2, "mlp": 1, "rf": 1, "logreg": 2}          |
| jma_7_21         | ma_crossover         | S2_regime  |        -5.689 |   -5.861 |     -0.172 | 0.419 |    1.000 |  0.518 | {"lgbm": 3, "rf": 1, "logreg": 2}                    |
| jma_7_21         | ma_crossover         | S3_sizing  |        -5.689 |   -6.400 |     -0.712 | 0.000 |    0.000 |  0.000 | {"mlp": 3, "logreg": 3}                              |
| jma_7_21         | ma_crossover         | S4_exit    |        -5.689 |   -6.113 |     -0.424 | 0.046 |    1.000 |  0.074 | {"logreg": 3, "mlp": 2, "et": 1}                     |
| jma_7_21         | ma_crossover         | S5_params  |        -5.689 |   -2.149 |      3.540 | 0.000 |    0.000 |  0.000 | {"rf": 4, "logreg": 2}                               |
| kama_10          | adaptive_ma          | S1_filter  |        -6.678 |   -4.149 |      2.530 | 0.000 |    0.000 |  0.000 | {"lgbm": 3, "mlp": 1, "logreg": 1, "rf": 1}          |
| kama_10          | adaptive_ma          | S2_regime  |        -6.678 |   -6.525 |      0.153 | 0.541 |    1.000 |  0.625 | {"rf": 2, "lgbm": 2, "logreg": 1, "et": 1}           |
| kama_10          | adaptive_ma          | S3_sizing  |        -6.678 |   -6.965 |     -0.286 | 0.210 |    1.000 |  0.283 | {"et": 1, "logreg": 3, "mlp": 2}                     |
| kama_10          | adaptive_ma          | S4_exit    |        -6.678 |   -7.120 |     -0.442 | 0.092 |    1.000 |  0.140 | {"rf": 2, "et": 1, "logreg": 2, "mlp": 1}            |
| kama_10          | adaptive_ma          | S5_params  |        -6.678 |   -4.336 |      2.343 | 0.000 |    0.000 |  0.000 | {"et": 1, "mlp": 1, "rf": 3, "logreg": 1}            |
| donchian_20      | breakout             | S1_filter  |        -1.119 |   -0.049 |      1.071 | 0.000 |    0.000 |  0.000 | {"lgbm": 2, "rf": 2, "logreg": 2}                    |
| donchian_20      | breakout             | S2_regime  |        -1.119 |   -1.938 |     -0.819 | 0.000 |    0.000 |  0.000 | {"logreg": 4, "mlp": 1, "lgbm": 1}                   |
| donchian_20      | breakout             | S3_sizing  |        -1.119 |   -1.358 |     -0.238 | 0.228 |    1.000 |  0.302 | {"lgbm": 1, "logreg": 5}                             |
| donchian_20      | breakout             | S4_exit    |        -1.119 |   -0.775 |      0.344 | 0.327 |    1.000 |  0.419 | {"logreg": 2, "mlp": 1, "lgbm": 1, "et": 1, "rf": 1} |
| donchian_20      | breakout             | S5_params  |        -1.119 |   -0.997 |      0.122 | 0.419 |    1.000 |  0.518 | {"mlp": 3, "rf": 1, "lgbm": 1, "logreg": 1}          |
| turtle_55_20     | breakout             | S1_filter  |        -0.513 |   -0.293 |      0.220 | 0.270 |    1.000 |  0.353 | {"mlp": 1, "lgbm": 2, "logreg": 3}                   |
| turtle_55_20     | breakout             | S2_regime  |        -0.513 |   -1.353 |     -0.840 | 0.000 |    0.000 |  0.000 | {"logreg": 4, "mlp": 1, "lgbm": 1}                   |
| turtle_55_20     | breakout             | S3_sizing  |        -0.513 |   -0.808 |     -0.294 | 0.191 |    1.000 |  0.262 | {"lgbm": 2, "et": 2, "rf": 1, "logreg": 1}           |
| turtle_55_20     | breakout             | S4_exit    |        -0.513 |   -0.342 |      0.172 | 0.612 |    1.000 |  0.685 | {"rf": 2, "logreg": 2, "mlp": 1, "et": 1}            |
| turtle_55_20     | breakout             | S5_params  |        -0.513 |   -0.194 |      0.320 | 0.152 |    1.000 |  0.219 | {"mlp": 1, "rf": 3, "logreg": 1, "et": 1}            |
| hma_55_slope     | adaptive_ma          | S1_filter  |        -1.468 |   -0.961 |      0.507 | 0.027 |    1.000 |  0.047 | {"logreg": 2, "lgbm": 1, "mlp": 1, "et": 1, "rf": 1} |
| hma_55_slope     | adaptive_ma          | S2_regime  |        -1.468 |   -2.296 |     -0.828 | 0.000 |    0.000 |  0.000 | {"lgbm": 4, "logreg": 2}                             |
| hma_55_slope     | adaptive_ma          | S3_sizing  |        -1.468 |   -1.982 |     -0.513 | 0.018 |    1.000 |  0.031 | {"lgbm": 2, "rf": 1, "logreg": 2, "et": 1}           |
| hma_55_slope     | adaptive_ma          | S4_exit    |        -1.468 |   -1.330 |      0.138 | 0.626 |    1.000 |  0.694 | {"lgbm": 2, "rf": 1, "mlp": 1, "logreg": 2}          |
| hma_55_slope     | adaptive_ma          | S5_params  |        -1.468 |   -1.308 |      0.161 | 0.642 |    1.000 |  0.703 | {"et": 4, "logreg": 1, "rf": 1}                      |
| turtle_20_10     | breakout             | S1_filter  |        -1.183 |   -0.257 |      0.926 | 0.000 |    0.000 |  0.000 | {"mlp": 1, "et": 2, "rf": 2, "logreg": 1}            |
| turtle_20_10     | breakout             | S2_regime  |        -1.183 |   -1.980 |     -0.797 | 0.000 |    0.000 |  0.000 | {"lgbm": 3, "logreg": 2, "mlp": 1}                   |
| turtle_20_10     | breakout             | S3_sizing  |        -1.183 |   -1.077 |      0.105 | 0.632 |    1.000 |  0.696 | {"lgbm": 2, "rf": 1, "et": 2, "logreg": 1}           |
| turtle_20_10     | breakout             | S4_exit    |        -1.183 |   -0.759 |      0.423 | 0.132 |    1.000 |  0.191 | {"logreg": 2, "et": 2, "mlp": 1, "lgbm": 1}          |
| turtle_20_10     | breakout             | S5_params  |        -1.183 |   -0.857 |      0.326 | 0.181 |    1.000 |  0.250 | {"et": 5, "lgbm": 1}                                 |
| vidya_14         | adaptive_ma          | S1_filter  |        -3.828 |   -2.516 |      1.312 | 0.000 |    0.000 |  0.000 | {"lgbm": 2, "mlp": 1, "logreg": 3}                   |
| vidya_14         | adaptive_ma          | S2_regime  |        -3.828 |   -4.574 |     -0.746 | 0.000 |    0.000 |  0.000 | {"rf": 1, "logreg": 3, "lgbm": 2}                    |
| vidya_14         | adaptive_ma          | S3_sizing  |        -3.828 |   -4.135 |     -0.307 | 0.160 |    1.000 |  0.228 | {"lgbm": 1, "logreg": 5}                             |
| vidya_14         | adaptive_ma          | S4_exit    |        -3.828 |   -3.842 |     -0.014 | 0.965 |    1.000 |  0.974 | {"logreg": 6}                                        |
| vidya_14         | adaptive_ma          | S5_params  |        -3.828 |   -2.400 |      1.428 | 0.000 |    0.000 |  0.000 | {"et": 4, "lgbm": 1, "logreg": 1}                    |
| t3_20            | adaptive_ma          | S1_filter  |        -3.585 |   -2.174 |      1.412 | 0.000 |    0.000 |  0.000 | {"lgbm": 3, "logreg": 1, "et": 1, "rf": 1}           |
| t3_20            | adaptive_ma          | S2_regime  |        -3.585 |   -4.328 |     -0.743 | 0.000 |    0.000 |  0.000 | {"lgbm": 4, "logreg": 2}                             |
| t3_20            | adaptive_ma          | S3_sizing  |        -3.585 |   -4.256 |     -0.671 | 0.001 |    0.056 |  0.001 | {"lgbm": 2, "logreg": 3, "mlp": 1}                   |
| t3_20            | adaptive_ma          | S4_exit    |        -3.585 |   -3.901 |     -0.316 | 0.179 |    1.000 |  0.250 | {"logreg": 3, "lgbm": 2, "rf": 1}                    |
| t3_20            | adaptive_ma          | S5_params  |        -3.585 |   -2.571 |      1.014 | 0.003 |    0.253 |  0.005 | {"lgbm": 1, "logreg": 1, "rf": 2, "et": 2}           |
| frama_16         | adaptive_ma          | S1_filter  |        -9.044 |   -5.320 |      3.724 | 0.000 |    0.000 |  0.000 | {"lgbm": 6}                                          |
| frama_16         | adaptive_ma          | S2_regime  |        -9.044 |   -7.494 |      1.549 | 0.000 |    0.000 |  0.000 | {"et": 4, "lgbm": 1, "logreg": 1}                    |
| frama_16         | adaptive_ma          | S3_sizing  |        -9.044 |   -9.250 |     -0.206 | 0.379 |    1.000 |  0.479 | {"logreg": 5, "lgbm": 1}                             |
| frama_16         | adaptive_ma          | S4_exit    |        -9.044 |   -8.837 |      0.206 | 0.463 |    1.000 |  0.553 | {"logreg": 2, "lgbm": 1, "et": 1, "rf": 2}           |
| frama_16         | adaptive_ma          | S5_params  |        -9.044 |   -7.283 |      1.760 | 0.000 |    0.000 |  0.000 | {"et": 2, "rf": 2, "lgbm": 2}                        |
| keltner_20       | breakout             | S1_filter  |        -0.431 |    0.680 |      1.111 | 0.000 |    0.000 |  0.000 | {"lgbm": 2, "rf": 1, "et": 1, "logreg": 2}           |
| keltner_20       | breakout             | S2_regime  |        -0.431 |   -1.171 |     -0.741 | 0.000 |    0.000 |  0.000 | {"lgbm": 3, "logreg": 2, "mlp": 1}                   |
| keltner_20       | breakout             | S3_sizing  |        -0.431 |   -0.242 |      0.189 | 0.394 |    1.000 |  0.496 | {"lgbm": 2, "et": 2, "logreg": 1, "rf": 1}           |
| keltner_20       | breakout             | S4_exit    |        -0.431 |   -0.025 |      0.406 | 0.101 |    1.000 |  0.151 | {"rf": 1, "et": 1, "mlp": 2, "lgbm": 1, "logreg": 1} |
| keltner_20       | breakout             | S5_params  |        -0.431 |   -0.840 |     -0.409 | 0.031 |    1.000 |  0.052 | {"lgbm": 1, "rf": 1, "logreg": 2, "mlp": 2}          |
| bollinger_20_2   | breakout             | S1_filter  |        -1.255 |   -0.090 |      1.165 | 0.000 |    0.000 |  0.000 | {"lgbm": 1, "et": 3, "mlp": 1, "rf": 1}              |
| bollinger_20_2   | breakout             | S2_regime  |        -1.255 |   -1.947 |     -0.693 | 0.000 |    0.000 |  0.000 | {"lgbm": 4, "logreg": 2}                             |
| bollinger_20_2   | breakout             | S3_sizing  |        -1.255 |   -0.783 |      0.472 | 0.018 |    1.000 |  0.031 | {"lgbm": 2, "rf": 2, "et": 2}                        |
| bollinger_20_2   | breakout             | S4_exit    |        -1.255 |   -0.593 |      0.662 | 0.005 |    0.470 |  0.010 | {"rf": 1, "lgbm": 5}                                 |
| bollinger_20_2   | breakout             | S5_params  |        -1.255 |   -0.523 |      0.732 | 0.004 |    0.384 |  0.008 | {"rf": 2, "logreg": 1, "et": 3}                      |
| abs_mom_12m      | time_series_momentum | S1_filter  |         0.061 |    0.061 |      0.000 | 1.000 |    1.000 |  1.000 | {"logreg": 6}                                        |
| abs_mom_12m      | time_series_momentum | S2_regime  |         0.061 |   -0.603 |     -0.664 | 0.000 |    0.000 |  0.000 | {"lgbm": 1, "mlp": 2, "logreg": 3}                   |
| abs_mom_12m      | time_series_momentum | S3_sizing  |         0.061 |    0.107 |      0.046 | 0.774 |    1.000 |  0.825 | {"logreg": 5, "lgbm": 1}                             |
| abs_mom_12m      | time_series_momentum | S4_exit    |         0.061 |    0.842 |      0.781 | 0.051 |    1.000 |  0.080 | {"mlp": 2, "et": 2, "lgbm": 1, "logreg": 1}          |
| abs_mom_12m      | time_series_momentum | S5_params  |         0.061 |   -0.128 |     -0.189 | 0.277 |    1.000 |  0.358 | {"rf": 1, "lgbm": 2, "logreg": 2, "mlp": 1}          |
| supertrend_10_3  | volatility_stop      | S1_filter  |        -0.647 |   -0.251 |      0.396 | 0.043 |    1.000 |  0.068 | {"lgbm": 1, "et": 1, "mlp": 2, "logreg": 1, "rf": 1} |
| supertrend_10_3  | volatility_stop      | S2_regime  |        -0.647 |   -1.551 |     -0.904 | 0.000 |    0.000 |  0.000 | {"lgbm": 2, "logreg": 3, "et": 1}                    |
| supertrend_10_3  | volatility_stop      | S3_sizing  |        -0.647 |   -1.302 |     -0.655 | 0.003 |    0.297 |  0.006 | {"lgbm": 1, "et": 1, "logreg": 4}                    |
| supertrend_10_3  | volatility_stop      | S4_exit    |        -0.647 |   -0.702 |     -0.055 | 0.866 |    1.000 |  0.903 | {"mlp": 1, "et": 1, "lgbm": 1, "logreg": 2, "rf": 1} |
| supertrend_10_3  | volatility_stop      | S5_params  |        -0.647 |   -0.674 |     -0.027 | 0.425 |    1.000 |  0.518 | {"logreg": 3, "lgbm": 2, "mlp": 1}                   |
| tsmom_252        | time_series_momentum | S1_filter  |        -1.116 |   -1.407 |     -0.291 | 0.169 |    1.000 |  0.239 | {"lgbm": 2, "rf": 2, "mlp": 1, "et": 1}              |
| tsmom_252        | time_series_momentum | S2_regime  |        -1.116 |   -1.929 |     -0.813 | 0.000 |    0.000 |  0.000 | {"rf": 1, "mlp": 1, "logreg": 3, "lgbm": 1}          |
| tsmom_252        | time_series_momentum | S3_sizing  |        -1.116 |   -1.921 |     -0.805 | 0.000 |    0.000 |  0.000 | {"lgbm": 1, "mlp": 1, "logreg": 4}                   |
| tsmom_252        | time_series_momentum | S4_exit    |        -1.116 |   -1.610 |     -0.494 | 0.174 |    1.000 |  0.244 | {"mlp": 1, "et": 4, "logreg": 1}                     |
| tsmom_252        | time_series_momentum | S5_params  |        -1.116 |   -1.358 |     -0.242 | 0.213 |    1.000 |  0.285 | {"lgbm": 2, "logreg": 1, "rf": 1, "et": 2}           |
| chande_kroll     | volatility_stop      | S1_filter  |        -1.580 |   -0.726 |      0.854 | 0.000 |    0.000 |  0.000 | {"lgbm": 1, "mlp": 1, "rf": 2, "et": 2}              |
| chande_kroll     | volatility_stop      | S2_regime  |        -1.580 |   -2.475 |     -0.895 | 0.000 |    0.000 |  0.000 | {"lgbm": 4, "logreg": 2}                             |
| chande_kroll     | volatility_stop      | S3_sizing  |        -1.580 |   -1.979 |     -0.398 | 0.048 |    1.000 |  0.075 | {"lgbm": 4, "rf": 1, "logreg": 1}                    |
| chande_kroll     | volatility_stop      | S4_exit    |        -1.580 |   -1.082 |      0.498 | 0.077 |    1.000 |  0.119 | {"lgbm": 4, "mlp": 1, "et": 1}                       |
| chande_kroll     | volatility_stop      | S5_params  |        -1.580 |   -0.307 |      1.273 | 0.001 |    0.108 |  0.002 | {"logreg": 2, "lgbm": 1, "et": 3}                    |
| psar             | volatility_stop      | S1_filter  |        -3.582 |   -1.838 |      1.744 | 0.000 |    0.000 |  0.000 | {"lgbm": 2, "logreg": 3, "mlp": 1}                   |
| psar             | volatility_stop      | S2_regime  |        -3.582 |   -3.995 |     -0.412 | 0.033 |    1.000 |  0.054 | {"lgbm": 2, "rf": 1, "logreg": 2, "et": 1}           |
| psar             | volatility_stop      | S3_sizing  |        -3.582 |   -4.595 |     -1.013 | 0.000 |    0.000 |  0.000 | {"lgbm": 3, "mlp": 2, "logreg": 1}                   |
| psar             | volatility_stop      | S4_exit    |        -3.582 |   -3.554 |      0.028 | 0.917 |    1.000 |  0.946 | {"mlp": 3, "et": 1, "lgbm": 2}                       |
| psar             | volatility_stop      | S5_params  |        -3.582 |   -1.544 |      2.038 | 0.000 |    0.000 |  0.000 | {"rf": 5, "logreg": 1}                               |
| chandelier_22_3  | volatility_stop      | S1_filter  |        -2.481 |   -1.473 |      1.009 | 0.000 |    0.000 |  0.000 | {"logreg": 2, "et": 2, "lgbm": 1, "mlp": 1}          |
| chandelier_22_3  | volatility_stop      | S2_regime  |        -2.481 |   -2.795 |     -0.313 | 0.111 |    1.000 |  0.164 | {"rf": 1, "logreg": 3, "lgbm": 2}                    |
| chandelier_22_3  | volatility_stop      | S3_sizing  |        -2.481 |   -3.526 |     -1.044 | 0.000 |    0.000 |  0.000 | {"lgbm": 2, "logreg": 2, "rf": 1, "et": 1}           |
| chandelier_22_3  | volatility_stop      | S4_exit    |        -2.481 |   -2.590 |     -0.108 | 0.720 |    1.000 |  0.779 | {"logreg": 3, "rf": 3}                               |
| chandelier_22_3  | volatility_stop      | S5_params  |        -2.481 |   -2.036 |      0.446 | 0.000 |    0.000 |  0.000 | {"mlp": 1, "lgbm": 5}                                |
| tsmom_multi      | time_series_momentum | S1_filter  |        -2.546 |   -1.832 |      0.714 | 0.001 |    0.108 |  0.002 | {"logreg": 3, "mlp": 2, "rf": 1}                     |
| tsmom_multi      | time_series_momentum | S2_regime  |        -2.546 |   -3.454 |     -0.908 | 0.000 |    0.000 |  0.000 | {"lgbm": 2, "mlp": 2, "logreg": 2}                   |
| tsmom_multi      | time_series_momentum | S3_sizing  |        -2.546 |   -2.699 |     -0.153 | 0.511 |    1.000 |  0.600 | {"lgbm": 2, "logreg": 4}                             |
| tsmom_multi      | time_series_momentum | S4_exit    |        -2.546 |   -3.487 |     -0.941 | 0.004 |    0.427 |  0.009 | {"et": 3, "lgbm": 2, "logreg": 1}                    |
| tsmom_multi      | time_series_momentum | S5_params  |        -2.546 |   -2.018 |      0.528 | 0.030 |    1.000 |  0.050 | {"et": 2, "rf": 1, "lgbm": 2, "logreg": 1}           |
| linreg_63        | time_series_momentum | S1_filter  |        -0.292 |    0.160 |      0.453 | 0.017 |    1.000 |  0.030 | {"lgbm": 3, "mlp": 1, "logreg": 2}                   |
| linreg_63        | time_series_momentum | S2_regime  |        -0.292 |   -1.119 |     -0.827 | 0.000 |    0.000 |  0.000 | {"logreg": 4, "mlp": 1, "lgbm": 1}                   |
| linreg_63        | time_series_momentum | S3_sizing  |        -0.292 |   -1.639 |     -1.347 | 0.000 |    0.000 |  0.000 | {"mlp": 2, "lgbm": 3, "logreg": 1}                   |
| linreg_63        | time_series_momentum | S4_exit    |        -0.292 |   -0.517 |     -0.225 | 0.444 |    1.000 |  0.538 | {"lgbm": 2, "mlp": 2, "et": 2}                       |
| linreg_63        | time_series_momentum | S5_params  |        -0.292 |   -0.356 |     -0.064 | 0.848 |    1.000 |  0.894 | {"logreg": 1, "mlp": 1, "et": 2, "rf": 2}            |
| macd_12_26_9     | oscillator_trend     | S1_filter  |        -3.041 |   -1.990 |      1.051 | 0.000 |    0.000 |  0.000 | {"logreg": 1, "mlp": 1, "lgbm": 2, "rf": 2}          |
| macd_12_26_9     | oscillator_trend     | S2_regime  |        -3.041 |   -3.701 |     -0.660 | 0.001 |    0.056 |  0.001 | {"lgbm": 1, "logreg": 4, "et": 1}                    |
| macd_12_26_9     | oscillator_trend     | S3_sizing  |        -3.041 |   -3.805 |     -0.764 | 0.000 |    0.000 |  0.000 | {"lgbm": 5, "mlp": 1}                                |
| macd_12_26_9     | oscillator_trend     | S4_exit    |        -3.041 |   -3.149 |     -0.108 | 0.593 |    1.000 |  0.668 | {"lgbm": 2, "logreg": 1, "mlp": 3}                   |
| macd_12_26_9     | oscillator_trend     | S5_params  |        -3.041 |   -1.408 |      1.633 | 0.000 |    0.000 |  0.000 | {"et": 4, "logreg": 1, "lgbm": 1}                    |
| adx_dmi_14       | directional          | S1_filter  |        -1.730 |   -0.655 |      1.075 | 0.000 |    0.000 |  0.000 | {"et": 1, "logreg": 2, "rf": 2, "mlp": 1}            |
| adx_dmi_14       | directional          | S2_regime  |        -1.730 |   -2.663 |     -0.933 | 0.000 |    0.000 |  0.000 | {"rf": 1, "lgbm": 2, "mlp": 2, "logreg": 1}          |
| adx_dmi_14       | directional          | S3_sizing  |        -1.730 |   -1.992 |     -0.262 | 0.274 |    1.000 |  0.356 | {"et": 2, "lgbm": 1, "logreg": 3}                    |
| adx_dmi_14       | directional          | S4_exit    |        -1.730 |   -1.554 |      0.176 | 0.557 |    1.000 |  0.632 | {"rf": 2, "lgbm": 1, "logreg": 3}                    |
| adx_dmi_14       | directional          | S5_params  |        -1.730 |   -0.440 |      1.290 | 0.000 |    0.000 |  0.000 | {"rf": 1, "lgbm": 1, "et": 4}                        |
| trix_15          | oscillator_trend     | S1_filter  |        -1.562 |   -0.516 |      1.046 | 0.000 |    0.000 |  0.000 | {"logreg": 4, "lgbm": 2}                             |
| trix_15          | oscillator_trend     | S2_regime  |        -1.562 |   -2.348 |     -0.786 | 0.000 |    0.000 |  0.000 | {"mlp": 1, "logreg": 2, "lgbm": 3}                   |
| trix_15          | oscillator_trend     | S3_sizing  |        -1.562 |   -2.376 |     -0.814 | 0.000 |    0.000 |  0.000 | {"lgbm": 3, "rf": 1, "logreg": 2}                    |
| trix_15          | oscillator_trend     | S4_exit    |        -1.562 |   -1.402 |      0.159 | 0.526 |    1.000 |  0.611 | {"lgbm": 1, "rf": 2, "mlp": 1, "logreg": 1, "et": 1} |
| trix_15          | oscillator_trend     | S5_params  |        -1.562 |   -1.252 |      0.309 | 0.557 |    1.000 |  0.632 | {"et": 3, "logreg": 2, "mlp": 1}                     |
| kst              | oscillator_trend     | S1_filter  |        -1.970 |   -1.141 |      0.830 | 0.000 |    0.000 |  0.000 | {"mlp": 2, "logreg": 2, "rf": 1, "lgbm": 1}          |
| kst              | oscillator_trend     | S2_regime  |        -1.970 |   -2.597 |     -0.626 | 0.000 |    0.000 |  0.000 | {"lgbm": 3, "mlp": 1, "logreg": 2}                   |
| kst              | oscillator_trend     | S3_sizing  |        -1.970 |   -2.637 |     -0.667 | 0.001 |    0.056 |  0.001 | {"lgbm": 5, "rf": 1}                                 |
| kst              | oscillator_trend     | S4_exit    |        -1.970 |   -1.923 |      0.047 | 0.865 |    1.000 |  0.903 | {"lgbm": 4, "logreg": 1, "mlp": 1}                   |
| kst              | oscillator_trend     | S5_params  |        -1.970 |   -1.744 |      0.226 | 0.617 |    1.000 |  0.688 | {"et": 3, "logreg": 1, "mlp": 1, "rf": 1}            |
| tsi_25_13        | oscillator_trend     | S1_filter  |        -3.506 |   -2.228 |      1.278 | 0.000 |    0.000 |  0.000 | {"lgbm": 1, "mlp": 3, "logreg": 1, "rf": 1}          |
| tsi_25_13        | oscillator_trend     | S2_regime  |        -3.506 |   -4.100 |     -0.594 | 0.003 |    0.253 |  0.005 | {"lgbm": 1, "logreg": 3, "rf": 1, "et": 1}           |
| tsi_25_13        | oscillator_trend     | S3_sizing  |        -3.506 |   -4.033 |     -0.527 | 0.003 |    0.297 |  0.006 | {"lgbm": 5, "mlp": 1}                                |
| tsi_25_13        | oscillator_trend     | S4_exit    |        -3.506 |   -3.435 |      0.071 | 0.747 |    1.000 |  0.800 | {"lgbm": 4, "mlp": 2}                                |
| tsi_25_13        | oscillator_trend     | S5_params  |        -3.506 |   -2.644 |      0.862 | 0.005 |    0.511 |  0.010 | {"et": 6}                                            |
| aroon_25         | directional          | S1_filter  |        -1.537 |   -0.362 |      1.175 | 0.000 |    0.000 |  0.000 | {"lgbm": 2, "et": 2, "logreg": 1, "mlp": 1}          |
| aroon_25         | directional          | S2_regime  |        -1.537 |   -2.310 |     -0.774 | 0.000 |    0.000 |  0.000 | {"lgbm": 3, "rf": 1, "logreg": 2}                    |
| aroon_25         | directional          | S3_sizing  |        -1.537 |   -1.796 |     -0.259 | 0.231 |    1.000 |  0.304 | {"lgbm": 1, "et": 1, "logreg": 3, "rf": 1}           |
| aroon_25         | directional          | S4_exit    |        -1.537 |   -1.120 |      0.417 | 0.114 |    1.000 |  0.168 | {"rf": 3, "lgbm": 1, "mlp": 1, "logreg": 1}          |
| aroon_25         | directional          | S5_params  |        -1.537 |   -1.086 |      0.451 | 0.196 |    1.000 |  0.266 | {"rf": 3, "logreg": 2, "lgbm": 1}                    |
| cci_20_100       | oscillator_trend     | S1_filter  |        -0.888 |   -0.139 |      0.748 | 0.001 |    0.056 |  0.001 | {"logreg": 1, "mlp": 1, "et": 1, "lgbm": 2, "rf": 1} |
| cci_20_100       | oscillator_trend     | S2_regime  |        -0.888 |   -1.738 |     -0.850 | 0.000 |    0.000 |  0.000 | {"lgbm": 4, "logreg": 2}                             |
| cci_20_100       | oscillator_trend     | S3_sizing  |        -0.888 |   -1.208 |     -0.320 | 0.127 |    1.000 |  0.185 | {"lgbm": 2, "et": 2, "rf": 1, "logreg": 1}           |
| cci_20_100       | oscillator_trend     | S4_exit    |        -0.888 |   -0.289 |      0.599 | 0.033 |    1.000 |  0.054 | {"lgbm": 1, "rf": 1, "mlp": 1, "logreg": 2, "et": 1} |
| cci_20_100       | oscillator_trend     | S5_params  |        -0.888 |   -0.234 |      0.654 | 0.029 |    1.000 |  0.048 | {"rf": 2, "logreg": 1, "et": 3}                      |
| elder_impulse    | oscillator_trend     | S1_filter  |       -11.628 |   -6.486 |      5.142 | 0.000 |    0.000 |  0.000 | {"lgbm": 5, "mlp": 1}                                |
| elder_impulse    | oscillator_trend     | S2_regime  |       -11.628 |  -10.231 |      1.396 | 0.000 |    0.000 |  0.000 | {"lgbm": 2, "rf": 1, "mlp": 1, "et": 2}              |
| elder_impulse    | oscillator_trend     | S3_sizing  |       -11.628 |  -10.794 |      0.834 | 0.001 |    0.108 |  0.002 | {"et": 1, "logreg": 5}                               |
| elder_impulse    | oscillator_trend     | S4_exit    |       -11.628 |  -11.063 |      0.565 | 0.042 |    1.000 |  0.068 | {"et": 1, "logreg": 2, "rf": 3}                      |
| elder_impulse    | oscillator_trend     | S5_params  |       -11.628 |   -9.198 |      2.430 | 0.000 |    0.000 |  0.000 | {"et": 3, "rf": 3}                                   |
| vortex_14        | directional          | S1_filter  |        -4.003 |   -2.306 |      1.697 | 0.000 |    0.000 |  0.000 | {"logreg": 2, "mlp": 1, "rf": 3}                     |
| vortex_14        | directional          | S2_regime  |        -4.003 |   -4.930 |     -0.927 | 0.000 |    0.000 |  0.000 | {"lgbm": 3, "logreg": 3}                             |
| vortex_14        | directional          | S3_sizing  |        -4.003 |   -4.883 |     -0.880 | 0.000 |    0.000 |  0.000 | {"lgbm": 4, "logreg": 2}                             |
| vortex_14        | directional          | S4_exit    |        -4.003 |   -4.083 |     -0.080 | 0.692 |    1.000 |  0.754 | {"lgbm": 2, "logreg": 1, "mlp": 3}                   |
| vortex_14        | directional          | S5_params  |        -4.003 |   -3.178 |      0.825 | 0.028 |    1.000 |  0.047 | {"lgbm": 1, "logreg": 1, "et": 2, "rf": 2}           |
| ichimoku         | directional          | S1_filter  |        -1.542 |   -0.386 |      1.156 | 0.000 |    0.000 |  0.000 | {"lgbm": 1, "logreg": 3, "et": 1, "rf": 1}           |
| ichimoku         | directional          | S2_regime  |        -1.542 |   -2.335 |     -0.793 | 0.000 |    0.000 |  0.000 | {"lgbm": 3, "logreg": 3}                             |
| ichimoku         | directional          | S3_sizing  |        -1.542 |   -1.918 |     -0.376 | 0.090 |    1.000 |  0.138 | {"lgbm": 2, "logreg": 2, "et": 2}                    |
| ichimoku         | directional          | S4_exit    |        -1.542 |   -1.512 |      0.029 | 0.887 |    1.000 |  0.921 | {"logreg": 3, "rf": 1, "mlp": 2}                     |
| ichimoku         | directional          | S5_params  |        -1.542 |   -0.916 |      0.626 | 0.015 |    1.000 |  0.028 | {"rf": 1, "lgbm": 2, "et": 2, "logreg": 1}           |
| heikin_ashi_3    | directional          | S1_filter  |        -4.707 |   -2.713 |      1.994 | 0.000 |    0.000 |  0.000 | {"lgbm": 4, "logreg": 2}                             |
| heikin_ashi_3    | directional          | S2_regime  |        -4.707 |   -5.303 |     -0.596 | 0.001 |    0.108 |  0.002 | {"lgbm": 4, "logreg": 2}                             |
| heikin_ashi_3    | directional          | S3_sizing  |        -4.707 |   -5.982 |     -1.276 | 0.000 |    0.000 |  0.000 | {"et": 1, "lgbm": 2, "rf": 1, "mlp": 2}              |
| heikin_ashi_3    | directional          | S4_exit    |        -4.707 |   -5.132 |     -0.425 | 0.013 |    1.000 |  0.023 | {"logreg": 2, "lgbm": 1, "rf": 1, "mlp": 2}          |
| heikin_ashi_3    | directional          | S5_params  |        -4.707 |   -1.136 |      3.571 | 0.000 |    0.000 |  0.000 | {"logreg": 2, "rf": 3, "lgbm": 1}                    |
| mcginley_14      | adaptive_ma          | S1_filter  |        -4.400 |   -2.835 |      1.565 | 0.000 |    0.000 |  0.000 | {"mlp": 2, "lgbm": 1, "logreg": 2, "et": 1}          |
| mcginley_14      | adaptive_ma          | S2_regime  |        -4.400 |   -5.217 |     -0.817 | 0.000 |    0.000 |  0.000 | {"lgbm": 4, "rf": 1, "logreg": 1}                    |
| mcginley_14      | adaptive_ma          | S3_sizing  |        -4.400 |   -4.388 |      0.011 | 0.960 |    1.000 |  0.974 | {"et": 1, "lgbm": 1, "logreg": 4}                    |
| mcginley_14      | adaptive_ma          | S4_exit    |        -4.400 |   -4.386 |      0.014 | 0.955 |    1.000 |  0.974 | {"logreg": 4, "lgbm": 1, "et": 1}                    |
| mcginley_14      | adaptive_ma          | S5_params  |        -4.400 |   -2.799 |      1.601 | 0.000 |    0.000 |  0.000 | {"et": 5, "logreg": 1}                               |