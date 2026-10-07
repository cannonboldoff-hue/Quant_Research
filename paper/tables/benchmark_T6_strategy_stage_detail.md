| strategy         | strategy_family      | ml_stage   |   base_sharpe |   sharpe |   d_sharpe |     p |   p_holm |   p_bh | selected_models                                       |
|:-----------------|:---------------------|:-----------|--------------:|---------:|-----------:|------:|---------:|-------:|:------------------------------------------------------|
| sma_50_200       | ma_crossover         | S1_filter  |         0.420 |    0.672 |      0.253 | 0.056 |    1.000 |  0.104 | {"et": 16, "logreg": 3, "rf": 2}                      |
| sma_50_200       | ma_crossover         | S2_regime  |         0.420 |    0.369 |     -0.050 | 0.565 |    1.000 |  0.668 | {"lgbm": 4, "rf": 4, "logreg": 8, "et": 3, "mlp": 2}  |
| sma_50_200       | ma_crossover         | S3_sizing  |         0.420 |    0.645 |      0.225 | 0.069 |    1.000 |  0.118 | {"logreg": 5, "lgbm": 2, "et": 7, "rf": 4, "mlp": 3}  |
| sma_50_200       | ma_crossover         | S4_exit    |         0.420 |    0.708 |      0.289 | 0.106 |    1.000 |  0.169 | {"mlp": 7, "et": 6, "lgbm": 4, "logreg": 1, "rf": 3}  |
| sma_50_200       | ma_crossover         | S5_params  |         0.420 |    0.521 |      0.101 | 0.323 |    1.000 |  0.450 | {"lgbm": 7, "rf": 2, "logreg": 6, "et": 3, "mlp": 3}  |
| dema_20_50       | ma_crossover         | S1_filter  |        -0.071 |    0.467 |      0.538 | 0.012 |    1.000 |  0.029 | {"et": 14, "mlp": 1, "rf": 2, "logreg": 3, "lgbm": 1} |
| dema_20_50       | ma_crossover         | S2_regime  |        -0.071 |   -0.191 |     -0.121 | 0.400 |    1.000 |  0.523 | {"lgbm": 5, "rf": 4, "logreg": 3, "mlp": 3, "et": 6}  |
| dema_20_50       | ma_crossover         | S3_sizing  |        -0.071 |    0.497 |      0.567 | 0.000 |    0.000 |  0.000 | {"lgbm": 2, "et": 9, "mlp": 3, "rf": 3, "logreg": 4}  |
| dema_20_50       | ma_crossover         | S4_exit    |        -0.071 |    0.389 |      0.460 | 0.007 |    0.896 |  0.020 | {"rf": 3, "mlp": 3, "lgbm": 5, "logreg": 3, "et": 7}  |
| dema_20_50       | ma_crossover         | S5_params  |        -0.071 |    0.299 |      0.369 | 0.090 |    1.000 |  0.149 | {"et": 6, "logreg": 7, "lgbm": 4, "rf": 2, "mlp": 2}  |
| triple_ma_4_9_18 | ma_crossover         | S1_filter  |        -0.311 |    0.229 |      0.539 | 0.000 |    0.000 |  0.000 | {"lgbm": 3, "et": 10, "mlp": 3, "rf": 2, "logreg": 3} |
| triple_ma_4_9_18 | ma_crossover         | S2_regime  |        -0.311 |   -0.218 |      0.093 | 0.525 |    1.000 |  0.640 | {"lgbm": 3, "rf": 3, "logreg": 5, "mlp": 4, "et": 6}  |
| triple_ma_4_9_18 | ma_crossover         | S3_sizing  |        -0.311 |    0.274 |      0.585 | 0.000 |    0.000 |  0.000 | {"logreg": 5, "rf": 6, "mlp": 2, "lgbm": 3, "et": 5}  |
| triple_ma_4_9_18 | ma_crossover         | S4_exit    |        -0.311 |    0.083 |      0.393 | 0.010 |    1.000 |  0.025 | {"logreg": 4, "et": 7, "rf": 5, "lgbm": 4, "mlp": 1}  |
| triple_ma_4_9_18 | ma_crossover         | S5_params  |        -0.311 |    0.050 |      0.360 | 0.011 |    1.000 |  0.028 | {"lgbm": 13, "rf": 1, "logreg": 3, "et": 2, "mlp": 2} |
| ema_12_26        | ma_crossover         | S1_filter  |         0.193 |    0.643 |      0.450 | 0.005 |    0.742 |  0.018 | {"rf": 5, "et": 10, "mlp": 3, "logreg": 2, "lgbm": 1} |
| ema_12_26        | ma_crossover         | S2_regime  |         0.193 |    0.072 |     -0.120 | 0.397 |    1.000 |  0.523 | {"lgbm": 3, "rf": 6, "logreg": 6, "mlp": 2, "et": 4}  |
| ema_12_26        | ma_crossover         | S3_sizing  |         0.193 |    0.679 |      0.486 | 0.001 |    0.157 |  0.004 | {"et": 12, "mlp": 1, "lgbm": 3, "rf": 3, "logreg": 2} |
| ema_12_26        | ma_crossover         | S4_exit    |         0.193 |    0.610 |      0.417 | 0.009 |    1.000 |  0.024 | {"et": 8, "lgbm": 5, "rf": 4, "logreg": 2, "mlp": 2}  |
| ema_12_26        | ma_crossover         | S5_params  |         0.193 |    0.306 |      0.114 | 0.325 |    1.000 |  0.450 | {"lgbm": 7, "logreg": 6, "et": 4, "mlp": 4}           |
| zlema_10_30      | ma_crossover         | S1_filter  |        -0.282 |    0.401 |      0.683 | 0.000 |    0.000 |  0.000 | {"et": 11, "mlp": 4, "rf": 3, "lgbm": 3}              |
| zlema_10_30      | ma_crossover         | S2_regime  |        -0.282 |   -0.139 |      0.143 | 0.224 |    1.000 |  0.325 | {"et": 9, "logreg": 6, "mlp": 4, "lgbm": 1, "rf": 1}  |
| zlema_10_30      | ma_crossover         | S3_sizing  |        -0.282 |    0.184 |      0.466 | 0.001 |    0.157 |  0.004 | {"logreg": 2, "et": 9, "lgbm": 6, "rf": 3, "mlp": 1}  |
| zlema_10_30      | ma_crossover         | S4_exit    |        -0.282 |    0.026 |      0.309 | 0.019 |    1.000 |  0.044 | {"logreg": 4, "rf": 7, "mlp": 4, "lgbm": 3, "et": 3}  |
| zlema_10_30      | ma_crossover         | S5_params  |        -0.282 |    0.092 |      0.374 | 0.061 |    1.000 |  0.111 | {"et": 5, "rf": 4, "lgbm": 8, "mlp": 2, "logreg": 2}  |
| jma_7_21         | ma_crossover         | S1_filter  |        -0.318 |    0.370 |      0.687 | 0.000 |    0.000 |  0.000 | {"logreg": 1, "et": 12, "mlp": 3, "rf": 3, "lgbm": 2} |
| jma_7_21         | ma_crossover         | S2_regime  |        -0.318 |   -0.199 |      0.119 | 0.366 |    1.000 |  0.500 | {"rf": 5, "mlp": 2, "logreg": 2, "lgbm": 7, "et": 5}  |
| jma_7_21         | ma_crossover         | S3_sizing  |        -0.318 |    0.120 |      0.438 | 0.001 |    0.084 |  0.003 | {"logreg": 4, "et": 7, "mlp": 1, "lgbm": 5, "rf": 4}  |
| jma_7_21         | ma_crossover         | S4_exit    |        -0.318 |   -0.025 |      0.293 | 0.025 |    1.000 |  0.054 | {"logreg": 5, "mlp": 3, "et": 6, "lgbm": 5, "rf": 2}  |
| jma_7_21         | ma_crossover         | S5_params  |        -0.318 |   -0.041 |      0.277 | 0.139 |    1.000 |  0.213 | {"lgbm": 10, "rf": 5, "et": 6}                        |
| faber_10m        | price_vs_ma          | S1_filter  |         0.849 |    0.832 |     -0.017 | 0.742 |    1.000 |  0.812 | {"mlp": 4, "rf": 6, "et": 4, "lgbm": 5, "logreg": 2}  |
| faber_10m        | price_vs_ma          | S2_regime  |         0.849 |    0.637 |     -0.212 | 0.024 |    1.000 |  0.052 | {"lgbm": 6, "logreg": 7, "et": 3, "rf": 2, "mlp": 3}  |
| faber_10m        | price_vs_ma          | S3_sizing  |         0.849 |    0.872 |      0.023 | 0.738 |    1.000 |  0.812 | {"mlp": 10, "logreg": 1, "lgbm": 4, "rf": 2, "et": 4} |
| faber_10m        | price_vs_ma          | S4_exit    |         0.849 |    0.433 |     -0.416 | 0.009 |    1.000 |  0.025 | {"lgbm": 5, "mlp": 6, "logreg": 2, "et": 6, "rf": 2}  |
| faber_10m        | price_vs_ma          | S5_params  |         0.849 |    0.841 |     -0.008 | 0.777 |    1.000 |  0.842 | {"lgbm": 5, "rf": 3, "et": 3, "logreg": 7, "mlp": 3}  |
| gmma             | ma_crossover         | S1_filter  |         0.229 |    0.473 |      0.244 | 0.099 |    1.000 |  0.161 | {"et": 14, "mlp": 2, "rf": 2, "lgbm": 2, "logreg": 1} |
| gmma             | ma_crossover         | S2_regime  |         0.229 |    0.127 |     -0.102 | 0.507 |    1.000 |  0.626 | {"lgbm": 3, "rf": 6, "logreg": 6, "et": 5, "mlp": 1}  |
| gmma             | ma_crossover         | S3_sizing  |         0.229 |    0.604 |      0.375 | 0.004 |    0.634 |  0.015 | {"et": 14, "rf": 4, "mlp": 1, "logreg": 1, "lgbm": 1} |
| gmma             | ma_crossover         | S4_exit    |         0.229 |    0.446 |      0.217 | 0.163 |    1.000 |  0.244 | {"et": 11, "rf": 4, "lgbm": 3, "logreg": 1, "mlp": 2} |
| gmma             | ma_crossover         | S5_params  |         0.229 |    0.378 |      0.149 | 0.150 |    1.000 |  0.228 | {"lgbm": 11, "logreg": 4, "rf": 1, "et": 2, "mlp": 3} |
| bll_vma_200      | price_vs_ma          | S1_filter  |         0.455 |    0.721 |      0.266 | 0.045 |    1.000 |  0.089 | {"et": 9, "logreg": 2, "rf": 4, "lgbm": 3, "mlp": 3}  |
| bll_vma_200      | price_vs_ma          | S2_regime  |         0.455 |    0.449 |     -0.006 | 0.962 |    1.000 |  0.967 | {"lgbm": 6, "rf": 9, "logreg": 4, "et": 2}            |
| bll_vma_200      | price_vs_ma          | S3_sizing  |         0.455 |    0.761 |      0.306 | 0.022 |    1.000 |  0.051 | {"et": 11, "logreg": 2, "lgbm": 4, "mlp": 1, "rf": 3} |
| bll_vma_200      | price_vs_ma          | S4_exit    |         0.455 |    0.620 |      0.165 | 0.333 |    1.000 |  0.457 | {"mlp": 5, "et": 6, "logreg": 3, "lgbm": 4, "rf": 3}  |
| bll_vma_200      | price_vs_ma          | S5_params  |         0.455 |    0.468 |      0.013 | 0.853 |    1.000 |  0.899 | {"mlp": 5, "et": 7, "lgbm": 3, "rf": 2, "logreg": 4}  |
| kama_10          | adaptive_ma          | S1_filter  |        -0.498 |    0.174 |      0.671 | 0.000 |    0.000 |  0.000 | {"et": 13, "mlp": 1, "rf": 6, "lgbm": 1}              |
| kama_10          | adaptive_ma          | S2_regime  |        -0.498 |   -0.409 |      0.089 | 0.534 |    1.000 |  0.647 | {"lgbm": 4, "rf": 6, "logreg": 3, "et": 5, "mlp": 3}  |
| kama_10          | adaptive_ma          | S3_sizing  |        -0.498 |    0.064 |      0.562 | 0.000 |    0.000 |  0.000 | {"et": 7, "lgbm": 5, "logreg": 5, "rf": 4}            |
| kama_10          | adaptive_ma          | S4_exit    |        -0.498 |   -0.243 |      0.254 | 0.064 |    1.000 |  0.113 | {"logreg": 9, "rf": 3, "lgbm": 2, "mlp": 2, "et": 5}  |
| kama_10          | adaptive_ma          | S5_params  |        -0.498 |   -0.180 |      0.317 | 0.001 |    0.084 |  0.003 | {"et": 4, "logreg": 1, "rf": 6, "mlp": 2, "lgbm": 8}  |
| vidya_14         | adaptive_ma          | S1_filter  |        -0.011 |    0.606 |      0.617 | 0.003 |    0.365 |  0.010 | {"et": 16, "mlp": 2, "rf": 3}                         |
| vidya_14         | adaptive_ma          | S2_regime  |        -0.011 |   -0.017 |     -0.006 | 0.962 |    1.000 |  0.967 | {"lgbm": 4, "rf": 6, "logreg": 6, "et": 5}            |
| vidya_14         | adaptive_ma          | S3_sizing  |        -0.011 |    0.458 |      0.469 | 0.001 |    0.084 |  0.003 | {"et": 15, "lgbm": 1, "logreg": 3, "rf": 1, "mlp": 1} |
| vidya_14         | adaptive_ma          | S4_exit    |        -0.011 |    0.248 |      0.259 | 0.070 |    1.000 |  0.118 | {"et": 13, "lgbm": 2, "mlp": 2, "logreg": 3, "rf": 1} |
| vidya_14         | adaptive_ma          | S5_params  |        -0.011 |    0.185 |      0.196 | 0.045 |    1.000 |  0.089 | {"et": 3, "rf": 8, "lgbm": 6, "mlp": 2, "logreg": 2}  |
| frama_16         | adaptive_ma          | S1_filter  |        -0.825 |   -0.110 |      0.716 | 0.000 |    0.000 |  0.000 | {"et": 14, "mlp": 2, "rf": 3, "lgbm": 1, "logreg": 1} |
| frama_16         | adaptive_ma          | S2_regime  |        -0.825 |   -0.715 |      0.110 | 0.421 |    1.000 |  0.541 | {"lgbm": 6, "et": 7, "logreg": 3, "mlp": 2, "rf": 3}  |
| frama_16         | adaptive_ma          | S3_sizing  |        -0.825 |   -0.396 |      0.429 | 0.001 |    0.084 |  0.003 | {"logreg": 6, "et": 6, "mlp": 1, "lgbm": 6, "rf": 2}  |
| frama_16         | adaptive_ma          | S4_exit    |        -0.825 |   -0.613 |      0.213 | 0.084 |    1.000 |  0.140 | {"logreg": 11, "mlp": 3, "lgbm": 3, "rf": 2, "et": 2} |
| frama_16         | adaptive_ma          | S5_params  |        -0.825 |   -0.624 |      0.201 | 0.063 |    1.000 |  0.113 | {"et": 6, "lgbm": 3, "mlp": 5, "logreg": 2, "rf": 5}  |
| hma_55_slope     | adaptive_ma          | S1_filter  |         0.025 |    0.531 |      0.506 | 0.001 |    0.157 |  0.004 | {"logreg": 4, "et": 12, "rf": 3, "mlp": 1, "lgbm": 1} |
| hma_55_slope     | adaptive_ma          | S2_regime  |         0.025 |    0.039 |      0.014 | 0.918 |    1.000 |  0.937 | {"lgbm": 3, "et": 6, "rf": 3, "logreg": 5, "mlp": 4}  |
| hma_55_slope     | adaptive_ma          | S3_sizing  |         0.025 |    0.531 |      0.506 | 0.000 |    0.000 |  0.000 | {"logreg": 4, "et": 9, "mlp": 2, "lgbm": 2, "rf": 4}  |
| hma_55_slope     | adaptive_ma          | S4_exit    |         0.025 |    0.345 |      0.320 | 0.048 |    1.000 |  0.092 | {"logreg": 1, "et": 6, "mlp": 6, "lgbm": 5, "rf": 3}  |
| hma_55_slope     | adaptive_ma          | S5_params  |         0.025 |    0.279 |      0.254 | 0.124 |    1.000 |  0.194 | {"lgbm": 8, "logreg": 5, "et": 1, "rf": 3, "mlp": 4}  |
| mcginley_14      | adaptive_ma          | S1_filter  |        -0.007 |    0.534 |      0.541 | 0.002 |    0.225 |  0.006 | {"et": 17, "mlp": 1, "rf": 2, "lgbm": 1}              |
| mcginley_14      | adaptive_ma          | S2_regime  |        -0.007 |   -0.026 |     -0.019 | 0.896 |    1.000 |  0.924 | {"lgbm": 3, "rf": 8, "logreg": 2, "et": 4, "mlp": 4}  |
| mcginley_14      | adaptive_ma          | S3_sizing  |        -0.007 |    0.469 |      0.476 | 0.000 |    0.000 |  0.000 | {"et": 10, "lgbm": 2, "logreg": 5, "rf": 4}           |
| mcginley_14      | adaptive_ma          | S4_exit    |        -0.007 |    0.227 |      0.233 | 0.100 |    1.000 |  0.161 | {"rf": 2, "et": 8, "lgbm": 5, "logreg": 5, "mlp": 1}  |
| mcginley_14      | adaptive_ma          | S5_params  |        -0.007 |    0.246 |      0.252 | 0.006 |    0.845 |  0.019 | {"mlp": 1, "lgbm": 6, "rf": 4, "et": 6, "logreg": 4}  |
| t3_20            | adaptive_ma          | S1_filter  |        -0.168 |    0.442 |      0.610 | 0.002 |    0.296 |  0.008 | {"lgbm": 3, "et": 10, "mlp": 1, "rf": 6, "logreg": 1} |
| t3_20            | adaptive_ma          | S2_regime  |        -0.168 |   -0.094 |      0.074 | 0.606 |    1.000 |  0.699 | {"lgbm": 6, "rf": 3, "mlp": 5, "logreg": 3, "et": 4}  |
| t3_20            | adaptive_ma          | S3_sizing  |        -0.168 |    0.349 |      0.517 | 0.001 |    0.084 |  0.003 | {"et": 9, "lgbm": 2, "logreg": 5, "rf": 4, "mlp": 1}  |
| t3_20            | adaptive_ma          | S4_exit    |        -0.168 |    0.075 |      0.244 | 0.148 |    1.000 |  0.226 | {"lgbm": 7, "logreg": 5, "et": 6, "rf": 1, "mlp": 2}  |
| t3_20            | adaptive_ma          | S5_params  |        -0.168 |    0.209 |      0.378 | 0.008 |    1.000 |  0.022 | {"lgbm": 7, "et": 4, "mlp": 4, "rf": 3, "logreg": 3}  |
| donchian_20      | breakout             | S1_filter  |         0.110 |    0.673 |      0.563 | 0.005 |    0.685 |  0.016 | {"et": 11, "logreg": 3, "lgbm": 5, "rf": 2}           |
| donchian_20      | breakout             | S2_regime  |         0.110 |    0.006 |     -0.104 | 0.471 |    1.000 |  0.593 | {"lgbm": 5, "rf": 5, "logreg": 4, "mlp": 2, "et": 5}  |
| donchian_20      | breakout             | S3_sizing  |         0.110 |    0.629 |      0.519 | 0.000 |    0.000 |  0.000 | {"et": 12, "lgbm": 5, "rf": 2, "logreg": 2}           |
| donchian_20      | breakout             | S4_exit    |         0.110 |    0.447 |      0.337 | 0.061 |    1.000 |  0.111 | {"rf": 4, "et": 7, "logreg": 3, "lgbm": 4, "mlp": 3}  |
| donchian_20      | breakout             | S5_params  |         0.110 |    0.072 |     -0.038 | 0.746 |    1.000 |  0.812 | {"mlp": 7, "logreg": 6, "et": 2, "lgbm": 4, "rf": 2}  |
| turtle_20_10     | breakout             | S1_filter  |        -0.019 |    0.541 |      0.560 | 0.001 |    0.157 |  0.004 | {"et": 9, "logreg": 6, "rf": 5, "lgbm": 1}            |
| turtle_20_10     | breakout             | S2_regime  |        -0.019 |   -0.106 |     -0.087 | 0.549 |    1.000 |  0.661 | {"lgbm": 5, "rf": 5, "logreg": 6, "mlp": 1, "et": 4}  |
| turtle_20_10     | breakout             | S3_sizing  |        -0.019 |    0.543 |      0.562 | 0.000 |    0.000 |  0.000 | {"rf": 4, "mlp": 2, "lgbm": 2, "et": 11, "logreg": 2} |
| turtle_20_10     | breakout             | S4_exit    |        -0.019 |    0.384 |      0.403 | 0.006 |    0.804 |  0.018 | {"lgbm": 2, "mlp": 5, "logreg": 4, "et": 8, "rf": 2}  |
| turtle_20_10     | breakout             | S5_params  |        -0.019 |    0.180 |      0.199 | 0.068 |    1.000 |  0.118 | {"logreg": 4, "lgbm": 8, "rf": 4, "mlp": 4, "et": 1}  |
| turtle_55_20     | breakout             | S1_filter  |         0.214 |    0.629 |      0.415 | 0.006 |    0.845 |  0.019 | {"et": 7, "mlp": 3, "lgbm": 5, "rf": 5, "logreg": 1}  |
| turtle_55_20     | breakout             | S2_regime  |         0.214 |    0.150 |     -0.064 | 0.676 |    1.000 |  0.767 | {"lgbm": 5, "rf": 7, "logreg": 3, "mlp": 1, "et": 5}  |
| turtle_55_20     | breakout             | S3_sizing  |         0.214 |    0.604 |      0.390 | 0.004 |    0.634 |  0.015 | {"et": 9, "rf": 4, "lgbm": 5, "mlp": 2, "logreg": 1}  |
| turtle_55_20     | breakout             | S4_exit    |         0.214 |    0.533 |      0.319 | 0.043 |    1.000 |  0.088 | {"et": 9, "mlp": 5, "lgbm": 4, "rf": 2, "logreg": 1}  |
| turtle_55_20     | breakout             | S5_params  |         0.214 |    0.325 |      0.111 | 0.312 |    1.000 |  0.442 | {"mlp": 6, "logreg": 4, "lgbm": 5, "rf": 2, "et": 4}  |
| keltner_20       | breakout             | S1_filter  |         0.035 |    0.651 |      0.617 | 0.000 |    0.000 |  0.000 | {"et": 8, "logreg": 3, "lgbm": 1, "rf": 9}            |
| keltner_20       | breakout             | S2_regime  |         0.035 |    0.072 |      0.037 | 0.819 |    1.000 |  0.873 | {"lgbm": 6, "rf": 6, "logreg": 5, "et": 4}            |
| keltner_20       | breakout             | S3_sizing  |         0.035 |    0.575 |      0.540 | 0.000 |    0.000 |  0.000 | {"lgbm": 5, "et": 4, "mlp": 2, "rf": 5, "logreg": 5}  |
| keltner_20       | breakout             | S4_exit    |         0.035 |    0.425 |      0.390 | 0.006 |    0.804 |  0.018 | {"lgbm": 4, "et": 6, "rf": 2, "logreg": 5, "mlp": 4}  |
| keltner_20       | breakout             | S5_params  |         0.035 |    0.271 |      0.236 | 0.023 |    1.000 |  0.052 | {"mlp": 4, "rf": 2, "lgbm": 5, "logreg": 4, "et": 6}  |
| bollinger_20_2   | breakout             | S1_filter  |        -0.076 |    0.398 |      0.474 | 0.004 |    0.634 |  0.015 | {"logreg": 5, "et": 11, "rf": 4, "mlp": 1}            |
| bollinger_20_2   | breakout             | S2_regime  |        -0.076 |   -0.095 |     -0.019 | 0.900 |    1.000 |  0.924 | {"lgbm": 4, "rf": 5, "logreg": 5, "et": 6, "mlp": 1}  |
| bollinger_20_2   | breakout             | S3_sizing  |        -0.076 |    0.406 |      0.482 | 0.000 |    0.000 |  0.000 | {"rf": 6, "et": 7, "lgbm": 3, "logreg": 5}            |
| bollinger_20_2   | breakout             | S4_exit    |        -0.076 |    0.248 |      0.324 | 0.019 |    1.000 |  0.045 | {"lgbm": 7, "rf": 6, "logreg": 3, "et": 3, "mlp": 2}  |
| bollinger_20_2   | breakout             | S5_params  |        -0.076 |    0.153 |      0.229 | 0.051 |    1.000 |  0.095 | {"lgbm": 9, "rf": 2, "et": 4, "mlp": 4, "logreg": 2}  |
| supertrend_10_3  | volatility_stop      | S1_filter  |         0.094 |    0.549 |      0.455 | 0.030 |    1.000 |  0.064 | {"et": 13, "lgbm": 1, "logreg": 2, "mlp": 3, "rf": 2} |
| supertrend_10_3  | volatility_stop      | S2_regime  |         0.094 |    0.017 |     -0.077 | 0.581 |    1.000 |  0.678 | {"rf": 6, "logreg": 4, "et": 5, "lgbm": 4, "mlp": 2}  |
| supertrend_10_3  | volatility_stop      | S3_sizing  |         0.094 |    0.660 |      0.566 | 0.000 |    0.000 |  0.000 | {"logreg": 4, "et": 11, "lgbm": 3, "rf": 1, "mlp": 2} |
| supertrend_10_3  | volatility_stop      | S4_exit    |         0.094 |    0.530 |      0.436 | 0.004 |    0.501 |  0.013 | {"logreg": 2, "et": 10, "mlp": 2, "lgbm": 5, "rf": 2} |
| supertrend_10_3  | volatility_stop      | S5_params  |         0.094 |    0.085 |     -0.009 | 0.570 |    1.000 |  0.670 | {"logreg": 3, "lgbm": 5, "rf": 5, "mlp": 6, "et": 2}  |
| psar             | volatility_stop      | S1_filter  |        -0.268 |    0.508 |      0.775 | 0.000 |    0.000 |  0.000 | {"et": 10, "rf": 4, "mlp": 3, "lgbm": 3, "logreg": 1} |
| psar             | volatility_stop      | S2_regime  |        -0.268 |   -0.234 |      0.033 | 0.816 |    1.000 |  0.873 | {"et": 8, "lgbm": 4, "rf": 4, "mlp": 3, "logreg": 2}  |
| psar             | volatility_stop      | S3_sizing  |        -0.268 |    0.152 |      0.419 | 0.004 |    0.501 |  0.013 | {"et": 9, "lgbm": 6, "logreg": 2, "rf": 3, "mlp": 1}  |
| psar             | volatility_stop      | S4_exit    |        -0.268 |    0.008 |      0.275 | 0.027 |    1.000 |  0.058 | {"logreg": 5, "et": 8, "rf": 5, "lgbm": 2, "mlp": 1}  |
| psar             | volatility_stop      | S5_params  |        -0.268 |   -0.047 |      0.221 | 0.069 |    1.000 |  0.118 | {"lgbm": 7, "rf": 4, "logreg": 3, "et": 5, "mlp": 2}  |
| chandelier_22_3  | volatility_stop      | S1_filter  |        -0.117 |    0.222 |      0.339 | 0.000 |    0.000 |  0.000 | {"mlp": 7, "et": 4, "rf": 1, "lgbm": 3, "logreg": 6}  |
| chandelier_22_3  | volatility_stop      | S2_regime  |        -0.117 |   -0.031 |      0.086 | 0.556 |    1.000 |  0.665 | {"lgbm": 4, "rf": 6, "logreg": 4, "et": 4, "mlp": 3}  |
| chandelier_22_3  | volatility_stop      | S3_sizing  |        -0.117 |    0.484 |      0.601 | 0.000 |    0.000 |  0.000 | {"et": 12, "lgbm": 3, "rf": 3, "mlp": 1, "logreg": 2} |
| chandelier_22_3  | volatility_stop      | S4_exit    |        -0.117 |    0.126 |      0.243 | 0.070 |    1.000 |  0.118 | {"lgbm": 6, "mlp": 2, "logreg": 3, "rf": 4, "et": 6}  |
| chandelier_22_3  | volatility_stop      | S5_params  |        -0.117 |   -0.142 |     -0.025 | 0.218 |    1.000 |  0.320 | {"lgbm": 4, "rf": 6, "mlp": 5, "et": 4, "logreg": 2}  |
| chande_kroll     | volatility_stop      | S1_filter  |         0.094 |    0.524 |      0.431 | 0.009 |    1.000 |  0.025 | {"et": 12, "mlp": 2, "lgbm": 4, "rf": 1, "logreg": 2} |
| chande_kroll     | volatility_stop      | S2_regime  |         0.094 |    0.146 |      0.052 | 0.723 |    1.000 |  0.806 | {"lgbm": 2, "rf": 4, "logreg": 5, "mlp": 5, "et": 5}  |
| chande_kroll     | volatility_stop      | S3_sizing  |         0.094 |    0.630 |      0.536 | 0.000 |    0.000 |  0.000 | {"et": 11, "lgbm": 3, "logreg": 4, "mlp": 1, "rf": 2} |
| chande_kroll     | volatility_stop      | S4_exit    |         0.094 |    0.390 |      0.296 | 0.041 |    1.000 |  0.083 | {"logreg": 1, "lgbm": 7, "et": 9, "rf": 4}            |
| chande_kroll     | volatility_stop      | S5_params  |         0.094 |    0.252 |      0.158 | 0.370 |    1.000 |  0.502 | {"mlp": 8, "lgbm": 5, "et": 2, "logreg": 6}           |
| tsmom_252        | time_series_momentum | S1_filter  |         0.488 |    0.562 |      0.074 | 0.559 |    1.000 |  0.665 | {"mlp": 2, "et": 12, "lgbm": 5, "rf": 1, "logreg": 1} |
| tsmom_252        | time_series_momentum | S2_regime  |         0.488 |    0.487 |     -0.000 | 1.000 |    1.000 |  1.000 | {"lgbm": 4, "logreg": 8, "mlp": 4, "rf": 4, "et": 1}  |
| tsmom_252        | time_series_momentum | S3_sizing  |         0.488 |    0.707 |      0.220 | 0.075 |    1.000 |  0.126 | {"mlp": 5, "logreg": 3, "rf": 5, "lgbm": 1, "et": 7}  |
| tsmom_252        | time_series_momentum | S4_exit    |         0.488 |    0.611 |      0.123 | 0.483 |    1.000 |  0.604 | {"mlp": 5, "et": 5, "logreg": 4, "rf": 5, "lgbm": 2}  |
| tsmom_252        | time_series_momentum | S5_params  |         0.488 |    0.400 |     -0.088 | 0.440 |    1.000 |  0.557 | {"lgbm": 8, "logreg": 6, "et": 3, "mlp": 3, "rf": 1}  |
| abs_mom_12m      | time_series_momentum | S1_filter  |         0.841 |    0.849 |      0.008 | 0.878 |    1.000 |  0.913 | {"mlp": 3, "rf": 7, "logreg": 4, "lgbm": 6, "et": 1}  |
| abs_mom_12m      | time_series_momentum | S2_regime  |         0.841 |    0.645 |     -0.196 | 0.038 |    1.000 |  0.079 | {"mlp": 5, "logreg": 7, "et": 2, "rf": 2, "lgbm": 5}  |
| abs_mom_12m      | time_series_momentum | S3_sizing  |         0.841 |    0.836 |     -0.006 | 0.932 |    1.000 |  0.947 | {"lgbm": 2, "mlp": 10, "logreg": 2, "rf": 3, "et": 4} |
| abs_mom_12m      | time_series_momentum | S4_exit    |         0.841 |    0.556 |     -0.285 | 0.061 |    1.000 |  0.111 | {"lgbm": 2, "logreg": 12, "mlp": 4, "et": 3}          |
| abs_mom_12m      | time_series_momentum | S5_params  |         0.841 |    0.724 |     -0.118 | 0.046 |    1.000 |  0.090 | {"lgbm": 9, "mlp": 3, "rf": 3, "logreg": 1, "et": 5}  |
| tsmom_multi      | time_series_momentum | S1_filter  |         0.348 |    0.695 |      0.348 | 0.013 |    1.000 |  0.030 | {"rf": 3, "et": 14, "logreg": 2, "mlp": 1, "lgbm": 1} |
| tsmom_multi      | time_series_momentum | S2_regime  |         0.348 |    0.319 |     -0.029 | 0.848 |    1.000 |  0.899 | {"lgbm": 3, "rf": 8, "logreg": 5, "et": 3, "mlp": 2}  |
| tsmom_multi      | time_series_momentum | S3_sizing  |         0.348 |    0.605 |      0.257 | 0.049 |    1.000 |  0.093 | {"lgbm": 3, "et": 7, "mlp": 2, "logreg": 2, "rf": 7}  |
| tsmom_multi      | time_series_momentum | S4_exit    |         0.348 |    0.639 |      0.291 | 0.050 |    1.000 |  0.095 | {"lgbm": 4, "et": 9, "mlp": 3, "logreg": 3, "rf": 2}  |
| tsmom_multi      | time_series_momentum | S5_params  |         0.348 |    0.468 |      0.120 | 0.180 |    1.000 |  0.267 | {"logreg": 5, "lgbm": 2, "et": 5, "rf": 3, "mlp": 6}  |
| linreg_63        | time_series_momentum | S1_filter  |         0.251 |    0.544 |      0.293 | 0.037 |    1.000 |  0.078 | {"et": 9, "mlp": 1, "logreg": 6, "rf": 3, "lgbm": 2}  |
| linreg_63        | time_series_momentum | S2_regime  |         0.251 |    0.144 |     -0.108 | 0.389 |    1.000 |  0.519 | {"rf": 9, "logreg": 4, "et": 2, "mlp": 2, "lgbm": 4}  |
| linreg_63        | time_series_momentum | S3_sizing  |         0.251 |    0.613 |      0.362 | 0.006 |    0.804 |  0.018 | {"et": 9, "rf": 3, "logreg": 2, "lgbm": 3, "mlp": 4}  |
| linreg_63        | time_series_momentum | S4_exit    |         0.251 |    0.475 |      0.224 | 0.231 |    1.000 |  0.334 | {"mlp": 6, "lgbm": 5, "logreg": 1, "et": 6, "rf": 3}  |
| linreg_63        | time_series_momentum | S5_params  |         0.251 |    0.511 |      0.260 | 0.114 |    1.000 |  0.181 | {"et": 5, "lgbm": 7, "rf": 4, "logreg": 4, "mlp": 1}  |
| trix_15          | oscillator_trend     | S1_filter  |        -0.107 |    0.507 |      0.615 | 0.001 |    0.084 |  0.003 | {"logreg": 4, "rf": 7, "et": 9, "lgbm": 1}            |
| trix_15          | oscillator_trend     | S2_regime  |        -0.107 |   -0.089 |      0.019 | 0.880 |    1.000 |  0.913 | {"et": 7, "rf": 3, "mlp": 6, "lgbm": 2, "logreg": 3}  |
| trix_15          | oscillator_trend     | S3_sizing  |        -0.107 |    0.440 |      0.547 | 0.000 |    0.000 |  0.000 | {"lgbm": 4, "et": 7, "mlp": 3, "logreg": 2, "rf": 5}  |
| trix_15          | oscillator_trend     | S4_exit    |        -0.107 |    0.320 |      0.427 | 0.005 |    0.685 |  0.016 | {"lgbm": 7, "et": 6, "rf": 4, "logreg": 2, "mlp": 2}  |
| trix_15          | oscillator_trend     | S5_params  |        -0.107 |    0.093 |      0.200 | 0.380 |    1.000 |  0.511 | {"mlp": 8, "logreg": 8, "lgbm": 3, "et": 2}           |
| macd_12_26_9     | oscillator_trend     | S1_filter  |        -0.109 |    0.541 |      0.650 | 0.001 |    0.084 |  0.003 | {"logreg": 1, "et": 8, "mlp": 5, "lgbm": 3, "rf": 4}  |
| macd_12_26_9     | oscillator_trend     | S2_regime  |        -0.109 |   -0.005 |      0.103 | 0.431 |    1.000 |  0.550 | {"et": 8, "rf": 3, "mlp": 5, "lgbm": 2, "logreg": 3}  |
| macd_12_26_9     | oscillator_trend     | S3_sizing  |        -0.109 |    0.372 |      0.481 | 0.001 |    0.084 |  0.003 | {"logreg": 3, "et": 6, "rf": 5, "lgbm": 3, "mlp": 4}  |
| macd_12_26_9     | oscillator_trend     | S4_exit    |        -0.109 |    0.337 |      0.446 | 0.000 |    0.000 |  0.000 | {"lgbm": 6, "logreg": 3, "rf": 7, "mlp": 3, "et": 2}  |
| macd_12_26_9     | oscillator_trend     | S5_params  |        -0.109 |    0.040 |      0.148 | 0.416 |    1.000 |  0.538 | {"lgbm": 7, "logreg": 4, "et": 5, "mlp": 2, "rf": 3}  |
| kst              | oscillator_trend     | S1_filter  |        -0.091 |    0.370 |      0.461 | 0.010 |    1.000 |  0.025 | {"et": 10, "mlp": 2, "rf": 3, "lgbm": 5, "logreg": 1} |
| kst              | oscillator_trend     | S2_regime  |        -0.091 |   -0.149 |     -0.058 | 0.672 |    1.000 |  0.766 | {"et": 10, "mlp": 3, "lgbm": 3, "rf": 2, "logreg": 3} |
| kst              | oscillator_trend     | S3_sizing  |        -0.091 |    0.460 |      0.551 | 0.000 |    0.000 |  0.000 | {"logreg": 3, "et": 9, "lgbm": 4, "mlp": 2, "rf": 3}  |
| kst              | oscillator_trend     | S4_exit    |        -0.091 |    0.461 |      0.552 | 0.001 |    0.157 |  0.004 | {"mlp": 4, "et": 7, "lgbm": 2, "rf": 6, "logreg": 2}  |
| kst              | oscillator_trend     | S5_params  |        -0.091 |    0.090 |      0.181 | 0.397 |    1.000 |  0.523 | {"logreg": 3, "et": 7, "mlp": 5, "rf": 4, "lgbm": 2}  |
| cci_20_100       | oscillator_trend     | S1_filter  |         0.050 |    0.534 |      0.484 | 0.018 |    1.000 |  0.042 | {"rf": 3, "et": 13, "mlp": 1, "logreg": 4}            |
| cci_20_100       | oscillator_trend     | S2_regime  |         0.050 |   -0.045 |     -0.095 | 0.521 |    1.000 |  0.640 | {"lgbm": 3, "rf": 5, "logreg": 6, "mlp": 3, "et": 4}  |
| cci_20_100       | oscillator_trend     | S3_sizing  |         0.050 |    0.543 |      0.494 | 0.001 |    0.084 |  0.003 | {"et": 11, "mlp": 4, "lgbm": 1, "rf": 3, "logreg": 2} |
| cci_20_100       | oscillator_trend     | S4_exit    |         0.050 |    0.401 |      0.351 | 0.035 |    1.000 |  0.075 | {"logreg": 1, "et": 5, "mlp": 2, "lgbm": 9, "rf": 4}  |
| cci_20_100       | oscillator_trend     | S5_params  |         0.050 |    0.194 |      0.144 | 0.326 |    1.000 |  0.450 | {"lgbm": 8, "mlp": 6, "rf": 3, "et": 3, "logreg": 1}  |
| tsi_25_13        | oscillator_trend     | S1_filter  |        -0.157 |    0.340 |      0.498 | 0.008 |    1.000 |  0.022 | {"logreg": 3, "lgbm": 1, "et": 7, "mlp": 4, "rf": 6}  |
| tsi_25_13        | oscillator_trend     | S2_regime  |        -0.157 |   -0.117 |      0.040 | 0.745 |    1.000 |  0.812 | {"et": 7, "logreg": 5, "mlp": 5, "lgbm": 2, "rf": 2}  |
| tsi_25_13        | oscillator_trend     | S3_sizing  |        -0.157 |    0.300 |      0.457 | 0.001 |    0.084 |  0.003 | {"logreg": 3, "et": 5, "lgbm": 3, "rf": 7, "mlp": 3}  |
| tsi_25_13        | oscillator_trend     | S4_exit    |        -0.157 |    0.275 |      0.433 | 0.003 |    0.435 |  0.011 | {"logreg": 4, "et": 6, "lgbm": 4, "rf": 3, "mlp": 4}  |
| tsi_25_13        | oscillator_trend     | S5_params  |        -0.157 |   -0.067 |      0.090 | 0.506 |    1.000 |  0.626 | {"mlp": 5, "rf": 4, "lgbm": 3, "et": 7, "logreg": 2}  |
| adx_dmi_14       | directional          | S1_filter  |        -0.193 |    0.368 |      0.561 | 0.000 |    0.000 |  0.000 | {"lgbm": 4, "rf": 4, "et": 9, "logreg": 3, "mlp": 1}  |
| adx_dmi_14       | directional          | S2_regime  |        -0.193 |   -0.098 |      0.095 | 0.626 |    1.000 |  0.719 | {"lgbm": 7, "rf": 3, "logreg": 5, "et": 6}            |
| adx_dmi_14       | directional          | S3_sizing  |        -0.193 |    0.285 |      0.479 | 0.000 |    0.000 |  0.000 | {"et": 9, "rf": 5, "lgbm": 6, "logreg": 1}            |
| adx_dmi_14       | directional          | S4_exit    |        -0.193 |    0.231 |      0.425 | 0.002 |    0.225 |  0.006 | {"lgbm": 6, "et": 5, "mlp": 2, "logreg": 3, "rf": 5}  |
| adx_dmi_14       | directional          | S5_params  |        -0.193 |   -0.126 |      0.067 | 0.686 |    1.000 |  0.774 | {"lgbm": 5, "et": 7, "mlp": 6, "rf": 2, "logreg": 1}  |
| aroon_25         | directional          | S1_filter  |         0.104 |    0.564 |      0.460 | 0.008 |    1.000 |  0.022 | {"logreg": 6, "et": 11, "rf": 2, "lgbm": 2}           |
| aroon_25         | directional          | S2_regime  |         0.104 |   -0.065 |     -0.169 | 0.241 |    1.000 |  0.346 | {"rf": 5, "lgbm": 4, "logreg": 5, "mlp": 2, "et": 5}  |
| aroon_25         | directional          | S3_sizing  |         0.104 |    0.604 |      0.499 | 0.001 |    0.084 |  0.003 | {"et": 11, "lgbm": 5, "rf": 1, "logreg": 4}           |
| aroon_25         | directional          | S4_exit    |         0.104 |    0.383 |      0.279 | 0.096 |    1.000 |  0.157 | {"mlp": 3, "et": 10, "logreg": 4, "rf": 1, "lgbm": 3} |
| aroon_25         | directional          | S5_params  |         0.104 |    0.311 |      0.207 | 0.263 |    1.000 |  0.374 | {"mlp": 6, "lgbm": 5, "rf": 3, "logreg": 4, "et": 3}  |
| elder_impulse    | oscillator_trend     | S1_filter  |        -0.720 |   -0.171 |      0.549 | 0.000 |    0.000 |  0.000 | {"logreg": 1, "lgbm": 7, "mlp": 3, "et": 8, "rf": 2}  |
| elder_impulse    | oscillator_trend     | S2_regime  |        -0.720 |   -0.417 |      0.303 | 0.013 |    1.000 |  0.032 | {"lgbm": 6, "mlp": 4, "logreg": 4, "rf": 5, "et": 2}  |
| elder_impulse    | oscillator_trend     | S3_sizing  |        -0.720 |   -0.292 |      0.428 | 0.001 |    0.157 |  0.004 | {"logreg": 6, "et": 5, "lgbm": 6, "rf": 4}            |
| elder_impulse    | oscillator_trend     | S4_exit    |        -0.720 |   -0.363 |      0.357 | 0.006 |    0.804 |  0.018 | {"et": 4, "mlp": 2, "logreg": 7, "lgbm": 2, "rf": 6}  |
| elder_impulse    | oscillator_trend     | S5_params  |        -0.720 |   -0.640 |      0.080 | 0.125 |    1.000 |  0.194 | {"lgbm": 4, "mlp": 1, "et": 5, "rf": 6, "logreg": 5}  |
| ichimoku         | directional          | S1_filter  |         0.194 |    0.581 |      0.387 | 0.009 |    1.000 |  0.024 | {"et": 14, "mlp": 2, "logreg": 3, "rf": 1, "lgbm": 1} |
| ichimoku         | directional          | S2_regime  |         0.194 |    0.112 |     -0.083 | 0.589 |    1.000 |  0.684 | {"lgbm": 6, "rf": 5, "logreg": 5, "mlp": 2, "et": 3}  |
| ichimoku         | directional          | S3_sizing  |         0.194 |    0.607 |      0.413 | 0.003 |    0.435 |  0.011 | {"rf": 6, "et": 7, "mlp": 2, "lgbm": 2, "logreg": 4}  |
| ichimoku         | directional          | S4_exit    |         0.194 |    0.403 |      0.209 | 0.170 |    1.000 |  0.253 | {"lgbm": 3, "et": 8, "mlp": 6, "logreg": 3, "rf": 1}  |
| ichimoku         | directional          | S5_params  |         0.194 |    0.359 |      0.165 | 0.126 |    1.000 |  0.194 | {"lgbm": 10, "logreg": 4, "rf": 1, "et": 2, "mlp": 4} |
| vortex_14        | directional          | S1_filter  |        -0.004 |    0.430 |      0.434 | 0.002 |    0.296 |  0.008 | {"rf": 3, "et": 10, "mlp": 1, "logreg": 5, "lgbm": 2} |
| vortex_14        | directional          | S2_regime  |        -0.004 |    0.017 |      0.021 | 0.878 |    1.000 |  0.913 | {"lgbm": 5, "rf": 3, "logreg": 6, "mlp": 4, "et": 3}  |
| vortex_14        | directional          | S3_sizing  |        -0.004 |    0.499 |      0.503 | 0.001 |    0.157 |  0.004 | {"rf": 7, "et": 7, "lgbm": 1, "mlp": 1, "logreg": 5}  |
| vortex_14        | directional          | S4_exit    |        -0.004 |    0.383 |      0.387 | 0.012 |    1.000 |  0.029 | {"rf": 8, "mlp": 5, "lgbm": 5, "logreg": 1, "et": 2}  |
| vortex_14        | directional          | S5_params  |        -0.004 |    0.060 |      0.064 | 0.694 |    1.000 |  0.778 | {"mlp": 2, "et": 6, "lgbm": 8, "logreg": 3, "rf": 2}  |
| heikin_ashi_3    | directional          | S1_filter  |        -0.382 |    0.186 |      0.567 | 0.004 |    0.634 |  0.015 | {"et": 12, "mlp": 1, "logreg": 5, "rf": 2, "lgbm": 1} |
| heikin_ashi_3    | directional          | S2_regime  |        -0.382 |   -0.349 |      0.032 | 0.813 |    1.000 |  0.873 | {"et": 6, "lgbm": 6, "mlp": 3, "rf": 3, "logreg": 3}  |
| heikin_ashi_3    | directional          | S3_sizing  |        -0.382 |    0.197 |      0.579 | 0.000 |    0.000 |  0.000 | {"logreg": 5, "et": 8, "lgbm": 4, "rf": 4}            |
| heikin_ashi_3    | directional          | S4_exit    |        -0.382 |   -0.264 |      0.118 | 0.409 |    1.000 |  0.532 | {"logreg": 8, "et": 4, "lgbm": 6, "rf": 3}            |
| heikin_ashi_3    | directional          | S5_params  |        -0.382 |    0.050 |      0.432 | 0.044 |    1.000 |  0.089 | {"lgbm": 11, "et": 4, "logreg": 4, "mlp": 2}          |