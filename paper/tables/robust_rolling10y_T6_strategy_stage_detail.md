| strategy         | strategy_family      | ml_stage   |   base_sharpe |   sharpe |   d_sharpe |     p |   p_holm |   p_bh | selected_models                                       |
|:-----------------|:---------------------|:-----------|--------------:|---------:|-----------:|------:|---------:|-------:|:------------------------------------------------------|
| faber_10m        | price_vs_ma          | S1_filter  |         0.849 |    0.848 |     -0.001 | 0.985 |    1.000 |  0.990 | {"mlp": 3, "rf": 9, "et": 4, "lgbm": 4, "logreg": 1}  |
| faber_10m        | price_vs_ma          | S2_regime  |         0.849 |    0.651 |     -0.198 | 0.068 |    1.000 |  0.124 | {"mlp": 2, "logreg": 7, "et": 5, "lgbm": 5, "rf": 2}  |
| faber_10m        | price_vs_ma          | S3_sizing  |         0.849 |    0.790 |     -0.059 | 0.426 |    1.000 |  0.513 | {"lgbm": 3, "mlp": 10, "logreg": 2, "rf": 4, "et": 2} |
| faber_10m        | price_vs_ma          | S4_exit    |         0.849 |    0.523 |     -0.327 | 0.032 |    1.000 |  0.067 | {"mlp": 3, "rf": 5, "logreg": 2, "lgbm": 3, "et": 8}  |
| faber_10m        | price_vs_ma          | S5_params  |         0.849 |    0.863 |      0.014 | 0.637 |    1.000 |  0.706 | {"mlp": 4, "rf": 3, "et": 4, "lgbm": 4, "logreg": 6}  |
| sma_50_200       | ma_crossover         | S1_filter  |         0.420 |    0.618 |      0.198 | 0.095 |    1.000 |  0.152 | {"logreg": 4, "et": 12, "mlp": 2, "rf": 2, "lgbm": 1} |
| sma_50_200       | ma_crossover         | S2_regime  |         0.420 |    0.192 |     -0.227 | 0.073 |    1.000 |  0.129 | {"et": 2, "lgbm": 5, "rf": 6, "logreg": 8}            |
| sma_50_200       | ma_crossover         | S3_sizing  |         0.420 |    0.615 |      0.195 | 0.118 |    1.000 |  0.176 | {"logreg": 5, "lgbm": 4, "mlp": 2, "rf": 3, "et": 7}  |
| sma_50_200       | ma_crossover         | S4_exit    |         0.420 |    0.585 |      0.166 | 0.372 |    1.000 |  0.462 | {"lgbm": 2, "rf": 5, "logreg": 2, "mlp": 9, "et": 3}  |
| sma_50_200       | ma_crossover         | S5_params  |         0.420 |    0.488 |      0.068 | 0.501 |    1.000 |  0.575 | {"lgbm": 9, "rf": 3, "logreg": 5, "et": 2, "mlp": 2}  |
| gmma             | ma_crossover         | S1_filter  |         0.229 |    0.491 |      0.263 | 0.086 |    1.000 |  0.143 | {"lgbm": 4, "rf": 3, "et": 12, "mlp": 1, "logreg": 1} |
| gmma             | ma_crossover         | S2_regime  |         0.229 |    0.093 |     -0.136 | 0.355 |    1.000 |  0.447 | {"et": 4, "rf": 4, "logreg": 8, "lgbm": 4, "mlp": 1}  |
| gmma             | ma_crossover         | S3_sizing  |         0.229 |    0.656 |      0.427 | 0.000 |    0.000 |  0.000 | {"et": 12, "lgbm": 3, "rf": 3, "mlp": 1, "logreg": 2} |
| gmma             | ma_crossover         | S4_exit    |         0.229 |    0.523 |      0.295 | 0.049 |    1.000 |  0.095 | {"et": 13, "rf": 4, "lgbm": 3, "logreg": 1}           |
| gmma             | ma_crossover         | S5_params  |         0.229 |    0.409 |      0.180 | 0.090 |    1.000 |  0.146 | {"mlp": 4, "logreg": 5, "rf": 1, "et": 4, "lgbm": 7}  |
| bll_vma_200      | price_vs_ma          | S1_filter  |         0.455 |    0.716 |      0.261 | 0.044 |    1.000 |  0.088 | {"et": 13, "logreg": 2, "lgbm": 3, "mlp": 1, "rf": 2} |
| bll_vma_200      | price_vs_ma          | S2_regime  |         0.455 |    0.307 |     -0.148 | 0.266 |    1.000 |  0.346 | {"et": 3, "rf": 6, "logreg": 7, "lgbm": 5}            |
| bll_vma_200      | price_vs_ma          | S3_sizing  |         0.455 |    0.763 |      0.308 | 0.021 |    1.000 |  0.045 | {"et": 13, "logreg": 1, "lgbm": 2, "mlp": 2, "rf": 3} |
| bll_vma_200      | price_vs_ma          | S4_exit    |         0.455 |    0.891 |      0.436 | 0.013 |    1.000 |  0.032 | {"lgbm": 7, "et": 6, "logreg": 2, "mlp": 2, "rf": 4}  |
| bll_vma_200      | price_vs_ma          | S5_params  |         0.455 |    0.453 |     -0.002 | 0.972 |    1.000 |  0.982 | {"mlp": 4, "logreg": 5, "rf": 3, "et": 4, "lgbm": 5}  |
| ema_12_26        | ma_crossover         | S1_filter  |         0.193 |    0.571 |      0.378 | 0.011 |    1.000 |  0.028 | {"lgbm": 2, "et": 12, "mlp": 1, "rf": 5, "logreg": 1} |
| ema_12_26        | ma_crossover         | S2_regime  |         0.193 |   -0.069 |     -0.261 | 0.058 |    1.000 |  0.108 | {"et": 5, "rf": 4, "logreg": 8, "lgbm": 2, "mlp": 2}  |
| ema_12_26        | ma_crossover         | S3_sizing  |         0.193 |    0.646 |      0.453 | 0.002 |    0.228 |  0.006 | {"et": 12, "mlp": 3, "lgbm": 1, "rf": 3, "logreg": 2} |
| ema_12_26        | ma_crossover         | S4_exit    |         0.193 |    0.391 |      0.198 | 0.251 |    1.000 |  0.328 | {"mlp": 4, "et": 5, "lgbm": 6, "rf": 5, "logreg": 1}  |
| ema_12_26        | ma_crossover         | S5_params  |         0.193 |    0.277 |      0.084 | 0.465 |    1.000 |  0.546 | {"lgbm": 11, "logreg": 5, "et": 3, "rf": 1, "mlp": 1} |
| dema_20_50       | ma_crossover         | S1_filter  |        -0.071 |    0.372 |      0.442 | 0.021 |    1.000 |  0.046 | {"rf": 3, "logreg": 8, "mlp": 2, "et": 6, "lgbm": 2}  |
| dema_20_50       | ma_crossover         | S2_regime  |        -0.071 |   -0.236 |     -0.165 | 0.202 |    1.000 |  0.272 | {"rf": 3, "et": 6, "logreg": 6, "lgbm": 4, "mlp": 2}  |
| dema_20_50       | ma_crossover         | S3_sizing  |        -0.071 |    0.470 |      0.540 | 0.000 |    0.000 |  0.000 | {"et": 12, "rf": 2, "mlp": 1, "lgbm": 2, "logreg": 4} |
| dema_20_50       | ma_crossover         | S4_exit    |        -0.071 |    0.351 |      0.422 | 0.017 |    1.000 |  0.040 | {"rf": 3, "et": 8, "mlp": 3, "lgbm": 3, "logreg": 4}  |
| dema_20_50       | ma_crossover         | S5_params  |        -0.071 |    0.305 |      0.376 | 0.072 |    1.000 |  0.129 | {"et": 6, "lgbm": 6, "logreg": 4, "mlp": 2, "rf": 3}  |
| triple_ma_4_9_18 | ma_crossover         | S1_filter  |        -0.311 |    0.255 |      0.566 | 0.000 |    0.000 |  0.000 | {"et": 12, "mlp": 4, "rf": 2, "lgbm": 1, "logreg": 2} |
| triple_ma_4_9_18 | ma_crossover         | S2_regime  |        -0.311 |   -0.273 |      0.038 | 0.792 |    1.000 |  0.849 | {"et": 3, "rf": 3, "mlp": 2, "logreg": 8, "lgbm": 5}  |
| triple_ma_4_9_18 | ma_crossover         | S3_sizing  |        -0.311 |    0.280 |      0.591 | 0.000 |    0.000 |  0.000 | {"rf": 7, "et": 8, "lgbm": 2, "mlp": 1, "logreg": 3}  |
| triple_ma_4_9_18 | ma_crossover         | S4_exit    |        -0.311 |    0.168 |      0.478 | 0.002 |    0.296 |  0.008 | {"logreg": 5, "et": 8, "rf": 5, "lgbm": 2, "mlp": 1}  |
| triple_ma_4_9_18 | ma_crossover         | S5_params  |        -0.311 |    0.032 |      0.343 | 0.017 |    1.000 |  0.040 | {"mlp": 3, "logreg": 5, "rf": 3, "et": 2, "lgbm": 8}  |
| zlema_10_30      | ma_crossover         | S1_filter  |        -0.282 |    0.205 |      0.487 | 0.009 |    1.000 |  0.023 | {"logreg": 1, "et": 11, "mlp": 6, "rf": 1, "lgbm": 2} |
| zlema_10_30      | ma_crossover         | S2_regime  |        -0.282 |   -0.052 |      0.230 | 0.041 |    1.000 |  0.082 | {"et": 7, "mlp": 1, "logreg": 9, "lgbm": 3, "rf": 1}  |
| zlema_10_30      | ma_crossover         | S3_sizing  |        -0.282 |    0.120 |      0.402 | 0.003 |    0.357 |  0.009 | {"logreg": 4, "et": 8, "lgbm": 1, "rf": 5, "mlp": 3}  |
| zlema_10_30      | ma_crossover         | S4_exit    |        -0.282 |    0.051 |      0.333 | 0.004 |    0.612 |  0.014 | {"logreg": 6, "lgbm": 2, "mlp": 6, "rf": 4, "et": 3}  |
| zlema_10_30      | ma_crossover         | S5_params  |        -0.282 |   -0.118 |      0.164 | 0.412 |    1.000 |  0.505 | {"et": 5, "rf": 4, "logreg": 3, "lgbm": 6, "mlp": 3}  |
| jma_7_21         | ma_crossover         | S1_filter  |        -0.318 |    0.409 |      0.726 | 0.000 |    0.000 |  0.000 | {"logreg": 2, "et": 8, "mlp": 4, "rf": 4, "lgbm": 3}  |
| jma_7_21         | ma_crossover         | S2_regime  |        -0.318 |   -0.135 |      0.183 | 0.160 |    1.000 |  0.224 | {"et": 5, "rf": 3, "mlp": 1, "logreg": 8, "lgbm": 4}  |
| jma_7_21         | ma_crossover         | S3_sizing  |        -0.318 |    0.084 |      0.401 | 0.002 |    0.228 |  0.006 | {"logreg": 5, "et": 5, "mlp": 1, "lgbm": 5, "rf": 5}  |
| jma_7_21         | ma_crossover         | S4_exit    |        -0.318 |   -0.001 |      0.317 | 0.008 |    1.000 |  0.023 | {"logreg": 10, "mlp": 1, "rf": 3, "lgbm": 3, "et": 4} |
| jma_7_21         | ma_crossover         | S5_params  |        -0.318 |   -0.083 |      0.235 | 0.211 |    1.000 |  0.282 | {"rf": 5, "logreg": 2, "lgbm": 8, "et": 5, "mlp": 1}  |
| donchian_20      | breakout             | S1_filter  |         0.110 |    0.590 |      0.480 | 0.015 |    1.000 |  0.035 | {"et": 13, "rf": 1, "mlp": 2, "logreg": 4, "lgbm": 1} |
| donchian_20      | breakout             | S2_regime  |         0.110 |   -0.098 |     -0.208 | 0.143 |    1.000 |  0.205 | {"et": 4, "rf": 3, "logreg": 8, "lgbm": 4, "mlp": 2}  |
| donchian_20      | breakout             | S3_sizing  |         0.110 |    0.603 |      0.493 | 0.001 |    0.083 |  0.003 | {"et": 12, "lgbm": 3, "rf": 4, "mlp": 1, "logreg": 1} |
| donchian_20      | breakout             | S4_exit    |         0.110 |    0.484 |      0.374 | 0.033 |    1.000 |  0.067 | {"rf": 5, "et": 8, "mlp": 4, "lgbm": 4}               |
| donchian_20      | breakout             | S5_params  |         0.110 |    0.061 |     -0.049 | 0.648 |    1.000 |  0.714 | {"mlp": 1, "lgbm": 10, "logreg": 4, "et": 4, "rf": 2} |
| kama_10          | adaptive_ma          | S1_filter  |        -0.498 |    0.085 |      0.582 | 0.002 |    0.228 |  0.006 | {"logreg": 1, "et": 12, "rf": 6, "lgbm": 2}           |
| kama_10          | adaptive_ma          | S2_regime  |        -0.498 |   -0.294 |      0.204 | 0.111 |    1.000 |  0.169 | {"et": 6, "rf": 3, "mlp": 1, "logreg": 7, "lgbm": 4}  |
| kama_10          | adaptive_ma          | S3_sizing  |        -0.498 |    0.035 |      0.533 | 0.000 |    0.000 |  0.000 | {"rf": 5, "et": 7, "lgbm": 1, "logreg": 7, "mlp": 1}  |
| kama_10          | adaptive_ma          | S4_exit    |        -0.498 |   -0.199 |      0.299 | 0.018 |    1.000 |  0.041 | {"logreg": 8, "rf": 3, "lgbm": 3, "et": 7}            |
| kama_10          | adaptive_ma          | S5_params  |        -0.498 |   -0.212 |      0.285 | 0.003 |    0.357 |  0.009 | {"lgbm": 6, "logreg": 4, "et": 6, "rf": 3, "mlp": 2}  |
| turtle_55_20     | breakout             | S1_filter  |         0.214 |    0.537 |      0.324 | 0.042 |    1.000 |  0.083 | {"rf": 8, "et": 7, "lgbm": 3, "mlp": 3}               |
| turtle_55_20     | breakout             | S2_regime  |         0.214 |    0.007 |     -0.207 | 0.159 |    1.000 |  0.224 | {"lgbm": 6, "rf": 2, "logreg": 8, "et": 5}            |
| turtle_55_20     | breakout             | S3_sizing  |         0.214 |    0.583 |      0.369 | 0.006 |    0.798 |  0.019 | {"et": 8, "rf": 6, "lgbm": 3, "mlp": 3, "logreg": 1}  |
| turtle_55_20     | breakout             | S4_exit    |         0.214 |    0.581 |      0.367 | 0.013 |    1.000 |  0.032 | {"et": 12, "lgbm": 6, "mlp": 2, "rf": 1}              |
| turtle_55_20     | breakout             | S5_params  |         0.214 |    0.294 |      0.080 | 0.457 |    1.000 |  0.539 | {"mlp": 4, "lgbm": 7, "rf": 2, "et": 5, "logreg": 3}  |
| turtle_20_10     | breakout             | S1_filter  |        -0.019 |    0.556 |      0.575 | 0.001 |    0.158 |  0.005 | {"rf": 5, "et": 10, "mlp": 2, "logreg": 3, "lgbm": 1} |
| turtle_20_10     | breakout             | S2_regime  |        -0.019 |   -0.163 |     -0.144 | 0.302 |    1.000 |  0.385 | {"et": 4, "rf": 2, "mlp": 3, "logreg": 6, "lgbm": 6}  |
| turtle_20_10     | breakout             | S3_sizing  |        -0.019 |    0.516 |      0.535 | 0.000 |    0.000 |  0.000 | {"lgbm": 2, "rf": 3, "mlp": 1, "logreg": 3, "et": 12} |
| turtle_20_10     | breakout             | S4_exit    |        -0.019 |    0.487 |      0.506 | 0.001 |    0.083 |  0.003 | {"rf": 4, "et": 7, "lgbm": 5, "logreg": 3, "mlp": 2}  |
| turtle_20_10     | breakout             | S5_params  |        -0.019 |    0.160 |      0.179 | 0.105 |    1.000 |  0.161 | {"logreg": 5, "et": 5, "lgbm": 8, "rf": 2, "mlp": 1}  |
| hma_55_slope     | adaptive_ma          | S1_filter  |         0.025 |    0.631 |      0.606 | 0.000 |    0.000 |  0.000 | {"et": 14, "rf": 4, "logreg": 2, "mlp": 1}            |
| hma_55_slope     | adaptive_ma          | S2_regime  |         0.025 |   -0.077 |     -0.102 | 0.483 |    1.000 |  0.561 | {"lgbm": 7, "rf": 2, "et": 2, "logreg": 9, "mlp": 1}  |
| hma_55_slope     | adaptive_ma          | S3_sizing  |         0.025 |    0.459 |      0.434 | 0.002 |    0.296 |  0.008 | {"logreg": 3, "et": 9, "lgbm": 2, "rf": 6, "mlp": 1}  |
| hma_55_slope     | adaptive_ma          | S4_exit    |         0.025 |    0.439 |      0.414 | 0.006 |    0.858 |  0.019 | {"logreg": 1, "et": 9, "lgbm": 7, "rf": 3, "mlp": 1}  |
| hma_55_slope     | adaptive_ma          | S5_params  |         0.025 |    0.247 |      0.222 | 0.170 |    1.000 |  0.237 | {"et": 2, "lgbm": 8, "logreg": 5, "rf": 3, "mlp": 3}  |
| vidya_14         | adaptive_ma          | S1_filter  |        -0.011 |    0.340 |      0.351 | 0.050 |    1.000 |  0.096 | {"mlp": 2, "et": 14, "logreg": 3, "lgbm": 1, "rf": 1} |
| vidya_14         | adaptive_ma          | S2_regime  |        -0.011 |   -0.246 |     -0.235 | 0.114 |    1.000 |  0.171 | {"et": 4, "rf": 4, "mlp": 2, "logreg": 7, "lgbm": 4}  |
| vidya_14         | adaptive_ma          | S3_sizing  |        -0.011 |    0.481 |      0.492 | 0.000 |    0.000 |  0.000 | {"et": 13, "mlp": 1, "lgbm": 1, "logreg": 4, "rf": 2} |
| vidya_14         | adaptive_ma          | S4_exit    |        -0.011 |    0.224 |      0.235 | 0.085 |    1.000 |  0.143 | {"et": 12, "mlp": 3, "lgbm": 3, "logreg": 3}          |
| vidya_14         | adaptive_ma          | S5_params  |        -0.011 |    0.154 |      0.165 | 0.086 |    1.000 |  0.143 | {"rf": 7, "et": 4, "logreg": 4, "lgbm": 3, "mlp": 3}  |
| t3_20            | adaptive_ma          | S1_filter  |        -0.168 |    0.379 |      0.547 | 0.004 |    0.483 |  0.012 | {"rf": 7, "et": 9, "mlp": 2, "lgbm": 2, "logreg": 1}  |
| t3_20            | adaptive_ma          | S2_regime  |        -0.168 |   -0.068 |      0.100 | 0.433 |    1.000 |  0.519 | {"et": 6, "rf": 4, "mlp": 2, "lgbm": 4, "logreg": 5}  |
| t3_20            | adaptive_ma          | S3_sizing  |        -0.168 |    0.353 |      0.521 | 0.000 |    0.000 |  0.000 | {"et": 9, "mlp": 1, "lgbm": 2, "logreg": 6, "rf": 3}  |
| t3_20            | adaptive_ma          | S4_exit    |        -0.168 |    0.081 |      0.249 | 0.099 |    1.000 |  0.157 | {"lgbm": 6, "et": 8, "rf": 1, "mlp": 3, "logreg": 3}  |
| t3_20            | adaptive_ma          | S5_params  |        -0.168 |    0.147 |      0.315 | 0.033 |    1.000 |  0.067 | {"lgbm": 8, "et": 5, "rf": 2, "logreg": 3, "mlp": 3}  |
| mcginley_14      | adaptive_ma          | S1_filter  |        -0.007 |    0.426 |      0.433 | 0.009 |    1.000 |  0.023 | {"lgbm": 4, "et": 15, "mlp": 1, "logreg": 1}          |
| mcginley_14      | adaptive_ma          | S2_regime  |        -0.007 |   -0.006 |      0.000 | 0.997 |    1.000 |  0.997 | {"et": 3, "rf": 1, "mlp": 2, "logreg": 8, "lgbm": 7}  |
| mcginley_14      | adaptive_ma          | S3_sizing  |        -0.007 |    0.468 |      0.475 | 0.000 |    0.000 |  0.000 | {"et": 11, "lgbm": 1, "logreg": 3, "rf": 4, "mlp": 2} |
| mcginley_14      | adaptive_ma          | S4_exit    |        -0.007 |    0.202 |      0.209 | 0.102 |    1.000 |  0.159 | {"mlp": 1, "et": 9, "lgbm": 4, "logreg": 4, "rf": 3}  |
| mcginley_14      | adaptive_ma          | S5_params  |        -0.007 |    0.273 |      0.279 | 0.001 |    0.158 |  0.005 | {"mlp": 1, "rf": 2, "et": 5, "lgbm": 7, "logreg": 6}  |
| frama_16         | adaptive_ma          | S1_filter  |        -0.825 |   -0.264 |      0.561 | 0.000 |    0.000 |  0.000 | {"mlp": 2, "et": 12, "rf": 4, "lgbm": 2, "logreg": 1} |
| frama_16         | adaptive_ma          | S2_regime  |        -0.825 |   -0.648 |      0.177 | 0.192 |    1.000 |  0.261 | {"et": 6, "rf": 7, "mlp": 1, "logreg": 4, "lgbm": 3}  |
| frama_16         | adaptive_ma          | S3_sizing  |        -0.825 |   -0.374 |      0.452 | 0.000 |    0.000 |  0.000 | {"logreg": 9, "et": 5, "mlp": 1, "lgbm": 4, "rf": 2}  |
| frama_16         | adaptive_ma          | S4_exit    |        -0.825 |   -0.507 |      0.318 | 0.006 |    0.858 |  0.019 | {"logreg": 10, "mlp": 3, "et": 5, "lgbm": 3}          |
| frama_16         | adaptive_ma          | S5_params  |        -0.825 |   -0.601 |      0.225 | 0.040 |    1.000 |  0.081 | {"logreg": 5, "rf": 5, "et": 6, "mlp": 4, "lgbm": 1}  |
| keltner_20       | breakout             | S1_filter  |         0.035 |    0.589 |      0.554 | 0.001 |    0.158 |  0.005 | {"rf": 9, "lgbm": 3, "logreg": 2, "mlp": 2, "et": 5}  |
| keltner_20       | breakout             | S2_regime  |         0.035 |    0.041 |      0.006 | 0.971 |    1.000 |  0.982 | {"rf": 6, "logreg": 6, "et": 4, "lgbm": 5}            |
| keltner_20       | breakout             | S3_sizing  |         0.035 |    0.531 |      0.496 | 0.000 |    0.000 |  0.000 | {"et": 11, "lgbm": 2, "rf": 4, "mlp": 1, "logreg": 3} |
| keltner_20       | breakout             | S4_exit    |         0.035 |    0.493 |      0.458 | 0.003 |    0.357 |  0.009 | {"rf": 4, "et": 5, "lgbm": 4, "logreg": 4, "mlp": 4}  |
| keltner_20       | breakout             | S5_params  |         0.035 |    0.262 |      0.227 | 0.025 |    1.000 |  0.055 | {"rf": 3, "et": 7, "logreg": 5, "lgbm": 4, "mlp": 2}  |
| bollinger_20_2   | breakout             | S1_filter  |        -0.076 |    0.530 |      0.606 | 0.000 |    0.000 |  0.000 | {"et": 15, "lgbm": 2, "logreg": 1, "rf": 2, "mlp": 1} |
| bollinger_20_2   | breakout             | S2_regime  |        -0.076 |   -0.171 |     -0.095 | 0.543 |    1.000 |  0.615 | {"lgbm": 8, "rf": 3, "et": 2, "mlp": 3, "logreg": 5}  |
| bollinger_20_2   | breakout             | S3_sizing  |        -0.076 |    0.422 |      0.498 | 0.000 |    0.000 |  0.000 | {"et": 10, "lgbm": 3, "logreg": 4, "rf": 4}           |
| bollinger_20_2   | breakout             | S4_exit    |        -0.076 |    0.315 |      0.391 | 0.002 |    0.296 |  0.008 | {"rf": 6, "lgbm": 7, "logreg": 3, "et": 4, "mlp": 1}  |
| bollinger_20_2   | breakout             | S5_params  |        -0.076 |    0.130 |      0.206 | 0.069 |    1.000 |  0.124 | {"lgbm": 6, "mlp": 6, "logreg": 2, "rf": 4, "et": 3}  |
| supertrend_10_3  | volatility_stop      | S1_filter  |         0.094 |    0.591 |      0.497 | 0.015 |    1.000 |  0.037 | {"et": 14, "mlp": 1, "lgbm": 3, "logreg": 2, "rf": 1} |
| supertrend_10_3  | volatility_stop      | S2_regime  |         0.094 |   -0.054 |     -0.148 | 0.279 |    1.000 |  0.358 | {"et": 5, "rf": 3, "logreg": 7, "lgbm": 6}            |
| supertrend_10_3  | volatility_stop      | S3_sizing  |         0.094 |    0.645 |      0.551 | 0.000 |    0.000 |  0.000 | {"logreg": 3, "et": 10, "lgbm": 3, "rf": 3, "mlp": 2} |
| supertrend_10_3  | volatility_stop      | S4_exit    |         0.094 |    0.633 |      0.540 | 0.000 |    0.000 |  0.000 | {"logreg": 1, "et": 11, "mlp": 2, "lgbm": 4, "rf": 3} |
| supertrend_10_3  | volatility_stop      | S5_params  |         0.094 |    0.093 |     -0.001 | 0.969 |    1.000 |  0.982 | {"logreg": 3, "lgbm": 4, "et": 2, "rf": 4, "mlp": 8}  |
| abs_mom_12m      | time_series_momentum | S1_filter  |         0.841 |    0.852 |      0.011 | 0.818 |    1.000 |  0.867 | {"et": 4, "lgbm": 6, "logreg": 2, "rf": 6, "mlp": 3}  |
| abs_mom_12m      | time_series_momentum | S2_regime  |         0.841 |    0.690 |     -0.152 | 0.105 |    1.000 |  0.161 | {"logreg": 10, "et": 5, "mlp": 5, "rf": 1}            |
| abs_mom_12m      | time_series_momentum | S3_sizing  |         0.841 |    0.801 |     -0.041 | 0.514 |    1.000 |  0.587 | {"et": 4, "mlp": 8, "lgbm": 4, "rf": 2, "logreg": 3}  |
| abs_mom_12m      | time_series_momentum | S4_exit    |         0.841 |    0.364 |     -0.477 | 0.011 |    1.000 |  0.027 | {"mlp": 7, "logreg": 7, "lgbm": 2, "et": 4, "rf": 1}  |
| abs_mom_12m      | time_series_momentum | S5_params  |         0.841 |    0.748 |     -0.093 | 0.089 |    1.000 |  0.146 | {"lgbm": 9, "mlp": 3, "rf": 2, "logreg": 1, "et": 6}  |
| tsmom_252        | time_series_momentum | S1_filter  |         0.488 |    0.609 |      0.122 | 0.216 |    1.000 |  0.287 | {"mlp": 7, "rf": 2, "et": 9, "lgbm": 1, "logreg": 2}  |
| tsmom_252        | time_series_momentum | S2_regime  |         0.488 |    0.355 |     -0.132 | 0.200 |    1.000 |  0.270 | {"rf": 3, "logreg": 9, "et": 4, "mlp": 3, "lgbm": 2}  |
| tsmom_252        | time_series_momentum | S3_sizing  |         0.488 |    0.678 |      0.190 | 0.129 |    1.000 |  0.188 | {"mlp": 2, "logreg": 6, "rf": 4, "et": 8, "lgbm": 1}  |
| tsmom_252        | time_series_momentum | S4_exit    |         0.488 |    0.700 |      0.213 | 0.172 |    1.000 |  0.238 | {"et": 9, "logreg": 4, "lgbm": 4, "rf": 2, "mlp": 2}  |
| tsmom_252        | time_series_momentum | S5_params  |         0.488 |    0.473 |     -0.015 | 0.888 |    1.000 |  0.921 | {"logreg": 9, "et": 2, "lgbm": 5, "mlp": 2, "rf": 3}  |
| chande_kroll     | volatility_stop      | S1_filter  |         0.094 |    0.572 |      0.478 | 0.002 |    0.296 |  0.008 | {"et": 13, "mlp": 3, "lgbm": 3, "rf": 2}              |
| chande_kroll     | volatility_stop      | S2_regime  |         0.094 |    0.053 |     -0.041 | 0.778 |    1.000 |  0.839 | {"et": 5, "rf": 2, "mlp": 2, "logreg": 8, "lgbm": 4}  |
| chande_kroll     | volatility_stop      | S3_sizing  |         0.094 |    0.609 |      0.515 | 0.000 |    0.000 |  0.000 | {"et": 10, "mlp": 3, "lgbm": 2, "rf": 4, "logreg": 2} |
| chande_kroll     | volatility_stop      | S4_exit    |         0.094 |    0.326 |      0.232 | 0.129 |    1.000 |  0.188 | {"logreg": 1, "et": 9, "mlp": 2, "lgbm": 5, "rf": 4}  |
| chande_kroll     | volatility_stop      | S5_params  |         0.094 |    0.305 |      0.211 | 0.236 |    1.000 |  0.311 | {"mlp": 8, "lgbm": 5, "logreg": 5, "et": 2, "rf": 1}  |
| chandelier_22_3  | volatility_stop      | S1_filter  |        -0.117 |    0.281 |      0.398 | 0.000 |    0.000 |  0.000 | {"lgbm": 7, "mlp": 5, "logreg": 6, "et": 1, "rf": 2}  |
| chandelier_22_3  | volatility_stop      | S2_regime  |        -0.117 |   -0.125 |     -0.008 | 0.956 |    1.000 |  0.981 | {"et": 3, "rf": 3, "logreg": 10, "lgbm": 4, "mlp": 1} |
| chandelier_22_3  | volatility_stop      | S3_sizing  |        -0.117 |    0.419 |      0.536 | 0.000 |    0.000 |  0.000 | {"et": 12, "mlp": 1, "lgbm": 1, "logreg": 5, "rf": 2} |
| chandelier_22_3  | volatility_stop      | S4_exit    |        -0.117 |    0.183 |      0.300 | 0.040 |    1.000 |  0.081 | {"logreg": 4, "mlp": 3, "et": 8, "lgbm": 2, "rf": 4}  |
| chandelier_22_3  | volatility_stop      | S5_params  |        -0.117 |   -0.134 |     -0.017 | 0.415 |    1.000 |  0.506 | {"lgbm": 7, "rf": 8, "et": 2, "mlp": 3, "logreg": 1}  |
| psar             | volatility_stop      | S1_filter  |        -0.268 |    0.372 |      0.639 | 0.000 |    0.000 |  0.000 | {"logreg": 1, "et": 12, "lgbm": 3, "rf": 4, "mlp": 1} |
| psar             | volatility_stop      | S2_regime  |        -0.268 |   -0.189 |      0.079 | 0.546 |    1.000 |  0.615 | {"et": 3, "rf": 4, "mlp": 2, "lgbm": 5, "logreg": 7}  |
| psar             | volatility_stop      | S3_sizing  |        -0.268 |    0.146 |      0.414 | 0.003 |    0.357 |  0.009 | {"et": 10, "lgbm": 1, "logreg": 5, "rf": 4, "mlp": 1} |
| psar             | volatility_stop      | S4_exit    |        -0.268 |   -0.057 |      0.210 | 0.135 |    1.000 |  0.194 | {"logreg": 5, "et": 11, "rf": 5}                      |
| psar             | volatility_stop      | S5_params  |        -0.268 |   -0.067 |      0.200 | 0.083 |    1.000 |  0.143 | {"lgbm": 6, "logreg": 3, "et": 6, "rf": 5, "mlp": 1}  |
| tsmom_multi      | time_series_momentum | S1_filter  |         0.348 |    0.637 |      0.290 | 0.018 |    1.000 |  0.041 | {"logreg": 3, "et": 12, "mlp": 5, "rf": 1}            |
| tsmom_multi      | time_series_momentum | S2_regime  |         0.348 |    0.317 |     -0.031 | 0.818 |    1.000 |  0.867 | {"lgbm": 6, "rf": 4, "logreg": 8, "mlp": 2, "et": 1}  |
| tsmom_multi      | time_series_momentum | S3_sizing  |         0.348 |    0.575 |      0.227 | 0.080 |    1.000 |  0.140 | {"lgbm": 1, "et": 8, "rf": 7, "mlp": 3, "logreg": 2}  |
| tsmom_multi      | time_series_momentum | S4_exit    |         0.348 |    0.571 |      0.223 | 0.152 |    1.000 |  0.216 | {"lgbm": 2, "et": 6, "rf": 9, "logreg": 1, "mlp": 3}  |
| tsmom_multi      | time_series_momentum | S5_params  |         0.348 |    0.419 |      0.072 | 0.445 |    1.000 |  0.529 | {"logreg": 7, "mlp": 4, "lgbm": 3, "rf": 3, "et": 4}  |
| linreg_63        | time_series_momentum | S1_filter  |         0.251 |    0.503 |      0.251 | 0.084 |    1.000 |  0.143 | {"rf": 2, "et": 12, "lgbm": 2, "mlp": 3, "logreg": 2} |
| linreg_63        | time_series_momentum | S2_regime  |         0.251 |    0.133 |     -0.119 | 0.366 |    1.000 |  0.458 | {"rf": 6, "logreg": 8, "lgbm": 4, "et": 3}            |
| linreg_63        | time_series_momentum | S3_sizing  |         0.251 |    0.521 |      0.270 | 0.054 |    1.000 |  0.102 | {"et": 10, "rf": 4, "lgbm": 4, "logreg": 2, "mlp": 1} |
| linreg_63        | time_series_momentum | S4_exit    |         0.251 |    0.550 |      0.299 | 0.084 |    1.000 |  0.143 | {"et": 9, "logreg": 2, "lgbm": 3, "rf": 5, "mlp": 2}  |
| linreg_63        | time_series_momentum | S5_params  |         0.251 |    0.498 |      0.247 | 0.112 |    1.000 |  0.169 | {"et": 5, "mlp": 2, "lgbm": 1, "logreg": 10, "rf": 3} |
| macd_12_26_9     | oscillator_trend     | S1_filter  |        -0.109 |    0.527 |      0.635 | 0.001 |    0.083 |  0.003 | {"et": 12, "mlp": 3, "lgbm": 3, "rf": 2, "logreg": 1} |
| macd_12_26_9     | oscillator_trend     | S2_regime  |        -0.109 |    0.051 |      0.160 | 0.175 |    1.000 |  0.241 | {"et": 6, "mlp": 3, "lgbm": 4, "logreg": 6, "rf": 2}  |
| macd_12_26_9     | oscillator_trend     | S3_sizing  |        -0.109 |    0.380 |      0.488 | 0.000 |    0.000 |  0.000 | {"et": 10, "lgbm": 3, "rf": 5, "logreg": 2, "mlp": 1} |
| macd_12_26_9     | oscillator_trend     | S4_exit    |        -0.109 |    0.427 |      0.536 | 0.001 |    0.158 |  0.005 | {"logreg": 6, "rf": 5, "lgbm": 2, "mlp": 2, "et": 6}  |
| macd_12_26_9     | oscillator_trend     | S5_params  |        -0.109 |   -0.035 |      0.074 | 0.675 |    1.000 |  0.738 | {"mlp": 6, "logreg": 3, "lgbm": 4, "rf": 4, "et": 4}  |
| adx_dmi_14       | directional          | S1_filter  |        -0.193 |    0.365 |      0.559 | 0.000 |    0.000 |  0.000 | {"et": 13, "logreg": 4, "rf": 2, "lgbm": 2}           |
| adx_dmi_14       | directional          | S2_regime  |        -0.193 |   -0.162 |      0.031 | 0.867 |    1.000 |  0.909 | {"lgbm": 5, "rf": 4, "logreg": 8, "et": 3, "mlp": 1}  |
| adx_dmi_14       | directional          | S3_sizing  |        -0.193 |    0.305 |      0.499 | 0.000 |    0.000 |  0.000 | {"rf": 3, "et": 12, "lgbm": 4, "mlp": 1, "logreg": 1} |
| adx_dmi_14       | directional          | S4_exit    |        -0.193 |    0.189 |      0.383 | 0.006 |    0.858 |  0.019 | {"lgbm": 5, "mlp": 5, "rf": 2, "et": 9}               |
| adx_dmi_14       | directional          | S5_params  |        -0.193 |   -0.163 |      0.030 | 0.863 |    1.000 |  0.909 | {"lgbm": 7, "et": 6, "rf": 3, "mlp": 2, "logreg": 3}  |
| trix_15          | oscillator_trend     | S1_filter  |        -0.107 |    0.354 |      0.462 | 0.007 |    0.903 |  0.020 | {"et": 13, "rf": 3, "mlp": 3, "logreg": 2}            |
| trix_15          | oscillator_trend     | S2_regime  |        -0.107 |   -0.099 |      0.008 | 0.949 |    1.000 |  0.979 | {"et": 8, "mlp": 5, "lgbm": 5, "rf": 1, "logreg": 2}  |
| trix_15          | oscillator_trend     | S3_sizing  |        -0.107 |    0.400 |      0.508 | 0.001 |    0.158 |  0.005 | {"et": 16, "lgbm": 1, "mlp": 3, "rf": 1}              |
| trix_15          | oscillator_trend     | S4_exit    |        -0.107 |    0.375 |      0.482 | 0.003 |    0.357 |  0.009 | {"lgbm": 7, "et": 7, "rf": 6, "mlp": 1}               |
| trix_15          | oscillator_trend     | S5_params  |        -0.107 |    0.098 |      0.205 | 0.378 |    1.000 |  0.467 | {"mlp": 3, "logreg": 5, "et": 4, "lgbm": 5, "rf": 4}  |
| kst              | oscillator_trend     | S1_filter  |        -0.091 |    0.490 |      0.581 | 0.001 |    0.083 |  0.003 | {"logreg": 3, "et": 12, "mlp": 3, "lgbm": 2, "rf": 1} |
| kst              | oscillator_trend     | S2_regime  |        -0.091 |   -0.140 |     -0.049 | 0.705 |    1.000 |  0.763 | {"et": 7, "mlp": 5, "rf": 2, "logreg": 3, "lgbm": 4}  |
| kst              | oscillator_trend     | S3_sizing  |        -0.091 |    0.439 |      0.531 | 0.000 |    0.000 |  0.000 | {"et": 13, "lgbm": 1, "mlp": 4, "logreg": 1, "rf": 2} |
| kst              | oscillator_trend     | S4_exit    |        -0.091 |    0.352 |      0.443 | 0.004 |    0.612 |  0.014 | {"logreg": 3, "et": 6, "lgbm": 5, "mlp": 1, "rf": 6}  |
| kst              | oscillator_trend     | S5_params  |        -0.091 |   -0.061 |      0.030 | 0.881 |    1.000 |  0.919 | {"logreg": 2, "et": 9, "mlp": 3, "lgbm": 6, "rf": 1}  |
| cci_20_100       | oscillator_trend     | S1_filter  |         0.050 |    0.497 |      0.447 | 0.019 |    1.000 |  0.043 | {"rf": 2, "et": 11, "mlp": 1, "logreg": 5, "lgbm": 2} |
| cci_20_100       | oscillator_trend     | S2_regime  |         0.050 |   -0.107 |     -0.157 | 0.274 |    1.000 |  0.353 | {"et": 4, "rf": 3, "logreg": 7, "lgbm": 5, "mlp": 2}  |
| cci_20_100       | oscillator_trend     | S3_sizing  |         0.050 |    0.588 |      0.538 | 0.000 |    0.000 |  0.000 | {"et": 11, "rf": 4, "lgbm": 3, "logreg": 3}           |
| cci_20_100       | oscillator_trend     | S4_exit    |         0.050 |    0.468 |      0.418 | 0.009 |    1.000 |  0.023 | {"mlp": 3, "et": 8, "rf": 2, "lgbm": 8}               |
| cci_20_100       | oscillator_trend     | S5_params  |         0.050 |    0.174 |      0.125 | 0.421 |    1.000 |  0.510 | {"mlp": 1, "logreg": 4, "et": 3, "lgbm": 8, "rf": 5}  |
| tsi_25_13        | oscillator_trend     | S1_filter  |        -0.157 |    0.373 |      0.530 | 0.005 |    0.737 |  0.017 | {"logreg": 1, "et": 10, "mlp": 5, "rf": 3, "lgbm": 2} |
| tsi_25_13        | oscillator_trend     | S2_regime  |        -0.157 |    0.028 |      0.185 | 0.097 |    1.000 |  0.155 | {"et": 6, "logreg": 8, "mlp": 1, "rf": 2, "lgbm": 4}  |
| tsi_25_13        | oscillator_trend     | S3_sizing  |        -0.157 |    0.282 |      0.439 | 0.004 |    0.548 |  0.013 | {"et": 12, "rf": 4, "mlp": 3, "lgbm": 1, "logreg": 1} |
| tsi_25_13        | oscillator_trend     | S4_exit    |        -0.157 |    0.208 |      0.366 | 0.019 |    1.000 |  0.043 | {"lgbm": 1, "logreg": 5, "et": 6, "rf": 5, "mlp": 4}  |
| tsi_25_13        | oscillator_trend     | S5_params  |        -0.157 |   -0.088 |      0.069 | 0.586 |    1.000 |  0.652 | {"mlp": 9, "lgbm": 4, "rf": 5, "et": 2, "logreg": 1}  |
| aroon_25         | directional          | S1_filter  |         0.104 |    0.494 |      0.390 | 0.021 |    1.000 |  0.046 | {"logreg": 4, "mlp": 2, "et": 10, "rf": 5}            |
| aroon_25         | directional          | S2_regime  |         0.104 |   -0.172 |     -0.277 | 0.051 |    1.000 |  0.097 | {"et": 5, "rf": 4, "logreg": 7, "lgbm": 4, "mlp": 1}  |
| aroon_25         | directional          | S3_sizing  |         0.104 |    0.597 |      0.493 | 0.000 |    0.000 |  0.000 | {"et": 13, "mlp": 1, "lgbm": 2, "rf": 2, "logreg": 3} |
| aroon_25         | directional          | S4_exit    |         0.104 |    0.425 |      0.321 | 0.051 |    1.000 |  0.097 | {"et": 10, "mlp": 2, "rf": 4, "lgbm": 3, "logreg": 2} |
| aroon_25         | directional          | S5_params  |         0.104 |    0.273 |      0.169 | 0.351 |    1.000 |  0.444 | {"mlp": 3, "logreg": 4, "lgbm": 10, "rf": 2, "et": 2} |
| elder_impulse    | oscillator_trend     | S1_filter  |        -0.720 |   -0.160 |      0.560 | 0.000 |    0.000 |  0.000 | {"logreg": 1, "lgbm": 4, "mlp": 4, "et": 5, "rf": 7}  |
| elder_impulse    | oscillator_trend     | S2_regime  |        -0.720 |   -0.517 |      0.203 | 0.086 |    1.000 |  0.143 | {"et": 3, "mlp": 3, "logreg": 9, "lgbm": 4, "rf": 2}  |
| elder_impulse    | oscillator_trend     | S3_sizing  |        -0.720 |   -0.289 |      0.431 | 0.001 |    0.083 |  0.003 | {"logreg": 6, "et": 4, "mlp": 2, "rf": 5, "lgbm": 4}  |
| elder_impulse    | oscillator_trend     | S4_exit    |        -0.720 |   -0.389 |      0.331 | 0.007 |    0.903 |  0.020 | {"logreg": 8, "mlp": 3, "rf": 5, "et": 2, "lgbm": 3}  |
| elder_impulse    | oscillator_trend     | S5_params  |        -0.720 |   -0.627 |      0.093 | 0.100 |    1.000 |  0.157 | {"lgbm": 3, "mlp": 2, "rf": 12, "et": 1, "logreg": 3} |
| ichimoku         | directional          | S1_filter  |         0.194 |    0.577 |      0.383 | 0.002 |    0.296 |  0.008 | {"et": 12, "lgbm": 3, "logreg": 3, "rf": 3}           |
| ichimoku         | directional          | S2_regime  |         0.194 |    0.093 |     -0.101 | 0.488 |    1.000 |  0.564 | {"rf": 5, "logreg": 8, "et": 4, "lgbm": 2, "mlp": 2}  |
| ichimoku         | directional          | S3_sizing  |         0.194 |    0.613 |      0.419 | 0.001 |    0.158 |  0.005 | {"rf": 5, "mlp": 2, "lgbm": 3, "et": 8, "logreg": 3}  |
| ichimoku         | directional          | S4_exit    |         0.194 |    0.541 |      0.347 | 0.010 |    1.000 |  0.026 | {"et": 11, "mlp": 3, "lgbm": 1, "logreg": 4, "rf": 2} |
| ichimoku         | directional          | S5_params  |         0.194 |    0.359 |      0.165 | 0.130 |    1.000 |  0.188 | {"rf": 3, "lgbm": 11, "logreg": 5, "mlp": 1, "et": 1} |
| vortex_14        | directional          | S1_filter  |        -0.004 |    0.459 |      0.463 | 0.002 |    0.228 |  0.006 | {"et": 11, "rf": 3, "mlp": 2, "logreg": 4, "lgbm": 1} |
| vortex_14        | directional          | S2_regime  |        -0.004 |   -0.098 |     -0.094 | 0.479 |    1.000 |  0.560 | {"et": 5, "rf": 2, "mlp": 1, "logreg": 7, "lgbm": 6}  |
| vortex_14        | directional          | S3_sizing  |        -0.004 |    0.518 |      0.522 | 0.000 |    0.000 |  0.000 | {"rf": 4, "et": 8, "lgbm": 2, "logreg": 5, "mlp": 2}  |
| vortex_14        | directional          | S4_exit    |        -0.004 |    0.388 |      0.392 | 0.009 |    1.000 |  0.023 | {"lgbm": 3, "rf": 3, "mlp": 6, "logreg": 4, "et": 5}  |
| vortex_14        | directional          | S5_params  |        -0.004 |    0.093 |      0.097 | 0.571 |    1.000 |  0.639 | {"logreg": 4, "et": 5, "lgbm": 9, "rf": 2, "mlp": 1}  |
| heikin_ashi_3    | directional          | S1_filter  |        -0.382 |    0.222 |      0.604 | 0.001 |    0.083 |  0.003 | {"et": 10, "mlp": 5, "logreg": 4, "rf": 1, "lgbm": 1} |
| heikin_ashi_3    | directional          | S2_regime  |        -0.382 |   -0.329 |      0.053 | 0.677 |    1.000 |  0.738 | {"et": 2, "rf": 3, "mlp": 1, "lgbm": 5, "logreg": 10} |
| heikin_ashi_3    | directional          | S3_sizing  |        -0.382 |    0.092 |      0.473 | 0.001 |    0.083 |  0.003 | {"logreg": 7, "et": 6, "lgbm": 2, "rf": 5, "mlp": 1}  |
| heikin_ashi_3    | directional          | S4_exit    |        -0.382 |   -0.023 |      0.358 | 0.012 |    1.000 |  0.030 | {"logreg": 8, "et": 7, "mlp": 2, "rf": 4}             |
| heikin_ashi_3    | directional          | S5_params  |        -0.382 |    0.030 |      0.412 | 0.059 |    1.000 |  0.108 | {"rf": 4, "logreg": 6, "mlp": 2, "et": 2, "lgbm": 7}  |