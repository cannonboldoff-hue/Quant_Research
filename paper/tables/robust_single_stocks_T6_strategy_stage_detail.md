| strategy         | strategy_family      | ml_stage   |   base_sharpe |   sharpe |   d_sharpe |     p |   p_holm |   p_bh | selected_models                                       |
|:-----------------|:---------------------|:-----------|--------------:|---------:|-----------:|------:|---------:|-------:|:------------------------------------------------------|
| faber_10m        | price_vs_ma          | S1_filter  |         1.178 |    1.102 |     -0.075 | 0.258 |    1.000 |  0.295 | {"rf": 5, "et": 2, "mlp": 9, "lgbm": 5}               |
| faber_10m        | price_vs_ma          | S2_regime  |         1.178 |    0.964 |     -0.214 | 0.022 |    1.000 |  0.032 | {"logreg": 9, "lgbm": 3, "et": 5, "mlp": 3, "rf": 1}  |
| faber_10m        | price_vs_ma          | S3_sizing  |         1.178 |    1.054 |     -0.124 | 0.075 |    1.000 |  0.098 | {"et": 2, "mlp": 6, "lgbm": 7, "logreg": 4, "rf": 2}  |
| faber_10m        | price_vs_ma          | S4_exit    |         1.178 |    0.759 |     -0.419 | 0.032 |    1.000 |  0.044 | {"logreg": 5, "et": 2, "mlp": 5, "lgbm": 6, "rf": 3}  |
| faber_10m        | price_vs_ma          | S5_params  |         1.178 |    1.199 |      0.021 | 0.513 |    1.000 |  0.550 | {"mlp": 9, "lgbm": 4, "et": 3, "rf": 2, "logreg": 3}  |
| sma_50_200       | ma_crossover         | S1_filter  |         0.568 |    0.988 |      0.420 | 0.004 |    0.273 |  0.006 | {"logreg": 7, "rf": 5, "mlp": 2, "et": 3, "lgbm": 4}  |
| sma_50_200       | ma_crossover         | S2_regime  |         0.568 |    0.388 |     -0.180 | 0.082 |    1.000 |  0.106 | {"lgbm": 8, "logreg": 8, "mlp": 4, "et": 1}           |
| sma_50_200       | ma_crossover         | S3_sizing  |         0.568 |    0.999 |      0.431 | 0.002 |    0.128 |  0.003 | {"rf": 4, "logreg": 4, "mlp": 2, "lgbm": 6, "et": 5}  |
| sma_50_200       | ma_crossover         | S4_exit    |         0.568 |    0.868 |      0.300 | 0.139 |    1.000 |  0.172 | {"et": 5, "mlp": 3, "logreg": 7, "lgbm": 3, "rf": 3}  |
| sma_50_200       | ma_crossover         | S5_params  |         0.568 |    0.907 |      0.339 | 0.010 |    0.680 |  0.015 | {"rf": 5, "mlp": 5, "et": 4, "lgbm": 5, "logreg": 2}  |
| gmma             | ma_crossover         | S1_filter  |         0.181 |    1.142 |      0.961 | 0.000 |    0.000 |  0.000 | {"et": 11, "lgbm": 5, "logreg": 2, "mlp": 1, "rf": 2} |
| gmma             | ma_crossover         | S2_regime  |         0.181 |    0.220 |      0.039 | 0.765 |    1.000 |  0.790 | {"logreg": 6, "et": 1, "lgbm": 7, "mlp": 4, "rf": 3}  |
| gmma             | ma_crossover         | S3_sizing  |         0.181 |    0.957 |      0.776 | 0.000 |    0.000 |  0.000 | {"mlp": 3, "et": 8, "lgbm": 5, "rf": 4, "logreg": 1}  |
| gmma             | ma_crossover         | S4_exit    |         0.181 |    0.928 |      0.747 | 0.000 |    0.000 |  0.000 | {"mlp": 1, "lgbm": 7, "logreg": 4, "rf": 3, "et": 6}  |
| gmma             | ma_crossover         | S5_params  |         0.181 |    0.478 |      0.297 | 0.004 |    0.337 |  0.007 | {"lgbm": 5, "logreg": 6, "rf": 4, "et": 1, "mlp": 5}  |
| bll_vma_200      | price_vs_ma          | S1_filter  |         0.517 |    1.112 |      0.595 | 0.000 |    0.000 |  0.000 | {"et": 11, "mlp": 4, "logreg": 4, "lgbm": 2}          |
| bll_vma_200      | price_vs_ma          | S2_regime  |         0.517 |    0.376 |     -0.141 | 0.179 |    1.000 |  0.212 | {"logreg": 9, "lgbm": 5, "mlp": 3, "et": 2, "rf": 2}  |
| bll_vma_200      | price_vs_ma          | S3_sizing  |         0.517 |    1.056 |      0.539 | 0.001 |    0.046 |  0.001 | {"mlp": 8, "lgbm": 4, "et": 4, "rf": 2, "logreg": 3}  |
| bll_vma_200      | price_vs_ma          | S4_exit    |         0.517 |    0.803 |      0.285 | 0.141 |    1.000 |  0.173 | {"rf": 6, "mlp": 2, "lgbm": 8, "et": 4, "logreg": 1}  |
| bll_vma_200      | price_vs_ma          | S5_params  |         0.517 |    0.745 |      0.227 | 0.018 |    1.000 |  0.026 | {"lgbm": 3, "mlp": 11, "et": 2, "rf": 3, "logreg": 2} |
| ema_12_26        | ma_crossover         | S1_filter  |         0.139 |    1.089 |      0.950 | 0.000 |    0.000 |  0.000 | {"et": 8, "lgbm": 3, "rf": 4, "logreg": 2, "mlp": 4}  |
| ema_12_26        | ma_crossover         | S2_regime  |         0.139 |    0.211 |      0.072 | 0.568 |    1.000 |  0.599 | {"logreg": 6, "et": 2, "lgbm": 5, "mlp": 4, "rf": 4}  |
| ema_12_26        | ma_crossover         | S3_sizing  |         0.139 |    1.024 |      0.884 | 0.000 |    0.000 |  0.000 | {"mlp": 1, "lgbm": 8, "rf": 4, "et": 6, "logreg": 2}  |
| ema_12_26        | ma_crossover         | S4_exit    |         0.139 |    0.869 |      0.730 | 0.000 |    0.000 |  0.000 | {"mlp": 2, "et": 8, "lgbm": 8, "logreg": 1, "rf": 2}  |
| ema_12_26        | ma_crossover         | S5_params  |         0.139 |    0.489 |      0.350 | 0.006 |    0.444 |  0.010 | {"lgbm": 7, "logreg": 7, "mlp": 4, "et": 3}           |
| dema_20_50       | ma_crossover         | S1_filter  |        -0.235 |    0.999 |      1.234 | 0.000 |    0.000 |  0.000 | {"et": 10, "lgbm": 2, "rf": 2, "logreg": 7}           |
| dema_20_50       | ma_crossover         | S2_regime  |        -0.235 |   -0.410 |     -0.175 | 0.154 |    1.000 |  0.185 | {"lgbm": 6, "logreg": 6, "mlp": 5, "rf": 3, "et": 1}  |
| dema_20_50       | ma_crossover         | S3_sizing  |        -0.235 |    0.699 |      0.935 | 0.000 |    0.000 |  0.000 | {"lgbm": 4, "et": 4, "mlp": 1, "logreg": 5, "rf": 7}  |
| dema_20_50       | ma_crossover         | S4_exit    |        -0.235 |    0.734 |      0.969 | 0.000 |    0.000 |  0.000 | {"lgbm": 7, "rf": 6, "logreg": 4, "et": 4}            |
| dema_20_50       | ma_crossover         | S5_params  |        -0.235 |    0.607 |      0.842 | 0.000 |    0.000 |  0.000 | {"logreg": 9, "mlp": 2, "et": 4, "rf": 1, "lgbm": 5}  |
| triple_ma_4_9_18 | ma_crossover         | S1_filter  |        -0.459 |    0.738 |      1.197 | 0.000 |    0.000 |  0.000 | {"et": 12, "logreg": 4, "rf": 5}                      |
| triple_ma_4_9_18 | ma_crossover         | S2_regime  |        -0.459 |   -0.337 |      0.122 | 0.415 |    1.000 |  0.455 | {"logreg": 5, "lgbm": 6, "et": 3, "mlp": 5, "rf": 2}  |
| triple_ma_4_9_18 | ma_crossover         | S3_sizing  |        -0.459 |    0.471 |      0.930 | 0.000 |    0.000 |  0.000 | {"et": 8, "lgbm": 3, "logreg": 3, "rf": 7}            |
| triple_ma_4_9_18 | ma_crossover         | S4_exit    |        -0.459 |    0.549 |      1.008 | 0.000 |    0.000 |  0.000 | {"rf": 6, "lgbm": 7, "mlp": 1, "logreg": 3, "et": 4}  |
| triple_ma_4_9_18 | ma_crossover         | S5_params  |        -0.459 |   -0.014 |      0.446 | 0.004 |    0.304 |  0.007 | {"logreg": 6, "lgbm": 6, "et": 4, "mlp": 3, "rf": 2}  |
| zlema_10_30      | ma_crossover         | S1_filter  |        -0.613 |    0.793 |      1.406 | 0.000 |    0.000 |  0.000 | {"et": 9, "rf": 5, "logreg": 2, "lgbm": 4, "mlp": 1}  |
| zlema_10_30      | ma_crossover         | S2_regime  |        -0.613 |   -0.490 |      0.123 | 0.364 |    1.000 |  0.404 | {"rf": 4, "logreg": 7, "lgbm": 5, "et": 4, "mlp": 1}  |
| zlema_10_30      | ma_crossover         | S3_sizing  |        -0.613 |    0.221 |      0.834 | 0.000 |    0.000 |  0.000 | {"et": 7, "rf": 3, "lgbm": 6, "logreg": 5}            |
| zlema_10_30      | ma_crossover         | S4_exit    |        -0.613 |   -0.094 |      0.519 | 0.006 |    0.468 |  0.010 | {"et": 8, "mlp": 5, "lgbm": 5, "rf": 2, "logreg": 1}  |
| zlema_10_30      | ma_crossover         | S5_params  |        -0.613 |    0.119 |      0.732 | 0.000 |    0.000 |  0.000 | {"et": 9, "mlp": 5, "lgbm": 1, "rf": 2, "logreg": 4}  |
| jma_7_21         | ma_crossover         | S1_filter  |        -0.565 |    0.785 |      1.350 | 0.000 |    0.000 |  0.000 | {"et": 13, "logreg": 3, "lgbm": 2, "rf": 3}           |
| jma_7_21         | ma_crossover         | S2_regime  |        -0.565 |   -0.225 |      0.340 | 0.011 |    0.715 |  0.016 | {"lgbm": 8, "logreg": 7, "mlp": 1, "et": 4, "rf": 1}  |
| jma_7_21         | ma_crossover         | S3_sizing  |        -0.565 |    0.287 |      0.852 | 0.000 |    0.000 |  0.000 | {"et": 7, "lgbm": 3, "logreg": 7, "rf": 4}            |
| jma_7_21         | ma_crossover         | S4_exit    |        -0.565 |   -0.032 |      0.533 | 0.004 |    0.273 |  0.006 | {"rf": 3, "logreg": 6, "et": 7, "lgbm": 3, "mlp": 2}  |
| jma_7_21         | ma_crossover         | S5_params  |        -0.565 |   -0.195 |      0.370 | 0.029 |    1.000 |  0.041 | {"lgbm": 2, "rf": 6, "et": 1, "mlp": 7, "logreg": 5}  |
| turtle_20_10     | breakout             | S1_filter  |        -0.089 |    1.025 |      1.114 | 0.000 |    0.000 |  0.000 | {"logreg": 3, "et": 6, "mlp": 3, "rf": 6, "lgbm": 3}  |
| turtle_20_10     | breakout             | S2_regime  |        -0.089 |    0.011 |      0.100 | 0.453 |    1.000 |  0.493 | {"logreg": 6, "et": 3, "lgbm": 6, "rf": 2, "mlp": 4}  |
| turtle_20_10     | breakout             | S3_sizing  |        -0.089 |    0.871 |      0.960 | 0.000 |    0.000 |  0.000 | {"mlp": 1, "et": 10, "lgbm": 2, "logreg": 3, "rf": 5} |
| turtle_20_10     | breakout             | S4_exit    |        -0.089 |    0.841 |      0.930 | 0.000 |    0.000 |  0.000 | {"mlp": 3, "lgbm": 6, "et": 7, "logreg": 1, "rf": 4}  |
| turtle_20_10     | breakout             | S5_params  |        -0.089 |    0.259 |      0.348 | 0.007 |    0.532 |  0.012 | {"logreg": 7, "et": 8, "rf": 4, "lgbm": 1, "mlp": 1}  |
| turtle_55_20     | breakout             | S1_filter  |         0.076 |    0.913 |      0.837 | 0.000 |    0.000 |  0.000 | {"rf": 4, "mlp": 6, "lgbm": 2, "logreg": 4, "et": 5}  |
| turtle_55_20     | breakout             | S2_regime  |         0.076 |    0.082 |      0.006 | 0.969 |    1.000 |  0.969 | {"logreg": 9, "lgbm": 3, "mlp": 5, "rf": 4}           |
| turtle_55_20     | breakout             | S3_sizing  |         0.076 |    0.849 |      0.773 | 0.000 |    0.000 |  0.000 | {"mlp": 2, "logreg": 4, "lgbm": 6, "rf": 2, "et": 7}  |
| turtle_55_20     | breakout             | S4_exit    |         0.076 |    0.873 |      0.797 | 0.000 |    0.000 |  0.000 | {"rf": 5, "lgbm": 8, "et": 8}                         |
| turtle_55_20     | breakout             | S5_params  |         0.076 |    0.372 |      0.296 | 0.018 |    1.000 |  0.026 | {"logreg": 8, "lgbm": 7, "et": 1, "mlp": 1, "rf": 4}  |
| donchian_20      | breakout             | S1_filter  |         0.008 |    0.991 |      0.983 | 0.000 |    0.000 |  0.000 | {"et": 11, "mlp": 2, "lgbm": 3, "rf": 4, "logreg": 1} |
| donchian_20      | breakout             | S2_regime  |         0.008 |   -0.115 |     -0.124 | 0.365 |    1.000 |  0.404 | {"mlp": 7, "logreg": 5, "et": 3, "rf": 3, "lgbm": 3}  |
| donchian_20      | breakout             | S3_sizing  |         0.008 |    0.860 |      0.852 | 0.000 |    0.000 |  0.000 | {"mlp": 3, "et": 8, "lgbm": 5, "rf": 3, "logreg": 2}  |
| donchian_20      | breakout             | S4_exit    |         0.008 |    0.886 |      0.877 | 0.000 |    0.000 |  0.000 | {"mlp": 4, "lgbm": 5, "et": 9, "logreg": 2, "rf": 1}  |
| donchian_20      | breakout             | S5_params  |         0.008 |    0.193 |      0.184 | 0.090 |    1.000 |  0.116 | {"lgbm": 7, "mlp": 5, "rf": 1, "logreg": 5, "et": 3}  |
| hma_55_slope     | adaptive_ma          | S1_filter  |        -0.001 |    1.053 |      1.054 | 0.000 |    0.000 |  0.000 | {"logreg": 4, "et": 7, "lgbm": 3, "rf": 5, "mlp": 2}  |
| hma_55_slope     | adaptive_ma          | S2_regime  |        -0.001 |   -0.032 |     -0.031 | 0.792 |    1.000 |  0.809 | {"mlp": 4, "logreg": 5, "rf": 2, "lgbm": 7, "et": 3}  |
| hma_55_slope     | adaptive_ma          | S3_sizing  |        -0.001 |    0.929 |      0.930 | 0.000 |    0.000 |  0.000 | {"rf": 5, "et": 6, "lgbm": 6, "logreg": 4}            |
| hma_55_slope     | adaptive_ma          | S4_exit    |        -0.001 |    0.751 |      0.752 | 0.000 |    0.000 |  0.000 | {"lgbm": 6, "rf": 4, "logreg": 3, "et": 7, "mlp": 1}  |
| hma_55_slope     | adaptive_ma          | S5_params  |        -0.001 |    0.336 |      0.338 | 0.054 |    1.000 |  0.073 | {"et": 14, "mlp": 2, "rf": 3, "lgbm": 1, "logreg": 1} |
| bollinger_20_2   | breakout             | S1_filter  |        -0.141 |    0.903 |      1.044 | 0.000 |    0.000 |  0.000 | {"rf": 7, "et": 7, "logreg": 1, "lgbm": 6}            |
| bollinger_20_2   | breakout             | S2_regime  |        -0.141 |    0.019 |      0.160 | 0.259 |    1.000 |  0.295 | {"mlp": 4, "logreg": 4, "rf": 4, "et": 4, "lgbm": 5}  |
| bollinger_20_2   | breakout             | S3_sizing  |        -0.141 |    0.755 |      0.896 | 0.000 |    0.000 |  0.000 | {"mlp": 2, "et": 9, "lgbm": 3, "logreg": 2, "rf": 5}  |
| bollinger_20_2   | breakout             | S4_exit    |        -0.141 |    0.845 |      0.987 | 0.000 |    0.000 |  0.000 | {"mlp": 3, "rf": 6, "lgbm": 7, "et": 4, "logreg": 1}  |
| bollinger_20_2   | breakout             | S5_params  |        -0.141 |    0.287 |      0.428 | 0.002 |    0.128 |  0.003 | {"et": 8, "logreg": 1, "lgbm": 7, "rf": 2, "mlp": 3}  |
| vidya_14         | adaptive_ma          | S1_filter  |        -0.169 |    0.884 |      1.053 | 0.000 |    0.000 |  0.000 | {"et": 12, "rf": 4, "logreg": 3, "lgbm": 2}           |
| vidya_14         | adaptive_ma          | S2_regime  |        -0.169 |    0.032 |      0.201 | 0.149 |    1.000 |  0.181 | {"mlp": 5, "logreg": 5, "et": 2, "lgbm": 9}           |
| vidya_14         | adaptive_ma          | S3_sizing  |        -0.169 |    0.828 |      0.997 | 0.000 |    0.000 |  0.000 | {"mlp": 1, "logreg": 7, "rf": 8, "et": 2, "lgbm": 3}  |
| vidya_14         | adaptive_ma          | S4_exit    |        -0.169 |    0.534 |      0.703 | 0.001 |    0.046 |  0.001 | {"lgbm": 6, "rf": 7, "mlp": 3, "logreg": 3, "et": 2}  |
| vidya_14         | adaptive_ma          | S5_params  |        -0.169 |    0.128 |      0.297 | 0.002 |    0.162 |  0.003 | {"logreg": 8, "lgbm": 2, "mlp": 3, "et": 4, "rf": 4}  |
| t3_20            | adaptive_ma          | S1_filter  |        -0.129 |    0.929 |      1.058 | 0.000 |    0.000 |  0.000 | {"rf": 2, "logreg": 6, "et": 9, "lgbm": 4}            |
| t3_20            | adaptive_ma          | S2_regime  |        -0.129 |    0.149 |      0.278 | 0.045 |    1.000 |  0.062 | {"lgbm": 9, "logreg": 6, "et": 3, "rf": 2, "mlp": 1}  |
| t3_20            | adaptive_ma          | S3_sizing  |        -0.129 |    0.850 |      0.979 | 0.000 |    0.000 |  0.000 | {"logreg": 7, "lgbm": 6, "rf": 4, "et": 4}            |
| t3_20            | adaptive_ma          | S4_exit    |        -0.129 |    0.792 |      0.921 | 0.000 |    0.000 |  0.000 | {"logreg": 6, "lgbm": 7, "rf": 4, "mlp": 1, "et": 3}  |
| t3_20            | adaptive_ma          | S5_params  |        -0.129 |    0.329 |      0.458 | 0.009 |    0.621 |  0.014 | {"et": 9, "lgbm": 6, "logreg": 4, "rf": 2}            |
| kama_10          | adaptive_ma          | S1_filter  |        -0.561 |    0.597 |      1.157 | 0.000 |    0.000 |  0.000 | {"et": 18, "mlp": 2, "lgbm": 1}                       |
| kama_10          | adaptive_ma          | S2_regime  |        -0.561 |   -0.110 |      0.451 | 0.001 |    0.046 |  0.001 | {"mlp": 2, "lgbm": 10, "logreg": 4, "et": 4, "rf": 1} |
| kama_10          | adaptive_ma          | S3_sizing  |        -0.561 |    0.438 |      0.999 | 0.000 |    0.000 |  0.000 | {"logreg": 7, "et": 5, "rf": 7, "lgbm": 1, "mlp": 1}  |
| kama_10          | adaptive_ma          | S4_exit    |        -0.561 |    0.265 |      0.826 | 0.000 |    0.000 |  0.000 | {"mlp": 3, "rf": 7, "et": 5, "lgbm": 4, "logreg": 2}  |
| kama_10          | adaptive_ma          | S5_params  |        -0.561 |   -0.240 |      0.321 | 0.002 |    0.128 |  0.003 | {"lgbm": 2, "rf": 8, "et": 6, "logreg": 4, "mlp": 1}  |
| frama_16         | adaptive_ma          | S1_filter  |        -0.856 |   -0.017 |      0.839 | 0.000 |    0.000 |  0.000 | {"et": 14, "mlp": 3, "rf": 1, "logreg": 2, "lgbm": 1} |
| frama_16         | adaptive_ma          | S2_regime  |        -0.856 |   -0.067 |      0.789 | 0.000 |    0.000 |  0.000 | {"mlp": 2, "logreg": 5, "et": 9, "rf": 1, "lgbm": 4}  |
| frama_16         | adaptive_ma          | S3_sizing  |        -0.856 |    0.128 |      0.985 | 0.000 |    0.000 |  0.000 | {"et": 11, "logreg": 3, "rf": 4, "lgbm": 3}           |
| frama_16         | adaptive_ma          | S4_exit    |        -0.856 |   -0.053 |      0.803 | 0.000 |    0.000 |  0.000 | {"rf": 3, "et": 7, "mlp": 2, "logreg": 2, "lgbm": 7}  |
| frama_16         | adaptive_ma          | S5_params  |        -0.856 |   -0.655 |      0.201 | 0.063 |    1.000 |  0.083 | {"mlp": 5, "et": 3, "rf": 5, "logreg": 2, "lgbm": 6}  |
| keltner_20       | breakout             | S1_filter  |        -0.130 |    0.861 |      0.991 | 0.000 |    0.000 |  0.000 | {"et": 3, "rf": 9, "logreg": 5, "lgbm": 3, "mlp": 1}  |
| keltner_20       | breakout             | S2_regime  |        -0.130 |   -0.160 |     -0.030 | 0.847 |    1.000 |  0.856 | {"logreg": 5, "et": 2, "mlp": 6, "lgbm": 6, "rf": 2}  |
| keltner_20       | breakout             | S3_sizing  |        -0.130 |    0.736 |      0.866 | 0.000 |    0.000 |  0.000 | {"mlp": 1, "et": 8, "logreg": 4, "lgbm": 1, "rf": 7}  |
| keltner_20       | breakout             | S4_exit    |        -0.130 |    0.852 |      0.982 | 0.000 |    0.000 |  0.000 | {"et": 11, "rf": 5, "lgbm": 4, "logreg": 1}           |
| keltner_20       | breakout             | S5_params  |        -0.130 |    0.285 |      0.416 | 0.000 |    0.000 |  0.000 | {"mlp": 7, "logreg": 3, "lgbm": 4, "et": 3, "rf": 4}  |
| abs_mom_12m      | time_series_momentum | S1_filter  |         1.123 |    1.158 |      0.035 | 0.510 |    1.000 |  0.549 | {"lgbm": 7, "logreg": 3, "mlp": 4, "et": 4, "rf": 3}  |
| abs_mom_12m      | time_series_momentum | S2_regime  |         1.123 |    0.972 |     -0.151 | 0.142 |    1.000 |  0.173 | {"logreg": 7, "lgbm": 3, "et": 4, "mlp": 5, "rf": 2}  |
| abs_mom_12m      | time_series_momentum | S3_sizing  |         1.123 |    1.076 |     -0.047 | 0.414 |    1.000 |  0.455 | {"logreg": 6, "mlp": 2, "lgbm": 8, "rf": 2, "et": 3}  |
| abs_mom_12m      | time_series_momentum | S4_exit    |         1.123 |    0.800 |     -0.322 | 0.097 |    1.000 |  0.124 | {"logreg": 2, "lgbm": 7, "rf": 3, "mlp": 5, "et": 4}  |
| abs_mom_12m      | time_series_momentum | S5_params  |         1.123 |    1.209 |      0.086 | 0.046 |    1.000 |  0.063 | {"logreg": 4, "rf": 1, "mlp": 8, "lgbm": 6, "et": 2}  |
| supertrend_10_3  | volatility_stop      | S1_filter  |        -0.043 |    1.050 |      1.093 | 0.000 |    0.000 |  0.000 | {"logreg": 9, "et": 4, "rf": 4, "lgbm": 2, "mlp": 2}  |
| supertrend_10_3  | volatility_stop      | S2_regime  |        -0.043 |   -0.016 |      0.027 | 0.812 |    1.000 |  0.825 | {"mlp": 5, "logreg": 5, "et": 1, "rf": 5, "lgbm": 5}  |
| supertrend_10_3  | volatility_stop      | S3_sizing  |        -0.043 |    0.898 |      0.941 | 0.000 |    0.000 |  0.000 | {"mlp": 1, "lgbm": 8, "rf": 4, "et": 6, "logreg": 2}  |
| supertrend_10_3  | volatility_stop      | S4_exit    |        -0.043 |    0.905 |      0.948 | 0.000 |    0.000 |  0.000 | {"mlp": 1, "lgbm": 7, "rf": 7, "logreg": 1, "et": 5}  |
| supertrend_10_3  | volatility_stop      | S5_params  |        -0.043 |    0.002 |      0.045 | 0.019 |    1.000 |  0.027 | {"lgbm": 11, "logreg": 3, "mlp": 7}                   |
| tsmom_252        | time_series_momentum | S1_filter  |         0.702 |    1.034 |      0.332 | 0.006 |    0.444 |  0.010 | {"logreg": 1, "rf": 4, "mlp": 2, "lgbm": 3, "et": 11} |
| tsmom_252        | time_series_momentum | S2_regime  |         0.702 |    0.581 |     -0.121 | 0.133 |    1.000 |  0.166 | {"mlp": 6, "lgbm": 5, "logreg": 4, "rf": 4, "et": 2}  |
| tsmom_252        | time_series_momentum | S3_sizing  |         0.702 |    1.042 |      0.340 | 0.007 |    0.532 |  0.012 | {"mlp": 5, "rf": 3, "lgbm": 7, "logreg": 5, "et": 1}  |
| tsmom_252        | time_series_momentum | S4_exit    |         0.702 |    0.964 |      0.262 | 0.132 |    1.000 |  0.165 | {"rf": 5, "lgbm": 9, "mlp": 4, "et": 3}               |
| tsmom_252        | time_series_momentum | S5_params  |         0.702 |    1.124 |      0.422 | 0.001 |    0.088 |  0.002 | {"logreg": 7, "mlp": 5, "lgbm": 8, "et": 1}           |
| chande_kroll     | volatility_stop      | S1_filter  |         0.096 |    1.138 |      1.041 | 0.000 |    0.000 |  0.000 | {"rf": 4, "mlp": 2, "logreg": 5, "lgbm": 5, "et": 5}  |
| chande_kroll     | volatility_stop      | S2_regime  |         0.096 |    0.287 |      0.190 | 0.156 |    1.000 |  0.187 | {"mlp": 6, "logreg": 5, "lgbm": 7, "rf": 2, "et": 1}  |
| chande_kroll     | volatility_stop      | S3_sizing  |         0.096 |    1.040 |      0.944 | 0.000 |    0.000 |  0.000 | {"mlp": 1, "et": 4, "lgbm": 5, "logreg": 6, "rf": 5}  |
| chande_kroll     | volatility_stop      | S4_exit    |         0.096 |    1.010 |      0.914 | 0.000 |    0.000 |  0.000 | {"et": 9, "rf": 6, "lgbm": 2, "mlp": 2, "logreg": 2}  |
| chande_kroll     | volatility_stop      | S5_params  |         0.096 |    0.362 |      0.266 | 0.111 |    1.000 |  0.140 | {"logreg": 5, "lgbm": 4, "et": 5, "mlp": 4, "rf": 3}  |
| psar             | volatility_stop      | S1_filter  |        -0.349 |    0.985 |      1.334 | 0.000 |    0.000 |  0.000 | {"et": 9, "mlp": 2, "lgbm": 6, "rf": 3, "logreg": 1}  |
| psar             | volatility_stop      | S2_regime  |        -0.349 |   -0.058 |      0.290 | 0.047 |    1.000 |  0.064 | {"mlp": 2, "logreg": 5, "et": 6, "rf": 2, "lgbm": 6}  |
| psar             | volatility_stop      | S3_sizing  |        -0.349 |    0.605 |      0.954 | 0.000 |    0.000 |  0.000 | {"mlp": 1, "et": 7, "lgbm": 6, "rf": 5, "logreg": 2}  |
| psar             | volatility_stop      | S4_exit    |        -0.349 |    0.234 |      0.583 | 0.001 |    0.088 |  0.002 | {"et": 8, "mlp": 3, "logreg": 1, "lgbm": 5, "rf": 4}  |
| psar             | volatility_stop      | S5_params  |        -0.349 |   -0.034 |      0.314 | 0.017 |    1.000 |  0.025 | {"et": 3, "rf": 7, "logreg": 7, "lgbm": 2, "mlp": 2}  |
| chandelier_22_3  | volatility_stop      | S1_filter  |        -0.219 |    0.497 |      0.716 | 0.000 |    0.000 |  0.000 | {"mlp": 5, "et": 9, "logreg": 2, "lgbm": 4, "rf": 1}  |
| chandelier_22_3  | volatility_stop      | S2_regime  |        -0.219 |   -0.021 |      0.197 | 0.134 |    1.000 |  0.166 | {"lgbm": 7, "logreg": 6, "rf": 3, "mlp": 2, "et": 3}  |
| chandelier_22_3  | volatility_stop      | S3_sizing  |        -0.219 |    0.773 |      0.991 | 0.000 |    0.000 |  0.000 | {"logreg": 5, "et": 4, "lgbm": 5, "rf": 7}            |
| chandelier_22_3  | volatility_stop      | S4_exit    |        -0.219 |    0.700 |      0.919 | 0.000 |    0.000 |  0.000 | {"logreg": 3, "et": 4, "rf": 3, "lgbm": 9, "mlp": 2}  |
| chandelier_22_3  | volatility_stop      | S5_params  |        -0.219 |   -0.175 |      0.044 | 0.242 |    1.000 |  0.281 | {"mlp": 6, "et": 3, "rf": 3, "lgbm": 5, "logreg": 4}  |
| tsmom_multi      | time_series_momentum | S1_filter  |         0.334 |    0.899 |      0.565 | 0.000 |    0.000 |  0.000 | {"logreg": 2, "et": 13, "lgbm": 4, "rf": 2}           |
| tsmom_multi      | time_series_momentum | S2_regime  |         0.334 |    0.317 |     -0.016 | 0.871 |    1.000 |  0.875 | {"et": 4, "logreg": 5, "lgbm": 6, "mlp": 4, "rf": 2}  |
| tsmom_multi      | time_series_momentum | S3_sizing  |         0.334 |    0.952 |      0.618 | 0.000 |    0.000 |  0.000 | {"rf": 6, "mlp": 1, "logreg": 6, "et": 7, "lgbm": 1}  |
| tsmom_multi      | time_series_momentum | S4_exit    |         0.334 |    0.848 |      0.515 | 0.002 |    0.128 |  0.003 | {"lgbm": 8, "rf": 7, "logreg": 2, "et": 3, "mlp": 1}  |
| tsmom_multi      | time_series_momentum | S5_params  |         0.334 |    0.668 |      0.335 | 0.001 |    0.046 |  0.001 | {"rf": 6, "et": 1, "lgbm": 6, "mlp": 4, "logreg": 4}  |
| linreg_63        | time_series_momentum | S1_filter  |         0.206 |    0.938 |      0.732 | 0.000 |    0.000 |  0.000 | {"rf": 8, "et": 7, "lgbm": 3, "logreg": 2, "mlp": 1}  |
| linreg_63        | time_series_momentum | S2_regime  |         0.206 |    0.084 |     -0.122 | 0.209 |    1.000 |  0.245 | {"logreg": 9, "lgbm": 6, "rf": 3, "mlp": 3}           |
| linreg_63        | time_series_momentum | S3_sizing  |         0.206 |    0.958 |      0.752 | 0.000 |    0.000 |  0.000 | {"mlp": 3, "logreg": 3, "rf": 5, "et": 6, "lgbm": 4}  |
| linreg_63        | time_series_momentum | S4_exit    |         0.206 |    0.813 |      0.606 | 0.001 |    0.088 |  0.002 | {"mlp": 6, "lgbm": 6, "rf": 2, "et": 3, "logreg": 4}  |
| linreg_63        | time_series_momentum | S5_params  |         0.206 |    0.688 |      0.482 | 0.010 |    0.680 |  0.015 | {"et": 5, "lgbm": 7, "logreg": 4, "mlp": 4, "rf": 1}  |
| macd_12_26_9     | oscillator_trend     | S1_filter  |        -0.355 |    0.970 |      1.325 | 0.000 |    0.000 |  0.000 | {"et": 11, "logreg": 3, "lgbm": 4, "rf": 2, "mlp": 1} |
| macd_12_26_9     | oscillator_trend     | S2_regime  |        -0.355 |   -0.185 |      0.170 | 0.201 |    1.000 |  0.236 | {"lgbm": 9, "rf": 3, "et": 4, "mlp": 2, "logreg": 3}  |
| macd_12_26_9     | oscillator_trend     | S3_sizing  |        -0.355 |    0.560 |      0.915 | 0.000 |    0.000 |  0.000 | {"logreg": 4, "et": 9, "lgbm": 5, "rf": 3}            |
| macd_12_26_9     | oscillator_trend     | S4_exit    |        -0.355 |    0.506 |      0.861 | 0.000 |    0.000 |  0.000 | {"logreg": 3, "lgbm": 5, "mlp": 1, "rf": 6, "et": 6}  |
| macd_12_26_9     | oscillator_trend     | S5_params  |        -0.355 |    0.194 |      0.549 | 0.010 |    0.680 |  0.015 | {"rf": 8, "et": 10, "mlp": 2, "logreg": 1}            |
| adx_dmi_14       | directional          | S1_filter  |        -0.322 |    0.683 |      1.004 | 0.000 |    0.000 |  0.000 | {"et": 15, "logreg": 4, "rf": 2}                      |
| adx_dmi_14       | directional          | S2_regime  |        -0.322 |   -0.384 |     -0.062 | 0.712 |    1.000 |  0.739 | {"mlp": 9, "logreg": 5, "rf": 4, "lgbm": 3}           |
| adx_dmi_14       | directional          | S3_sizing  |        -0.322 |    0.424 |      0.746 | 0.000 |    0.000 |  0.000 | {"mlp": 1, "et": 9, "lgbm": 3, "rf": 7, "logreg": 1}  |
| adx_dmi_14       | directional          | S4_exit    |        -0.322 |    0.389 |      0.711 | 0.000 |    0.000 |  0.000 | {"et": 6, "lgbm": 8, "mlp": 3, "rf": 4}               |
| adx_dmi_14       | directional          | S5_params  |        -0.322 |    0.242 |      0.564 | 0.002 |    0.162 |  0.003 | {"lgbm": 14, "mlp": 2, "logreg": 2, "et": 3}          |
| trix_15          | oscillator_trend     | S1_filter  |        -0.256 |    0.965 |      1.221 | 0.000 |    0.000 |  0.000 | {"et": 11, "logreg": 4, "rf": 2, "lgbm": 3, "mlp": 1} |
| trix_15          | oscillator_trend     | S2_regime  |        -0.256 |   -0.179 |      0.077 | 0.580 |    1.000 |  0.608 | {"lgbm": 7, "logreg": 1, "rf": 5, "et": 7, "mlp": 1}  |
| trix_15          | oscillator_trend     | S3_sizing  |        -0.256 |    0.632 |      0.888 | 0.000 |    0.000 |  0.000 | {"logreg": 4, "et": 7, "mlp": 1, "lgbm": 7, "rf": 2}  |
| trix_15          | oscillator_trend     | S4_exit    |        -0.256 |    0.630 |      0.886 | 0.000 |    0.000 |  0.000 | {"logreg": 4, "rf": 6, "lgbm": 4, "mlp": 2, "et": 5}  |
| trix_15          | oscillator_trend     | S5_params  |        -0.256 |    0.198 |      0.454 | 0.060 |    1.000 |  0.080 | {"logreg": 3, "lgbm": 11, "rf": 2, "mlp": 5}          |
| kst              | oscillator_trend     | S1_filter  |        -0.184 |    1.033 |      1.216 | 0.000 |    0.000 |  0.000 | {"et": 11, "logreg": 2, "rf": 6, "mlp": 2}            |
| kst              | oscillator_trend     | S2_regime  |        -0.184 |   -0.088 |      0.095 | 0.462 |    1.000 |  0.500 | {"logreg": 6, "lgbm": 7, "et": 6, "rf": 2}            |
| kst              | oscillator_trend     | S3_sizing  |        -0.184 |    0.727 |      0.911 | 0.000 |    0.000 |  0.000 | {"et": 10, "lgbm": 5, "logreg": 2, "mlp": 1, "rf": 3} |
| kst              | oscillator_trend     | S4_exit    |        -0.184 |    0.734 |      0.917 | 0.000 |    0.000 |  0.000 | {"logreg": 5, "et": 4, "rf": 8, "lgbm": 2, "mlp": 2}  |
| kst              | oscillator_trend     | S5_params  |        -0.184 |    0.107 |      0.291 | 0.176 |    1.000 |  0.210 | {"et": 9, "rf": 2, "lgbm": 4, "mlp": 6}               |
| cci_20_100       | oscillator_trend     | S1_filter  |         0.079 |    1.167 |      1.088 | 0.000 |    0.000 |  0.000 | {"rf": 6, "logreg": 9, "lgbm": 1, "et": 4, "mlp": 1}  |
| cci_20_100       | oscillator_trend     | S2_regime  |         0.079 |    0.196 |      0.116 | 0.321 |    1.000 |  0.363 | {"mlp": 4, "logreg": 4, "rf": 2, "et": 3, "lgbm": 8}  |
| cci_20_100       | oscillator_trend     | S3_sizing  |         0.079 |    0.941 |      0.862 | 0.000 |    0.000 |  0.000 | {"mlp": 2, "et": 5, "lgbm": 5, "logreg": 5, "rf": 4}  |
| cci_20_100       | oscillator_trend     | S4_exit    |         0.079 |    0.906 |      0.827 | 0.000 |    0.000 |  0.000 | {"mlp": 3, "et": 4, "lgbm": 4, "logreg": 3, "rf": 7}  |
| cci_20_100       | oscillator_trend     | S5_params  |         0.079 |    0.213 |      0.134 | 0.351 |    1.000 |  0.396 | {"rf": 5, "logreg": 6, "mlp": 1, "et": 7, "lgbm": 2}  |
| tsi_25_13        | oscillator_trend     | S1_filter  |        -0.373 |    0.804 |      1.177 | 0.000 |    0.000 |  0.000 | {"logreg": 4, "mlp": 2, "et": 8, "lgbm": 4, "rf": 3}  |
| tsi_25_13        | oscillator_trend     | S2_regime  |        -0.373 |   -0.160 |      0.213 | 0.086 |    1.000 |  0.111 | {"logreg": 5, "rf": 2, "lgbm": 8, "et": 6}            |
| tsi_25_13        | oscillator_trend     | S3_sizing  |        -0.373 |    0.548 |      0.921 | 0.000 |    0.000 |  0.000 | {"logreg": 4, "et": 6, "lgbm": 6, "rf": 4, "mlp": 1}  |
| tsi_25_13        | oscillator_trend     | S4_exit    |        -0.373 |    0.434 |      0.807 | 0.000 |    0.000 |  0.000 | {"logreg": 3, "rf": 9, "mlp": 3, "lgbm": 2, "et": 4}  |
| tsi_25_13        | oscillator_trend     | S5_params  |        -0.373 |   -0.194 |      0.180 | 0.256 |    1.000 |  0.295 | {"lgbm": 2, "logreg": 6, "mlp": 2, "et": 8, "rf": 3}  |
| aroon_25         | directional          | S1_filter  |         0.099 |    1.099 |      1.000 | 0.000 |    0.000 |  0.000 | {"rf": 5, "logreg": 8, "et": 5, "mlp": 1, "lgbm": 2}  |
| aroon_25         | directional          | S2_regime  |         0.099 |    0.025 |     -0.074 | 0.530 |    1.000 |  0.565 | {"logreg": 6, "rf": 3, "et": 6, "mlp": 3, "lgbm": 3}  |
| aroon_25         | directional          | S3_sizing  |         0.099 |    0.976 |      0.877 | 0.000 |    0.000 |  0.000 | {"et": 7, "lgbm": 7, "rf": 5, "logreg": 2}            |
| aroon_25         | directional          | S4_exit    |         0.099 |    1.014 |      0.915 | 0.000 |    0.000 |  0.000 | {"rf": 10, "mlp": 2, "et": 5, "lgbm": 3, "logreg": 1} |
| aroon_25         | directional          | S5_params  |         0.099 |    0.440 |      0.341 | 0.052 |    1.000 |  0.070 | {"lgbm": 8, "mlp": 3, "logreg": 3, "rf": 3, "et": 4}  |
| vortex_14        | directional          | S1_filter  |        -0.155 |    0.907 |      1.062 | 0.000 |    0.000 |  0.000 | {"mlp": 2, "rf": 6, "et": 10, "logreg": 3}            |
| vortex_14        | directional          | S2_regime  |        -0.155 |   -0.069 |      0.086 | 0.559 |    1.000 |  0.592 | {"logreg": 7, "rf": 2, "lgbm": 8, "et": 1, "mlp": 3}  |
| vortex_14        | directional          | S3_sizing  |        -0.155 |    0.813 |      0.969 | 0.000 |    0.000 |  0.000 | {"rf": 4, "et": 9, "logreg": 3, "lgbm": 5}            |
| vortex_14        | directional          | S4_exit    |        -0.155 |    0.677 |      0.832 | 0.000 |    0.000 |  0.000 | {"rf": 7, "lgbm": 6, "mlp": 3, "et": 5}               |
| vortex_14        | directional          | S5_params  |        -0.155 |    0.215 |      0.371 | 0.024 |    1.000 |  0.033 | {"et": 5, "logreg": 1, "rf": 6, "lgbm": 5, "mlp": 4}  |
| elder_impulse    | oscillator_trend     | S1_filter  |        -0.861 |    0.010 |      0.871 | 0.000 |    0.000 |  0.000 | {"et": 10, "logreg": 8, "mlp": 2, "lgbm": 1}          |
| elder_impulse    | oscillator_trend     | S2_regime  |        -0.861 |   -0.279 |      0.581 | 0.000 |    0.000 |  0.000 | {"rf": 2, "logreg": 8, "lgbm": 8, "et": 3}            |
| elder_impulse    | oscillator_trend     | S3_sizing  |        -0.861 |   -0.004 |      0.857 | 0.000 |    0.000 |  0.000 | {"rf": 4, "et": 3, "logreg": 10, "lgbm": 4}           |
| elder_impulse    | oscillator_trend     | S4_exit    |        -0.861 |    0.008 |      0.869 | 0.000 |    0.000 |  0.000 | {"et": 6, "rf": 5, "lgbm": 5, "logreg": 4, "mlp": 1}  |
| elder_impulse    | oscillator_trend     | S5_params  |        -0.861 |   -0.792 |      0.069 | 0.355 |    1.000 |  0.398 | {"rf": 2, "logreg": 5, "et": 11, "lgbm": 1, "mlp": 2} |
| ichimoku         | directional          | S1_filter  |         0.173 |    1.056 |      0.883 | 0.000 |    0.000 |  0.000 | {"mlp": 2, "rf": 7, "et": 7, "logreg": 5}             |
| ichimoku         | directional          | S2_regime  |         0.173 |    0.208 |      0.035 | 0.792 |    1.000 |  0.809 | {"mlp": 5, "logreg": 5, "et": 4, "lgbm": 4, "rf": 3}  |
| ichimoku         | directional          | S3_sizing  |         0.173 |    0.988 |      0.815 | 0.000 |    0.000 |  0.000 | {"et": 8, "lgbm": 3, "logreg": 4, "rf": 4, "mlp": 2}  |
| ichimoku         | directional          | S4_exit    |         0.173 |    1.048 |      0.875 | 0.000 |    0.000 |  0.000 | {"mlp": 2, "logreg": 3, "et": 8, "rf": 6, "lgbm": 2}  |
| ichimoku         | directional          | S5_params  |         0.173 |    0.484 |      0.311 | 0.011 |    0.715 |  0.016 | {"mlp": 1, "rf": 6, "logreg": 8, "lgbm": 5, "et": 1}  |
| heikin_ashi_3    | directional          | S1_filter  |        -0.586 |    0.785 |      1.371 | 0.000 |    0.000 |  0.000 | {"et": 9, "logreg": 6, "mlp": 2, "rf": 4}             |
| heikin_ashi_3    | directional          | S2_regime  |        -0.586 |   -0.527 |      0.059 | 0.621 |    1.000 |  0.648 | {"lgbm": 9, "logreg": 7, "et": 3, "mlp": 2}           |
| heikin_ashi_3    | directional          | S3_sizing  |        -0.586 |    0.355 |      0.941 | 0.000 |    0.000 |  0.000 | {"et": 13, "mlp": 2, "lgbm": 3, "rf": 2, "logreg": 1} |
| heikin_ashi_3    | directional          | S4_exit    |        -0.586 |    0.072 |      0.658 | 0.001 |    0.046 |  0.001 | {"mlp": 3, "et": 11, "rf": 3, "logreg": 1, "lgbm": 3} |
| heikin_ashi_3    | directional          | S5_params  |        -0.586 |    0.058 |      0.643 | 0.002 |    0.162 |  0.003 | {"logreg": 6, "et": 2, "mlp": 5, "rf": 2, "lgbm": 6}  |
| mcginley_14      | adaptive_ma          | S1_filter  |         0.114 |    0.993 |      0.879 | 0.000 |    0.000 |  0.000 | {"rf": 4, "et": 15, "logreg": 2}                      |
| mcginley_14      | adaptive_ma          | S2_regime  |         0.114 |    0.420 |      0.306 | 0.029 |    1.000 |  0.041 | {"logreg": 5, "lgbm": 9, "et": 2, "rf": 2, "mlp": 3}  |
| mcginley_14      | adaptive_ma          | S3_sizing  |         0.114 |    1.012 |      0.897 | 0.000 |    0.000 |  0.000 | {"mlp": 2, "logreg": 6, "lgbm": 5, "rf": 5, "et": 3}  |
| mcginley_14      | adaptive_ma          | S4_exit    |         0.114 |    0.706 |      0.591 | 0.000 |    0.000 |  0.000 | {"logreg": 6, "et": 4, "lgbm": 3, "rf": 5, "mlp": 3}  |
| mcginley_14      | adaptive_ma          | S5_params  |         0.114 |    0.511 |      0.396 | 0.000 |    0.000 |  0.000 | {"et": 4, "mlp": 6, "lgbm": 2, "rf": 7, "logreg": 2}  |