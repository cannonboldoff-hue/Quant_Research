| strategy         | strategy_family      | ml_stage   |   base_sharpe |   sharpe |   d_sharpe |     p |   p_holm |   p_bh | selected_models                                      |
|:-----------------|:---------------------|:-----------|--------------:|---------:|-----------:|------:|---------:|-------:|:-----------------------------------------------------|
| faber_10m        | price_vs_ma          | S1_filter  |         0.608 |    0.608 |      0.000 | 1.000 |    1.000 |  1.000 | {"logreg": 6}                                        |
| faber_10m        | price_vs_ma          | S2_regime  |         0.608 |    0.085 |     -0.523 | 0.002 |    0.282 |  0.037 | {"mlp": 2, "lgbm": 1, "logreg": 1, "et": 2}          |
| faber_10m        | price_vs_ma          | S3_sizing  |         0.608 |    0.414 |     -0.194 | 0.252 |    1.000 |  0.603 | {"lgbm": 2, "et": 2, "logreg": 2}                    |
| faber_10m        | price_vs_ma          | S4_exit    |         0.608 |    0.967 |      0.359 | 0.332 |    1.000 |  0.670 | {"mlp": 1, "lgbm": 1, "et": 2, "rf": 1, "logreg": 1} |
| faber_10m        | price_vs_ma          | S5_params  |         0.608 |    0.599 |     -0.009 | 0.686 |    1.000 |  0.905 | {"rf": 3, "logreg": 3}                               |
| bll_vma_200      | price_vs_ma          | S1_filter  |         0.812 |    0.738 |     -0.074 | 0.673 |    1.000 |  0.905 | {"mlp": 2, "rf": 3, "lgbm": 1}                       |
| bll_vma_200      | price_vs_ma          | S2_regime  |         0.812 |    0.562 |     -0.250 | 0.215 |    1.000 |  0.603 | {"mlp": 3, "rf": 1, "et": 2}                         |
| bll_vma_200      | price_vs_ma          | S3_sizing  |         0.812 |    0.398 |     -0.414 | 0.030 |    1.000 |  0.220 | {"lgbm": 3, "logreg": 1, "mlp": 2}                   |
| bll_vma_200      | price_vs_ma          | S4_exit    |         0.812 |    0.507 |     -0.305 | 0.320 |    1.000 |  0.664 | {"mlp": 4, "logreg": 1, "lgbm": 1}                   |
| bll_vma_200      | price_vs_ma          | S5_params  |         0.812 |    0.749 |     -0.063 | 0.395 |    1.000 |  0.698 | {"et": 4, "lgbm": 1, "rf": 1}                        |
| dema_20_50       | ma_crossover         | S1_filter  |         0.854 |    1.115 |      0.260 | 0.254 |    1.000 |  0.603 | {"logreg": 1, "mlp": 2, "et": 1, "rf": 2}            |
| dema_20_50       | ma_crossover         | S2_regime  |         0.854 |    0.567 |     -0.287 | 0.108 |    1.000 |  0.458 | {"rf": 1, "mlp": 1, "logreg": 1, "lgbm": 1, "et": 2} |
| dema_20_50       | ma_crossover         | S3_sizing  |         0.854 |    0.614 |     -0.240 | 0.197 |    1.000 |  0.599 | {"lgbm": 1, "logreg": 2, "rf": 2, "et": 1}           |
| dema_20_50       | ma_crossover         | S4_exit    |         0.854 |    0.907 |      0.053 | 0.857 |    1.000 |  0.967 | {"logreg": 4, "et": 2}                               |
| dema_20_50       | ma_crossover         | S5_params  |         0.854 |    0.473 |     -0.381 | 0.244 |    1.000 |  0.603 | {"logreg": 2, "et": 1, "lgbm": 3}                    |
| ema_12_26        | ma_crossover         | S1_filter  |         0.518 |    0.729 |      0.211 | 0.294 |    1.000 |  0.637 | {"logreg": 3, "mlp": 2, "et": 1}                     |
| ema_12_26        | ma_crossover         | S2_regime  |         0.518 |    0.335 |     -0.183 | 0.319 |    1.000 |  0.664 | {"rf": 1, "mlp": 1, "logreg": 1, "lgbm": 1, "et": 2} |
| ema_12_26        | ma_crossover         | S3_sizing  |         0.518 |    0.528 |      0.010 | 0.963 |    1.000 |  0.999 | {"lgbm": 1, "rf": 1, "logreg": 2, "et": 2}           |
| ema_12_26        | ma_crossover         | S4_exit    |         0.518 |    0.790 |      0.272 | 0.340 |    1.000 |  0.670 | {"lgbm": 1, "rf": 1, "logreg": 3, "et": 1}           |
| ema_12_26        | ma_crossover         | S5_params  |         0.518 |    0.738 |      0.220 | 0.293 |    1.000 |  0.637 | {"lgbm": 5, "logreg": 1}                             |
| triple_ma_4_9_18 | ma_crossover         | S1_filter  |        -0.144 |    0.213 |      0.357 | 0.058 |    1.000 |  0.311 | {"et": 3, "mlp": 1, "rf": 1, "logreg": 1}            |
| triple_ma_4_9_18 | ma_crossover         | S2_regime  |        -0.144 |   -0.004 |      0.140 | 0.512 |    1.000 |  0.792 | {"rf": 2, "lgbm": 2, "mlp": 1, "et": 1}              |
| triple_ma_4_9_18 | ma_crossover         | S3_sizing  |        -0.144 |   -0.141 |      0.003 | 0.985 |    1.000 |  0.999 | {"lgbm": 2, "et": 1, "logreg": 1, "mlp": 1, "rf": 1} |
| triple_ma_4_9_18 | ma_crossover         | S4_exit    |        -0.144 |    0.059 |      0.203 | 0.154 |    1.000 |  0.509 | {"lgbm": 2, "et": 1, "logreg": 1, "mlp": 1, "rf": 1} |
| triple_ma_4_9_18 | ma_crossover         | S5_params  |        -0.144 |    0.366 |      0.510 | 0.067 |    1.000 |  0.316 | {"lgbm": 1, "logreg": 1, "et": 3, "mlp": 1}          |
| sma_50_200       | ma_crossover         | S1_filter  |         0.663 |    0.572 |     -0.091 | 0.510 |    1.000 |  0.792 | {"mlp": 3, "rf": 2, "lgbm": 1}                       |
| sma_50_200       | ma_crossover         | S2_regime  |         0.663 |    0.487 |     -0.176 | 0.238 |    1.000 |  0.603 | {"mlp": 3, "logreg": 1, "et": 2}                     |
| sma_50_200       | ma_crossover         | S3_sizing  |         0.663 |    0.124 |     -0.540 | 0.004 |    0.736 |  0.065 | {"lgbm": 2, "et": 1, "logreg": 1, "mlp": 2}          |
| sma_50_200       | ma_crossover         | S4_exit    |         0.663 |    0.410 |     -0.254 | 0.479 |    1.000 |  0.771 | {"rf": 2, "mlp": 1, "logreg": 2, "lgbm": 1}          |
| sma_50_200       | ma_crossover         | S5_params  |         0.663 |    0.613 |     -0.050 | 0.775 |    1.000 |  0.912 | {"et": 1, "mlp": 4, "rf": 1}                         |
| gmma             | ma_crossover         | S1_filter  |         0.631 |    0.846 |      0.215 | 0.250 |    1.000 |  0.603 | {"rf": 2, "et": 2, "mlp": 1, "logreg": 1}            |
| gmma             | ma_crossover         | S2_regime  |         0.631 |    0.503 |     -0.128 | 0.477 |    1.000 |  0.771 | {"rf": 1, "mlp": 1, "logreg": 1, "lgbm": 1, "et": 2} |
| gmma             | ma_crossover         | S3_sizing  |         0.631 |    0.720 |      0.089 | 0.622 |    1.000 |  0.885 | {"lgbm": 2, "logreg": 3, "et": 1}                    |
| gmma             | ma_crossover         | S4_exit    |         0.631 |    0.711 |      0.080 | 0.743 |    1.000 |  0.909 | {"lgbm": 4, "logreg": 1, "mlp": 1}                   |
| gmma             | ma_crossover         | S5_params  |         0.631 |    0.603 |     -0.028 | 0.868 |    1.000 |  0.967 | {"mlp": 1, "lgbm": 4, "logreg": 1}                   |
| jma_7_21         | ma_crossover         | S1_filter  |        -0.602 |    0.159 |      0.761 | 0.004 |    0.823 |  0.067 | {"rf": 3, "logreg": 1, "et": 1, "mlp": 1}            |
| jma_7_21         | ma_crossover         | S2_regime  |        -0.602 |   -0.392 |      0.210 | 0.371 |    1.000 |  0.695 | {"et": 3, "mlp": 2, "lgbm": 1}                       |
| jma_7_21         | ma_crossover         | S3_sizing  |        -0.602 |   -0.327 |      0.275 | 0.166 |    1.000 |  0.521 | {"mlp": 1, "et": 2, "logreg": 1, "rf": 1, "lgbm": 1} |
| jma_7_21         | ma_crossover         | S4_exit    |        -0.602 |   -0.426 |      0.176 | 0.385 |    1.000 |  0.698 | {"rf": 2, "et": 1, "logreg": 2, "lgbm": 1}           |
| jma_7_21         | ma_crossover         | S5_params  |        -0.602 |    0.159 |      0.761 | 0.047 |    1.000 |  0.270 | {"et": 3, "logreg": 1, "rf": 1, "lgbm": 1}           |
| zlema_10_30      | ma_crossover         | S1_filter  |        -0.324 |    0.553 |      0.877 | 0.001 |    0.190 |  0.028 | {"lgbm": 1, "logreg": 2, "mlp": 1, "rf": 2}          |
| zlema_10_30      | ma_crossover         | S2_regime  |        -0.324 |   -0.240 |      0.085 | 0.714 |    1.000 |  0.906 | {"et": 3, "mlp": 2, "rf": 1}                         |
| zlema_10_30      | ma_crossover         | S3_sizing  |        -0.324 |   -0.276 |      0.049 | 0.781 |    1.000 |  0.912 | {"rf": 3, "et": 1, "lgbm": 1, "mlp": 1}              |
| zlema_10_30      | ma_crossover         | S4_exit    |        -0.324 |   -0.150 |      0.175 | 0.383 |    1.000 |  0.698 | {"rf": 3, "et": 1, "mlp": 2}                         |
| zlema_10_30      | ma_crossover         | S5_params  |        -0.324 |    0.376 |      0.701 | 0.114 |    1.000 |  0.463 | {"lgbm": 1, "rf": 2, "et": 3}                        |
| kama_10          | adaptive_ma          | S1_filter  |        -1.053 |   -0.396 |      0.657 | 0.002 |    0.374 |  0.039 | {"rf": 3, "lgbm": 1, "et": 1, "mlp": 1}              |
| kama_10          | adaptive_ma          | S2_regime  |        -1.053 |   -0.514 |      0.539 | 0.029 |    1.000 |  0.220 | {"rf": 2, "mlp": 2, "et": 1, "lgbm": 1}              |
| kama_10          | adaptive_ma          | S3_sizing  |        -1.053 |   -0.986 |      0.067 | 0.715 |    1.000 |  0.906 | {"lgbm": 1, "logreg": 2, "mlp": 1, "et": 2}          |
| kama_10          | adaptive_ma          | S4_exit    |        -1.053 |   -0.982 |      0.071 | 0.738 |    1.000 |  0.909 | {"rf": 2, "mlp": 2, "logreg": 2}                     |
| kama_10          | adaptive_ma          | S5_params  |        -1.053 |   -0.416 |      0.638 | 0.001 |    0.190 |  0.028 | {"mlp": 1, "et": 3, "lgbm": 1, "logreg": 1}          |
| turtle_55_20     | breakout             | S1_filter  |         0.455 |    0.594 |      0.139 | 0.564 |    1.000 |  0.846 | {"mlp": 3, "et": 1, "logreg": 2}                     |
| turtle_55_20     | breakout             | S2_regime  |         0.455 |    0.185 |     -0.270 | 0.142 |    1.000 |  0.493 | {"rf": 1, "mlp": 1, "logreg": 1, "et": 3}            |
| turtle_55_20     | breakout             | S3_sizing  |         0.455 |    0.322 |     -0.133 | 0.486 |    1.000 |  0.771 | {"lgbm": 3, "mlp": 1, "rf": 2}                       |
| turtle_55_20     | breakout             | S4_exit    |         0.455 |    0.509 |      0.054 | 0.849 |    1.000 |  0.967 | {"lgbm": 3, "mlp": 1, "rf": 2}                       |
| turtle_55_20     | breakout             | S5_params  |         0.455 |    0.436 |     -0.019 | 0.927 |    1.000 |  0.977 | {"lgbm": 2, "mlp": 1, "et": 2, "logreg": 1}          |
| vidya_14         | adaptive_ma          | S1_filter  |        -0.133 |    0.159 |      0.291 | 0.141 |    1.000 |  0.493 | {"logreg": 1, "lgbm": 1, "et": 3, "mlp": 1}          |
| vidya_14         | adaptive_ma          | S2_regime  |        -0.133 |   -0.215 |     -0.083 | 0.725 |    1.000 |  0.906 | {"rf": 1, "mlp": 2, "lgbm": 2, "et": 1}              |
| vidya_14         | adaptive_ma          | S3_sizing  |        -0.133 |    0.104 |      0.237 | 0.143 |    1.000 |  0.493 | {"lgbm": 3, "logreg": 3}                             |
| vidya_14         | adaptive_ma          | S4_exit    |        -0.133 |    0.026 |      0.158 | 0.465 |    1.000 |  0.771 | {"rf": 1, "lgbm": 2, "logreg": 2, "mlp": 1}          |
| vidya_14         | adaptive_ma          | S5_params  |        -0.133 |    0.173 |      0.305 | 0.062 |    1.000 |  0.312 | {"logreg": 1, "mlp": 1, "rf": 1, "lgbm": 2, "et": 1} |
| hma_55_slope     | adaptive_ma          | S1_filter  |         0.687 |    0.844 |      0.157 | 0.398 |    1.000 |  0.698 | {"rf": 2, "mlp": 2, "lgbm": 2}                       |
| hma_55_slope     | adaptive_ma          | S2_regime  |         0.687 |    0.505 |     -0.182 | 0.265 |    1.000 |  0.609 | {"et": 2, "lgbm": 2, "logreg": 1, "rf": 1}           |
| hma_55_slope     | adaptive_ma          | S3_sizing  |         0.687 |    0.491 |     -0.196 | 0.258 |    1.000 |  0.605 | {"lgbm": 4, "mlp": 2}                                |
| hma_55_slope     | adaptive_ma          | S4_exit    |         0.687 |    0.834 |      0.147 | 0.507 |    1.000 |  0.792 | {"rf": 1, "et": 1, "mlp": 2, "logreg": 2}            |
| hma_55_slope     | adaptive_ma          | S5_params  |         0.687 |    0.702 |      0.015 | 0.971 |    1.000 |  0.999 | {"lgbm": 1, "logreg": 1, "mlp": 3, "rf": 1}          |
| frama_16         | adaptive_ma          | S1_filter  |        -1.194 |   -0.527 |      0.667 | 0.004 |    0.647 |  0.062 | {"logreg": 3, "mlp": 2, "et": 1}                     |
| frama_16         | adaptive_ma          | S2_regime  |        -1.194 |   -0.676 |      0.518 | 0.043 |    1.000 |  0.270 | {"rf": 2, "mlp": 2, "et": 1, "lgbm": 1}              |
| frama_16         | adaptive_ma          | S3_sizing  |        -1.194 |   -0.891 |      0.303 | 0.100 |    1.000 |  0.441 | {"lgbm": 4, "mlp": 1, "rf": 1}                       |
| frama_16         | adaptive_ma          | S4_exit    |        -1.194 |   -0.875 |      0.319 | 0.070 |    1.000 |  0.323 | {"rf": 3, "lgbm": 2, "et": 1}                        |
| frama_16         | adaptive_ma          | S5_params  |        -1.194 |   -0.947 |      0.247 | 0.210 |    1.000 |  0.603 | {"rf": 2, "et": 1, "lgbm": 3}                        |
| turtle_20_10     | breakout             | S1_filter  |         0.462 |    0.828 |      0.366 | 0.078 |    1.000 |  0.354 | {"logreg": 3, "et": 1, "rf": 2}                      |
| turtle_20_10     | breakout             | S2_regime  |         0.462 |    0.322 |     -0.140 | 0.449 |    1.000 |  0.756 | {"rf": 1, "mlp": 1, "logreg": 1, "lgbm": 2, "et": 1} |
| turtle_20_10     | breakout             | S3_sizing  |         0.462 |    0.710 |      0.248 | 0.171 |    1.000 |  0.528 | {"lgbm": 2, "logreg": 1, "et": 3}                    |
| turtle_20_10     | breakout             | S4_exit    |         0.462 |    1.062 |      0.600 | 0.021 |    1.000 |  0.195 | {"lgbm": 2, "rf": 2, "et": 1, "logreg": 1}           |
| turtle_20_10     | breakout             | S5_params  |         0.462 |    0.550 |      0.088 | 0.664 |    1.000 |  0.904 | {"lgbm": 3, "rf": 2, "et": 1}                        |
| mcginley_14      | adaptive_ma          | S1_filter  |        -0.141 |    0.088 |      0.229 | 0.244 |    1.000 |  0.603 | {"logreg": 3, "mlp": 2, "et": 1}                     |
| mcginley_14      | adaptive_ma          | S2_regime  |        -0.141 |   -0.225 |     -0.083 | 0.657 |    1.000 |  0.904 | {"et": 2, "mlp": 3, "lgbm": 1}                       |
| mcginley_14      | adaptive_ma          | S3_sizing  |        -0.141 |   -0.366 |     -0.225 | 0.247 |    1.000 |  0.603 | {"lgbm": 2, "et": 2, "logreg": 2}                    |
| mcginley_14      | adaptive_ma          | S4_exit    |        -0.141 |   -0.356 |     -0.215 | 0.344 |    1.000 |  0.670 | {"et": 2, "mlp": 1, "logreg": 2, "lgbm": 1}          |
| mcginley_14      | adaptive_ma          | S5_params  |        -0.141 |    0.090 |      0.231 | 0.230 |    1.000 |  0.603 | {"logreg": 3, "mlp": 1, "et": 2}                     |
| donchian_20      | breakout             | S1_filter  |         0.484 |    0.520 |      0.036 | 0.885 |    1.000 |  0.968 | {"logreg": 2, "et": 2, "rf": 1, "lgbm": 1}           |
| donchian_20      | breakout             | S2_regime  |         0.484 |    0.234 |     -0.250 | 0.149 |    1.000 |  0.503 | {"rf": 1, "mlp": 1, "logreg": 1, "lgbm": 1, "et": 2} |
| donchian_20      | breakout             | S3_sizing  |         0.484 |    0.393 |     -0.091 | 0.626 |    1.000 |  0.885 | {"lgbm": 2, "et": 2, "logreg": 2}                    |
| donchian_20      | breakout             | S4_exit    |         0.484 |    0.460 |     -0.024 | 0.916 |    1.000 |  0.976 | {"logreg": 3, "mlp": 1, "et": 1, "lgbm": 1}          |
| donchian_20      | breakout             | S5_params  |         0.484 |    0.445 |     -0.039 | 0.777 |    1.000 |  0.912 | {"et": 3, "logreg": 2, "mlp": 1}                     |
| t3_20            | adaptive_ma          | S1_filter  |         0.289 |    0.717 |      0.429 | 0.035 |    1.000 |  0.244 | {"lgbm": 4, "mlp": 1, "rf": 1}                       |
| t3_20            | adaptive_ma          | S2_regime  |         0.289 |    0.363 |      0.074 | 0.697 |    1.000 |  0.905 | {"rf": 2, "mlp": 1, "logreg": 1, "et": 1, "lgbm": 1} |
| t3_20            | adaptive_ma          | S3_sizing  |         0.289 |    0.087 |     -0.202 | 0.252 |    1.000 |  0.603 | {"lgbm": 4, "et": 1, "mlp": 1}                       |
| t3_20            | adaptive_ma          | S4_exit    |         0.289 |    0.150 |     -0.139 | 0.423 |    1.000 |  0.736 | {"rf": 1, "et": 1, "mlp": 2, "lgbm": 1, "logreg": 1} |
| t3_20            | adaptive_ma          | S5_params  |         0.289 |    0.386 |      0.097 | 0.723 |    1.000 |  0.906 | {"logreg": 2, "lgbm": 1, "mlp": 2, "et": 1}          |
| keltner_20       | breakout             | S1_filter  |         0.704 |    1.304 |      0.600 | 0.011 |    1.000 |  0.132 | {"mlp": 1, "rf": 2, "logreg": 1, "et": 2}            |
| keltner_20       | breakout             | S2_regime  |         0.704 |    0.556 |     -0.148 | 0.483 |    1.000 |  0.771 | {"rf": 2, "mlp": 1, "logreg": 2, "et": 1}            |
| keltner_20       | breakout             | S3_sizing  |         0.704 |    0.816 |      0.112 | 0.584 |    1.000 |  0.857 | {"lgbm": 2, "et": 2, "rf": 1, "logreg": 1}           |
| keltner_20       | breakout             | S4_exit    |         0.704 |    0.894 |      0.189 | 0.475 |    1.000 |  0.771 | {"rf": 1, "et": 2, "logreg": 2, "lgbm": 1}           |
| keltner_20       | breakout             | S5_params  |         0.704 |    0.790 |      0.086 | 0.592 |    1.000 |  0.857 | {"lgbm": 2, "rf": 1, "et": 3}                        |
| bollinger_20_2   | breakout             | S1_filter  |         0.550 |    1.141 |      0.592 | 0.001 |    0.096 |  0.019 | {"lgbm": 2, "et": 4}                                 |
| bollinger_20_2   | breakout             | S2_regime  |         0.550 |    0.526 |     -0.024 | 0.908 |    1.000 |  0.976 | {"rf": 1, "mlp": 2, "logreg": 1, "et": 1, "lgbm": 1} |
| bollinger_20_2   | breakout             | S3_sizing  |         0.550 |    0.671 |      0.122 | 0.450 |    1.000 |  0.756 | {"rf": 2, "lgbm": 2, "et": 2}                        |
| bollinger_20_2   | breakout             | S4_exit    |         0.550 |    0.855 |      0.305 | 0.121 |    1.000 |  0.472 | {"rf": 2, "et": 2, "lgbm": 2}                        |
| bollinger_20_2   | breakout             | S5_params  |         0.550 |    0.788 |      0.238 | 0.250 |    1.000 |  0.603 | {"lgbm": 2, "logreg": 2, "et": 2}                    |
| abs_mom_12m      | time_series_momentum | S1_filter  |         0.407 |    0.407 |      0.000 | 1.000 |    1.000 |  1.000 | {"logreg": 6}                                        |
| abs_mom_12m      | time_series_momentum | S2_regime  |         0.407 |   -0.105 |     -0.512 | 0.002 |    0.374 |  0.039 | {"logreg": 3, "lgbm": 1, "et": 2}                    |
| abs_mom_12m      | time_series_momentum | S3_sizing  |         0.407 |    0.164 |     -0.243 | 0.113 |    1.000 |  0.463 | {"lgbm": 3, "et": 1, "logreg": 1, "rf": 1}           |
| abs_mom_12m      | time_series_momentum | S4_exit    |         0.407 |    0.299 |     -0.109 | 0.778 |    1.000 |  0.912 | {"logreg": 1, "et": 4, "rf": 1}                      |
| abs_mom_12m      | time_series_momentum | S5_params  |         0.407 |    0.392 |     -0.015 | 0.830 |    1.000 |  0.958 | {"logreg": 2, "mlp": 1, "rf": 1, "lgbm": 2}          |
| supertrend_10_3  | volatility_stop      | S1_filter  |         0.558 |    1.073 |      0.515 | 0.015 |    1.000 |  0.149 | {"rf": 1, "logreg": 2, "lgbm": 3}                    |
| supertrend_10_3  | volatility_stop      | S2_regime  |         0.558 |    0.523 |     -0.035 | 0.877 |    1.000 |  0.968 | {"rf": 1, "mlp": 4, "et": 1}                         |
| supertrend_10_3  | volatility_stop      | S3_sizing  |         0.558 |    0.491 |     -0.067 | 0.746 |    1.000 |  0.909 | {"lgbm": 2, "et": 2, "logreg": 2}                    |
| supertrend_10_3  | volatility_stop      | S4_exit    |         0.558 |    0.610 |      0.053 | 0.865 |    1.000 |  0.967 | {"lgbm": 2, "et": 2, "logreg": 2}                    |
| supertrend_10_3  | volatility_stop      | S5_params  |         0.558 |    0.549 |     -0.008 | 0.683 |    1.000 |  0.905 | {"mlp": 2, "lgbm": 2, "logreg": 2}                   |
| tsmom_252        | time_series_momentum | S1_filter  |         0.227 |    0.082 |     -0.146 | 0.525 |    1.000 |  0.800 | {"et": 1, "mlp": 4, "logreg": 1}                     |
| tsmom_252        | time_series_momentum | S2_regime  |         0.227 |   -0.188 |     -0.415 | 0.027 |    1.000 |  0.220 | {"mlp": 2, "et": 3, "lgbm": 1}                       |
| tsmom_252        | time_series_momentum | S3_sizing  |         0.227 |   -0.202 |     -0.430 | 0.020 |    1.000 |  0.195 | {"lgbm": 1, "mlp": 2, "logreg": 2, "et": 1}          |
| tsmom_252        | time_series_momentum | S4_exit    |         0.227 |   -0.212 |     -0.439 | 0.282 |    1.000 |  0.633 | {"lgbm": 2, "mlp": 2, "logreg": 1, "rf": 1}          |
| tsmom_252        | time_series_momentum | S5_params  |         0.227 |    0.315 |      0.088 | 0.585 |    1.000 |  0.857 | {"mlp": 1, "logreg": 3, "et": 1, "lgbm": 1}          |
| psar             | volatility_stop      | S1_filter  |        -0.031 |    0.790 |      0.821 | 0.001 |    0.096 |  0.019 | {"lgbm": 2, "et": 2, "logreg": 1, "rf": 1}           |
| psar             | volatility_stop      | S2_regime  |        -0.031 |   -0.052 |     -0.021 | 0.920 |    1.000 |  0.976 | {"et": 3, "mlp": 2, "lgbm": 1}                       |
| psar             | volatility_stop      | S3_sizing  |        -0.031 |    0.055 |      0.086 | 0.605 |    1.000 |  0.867 | {"lgbm": 1, "et": 2, "rf": 2, "mlp": 1}              |
| psar             | volatility_stop      | S4_exit    |        -0.031 |    0.261 |      0.293 | 0.141 |    1.000 |  0.493 | {"rf": 5, "mlp": 1}                                  |
| psar             | volatility_stop      | S5_params  |        -0.031 |    0.233 |      0.265 | 0.329 |    1.000 |  0.670 | {"mlp": 2, "logreg": 1, "et": 3}                     |
| chande_kroll     | volatility_stop      | S1_filter  |         0.708 |    0.836 |      0.129 | 0.544 |    1.000 |  0.823 | {"lgbm": 2, "et": 2, "mlp": 1, "logreg": 1}          |
| chande_kroll     | volatility_stop      | S2_regime  |         0.708 |    0.606 |     -0.102 | 0.593 |    1.000 |  0.857 | {"rf": 1, "lgbm": 2, "logreg": 1, "mlp": 1, "et": 1} |
| chande_kroll     | volatility_stop      | S3_sizing  |         0.708 |    0.677 |     -0.031 | 0.860 |    1.000 |  0.967 | {"lgbm": 4, "mlp": 1, "et": 1}                       |
| chande_kroll     | volatility_stop      | S4_exit    |         0.708 |    1.057 |      0.349 | 0.121 |    1.000 |  0.472 | {"lgbm": 1, "mlp": 3, "logreg": 1, "rf": 1}          |
| chande_kroll     | volatility_stop      | S5_params  |         0.708 |    0.686 |     -0.021 | 0.959 |    1.000 |  0.999 | {"lgbm": 2, "logreg": 1, "rf": 2, "mlp": 1}          |
| chandelier_22_3  | volatility_stop      | S1_filter  |         0.090 |    0.789 |      0.700 | 0.000 |    0.000 |  0.000 | {"logreg": 1, "lgbm": 3, "rf": 2}                    |
| chandelier_22_3  | volatility_stop      | S2_regime  |         0.090 |    0.408 |      0.318 | 0.144 |    1.000 |  0.493 | {"rf": 1, "mlp": 3, "et": 1, "lgbm": 1}              |
| chandelier_22_3  | volatility_stop      | S3_sizing  |         0.090 |    0.110 |      0.020 | 0.912 |    1.000 |  0.976 | {"lgbm": 1, "logreg": 3, "mlp": 1, "rf": 1}          |
| chandelier_22_3  | volatility_stop      | S4_exit    |         0.090 |   -0.018 |     -0.108 | 0.657 |    1.000 |  0.904 | {"rf": 2, "lgbm": 1, "logreg": 2, "mlp": 1}          |
| chandelier_22_3  | volatility_stop      | S5_params  |         0.090 |    0.132 |      0.042 | 0.046 |    1.000 |  0.270 | {"et": 1, "logreg": 2, "lgbm": 1, "rf": 2}           |
| tsmom_multi      | time_series_momentum | S1_filter  |         0.175 |    0.574 |      0.398 | 0.044 |    1.000 |  0.270 | {"lgbm": 2, "mlp": 3, "et": 1}                       |
| tsmom_multi      | time_series_momentum | S2_regime  |         0.175 |   -0.152 |     -0.327 | 0.102 |    1.000 |  0.442 | {"et": 3, "mlp": 2, "lgbm": 1}                       |
| tsmom_multi      | time_series_momentum | S3_sizing  |         0.175 |   -0.041 |     -0.217 | 0.228 |    1.000 |  0.603 | {"lgbm": 2, "et": 1, "logreg": 2, "rf": 1}           |
| tsmom_multi      | time_series_momentum | S4_exit    |         0.175 |    0.210 |      0.035 | 0.889 |    1.000 |  0.968 | {"logreg": 2, "mlp": 2, "et": 1, "lgbm": 1}          |
| tsmom_multi      | time_series_momentum | S5_params  |         0.175 |    0.399 |      0.223 | 0.344 |    1.000 |  0.670 | {"lgbm": 1, "rf": 2, "mlp": 1, "logreg": 1, "et": 1} |
| linreg_63        | time_series_momentum | S1_filter  |         0.473 |    0.285 |     -0.187 | 0.305 |    1.000 |  0.654 | {"logreg": 1, "rf": 2, "lgbm": 1, "mlp": 2}          |
| linreg_63        | time_series_momentum | S2_regime  |         0.473 |    0.206 |     -0.266 | 0.138 |    1.000 |  0.493 | {"mlp": 2, "logreg": 1, "et": 3}                     |
| linreg_63        | time_series_momentum | S3_sizing  |         0.473 |    0.008 |     -0.465 | 0.009 |    1.000 |  0.123 | {"lgbm": 1, "logreg": 3, "mlp": 2}                   |
| linreg_63        | time_series_momentum | S4_exit    |         0.473 |    0.373 |     -0.100 | 0.767 |    1.000 |  0.912 | {"lgbm": 3, "logreg": 1, "rf": 2}                    |
| linreg_63        | time_series_momentum | S5_params  |         0.473 |    0.305 |     -0.168 | 0.480 |    1.000 |  0.771 | {"logreg": 2, "et": 1, "mlp": 2, "rf": 1}            |
| macd_12_26_9     | oscillator_trend     | S1_filter  |        -0.169 |    0.317 |      0.486 | 0.042 |    1.000 |  0.270 | {"lgbm": 2, "logreg": 1, "rf": 2, "et": 1}           |
| macd_12_26_9     | oscillator_trend     | S2_regime  |        -0.169 |   -0.079 |      0.090 | 0.662 |    1.000 |  0.904 | {"et": 2, "logreg": 1, "rf": 1, "mlp": 1, "lgbm": 1} |
| macd_12_26_9     | oscillator_trend     | S3_sizing  |        -0.169 |   -0.166 |      0.004 | 0.981 |    1.000 |  0.999 | {"lgbm": 1, "et": 3, "mlp": 1, "logreg": 1}          |
| macd_12_26_9     | oscillator_trend     | S4_exit    |        -0.169 |   -0.146 |      0.023 | 0.918 |    1.000 |  0.976 | {"rf": 2, "et": 1, "mlp": 3}                         |
| macd_12_26_9     | oscillator_trend     | S5_params  |        -0.169 |    0.627 |      0.796 | 0.030 |    1.000 |  0.220 | {"lgbm": 1, "logreg": 1, "et": 2, "mlp": 2}          |
| trix_15          | oscillator_trend     | S1_filter  |         0.309 |    0.604 |      0.295 | 0.219 |    1.000 |  0.603 | {"lgbm": 2, "et": 1, "mlp": 3}                       |
| trix_15          | oscillator_trend     | S2_regime  |         0.309 |    0.127 |     -0.182 | 0.397 |    1.000 |  0.698 | {"et": 2, "logreg": 1, "rf": 2, "lgbm": 1}           |
| trix_15          | oscillator_trend     | S3_sizing  |         0.309 |   -0.156 |     -0.465 | 0.014 |    1.000 |  0.149 | {"lgbm": 3, "et": 2, "mlp": 1}                       |
| trix_15          | oscillator_trend     | S4_exit    |         0.309 |    0.023 |     -0.286 | 0.243 |    1.000 |  0.603 | {"lgbm": 3, "et": 1, "logreg": 1, "rf": 1}           |
| trix_15          | oscillator_trend     | S5_params  |         0.309 |    0.497 |      0.187 | 0.694 |    1.000 |  0.905 | {"lgbm": 2, "rf": 1, "logreg": 2, "et": 1}           |
| adx_dmi_14       | directional          | S1_filter  |         0.247 |    0.417 |      0.169 | 0.273 |    1.000 |  0.619 | {"mlp": 3, "et": 1, "rf": 1, "lgbm": 1}              |
| adx_dmi_14       | directional          | S2_regime  |         0.247 |    0.004 |     -0.243 | 0.349 |    1.000 |  0.674 | {"rf": 1, "mlp": 2, "lgbm": 2, "logreg": 1}          |
| adx_dmi_14       | directional          | S3_sizing  |         0.247 |    0.221 |     -0.026 | 0.887 |    1.000 |  0.968 | {"lgbm": 2, "et": 3, "rf": 1}                        |
| adx_dmi_14       | directional          | S4_exit    |         0.247 |    0.236 |     -0.012 | 0.951 |    1.000 |  0.996 | {"rf": 3, "et": 2, "logreg": 1}                      |
| adx_dmi_14       | directional          | S5_params  |         0.247 |    0.488 |      0.240 | 0.368 |    1.000 |  0.695 | {"lgbm": 4, "et": 2}                                 |
| kst              | oscillator_trend     | S1_filter  |         0.088 |    0.391 |      0.303 | 0.166 |    1.000 |  0.521 | {"lgbm": 4, "rf": 1, "mlp": 1}                       |
| kst              | oscillator_trend     | S2_regime  |         0.088 |   -0.006 |     -0.094 | 0.667 |    1.000 |  0.904 | {"lgbm": 2, "logreg": 1, "rf": 2, "et": 1}           |
| kst              | oscillator_trend     | S3_sizing  |         0.088 |   -0.345 |     -0.433 | 0.025 |    1.000 |  0.217 | {"lgbm": 2, "et": 2, "mlp": 1, "logreg": 1}          |
| kst              | oscillator_trend     | S4_exit    |         0.088 |   -0.208 |     -0.296 | 0.230 |    1.000 |  0.603 | {"logreg": 3, "et": 1, "mlp": 1, "rf": 1}            |
| kst              | oscillator_trend     | S5_params  |         0.088 |    0.605 |      0.517 | 0.232 |    1.000 |  0.603 | {"lgbm": 1, "et": 4, "rf": 1}                        |
| tsi_25_13        | oscillator_trend     | S1_filter  |         0.051 |    0.696 |      0.645 | 0.007 |    1.000 |  0.104 | {"logreg": 2, "et": 2, "mlp": 2}                     |
| tsi_25_13        | oscillator_trend     | S2_regime  |         0.051 |    0.137 |      0.086 | 0.708 |    1.000 |  0.906 | {"et": 1, "logreg": 2, "rf": 1, "mlp": 1, "lgbm": 1} |
| tsi_25_13        | oscillator_trend     | S3_sizing  |         0.051 |    0.053 |      0.002 | 0.989 |    1.000 |  0.999 | {"lgbm": 2, "mlp": 1, "logreg": 1, "rf": 2}          |
| tsi_25_13        | oscillator_trend     | S4_exit    |         0.051 |    0.462 |      0.411 | 0.046 |    1.000 |  0.270 | {"rf": 1, "et": 1, "mlp": 1, "logreg": 3}            |
| tsi_25_13        | oscillator_trend     | S5_params  |         0.051 |   -0.041 |     -0.092 | 0.732 |    1.000 |  0.909 | {"mlp": 3, "logreg": 1, "lgbm": 1, "et": 1}          |
| cci_20_100       | oscillator_trend     | S1_filter  |         0.557 |    0.912 |      0.355 | 0.062 |    1.000 |  0.312 | {"logreg": 2, "et": 2, "mlp": 2}                     |
| cci_20_100       | oscillator_trend     | S2_regime  |         0.557 |    0.437 |     -0.120 | 0.571 |    1.000 |  0.849 | {"rf": 1, "mlp": 2, "logreg": 1, "et": 1, "lgbm": 1} |
| cci_20_100       | oscillator_trend     | S3_sizing  |         0.557 |    0.387 |     -0.170 | 0.319 |    1.000 |  0.664 | {"lgbm": 3, "et": 1, "mlp": 2}                       |
| cci_20_100       | oscillator_trend     | S4_exit    |         0.557 |    0.461 |     -0.096 | 0.693 |    1.000 |  0.905 | {"rf": 1, "et": 1, "logreg": 1, "mlp": 1, "lgbm": 2} |
| cci_20_100       | oscillator_trend     | S5_params  |         0.557 |    0.765 |      0.208 | 0.438 |    1.000 |  0.755 | {"lgbm": 1, "rf": 1, "et": 1, "logreg": 3}           |
| aroon_25         | directional          | S1_filter  |         0.273 |    0.238 |     -0.035 | 0.866 |    1.000 |  0.967 | {"rf": 3, "lgbm": 2, "et": 1}                        |
| aroon_25         | directional          | S2_regime  |         0.273 |    0.047 |     -0.226 | 0.226 |    1.000 |  0.603 | {"rf": 2, "mlp": 1, "logreg": 1, "et": 2}            |
| aroon_25         | directional          | S3_sizing  |         0.273 |    0.064 |     -0.208 | 0.266 |    1.000 |  0.609 | {"logreg": 2, "et": 1, "lgbm": 2, "rf": 1}           |
| aroon_25         | directional          | S4_exit    |         0.273 |    0.343 |      0.071 | 0.781 |    1.000 |  0.912 | {"lgbm": 2, "et": 1, "logreg": 3}                    |
| aroon_25         | directional          | S5_params  |         0.273 |    0.548 |      0.275 | 0.445 |    1.000 |  0.756 | {"mlp": 2, "lgbm": 3, "rf": 1}                       |
| elder_impulse    | oscillator_trend     | S1_filter  |        -1.776 |   -0.499 |      1.277 | 0.000 |    0.000 |  0.000 | {"lgbm": 1, "et": 2, "mlp": 2, "rf": 1}              |
| elder_impulse    | oscillator_trend     | S2_regime  |        -1.776 |   -1.248 |      0.528 | 0.011 |    1.000 |  0.132 | {"et": 2, "mlp": 3, "lgbm": 1}                       |
| elder_impulse    | oscillator_trend     | S3_sizing  |        -1.776 |   -1.497 |      0.279 | 0.143 |    1.000 |  0.493 | {"rf": 3, "logreg": 2, "mlp": 1}                     |
| elder_impulse    | oscillator_trend     | S4_exit    |        -1.776 |   -1.578 |      0.198 | 0.291 |    1.000 |  0.637 | {"rf": 2, "logreg": 2, "mlp": 1, "lgbm": 1}          |
| elder_impulse    | oscillator_trend     | S5_params  |        -1.776 |   -1.054 |      0.721 | 0.000 |    0.000 |  0.000 | {"et": 4, "rf": 1, "lgbm": 1}                        |
| ichimoku         | directional          | S1_filter  |         0.618 |    0.847 |      0.229 | 0.163 |    1.000 |  0.521 | {"logreg": 1, "lgbm": 3, "rf": 1, "et": 1}           |
| ichimoku         | directional          | S2_regime  |         0.618 |    0.498 |     -0.119 | 0.525 |    1.000 |  0.800 | {"et": 3, "mlp": 1, "lgbm": 2}                       |
| ichimoku         | directional          | S3_sizing  |         0.618 |    0.553 |     -0.065 | 0.722 |    1.000 |  0.906 | {"lgbm": 2, "mlp": 2, "et": 1, "logreg": 1}          |
| ichimoku         | directional          | S4_exit    |         0.618 |    0.508 |     -0.110 | 0.630 |    1.000 |  0.885 | {"logreg": 4, "rf": 1, "mlp": 1}                     |
| ichimoku         | directional          | S5_params  |         0.618 |    0.429 |     -0.188 | 0.373 |    1.000 |  0.695 | {"mlp": 1, "rf": 1, "lgbm": 3, "logreg": 1}          |
| vortex_14        | directional          | S1_filter  |         0.050 |    0.428 |      0.379 | 0.059 |    1.000 |  0.311 | {"rf": 1, "et": 2, "lgbm": 2, "logreg": 1}           |
| vortex_14        | directional          | S2_regime  |         0.050 |   -0.134 |     -0.184 | 0.395 |    1.000 |  0.698 | {"rf": 2, "mlp": 2, "et": 1, "lgbm": 1}              |
| vortex_14        | directional          | S3_sizing  |         0.050 |   -0.303 |     -0.353 | 0.058 |    1.000 |  0.311 | {"lgbm": 2, "et": 2, "mlp": 2}                       |
| vortex_14        | directional          | S4_exit    |         0.050 |   -0.277 |     -0.326 | 0.066 |    1.000 |  0.316 | {"lgbm": 1, "et": 2, "logreg": 2, "mlp": 1}          |
| vortex_14        | directional          | S5_params  |         0.050 |   -0.019 |     -0.068 | 0.828 |    1.000 |  0.958 | {"et": 1, "logreg": 1, "mlp": 2, "rf": 1, "lgbm": 1} |
| heikin_ashi_3    | directional          | S1_filter  |        -0.506 |    0.039 |      0.545 | 0.030 |    1.000 |  0.220 | {"lgbm": 1, "rf": 4, "mlp": 1}                       |
| heikin_ashi_3    | directional          | S2_regime  |        -0.506 |   -0.436 |      0.070 | 0.774 |    1.000 |  0.912 | {"et": 3, "mlp": 2, "lgbm": 1}                       |
| heikin_ashi_3    | directional          | S3_sizing  |        -0.506 |   -0.508 |     -0.002 | 0.989 |    1.000 |  0.999 | {"lgbm": 2, "logreg": 1, "mlp": 1, "et": 1, "rf": 1} |
| heikin_ashi_3    | directional          | S4_exit    |        -0.506 |   -0.309 |      0.197 | 0.334 |    1.000 |  0.670 | {"lgbm": 1, "logreg": 2, "rf": 2, "et": 1}           |
| heikin_ashi_3    | directional          | S5_params  |        -0.506 |   -0.149 |      0.357 | 0.374 |    1.000 |  0.695 | {"rf": 1, "logreg": 3, "mlp": 1, "lgbm": 1}          |