| feature         | family            | definition                                                   |
|:----------------|:------------------|:-------------------------------------------------------------|
| mom_5           | momentum          | 5-bar log return / (vol*sqrt(5))                             |
| mom_21          | momentum          | 21-bar vol-scaled log return                                 |
| mom_63          | momentum          | 63-bar vol-scaled log return                                 |
| mom_126         | momentum          | 126-bar vol-scaled log return                                |
| mom_252         | momentum          | 252-bar vol-scaled log return (Moskowitz et al. 2012)        |
| mom_252_21      | momentum          | 12-1 momentum, vol-scaled (Jegadeesh & Titman 1993)          |
| vol_20          | volatility        | 20-bar realised vol (annualised)                             |
| vol_ratio       | volatility        | realised vol 20 / realised vol 120                           |
| vol_of_vol      | volatility        | std of 20-bar vol over 60 bars / mean                        |
| parkinson_ratio | volatility        | Parkinson (1980) vol / close-close vol                       |
| gk_ratio        | volatility        | Garman-Klass (1980) vol / close-close vol                    |
| rs_ratio        | volatility        | Rogers-Satchell (1991) vol / close-close vol                 |
| yz_ratio        | volatility        | Yang-Zhang (2000) vol / close-close vol                      |
| atr_pct         | volatility        | ATR(14) / close (Wilder 1978)                                |
| adx             | trend_strength    | ADX(14) / 100 (Wilder 1978)                                  |
| di_diff         | trend_direction   | (+DI - -DI)/(+DI + -DI)                                      |
| aroon_osc       | trend_direction   | (Aroon up - down)/100, n=25 (Chande 1995)                    |
| rsi             | oscillator        | RSI(14)/100 - 0.5 (Wilder 1978)                              |
| macd_hist       | trend_direction   | MACD(12,26,9) histogram / ATR (Appel)                        |
| macd_line       | trend_direction   | MACD line / ATR                                              |
| stoch_k         | oscillator        | Stochastic %K(14)/100 - 0.5 (Lane)                           |
| cci             | oscillator        | CCI(20)/100 (Lambert 1980)                                   |
| trix            | trend_direction   | TRIX(15) z-scored by 252-bar std (Hutson 1983)               |
| vortex_diff     | trend_direction   | VI+ - VI- (Botes & Siepman 2010)                             |
| er_10           | trend_strength    | Kaufman efficiency ratio, 10 bars                            |
| er_63           | trend_strength    | Kaufman efficiency ratio, 63 bars                            |
| chop            | trend_strength    | Choppiness index(14)/100 (Dreiss)                            |
| lr_tstat_63     | trend_strength    | t-stat of 63-bar log-price regression slope / 10             |
| lr_r2_63        | trend_strength    | R^2 of 63-bar log-price regression                           |
| lr_r2_126       | trend_strength    | R^2 of 126-bar log-price regression                          |
| dist_sma50      | trend_direction   | (close - SMA50)/ATR                                          |
| dist_sma200     | trend_direction   | (close - SMA200)/ATR                                         |
| sma50_200       | trend_direction   | SMA50/SMA200 - 1, vol-scaled                                 |
| bb_pctb         | channel           | Bollinger %b(20,2) - 0.5                                     |
| bb_width        | volatility        | Bollinger bandwidth / its 252-bar median                     |
| donch_pos_20    | channel           | position in 20-bar Donchian channel - 0.5                    |
| donch_pos_55    | channel           | position in 55-bar Donchian channel - 0.5                    |
| keltner_pos     | channel           | (close - EMA20)/(2*ATR10)                                    |
| ichimoku_dist   | trend_direction   | (close - cloud mid)/ATR (Hosoda)                             |
| psar_dist       | trend_direction   | (close - PSAR)/ATR (Wilder 1978)                             |
| supertrend_dist | trend_direction   | (close - SuperTrend line)/ATR                                |
| kama_dist       | trend_direction   | (close - KAMA(10,2,30))/ATR (Kaufman)                        |
| hma_slope       | trend_direction   | 5-bar change of HMA(55)/ATR (Hull)                           |
| tsi             | momentum          | TSI(25,13)/100 (Blau 1991)                                   |
| kst             | momentum          | KST - signal, z-scored (Pring)                               |
| var_ratio_5     | serial_dependence | variance ratio VR(5) over 126 bars - 1 (Lo & MacKinlay 1988) |
| autocorr_1      | serial_dependence | lag-1 return autocorrelation over 63 bars                    |
| skew_63         | distribution      | 63-bar return skewness                                       |
| kurt_63         | distribution      | 63-bar return excess kurtosis / 10                           |
| dd_252          | drawdown          | close / 252-bar max close - 1, vol-scaled                    |
| days_since_high | drawdown          | bars since 252-bar high / 252                                |
| ulcer_14        | drawdown          | Ulcer index(14) / vol                                        |
| vol_z           | volume            | log-volume z-score (20); 0 if no volume                      |
| obv_slope       | volume            | 20-bar OBV change / 20-bar volume sum                        |
| cmf             | volume            | Chaikin money flow(20)                                       |
| mfi             | volume            | MFI(14)/100 - 0.5                                            |