# RESEARCH_PAPERS.md — Publishable Research Discovered in This Repository

**Scope analyzed:** 238 notebooks (73 unique strategies, 47 archived duplicates, 100+ research/support notebooks), the consolidated `qresearch` library (`src/qresearch/`), and all docs. Analysis date: 2026-07-13.

**How to read the scores (1–10):** Novelty = distance from published state of the art. Academic = citation/venue potential. Industry = value to practitioners/funds. Publishability = odds a competent write-up gets accepted somewhere reputable. Practical = direct P&L/tooling value. Difficulty = work remaining to reach publication quality (10 = hardest). Patent = realistic patentability (financial/trading methods score low by law — see Patent section).

**Status:** pruned from an original 30-candidate list down to 16 survivors (P26, P24, P9, P12, P7, P10, P11, P14, P5, P16, P21, P30, P13, P28 cut — weakest score/novelty combination, or folded into stronger siblings). Remaining papers ranked by expected research impact.

---

## Ranked Paper Candidates

### P1. How Wrong Are Higher-Timeframe Backtests? Quantifying Intrabar Exit Bias with Dual-Timeframe Simulation

| Novelty | Academic | Industry | Publishability | Practical | Difficulty | Patent |
|---|---|---|---|---|---|---|
| 7 | 7 | 9 | 8 | 9 | 4 | 2 |

**Core contribution.** The repo's signature engineering pattern: signals generated on 1H/4H bars, then mapped onto 1-minute bars for exit simulation (`map_signals_to_1min`, used in 7 notebooks; consolidated as `qresearch.signals.generators.map_signals_to_timeframe`). The Cython engine (`cy_backtest_trades` in `multi_1h_jma_atr_013/014/023/034/037/038/040/043`) adds execution semantics almost never modeled in academic backtests: entry-bar stop-loss checks, stop-loss *confirmed by close* with exit at *next open*, and intrabar trailing stops. The paper: run the same signal set through (i) naive HTF close-to-close exits, (ii) HTF OHLC-touch exits, (iii) full 1-minute-path exits, and measure the distribution of Sharpe/max-DD/win-rate inflation. Nobody has published a systematic, cross-asset quantification of this specific bias with SL/TP/trailing exits.

**Why novel.** The look-ahead/`bar-magnifier` problem is known folklore (TradingView's "bar magnifier", vectorbt's `from_signals` caveats), but there is no peer-reviewed measurement of how large the bias is as a function of TP/SL width, timeframe ratio, volatility regime, and asset class. You already have the engine, the signal families, and 5 markets of data.

**Evidence in repo.** `src/qresearch/signals/generators.py:36` (`map_signals_to_timeframe`), `src/qresearch/backtest/engine.py`, Cython cells in `notebooks/05_strategies/by_market/crypto/multi_1h_jma_atr_040.ipynb` (cell 12: delayed SL-close-confirmed exit logic), `multi_1m_jma_atr_016/018/039/042` (1m variants of the same strategies — a natural paired experiment).

**State of the art / gap.** Bailey & López de Prado cover backtest *overfitting*; Harvey & Liu cover multiple testing; almost nothing peer-reviewed covers *path-resolution* bias for intrabar exits (the closest are practitioner blog posts and the ambiguous-bar discussion in Pardo's walk-forward book). Gap: a controlled, reproducible measurement.

**Needed before publication.** (1) Formalize the three exit-resolution levels; (2) grid over TP/SL multiples × timeframes × assets; (3) statistical tests on metric inflation (paired bootstrap on trade sets); (4) an analytical approximation (expected first-passage within a bar under Brownian bridge assumptions — gives the paper theoretical weight); (5) tick-data spot-check on one market to bound the residual 1m-vs-tick error.

**Venue.** Journal of Financial Data Science, or Quantitative Finance; arXiv q-fin.TR + SSRN preprint. **Publication likelihood:** high (75%) — clean question, clean answer, reproducible. **Effort:** 2–3 months.

---

### P2. Volatility-Gated Normalized Adaptive Moving Averages: A Cross-Asset Family Study

| Novelty | Academic | Industry | Publishability | Practical | Difficulty | Patent |
|---|---|---|---|---|---|---|
| 6 | 6 | 8 | 7 | 8 | 5 | 3 |

**Core contribution.** The repo's dominant signal template, repeated with disciplined variation across ~40 notebooks: (1) an adaptive smoother — JMA-style EMA, KAMA, CFB-adaptive Jurik DMX, or KAMA-on-z-scored-log-returns; (2) rolling z-score normalization of the smoother output; (3) entry only when a *range-based volatility estimator* (Rogers-Satchell or Yang-Zhang) exceeds a rolling percentile threshold; (4) ATR- or percent-based SL/TP with trailing. This is a coherent, parameterized *family*, evaluated on crypto, forex, Indian equities, commodities, and options across 1m/15m/1H/4H. The paper: define the family formally, run a unified evaluation with walk-forward and deflated Sharpe, and ablate each component (adaptive vs plain MA; z-score vs raw; RS/YZ gate vs no gate vs ATR gate).

**Why novel.** Each ingredient exists in the literature (KAMA — Kaufman; RS/YZ estimators — Rogers-Satchell 1991, Yang-Zhang 2000; trend filters — Moskowitz et al.). The *composition* — using range-based volatility **percentile gates** as a trade-admission regime filter on **z-score-normalized adaptive smoothers**, evaluated cross-asset — has no direct published counterpart. The volatility-gate ablation alone is a publishable finding either way (works or doesn't).

**Evidence in repo.** Canonical implementation: `notebooks/05_strategies/by_market/crypto/multi_1h_jma_atr_013.ipynb` (cell 8: `compute_jma_indicator` — normalized JMA + RS-vol percentile gate). KAMA/Numba variant with the identical gate: `notebooks/05_strategies/by_market/_unclassified/multi_15m_strat_041.ipynb`. Return-domain variant with Yang-Zhang gate: `multi_1h_strat_032.ipynb` (`compute_ltpkamadsl_indicator`). CFB-Jurik DMX variant: `indian_equities/multi_1h_strat_036.ipynb` (`cfba_jdmx_histogram`). Grid-search harness with per-side quantile stats: `multi_1h_jma_atr_008.ipynb` (`evaluate_params`, `compute_stats`). Library: `src/qresearch/indicators/` (moving_averages, volatility, dsl), `src/qresearch/optimize/walk_forward.py`.

**State of the art / gap.** Trend-following literature (Moskowitz/Ooi/Pedersen 2012; Baltas & Kosowski) uses simple MA/returns signals and volatility *scaling*; regime-filter literature uses HMMs or realized-vol thresholds on close-to-close vol. Gap: range-based estimators as *admission gates* (not sizing), and adaptive-MA normalization, tested jointly and honestly.

**Needed before publication.** (1) Unify all variants under one notation; (2) out-of-sample protocol: anchored walk-forward (already in `walk_forward_optimize`) + deflated Sharpe ratio + PBO (Bailey et al.) — the ROADMAP already lists this; (3) baselines: buy-and-hold, MACD, 12-1 momentum, vol-scaled trend following; (4) transaction-cost sensitivity per market; (5) component ablations; (6) subperiod robustness (2023–2025 crypto data is one regime — extend history).

**Venue.** Quantitative Finance, Journal of Systematic Investing, or SSRN + arXiv; strong candidate for a practitioner journal (JOIM). **Publication likelihood:** medium-high (60%) — hinges on honest OOS results surviving deflation. **Effort:** 3–4 months.

---

### P3. Funding-Adjusted Statistical Arbitrage Between Perpetual-Futures Ratios and Spot Crosses

| Novelty | Academic | Industry | Publishability | Practical | Difficulty | Patent |
|---|---|---|---|---|---|---|
| 8 | 7 | 8 | 7 | 7 | 6 | 4 |

**Core contribution.** `crypto_1m_strat_003.ipynb` builds a spread between the **log ratio of ETH-perp/BTC-perp** and the **spot ETH/BTC cross** — a triangular perp-vs-spot basis — filters it with a Kalman filter (state = fair spread), and trades the z-score with an "**effective z**" adjusted by the **funding-rate differential** between the two perps, with beta-weighted notional legs and a JIT-compiled backtest (entry-z / exit-z / max-z / max-hold grid). Funding history is fetched and merged at source (`fetch_funding_okx_safe`, `merge_funding`).

**Why novel.** Academic crypto stat-arb papers trade coin pairs on prices; the basis literature covers single-asset perp-vs-spot (funding-rate carry, e.g. "The crypto carry trade"). Trading the *cross-asset relative basis* (perp ratio vs spot cross) with funding-differential-adjusted entry thresholds is a real gap — it isolates relative funding pressure from directional carry. This is the most genuinely original *strategy* idea in the corpus.

**Evidence in repo.** `notebooks/05_strategies/by_market/crypto/crypto_1m_strat_003.ipynb` (cells 5–7: funding fetch/merge; 13–14: Kalman spread + funding_diff + effective z; 21: Numba backtest with the four-threshold exit logic). Related funding infrastructure: `02_data_cleaning/data-cleaning_014.ipynb`.

**State of the art / gap.** Kalman pairs trading is standard (Avellaneda-Lee lineage); funding-rate premia documented (e.g., papers on perp funding as sentiment). Gap: (a) triangular perp/spot basis as tradable spread; (b) funding differential as *threshold modifier* rather than P&L accrual only; (c) 1m/5m execution granularity with taker fees.

**Needed before publication.** (1) Accrue funding payments explicitly in P&L (currently only threshold-side); (2) extend to more pairs (SOL/ETH, alt triangles) and both OKX + Binance for venue robustness; (3) cointegration/stationarity testing of the spread (ADF — `us_equities_strat_007.ipynb` already has `adf_test` code to reuse); (4) capacity/slippage analysis with order-book snapshots; (5) benchmark vs plain price-pairs Kalman stat-arb and vs pure funding carry; (6) regime split: bull/bear/chop 2021–2025.

**Venue.** Journal of Financial Data Science, Digital Finance, or Journal of Alternative Investments; arXiv q-fin.TR. Crypto conferences (Crypto Asset Lab, FC workshops). **Publication likelihood:** medium-high (65%). **Effort:** 3–4 months (funding P&L accrual is the main gap).

---

### P4. Do "Smart Money Concepts" Survive Quantification? An Empirical Test on Tick-Level Data

| Novelty | Academic | Industry | Publishability | Practical | Difficulty | Patent |
|---|---|---|---|---|---|---|
| 8 | 7 | 6 | 8 | 5 | 6 | 1 |

**Core contribution.** `indian_equities_strat_001.ipynb` does something rare: it converts the retail-trading folklore of ICT/SMC — Fair Value Gaps, Order Blocks, Liquidity Sweeps, Optimal Trade Entry (61.8–79% fib zone), session timing ("Power of Three") — into *precise, testable rules on tick/quote data* (LTP, BuyPrice/SellPrice, LTQ from Indian market feeds), then requires all five conditions jointly. The consolidated library even has a dedicated module (`src/qresearch/signals/smc.py`). The paper: formalize each SMC primitive, test each in isolation and jointly on tick data with event-study methodology, and report whether any has predictive content beyond chance.

**Why novel.** SMC has millions of retail followers and essentially zero peer-reviewed scrutiny. A rigorous null-result paper is publishable; a positive result doubly so. Using *quote-level* data (bid/ask, not just OHLC) to define liquidity sweeps is a methodological step beyond the handful of blog-level "SMC backtests."

**Evidence in repo.** `notebooks/05_strategies/by_market/indian_equities/indian_equities_strat_001.ipynb` (cell 4: `identify_fvg`, `find_order_blocks`, `identify_liquidity_sweeps`, `calculate_fibonacci_levels`, `filter_session`, joint `apply_trading_logic`); `src/qresearch/signals/smc.py`.

**State of the art / gap.** Academic microstructure covers order-flow imbalance, sweep detection (e.g., intermarket sweep orders in US equities), and technical-analysis tests (Lo, Mamaysky & Wang 2000 for chart patterns). Nobody has run the LMW-style program on SMC primitives. Clear gap.

**Needed before publication.** (1) Reconcile definitions with the "canonical" SMC literature (document rule choices, test sensitivity); (2) event studies: forward returns after each primitive fires, with bootstrap nulls (shuffle timestamps, surrogate series); (3) multiple-testing control (the joint-condition strategy is a 5-way selection — use White's reality check / Romano-Wolf); (4) more symbols and dates; (5) compare against random entries with identical exit rules (isolates entry alpha from exit engineering).

**Venue.** Journal of Behavioral Finance, International Review of Financial Analysis, or Quantitative Finance; very strong SSRN/arXiv interest regardless. **Publication likelihood:** high (70%) — the question is inherently interesting and null results are acceptable here. **Effort:** 3 months (mostly data + statistics; definitions exist).

---

### P6. Session-Aligned Bar Construction: The 75-Minute Candle and Bar-Alignment Effects in Intraday Backtests

| Novelty | Academic | Industry | Publishability | Practical | Difficulty | Patent |
|---|---|---|---|---|---|---|
| 7 | 5 | 6 | 6 | 6 | 4 | 1 |

**Core contribution.** Scattered through the Indian-equities notebooks is an unusually thoughtful treatment of *bar construction*: 75-minute candles that tile the NSE session exactly (375 min = 5 × 75, `create_75min_candles` in `multi_strat_052.ipynb`), custom resampling with session offsets (`resample_1hr_custom` variants — the 1H bars of a 09:15-open market are misaligned in every standard resampler), and HalfTrend with *session reset* (`halftrend_with_session_reset`, `indian_equities_1h_strat_003.ipynb`) so indicator state does not leak across the overnight gap. The paper: measure how much intraday-strategy performance and signal statistics change purely as a function of (a) bar phase/offset, (b) session-tiling vs non-tiling bar lengths, (c) indicator state reset at session boundaries — across NSE and 24/7 crypto as a control.

**Why novel.** Timeframe-choice studies exist; *bar-phase and session-alignment* effects are essentially unstudied (closest: literature on sampling frequency for realized volatility, and practitioner lore about "which hourly close"). The NSE 75-minute tiling is a neat, concrete hook, and the crypto control (no sessions) makes the design clean.

**Evidence.** `notebooks/05_strategies/by_market/indian_equities/multi_strat_052.ipynb` (`create_75min_candles`, `normalized_dema_strategy`); `indian_equities_1h_strat_003.ipynb` (`halftrend_with_session_reset`, `backtest_halftrend`); ubiquitous `resample_1hr_custom` (40+ notebooks); `src/qresearch/data/preprocess.py`.

**Needed.** (1) Systematic phase-shift experiment (offset 0..k for k-minute bars); (2) session-reset ablation on 3+ indicators; (3) significance via stationary bootstrap; (4) tie to P1's engine for exit realism.

**Venue.** Journal of Financial Data Science or International Journal of Forecasting (bar-sampling angle); good workshop paper (ICAIF). **Publication likelihood:** medium (55%). **Effort:** 1.5–2 months (smallest marginal work — infrastructure exists).

---

### P8. Indicators in Return Space: Z-Scored Log High-Pass Signals with Yang-Zhang Volatility Admission (LTPKAMADSL)

| Novelty | Academic | Industry | Publishability | Practical | Difficulty | Patent |
|---|---|---|---|---|---|---|
| 6 | 4 | 6 | 5 | 6 | 4 | 2 |

**Core contribution.** `multi_1h_strat_031/032.ipynb`: instead of smoothing *price*, compute log(prev_close/close) (a high-pass of log price), rolling-z-score it, run KAMA *on the z-scored return stream*, and trade z vs KAMA crossovers gated by Yang-Zhang volatility above its median. The domain shift (adaptive smoothing of normalized returns rather than price) changes the indicator's stationarity properties — that is an articulable, testable claim: return-domain indicators transfer across assets/regimes with less re-tuning than price-domain ones.

**Why novel.** "Normalize your features" is standard ML practice, but the technical-indicator literature almost universally operates on price. A study showing parameter *transferability* (fit on asset A, deploy on asset B) improves in return space would be genuinely useful and is not in the literature.

**Evidence.** `notebooks/05_strategies/by_market/crypto/multi_1h_strat_032.ipynb` (cell 9, full implementation incl. Yang-Zhang); `multi_1h_strat_031.ipynb`; LHP-DSL relatives in `multi_1h_strat_022/028/029/030`, `forex_1h_strat_004.ipynb` (`compute_lhp_dsl_indicator`); `src/qresearch/indicators/dsl.py`.

**Needed.** Cross-asset transfer experiment (the repo's 5 markets are ideal); price-domain vs return-domain matched pairs; sensitivity to z-window. Best merged into P2 as its most interesting ablation, or standalone short paper.

**Venue.** Journal of Systematic Investing / SSRN; section of P2 otherwise. **Publication likelihood:** medium (50%) standalone. **Effort:** 1.5 months.

---

## Patent Assessment

Blunt reality: **pure trading algorithms are effectively unpatentable** in the US (Alice v. CLS Bank — abstract ideas) and explicitly excluded in India (Section 3(k), "business method / algorithm per se") and the EPO. Score ceilings above reflect this. What *could* clear the bar is a specific technical implementation, and even then trade secret is almost always the better vehicle:

1. **Dual-timeframe backtest engine with close-confirmed delayed-exit semantics** (P1 machinery: `cy_backtest_trades`' entry-bar SL + SL-close-confirm + next-open exit + trailing state machine). As a *simulation system* claim, borderline; as trade secret / open-source moat, valuable. **Recommendation: publish (P1) or open-source — the defensive publication kills competitors' patent options and earns citations.**
2. **Funding-adjusted effective-z threshold modulation** (P3). The most commercially valuable idea in the repo. **Recommendation: trade secret while trading it; publish only after capacity is consumed or edge decays.**

If patent filing is still desired, the only plausibly claimable subject matter is the *data-processing pipeline* of item 1 framed as a technical improvement in simulation fidelity/throughput (Cython/Numba dual-resolution engine) — consult a patent attorney before any public disclosure, since P1 publication forecloses it.

---

## Top Paper Roadmap — Wave 1 (superseded by the Combined Roadmap v2 at the end of this file)

| # | Paper | Score* | Why this rank |
|---|-------|--------|---------------|
| 1 | **P1** Intrabar exit bias quantification | 8.1 | Highest publishability × impact; infrastructure done; every backtester cites it |
| 2 | **P4** SMC quantification | 7.6 | Huge audience, empty literature, null-result-safe |
| 3 | **P3** Funding-adjusted perp/spot basis arb | 7.4 | Most original strategy; commercially sensitive — decide trade-secret vs publish first |
| 4 | **P2** Volatility-gated adaptive MA family | 7.2 | The corpus's core; needs deflated-Sharpe rigor to land |
| 5 | **P6** Session-aligned bars / 75-min candles | 6.3 | Cheapest to finish; novel micro-question |
| 6 | **P8** Return-domain indicators (or as P2's key ablation) | 5.2 | Best as P2 section; standalone only if P2 splits |

*Score = 0.35·Publishability + 0.25·Novelty + 0.2·Academic + 0.2·Industry.

---

## 12-Month Publication Plan

**Standing infrastructure (Month 1, prerequisite for everything):** implement deflated Sharpe + CSCV/PBO in `qresearch.backtest.metrics` (roadmap item already); add funding-P&L accrual to the engine; fix the Hilbert `mode='same'` causality leak; freeze a versioned data snapshot per market for reproducibility.

| Months | Milestone |
|--------|-----------|
| **1–3** | **P1** (intrabar bias): run 3-resolution experiments across 5 markets; Brownian-bridge analytical section; submit to arXiv/SSRN by end of M3, then JFDS. |
| **3–5** | **P6** (session-aligned bars): reuses P1 engine while it is warm; phase-shift + session-reset ablations; workshop or JFDS short paper. |
| **4–7** | **P4** (SMC/folklore): event studies + Romano-Wolf corrections on tick data; target International Review of Financial Analysis; SSRN preprint M6. |
| **5–8** | **P2** (adaptive-MA family): honest OOS with deflated Sharpe + PBO. Submit to Quantitative Finance, M8. |
| **7–9** | **Decision gate on P3** (basis arb): if live/paper trading shows persistent edge → keep as trade secret, defer paper. If edge marginal → finish funding-accrual backtests, venue-robustness (OKX+Binance), submit to Digital Finance / JFDS by M9. |
| **11–12** | Revisions cycle for P1/P4 referee reports; consolidate P8 material either into P2 revisions or a standalone SSRN companion note. |

**Cross-cutting reviewer-proofing checklist (apply to every paper):** deflated Sharpe + PBO on any optimized result; anchored walk-forward with embargo; per-market realistic costs (maker/taker, STT/brokerage for India, funding for perps); stationary/block bootstrap for significance; subperiod (regime) splits; data-snooping disclosure (the grids in this repo are large — report *all* trials, the `jma_hypertune_log.csv` pattern of logging every combo is exactly right, keep it); public code + frozen data snapshot via the `qresearch` package.

---
---

# SECOND WAVE — Papers from Recombination

**Constraint honored throughout:** every candidate below runs on data, fetchers, and engines already in this repository. No new external datasets.

## Why the first pass wasn't the right ceiling

Paper count does **not** scale with markets × strategies × indicators. Forty JMA/KAMA variant notebooks compress into *one* paper because they answer one question ("does this signal family work?") forty times; a new market or timeframe is a robustness *table*, not a new paper. Paper count scales with **distinct research questions**, and the corpus is question-rich along six axes the first pass under-mined. These axes are repeatable generators — apply them to any future notebook and new papers fall out:

| Axis | Generator question | Wave-2 papers |
|---|---|---|
| **A. Decomposition** | What does every backtest conflate that we can split? (entry vs exit, semantics vs signal, geometry vs alpha) | P15, P17 |
| **B. Transfer** | Fit *here*, test *there* — across markets, timeframes, domains, signal families | P18–P20 |
| **C. Synthesis** | Which existing components have never been combined? | P22, P23 |
| **D. Meta-science** | The corpus itself as dataset: what do 3 years and 238 notebooks reveal about the research process? | P25 |
| **E. Theory vs data** | Which model predicts what the 1-minute paths actually did? | P27 |
| **F. Method transplant** | Which established method from another field has never touched this data? | P29 |

**The single most important operational insight:** one unified compute campaign — *all signal families × all 5 markets × a common bracket-exit grid, run through the unified engine at all three exit resolutions* — simultaneously produces the raw material for P1, P15, P17, P18, P19, P22, P23, and P25. Build the campaign once (~2–3 weeks of engineering on top of `qresearch`), and most of these papers become analysis-and-writing exercises over one shared trade-book database.

---

### P15. Entries or Exits? Decomposing Intraday Trend-Following P&L with Random-Entry Baselines

| Novelty | Academic | Industry | Publishability | Practical | Difficulty | Patent |
|---|---|---|---|---|---|---|
| 8 | 7 | 9 | 8 | 9 | 3 | 1 |

**Core contribution.** Every strategy notebook in this corpus pairs a signal with an engineered exit stack (ATR/percent brackets, trailing, close-confirmed SL, EOD flatten). Which side earns the money? Design: (i) each signal family with its exit stack; (ii) *random entries, timestamp- and direction-matched in count, with the identical exit stack*; (iii) each family with naive close-to-close exits. The gap between (ii) and buy-and-hold measures pure exit engineering; the gap between (i) and (ii) is genuine entry alpha. Run per market × timeframe. Practitioner folklore ("exits matter more than entries") has never been measured at this scale with matched randomization.

**Evidence.** All of `05_strategies/` (10+ families); exit stack in `cy_backtest_trades` (`multi_1h_jma_atr_040.ipynb` cell 12); random-entry harness is ~50 lines on top of `qresearch.backtest.engine`.

**SOTA / gap.** Random-entry benchmarking appears in scattered practitioner posts (Tharp-style "random entry with good exits") and nowhere in peer-reviewed literature with proper matched bootstrap inference. Empty field, cheap experiment, provocative result either way.

**Needed.** Matched-count/direction randomization with block bootstrap (1,000 draws); decomposition table per family/market; significance via percentile ranks of real entries inside the random distribution.

**Venue.** Journal of Portfolio Management or JFDS. **Likelihood:** high (75%). **Effort:** 1 month on top of the unified campaign.

---

### P17. Exit Semantics Matter: Touch-SL vs Close-Confirmed-SL vs Next-Open Execution

| Novelty | Academic | Industry | Publishability | Practical | Difficulty | Patent |
|---|---|---|---|---|---|---|
| 7 | 6 | 8 | 7 | 8 | 2 | 2 |

**Core contribution.** The Cython engine already implements three stop semantics — intrabar touch, close-confirmation with next-open exit, and entry-bar SL — that correspond to real broker order types (stop-market vs stop-on-close vs delayed manual exit). Hold signal and geometry fixed; vary only semantics; measure P&L, whipsaw rate, and gap-risk exposure per market (gappy NSE equities vs continuous crypto is the natural contrast). Companion/spin-off of P1 focused on the *semantics* dimension rather than the *resolution* dimension — publishable separately because the audience (practitioners choosing order types) and the experiment differ.

**Evidence.** All three semantics coexist in `cy_backtest_trades` (`multi_1h_jma_atr_040.ipynb` cell 12: `exit_next_open`, entry-SL, touch-SL, trailing).

**Needed.** Pure ablation harness (semantics flag already effectively in code); overnight-gap stratification for NSE; bootstrap CIs. Cheapest paper in wave 2.

**Venue.** JFDS short paper / Journal of Trading-style. **Likelihood:** 65%. **Effort:** 3 weeks post-campaign.

---

### P18. The Alpha Transferability Matrix: Where Do Technical Signals Travel?

| Novelty | Academic | Industry | Publishability | Practical | Difficulty | Patent |
|---|---|---|---|---|---|---|
| 8 | 7 | 8 | 7 | 8 | 5 | 1 |

**Core contribution.** Fit each signal family's parameters on market i (walk-forward, `walk_forward_optimize`), deploy frozen on market j → a full N×N transfer matrix per family. Then *explain* the matrix: regress transfer success on market-quality covariates computable from existing OHLCV — Hurst exponent, volatility clustering (GARCH persistence), return autocorrelation, gap frequency, session structure. Output: "trend signals tuned on crypto transfer to forex but not NSE equities, and *here is the market property that predicts it*." Transforms the repo's 5-market breadth from robustness decoration into the paper's actual subject.

**Evidence.** Same families implemented across `by_market/crypto`, `forex`, `indian_equities`, `commodities`, `options`, `us_equities`; `src/qresearch/optimize/walk_forward.py`; covariates from raw OHLCV already loaded by `qresearch.data`.

**SOTA / gap.** Cross-asset momentum universality (Moskowitz et al.; Baltas-Kosowski) covers monthly futures momentum. Intraday technical-signal transferability with market-quality attribution: unstudied.

**Needed.** Frozen-parameter transfer protocol; covariate panel; regularized regression/attribution with market-level clustering (only ~6 markets — treat attribution as suggestive, lean on the matrix itself).

**Venue.** Quantitative Finance or JFDS; flagship potential. **Likelihood:** 65%. **Effort:** 2 months post-campaign.

---

### P19. Timeframe Scaling Laws for Adaptive Moving-Average Signals

| Novelty | Academic | Industry | Publishability | Practical | Difficulty | Patent |
|---|---|---|---|---|---|---|
| 7 | 6 | 7 | 6 | 7 | 4 | 1 |

**Core contribution.** The corpus implements the *same* families at 1m/15m/1H/4H (e.g., `multi_1m_jma_atr_016/018/039/042` vs `multi_1h_jma_atr_*` vs `multi_4h_jma_atr_015/017/021/051` — deliberate timeframe replications of one strategy). Question: how do optimal lookback, optimal bracket width, Sharpe, and cost drag scale with bar size? Test the null of √t diffusive scaling (optimal SL distance ∝ √timeframe; Sharpe per trade flat) against data; deviations localize where timeframes contain *structurally different* alpha rather than rescaled noise.

**Evidence.** Timeframe-replicated notebooks above; `resample_ohlcv` makes arbitrary intermediate timeframes free; costs constant per trade → natural cost-vs-alpha scaling story.

**SOTA / gap.** Signature-plot literature covers *volatility* scaling; nothing published on *trading-rule parameter* scaling. Clean, physics-flavored question.

**Needed.** Log-spaced timeframe sweep (1m→1d); per-timeframe walk-forward optima; fit scaling exponents with bootstrap CIs.

**Venue.** Quantitative Finance (fits its econophysics lineage). **Likelihood:** 60%. **Effort:** 1.5 months post-campaign.

---

### P20. Return-Domain vs Price-Domain Indicators: A Parameter-Transfer Experiment

*(P8 promoted from ablation to standalone transfer study — kept as separate entry; if P2 lands first, fold back.)*

| Novelty | Academic | Industry | Publishability | Practical | Difficulty | Patent |
|---|---|---|---|---|---|---|
| 6 | 5 | 7 | 6 | 7 | 3 | 1 |

**Core contribution.** Matched pairs: KAMA-on-price vs KAMA-on-z-scored-log-returns (`compute_ltpkamadsl_indicator`, `multi_1h_strat_032.ipynb`), JMA-on-price vs normalized-JMA (`multi_1h_jma_atr_013.ipynb`) — fit on asset A, freeze, deploy on assets B..N. Hypothesis: return-domain/normalized indicators transfer with less degradation because their input distribution is closer to stationary across assets. This is the *mechanism paper* for whatever P18 finds. **Evidence:** files above + `src/qresearch/indicators/dsl.py`. **Venue:** JSI/SSRN. **Likelihood:** 55%. **Effort:** 1 month (shares P18's harness).

---

### P22. Signal DNA: Does Indicator Diversity Buy Portfolio Diversification?

| Novelty | Academic | Industry | Publishability | Practical | Difficulty | Patent |
|---|---|---|---|---|---|---|
| 7 | 6 | 9 | 7 | 9 | 4 | 2 |

**Core contribution.** The corpus holds 10+ genuinely distinct signal *mechanisms* (JMA z-score, KAMA, LHP-DSL, CFB-JDMX, Hilbert phase, SAMX, HalfTrend, Kalman trend, quad-stoch divergence, adaptive momentum, SMC). Compute the correlation/co-occurrence structure of their *trade streams* (position overlap, P&L correlation, signal-timing correlation) on identical data. Questions: do mechanistically different indicators actually fire independently, or are they all reparameterized trend detectors (the "indicator illusion")? What does an equal-risk ensemble of families deliver vs the best single family, after costs? Cluster the families by trade-stream similarity → an empirical taxonomy of technical signals ("signal DNA" dendrogram).

**Evidence.** All families in `05_strategies/`; correlation machinery in `07_portfolio_construction/portfolio-construction_001/002.ipynb` (`daily_correlations`); ticker-weight allocation pattern in `multi_1h_strat_035.ipynb`.

**SOTA / gap.** Factor-crowding literature exists at the *fund* level; nothing at the *indicator-mechanism* level. The taxonomy figure alone will be widely shared.

**Needed.** Unified campaign trade books; overlap/correlation metrics with block bootstrap; hierarchical clustering; ensemble backtest with per-family vol targeting.

**Venue.** JPM (practitioner appeal) or JFDS. **Likelihood:** 65%. **Effort:** 1.5 months post-campaign.

---

### P23. When Do Volatility Gates Help? A Meta-Analysis Across Estimators, Signals, and Markets

| Novelty | Academic | Industry | Publishability | Practical | Difficulty | Patent |
|---|---|---|---|---|---|---|
| 6 | 6 | 8 | 7 | 8 | 3 | 1 |

**Core contribution.** Standalone generalization of P2's ablation: cross the four range-based estimators already implemented (Rogers-Satchell — `multi_1h_jma_atr_013`; Yang-Zhang — `multi_1h_strat_032`; ATR — everywhere; Parkinson — trivial addition to `qresearch.indicators.volatility`) × gate direction (trade-when-high vs trade-when-low) × percentile threshold × all families × 5 markets. Output: a meta-analytic answer to "should trend entries be volatility-gated, with which estimator, and does the answer flip by market/mechanism?" The corpus is uniquely positioned: the gate is already the family's design axis.

**Evidence.** Files above; `src/qresearch/indicators/volatility.py`.

**SOTA / gap.** Vol-managed portfolio literature (Moreira-Muir) covers *sizing* at monthly horizons; admission *gating* of intraday technical entries is unpublished territory.

**Needed.** Gate-factorial design inside the unified campaign; forest-plot style meta-analytic summary with market/family random effects.

**Venue.** JFDS. **Likelihood:** 65%. **Effort:** 1 month post-campaign.

---

### P25. The Strategy Graveyard: A Census of 73 Retail Quant Strategies Under Deflated Evaluation

| Novelty | Academic | Industry | Publishability | Practical | Difficulty | Patent |
|---|---|---|---|---|---|---|
| 8 | 8 | 8 | 8 | 7 | 6 | 1 |

**Core contribution.** Run *all 73 unique strategies* through the unified engine with realistic costs, then apply the full multiple-testing correction stack (deflated Sharpe, PBO, Romano-Wolf across the whole corpus). Report the survival curve: of everything a dedicated retail quant built in 3 years, what fraction survives honest evaluation — by mechanism, market, and timeframe? This is the empirical companion Harvey/Bailey-style methodology papers have never had: their corrections applied to a *complete, timestamped, unselected* research corpus (the notebooks are the pre-registration — every attempt is on disk, including failures and duplicates; no survivor curation possible). Flagship candidate.

**Evidence.** Entire `notebooks/05_strategies/` tree + `docs/STRATEGY_INDEX.md` (the census frame); `docs/notebook_mapping.csv` (provenance/timestamps); `rank_and_score` + `walk_forward_optimize` (the selection procedure to audit); grid logs pattern (`jma_hypertune_log.csv`).

**SOTA / gap.** "Most claimed research findings in finance are false" (Harvey) argued from published-paper meta-data; no one has opened a complete private research corpus and counted. Unique dataset = unique paper.

**Needed.** Port each strategy's signal function to the unified engine (the heavy lift — ~73 functions, many already consolidated in `qresearch`); DSR/PBO implementation (roadmap item); corpus-wide familywise error control; per-mechanism survival breakdown.

**Venue.** Flagship: submit to Journal of Finance-adjacent econometrics outlets is unrealistic, but Review of Financial Studies' data-oriented siblings, Journal of Financial Data Science (certain), and *massive* SSRN/arXiv+Twitter reach. **Likelihood:** 75% at JFDS; the story sells itself. **Effort:** 3 months (the porting is the cost — but it *is* the unified campaign, shared with other papers).

---

### P27. Which Diffusion Predicts Intrabar Outcomes? First-Passage Theory Meets One-Minute Ground Truth

| Novelty | Academic | Industry | Publishability | Practical | Difficulty | Patent |
|---|---|---|---|---|---|---|
| 7 | 7 | 6 | 7 | 6 | 6 | 1 |

**Core contribution.** Wave-1 P1's theory arm, promoted: given an H1 bar's OHLC, what is the probability TP is hit before SL — and which model predicts the *empirically observed* hit order in the 1m path? Candidates: GBM first-passage, Brownian bridge conditioned on OHLC, jump-diffusion, and a nonparametric resampled-path baseline. The 1m data supplies exact hit-order ground truth for millions of bar-bracket events across 5 markets. Deliverable: calibrated "ambiguous bar" resolution probabilities — directly usable by every backtesting-software vendor (vectorbt, backtesting.py currently guess).

**Evidence.** 1m datasets already in-corpus (`multi_1m_jma_atr_*` load them); bracket event generator = unified campaign; OHLC-conditioning math is self-contained.

**SOTA / gap.** First-passage under bridge conditioning is solved math; its *empirical validation against real intrabar paths at scale* is unpublished. Also directly fixes a known open issue in every OHLC backtester.

**Needed.** Event dataset (bar + bracket + observed outcome); model likelihood comparison (Brier/log scores); regime and asset-class stratification; release the calibrated resolver as a `qresearch` module (adoption vector).

**Venue.** Quantitative Finance (theory+empirics fit) or Journal of Derivatives-adjacent. **Likelihood:** 60%. **Effort:** 2 months.

---

### P29. Rainflow Counting for Financial Drawdowns: Fatigue Analysis of Equity Curves

| Novelty | Academic | Industry | Publishability | Practical | Difficulty | Patent |
|---|---|---|---|---|---|---|
| 9 | 7 | 6 | 7 | 6 | 5 | 3 |

**Core contribution.** The repo contains the *name* (`rainflow_hilbert`) but not the algorithm — implement **actual rainflow cycle counting** (ASTM E1049, the standard fatigue-analysis method for decomposing a load history into closed stress cycles) on equity curves and price series. Deliverables: (a) a drawdown-cycle spectrum per strategy — the full distribution of nested drawdown cycles, of which max-drawdown is only the single largest — enabling a "fatigue damage" risk index (Miner's-rule analog: cumulative ∑(cycle amplitude^k)) that discriminates between strategies with identical max-DD but very different bleed profiles; (b) rainflow spectra of *price* as a regime feature (trending vs choppy markets have distinct cycle-amplitude distributions — feeds P23's gates). Cross-disciplinary transplants of exactly this kind (e.g., Omega ratio from engineering) historically do very well.

**Why highest wave-2 novelty.** Rainflow is 50 years old in mechanical engineering, has an ASTM standard and Python implementations, and has — as far as the literature shows — never been applied to financial risk. The analogy is exact: an equity curve *is* a load history; investor pain *is* fatigue accumulation from repeated drawdown cycles, not just the single worst excursion.

**Evidence.** Equity curves from every backtest (`compute_drawdown` in `risk-management_005.ipynb`, `max_drawdown` in `qresearch.backtest.metrics`); the misnamed `rainflow_hilbert` (`multi_1h_strat_035.ipynb`) as the origin story; unified campaign supplies hundreds of equity curves to characterize.

**Needed.** ASTM-conformant rainflow implementation (~100 lines or the `rainflow` PyPI package); fatigue-index definition + exponent calibration; validation: does the index predict future drawdown behavior / investor-relevant outcomes better than max-DD, Ulcer index, CED? Comparison set: Ulcer, pain index, conditional expected drawdown.

**Venue.** Journal of Risk, Quantitative Finance, or Journal of Portfolio Management (risk-metric papers land well there). **Likelihood:** 65% — novelty carries it if the empirical section is competent. **Effort:** 1.5–2 months.

---

### Also viable, held in reserve (not scored to keep the list honest)

- **R1. RL exit policies** — `16_reinforcement_learning/` is nearly empty; the unified engine is a ready gym environment: learned exit policy vs bracket exits, judged with P25's deflation stack. ICAIF-grade if the learned policy beats brackets after costs; compute-hungry, so schedule after the campaign. Evidence: `reinforcement-learning_001.ipynb` (stub), engine.
- **R2. Intraday seasonality atlas** — `assign_time_slot` (`visualization_010.ipynb`) + campaign trade books → time-of-day alpha maps, NSE session structure vs 24/7 crypto vs forex sessions. Likely a strong *section* of P18 rather than standalone; promote only if the maps show sharp structure.
- **R3. Halftrend session-reset ablation** — already inside wave-1 P6's design; listed here only to note it should not be double-counted.

---

## Combined Roadmap v2 — Ordered by Novelty-per-Effort Under the Max-Impact Goal

The unified campaign (UC) is the backbone: **build once (M1–M3), feed every downstream paper.** UC = all portable signal families × 5 markets × bracket/geometry/semantics/gate factorial × 3 exit resolutions, all trades logged to one database with full provenance.

| Rank | Paper | Depends on | Marginal effort after deps | Why here |
|---|---|---|---|---|
| 1 | P1 Intrabar exit bias | UC | writing only | Anchor paper, defines UC |
| 2 | P15 Entries vs exits | UC | +1 mo | Highest impact-per-effort in corpus |
| 3 | P25 Strategy graveyard | UC (is UC) | +1 mo analysis | Flagship; UC porting *is* this paper |
| 4 | P29 Rainflow drawdowns | UC equity curves | 1.5 mo | Highest novelty; independent of market data quirks |
| 5 | P17 Exit semantics | UC | +3 wk | Near-free spin-off |
| 6 | P4 SMC quantification | tick data (have) | 3 mo | Empty literature, null-safe |
| 7 | P22 Signal DNA | UC | +1.5 mo | Practitioner-viral figure |
| 8 | P18 Transferability matrix | UC | +2 mo | Turns 5-market breadth into the subject |
| 9 | P23 Vol-gate meta-analysis | UC | +1 mo | Generalizes P2's best part |
| 10 | P27 First-passage validation | UC events | 2 mo | Theory weight; fixes real tooling gap |
| 11 | P3 Funding basis arb | own data (have) | 3 mo | Gate on trade-secret decision first |
| 12 | P2 VG-NAMA family | UC | +1 mo | After P23/P15, becomes mostly assembly |

Second shelf: P6 session bars, P19 timeframe scaling, P20 return-domain transfer.

**Realistic yield:** 16 scored candidates + 3 reserve ≈ 19-paper pipeline; at plausible per-paper likelihoods, expect **7–9 eventual acceptances** with SSRN/arXiv preprints for all.

## 12-Month Plan v2

| Months | Work | Papers moving |
|---|---|---|
| **1–3** | **Build the unified campaign**: port all portable strategy signal functions into `qresearch`, implement DSR/PBO + Romano-Wolf, run the full factorial, log every trade to one parquet store. In parallel (no data dependency): P29 rainflow implementation on existing equity curves. | P29 draft |
| **3–5** | Harvest wave A: P1, P15, P17 analyses off the campaign store. P29 empirical section off campaign equity curves → submit. | P1, P15 preprints M5; P17, P29 submitted |
| **5–7** | P25 graveyard analysis + writing (the campaign already ran everything). P4 SMC event studies on tick data. | P25 preprint M7; P4 submitted M7 |
| **7–9** | P22 signal-DNA + P23 vol-gate meta-analysis (shared campaign store). P18 transferability matrix runs. P3 trade-secret gate decision → publish or defer. | P22, P23 submitted M9 |
| **9–11** | P18 writing; P27 first-passage validation; P2 assembled from P15+P23 remnants. | P18, P27 submitted/preprinted M11 |
| **11–12** | Referee-report cycle for the M3–M7 submissions; second-shelf triage for year 2. | Revisions + P2 preprint |

**Yield at M12:** ~8 papers submitted or preprinted, 2–3 in revision/accepted, second-shelf queue (3 more) fully de-risked because the campaign data already exists.

**Discipline rule to keep novelty high:** before starting any new candidate, state its research question in one sentence *without naming a strategy, market, or indicator*. If impossible, it is a robustness section of an existing paper, not a new paper.

---
---

# Combined Ranking — All 16 Surviving Papers by Score, Novelty Tie-Break

Score = 0.35·Publishability + 0.25·Novelty + 0.2·Academic + 0.2·Industry. Ties broken by Novelty, then Difficulty (lower = easier, ranked higher).

| Rank | Paper | Title | Score | Novelty | Academic | Industry | Publishability | Difficulty |
|---|---|---|---|---|---|---|---|---|
| 1 | P15 | Entries or Exits? Decomposing Intraday P&L with Random-Entry Baselines | 8.00 | 8 | 7 | 9 | 8 | 3 |
| 2 | P25 | The Strategy Graveyard: 73 Strategies Under Deflated Evaluation | 8.00 | 8 | 8 | 8 | 8 | 6 |
| 3 | P1 | Quantifying Intrabar Exit Bias with Dual-Timeframe Simulation | 7.75 | 7 | 7 | 9 | 8 | 4 |
| 4 | P3 | Funding-Adjusted Perp/Spot Basis Stat-Arb | 7.45 | 8 | 7 | 8 | 7 | 6 |
| 5 | P18 | The Alpha Transferability Matrix | 7.45 | 8 | 7 | 8 | 7 | 5 |
| 6 | P4 | Do Smart Money Concepts Survive Quantification? | 7.40 | 8 | 7 | 6 | 8 | 6 |
| 7 | P29 | Rainflow Counting for Financial Drawdowns | 7.30 | 9 | 7 | 6 | 7 | 5 |
| 8 | P22 | Signal DNA: Does Indicator Diversity Buy Diversification? | 7.20 | 7 | 6 | 9 | 7 | 4 |
| 9 | P17 | Exit Semantics: Touch-SL vs Close-Confirmed vs Next-Open | 7.00 | 7 | 6 | 8 | 7 | 2 |
| 10 | P27 | First-Passage Theory Meets One-Minute Ground Truth | 6.80 | 7 | 7 | 6 | 7 | 6 |
| 11 | P2 | Volatility-Gated Normalized Adaptive MA Family | 6.75 | 6 | 6 | 8 | 7 | 5 |
| 12 | P23 | When Do Volatility Gates Help? A Meta-Analysis | 6.75 | 6 | 6 | 8 | 7 | 3 |
| 13 | P19 | Timeframe Scaling Laws for Adaptive MA Signals | 6.45 | 7 | 6 | 7 | 6 | 4 |
| 14 | P6 | Session-Aligned Bar Construction (75-min candles) | 6.05 | 7 | 5 | 6 | 6 | 4 |
| 15 | P20 | Return-Domain vs Price-Domain Indicators (transfer) | 6.00 | 6 | 5 | 7 | 6 | 3 |
| 16 | P8 | Indicators in Return Space (LTPKAMADSL) | 5.25 | 6 | 4 | 6 | 5 | 4 |

**Reading it.** Top 3 (P15, P25, P1) share one root cause: highest Industry score (9) plus low-to-moderate difficulty — they're the papers that pay off fastest per unit of the Unified Campaign. P29 has the single highest novelty (9, genuine cross-discipline transplant) but ranks 7th because Industry value (6) and Difficulty (5) drag the composite down. P8 sits last — well-trodden ground, modest novelty, thin standalone Industry value — the doc already recommends folding it into P2.

**Novelty-8 tier** (excluding P29 at 9): P3, P4, P15, P18, P25.

---
---

# Execution Plan — How to Actually Write These Papers, Phase by Phase

The sections above answer *what* to write and *when* (12-Month Plan v2). This section answers *how*: the concrete build/write/submit mechanics, in order, so picking up this doc cold is enough to start work today.

## Phase 0 — Foundations (before touching any paper) · ~2 weeks

Nothing below is publishable without these in place first; every paper's "Needed before publication" list leans on them.

- [ ] **Stats library** in `qresearch.stats` (doesn't exist yet — check `src/qresearch/` first, don't duplicate if it's hiding under `optimize/` or `backtest/metrics.py`):
  - [ ] Deflated Sharpe Ratio (Bailey & López de Prado)
  - [ ] Probability of Backtest Overfitting (PBO, combinatorially symmetric cross-validation)
  - [ ] Romano-Wolf stepdown for multiple-testing control
  - [ ] Paired bootstrap on trade sets (for P1-style metric-inflation comparisons)
- [ ] **Provenance/logging schema**: one parquet (or DuckDB) store — `strategy_id, market, timeframe, params, trade_id, entry_ts, exit_ts, pnl, exit_reason` — every downstream paper queries this one table. Decide the schema once, in writing, before the first backtest run.
- [ ] **Reference manager**: Zotero or Paperpile, one shared library, tagged by paper ID (P1, P2, …) so citations don't get re-hunted per paper.
- [ ] **Paper template**: one LaTeX (or Quarto/Markdown→PDF) skeleton — abstract, intro, related work, method, data, results, robustness, limitations, references — cloned per paper so formatting isn't reinvented 16 times.
- [ ] **Target-venue list locked**: pull each paper's "Venue" line above into a tracker (spreadsheet or `docs/SUBMISSION_TRACKER.md`) with author-guidelines links and word/page limits per venue — check before writing, not after.

## Phase 1 — Build the Unified Campaign (UC) · Months 1–3

This is the single highest-leverage step: 12 of 16 papers depend on it. Do this once, well.

1. **Inventory the portable signal functions.** Cross-reference `docs/STRATEGY_INDEX.md` and `docs/CODE_MAP.md` against `src/qresearch/signals/` — list every notebook strategy not yet consolidated into the library.
2. **Port each strategy's signal function** into `qresearch.signals` (the ~73-function lift `P25` already names as its own cost — shared, not duplicated, effort).
3. **Wire the factorial harness**: signal family × 5 markets × bracket geometry × exit semantics (touch-SL / close-confirmed / next-open, per P17) × volatility gate (per P23) × 3 exit resolutions (per P1).
4. **Run it once, fully**, writing every trade to the Phase 0 provenance store.
5. **Sanity-check the store**: row counts per strategy/market match expectations, no silent drops, spot-check 3 strategies' results against their original notebook output.
6. In parallel (no data dependency, can start Week 1): implement **P29's rainflow module** (ASTM E1049 or the `rainflow` PyPI package) against equity curves that already exist from current backtests — doesn't need to wait on the UC.

Exit criterion for Phase 1: one queryable table that every paper below reads from, plus a rainflow-annotated equity-curve set for P29.

## Phase 2 — Per-Paper Pipeline (repeat for each paper, in rank order)

Same seven steps for every paper — P15 first, then P25, P1, P29, P17, … per the ranked table above. Don't start step 3 on paper N+1 before paper N clears step 5 (peer feedback on paper N often changes how paper N+1's stats section gets written).

1. **Freeze the research question** — one sentence, no strategy/market/indicator named (the doc's own novelty-discipline rule, line ~410). If it can't be stated that way, it's a robustness section of an existing paper, not a new one.
2. **Run the paper-specific analysis** against the UC store (or, for P3/P4, against the standalone datasets already in-repo) — this is just a query + stats script, since the expensive backtesting already happened in Phase 1.
3. **Apply the "Needed before publication" checklist** listed under that paper's entry above, item by item.
4. **Generate figures/tables** — one script per paper in `scripts/papers/p{N}_*.py`, so figures are regenerable, not hand-made in Excel.
5. **Draft** using the Phase 0 template. Related-work section pulls from the "State of the art / gap" paragraph already written above — that's the literature review's first draft, not a blank page.
6. **Internal review pass**: reread against the paper's own novelty/gap claim — does the empirical section actually support the one-sentence research question from step 1?
7. **Preprint to arXiv (q-fin) + SSRN**, then submit to the "Venue" named above. Log submission date and status in `docs/SUBMISSION_TRACKER.md`.

## Phase 3 — Sequencing

Follow the ranked order already computed in "Combined Roadmap v2" and the 12-Month Plan v2 table above — don't re-derive it. Two things override strict rank order:

- **P3** (funding basis arb) has a trade-secret gate — decide *before* Month 9 whether this is published or kept proprietary. This is a business decision, not a research one; make it early so it doesn't block the pipeline.
- **P8** is a fold-in, not a standalone paper — don't schedule separate writing time for it; it surfaces as a section inside P2.

## Phase 4 — Revision & Second Shelf · Months 11–12+

- [ ] Track referee reports for Month 5–7 submissions (P1, P15, P17, P29 land first); budget 2–4 weeks per revision round.
- [ ] Triage second-shelf papers (P6, P19, P20) for year 2 — they're fully de-risked once the UC store exists, so this is scheduling, not new infra work.
- [ ] Re-run the novelty-discipline one-sentence test on anything new that surfaces from notebooks not yet classified, before adding it to the pipeline.

## Tracking

Keep a running per-paper status line (not a new file unless this grows unwieldy):

| Paper | Phase-2 step reached | Blocked on | Target submit month |
|---|---|---|---|
| P15 | — | UC (Phase 1) | 5 |
| P25 | — | UC (Phase 1) | 7 |
| P1 | — | UC (Phase 1) | 5 |
| P29 | — | equity curves (have) | 3 |
| … | | | |

Update this table as papers move; it's the single glance-check for "what's next."
