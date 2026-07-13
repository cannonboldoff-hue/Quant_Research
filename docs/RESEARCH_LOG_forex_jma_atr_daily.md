# Research Log: Forex JMA+ATR Daily — Multi-Pair Alpha Search

Status as of 2026-07-13: **provisionally validated candidate, not yet paper-traded.**
Strategy: `jma_signals` (JMA crossover) + ATR-based stop/target, daily bars, walk-forward
selected params per ticker per fold (`src/qresearch/optimize/finalist_selection.py`).

## TL;DR — where this landed

Best candidate found: **AUDNZD + CADJPY + AUDUSD + NZDUSD**, equal-weight combined,
daily JMA+ATR, walk-forward OOS trades only.

| | Sharpe | maxDD | CI vs 0 (bootstrap) | deflated Sharpe (n_trials=28, corrected for the 28-pair screen) |
|---|---|---|---|---|
| top3 (AUDNZD,CADJPY,AUDUSD) | 4.15 | -23.5% | [0.159, 0.375] sig | 0.998 |
| **top4 (+NZDUSD, recommended)** | 3.75 | -24.3% | [0.157, 0.327] sig | **1.000** |

This cleared every backtest-side check available: bootstrap significance, the
multiple-testing-corrected deflated Sharpe bar (0.95), split-robustness across
4 nested holdout windows, and manual trade-by-trade inspection (no data
artifacts, no single-trade blowups, realistic prices throughout). What it has
**not** cleared: any live/forward/paper-traded data. Next step is paper-trading,
not more backtesting — see "Open items" below.

## How we got here (chronological)

1. **3-market combo (commodities+indices+forex)**, tangency/equal/inverse-vol
   weighting — gate (CI vs best single sleeve) failed at every split, even
   after pulling 2000-2026 history from Yahoo (was 2020-2025 only originally).
   `scripts/build_portfolio_subset.py`.
2. **Single-market solo runs** — forex alone: Sharpe 1.65-1.75, maxDD only
   -15% to -17%, cleanest of the three markets. Commodities alone: Sharpe
   ~1.4 but maxDD -72% (driven by unadjusted continuous-futures data, e.g.
   WTI's real April-2020 negative-price day). Indices alone: weakest, Sharpe
   0.78-1.56.
3. **Forex 8-pair sleeve broken down per-pair** — the aggregate Sharpe 1.44 was
   NOT broad: dropping NZDUSD+AUDUSD (its top 2 contributors) killed
   significance entirely (CI [-0.036, 0.110]). NZDUSD+AUDUSD alone: Sharpe
   3.36, CI [0.112, 0.321], deflated Sharpe 0.996-1.000 (n_trials=1).
4. **Intraday check (4h/6h/12h) on NZDUSD+AUDUSD** — edge does NOT survive:
   4h Sharpe -0.56 (negative), 6h 0.57, 12h 2.24, none significant. Daily-only
   phenomenon — not disqualifying (trend-following commonly needs daily+ bars
   to beat noise) but means "works at every timeframe" is NOT a claim you can
   make here.
5. **Trade-level manual inspection of NZDUSD+AUDUSD** — clean. The -21.8%
   full-period max drawdown is a genuine 2011-2015 grind of small stop-outs,
   not an artifact. The apparent "2024 outlier" (Sharpe 14, CAGR 204%) is a
   **small-sample annualization artifact** — only 14 trading days that year,
   individually-normal trades (2-5% wins) compounded via `(1+r)^252` into a
   meaningless number. Not a data bug.
6. **2026 forward check** — we already had 2026 data (Yahoo fetched through
   today, 2026-07-13). 6 trades / 5 days in 2026 so far, net +7.98%, no red
   flags — but far too thin to be validation, just "hasn't broken yet."
7. **28-pair screen** — extended to ALL C(8,2)=28 major+cross pairs of
   EUR/GBP/USD/JPY/CHF/CAD/AUD/NZD (fetched via `scripts/fetch_yahoo_daily.py`,
   `FOREX_SPECS` in `src/qresearch/backtest/costs.py` extended with 20 new
   entries — **caveat: those 20 pairs' spread/swap costs are rough estimates,
   not sourced/verified like the original 8**). Naive per-pair bootstrap CI
   flagged 4/28 significant (AUDNZD, CADJPY, AUDUSD, NZDUSD) — but naive CI
   doesn't correct for having tested 28 candidates. Recomputed with
   `deflated_sharpe(n_trials=28)`: none of the 4 individually cleared 0.95
   (best was AUDNZD at 0.858).
8. **Combined top3/top4** — combining the (individually marginal-after-correction)
   candidates pushed the corrected deflated Sharpe to 0.998-1.000 — genuine
   near-zero cross-correlation (0.005 to 0.13) means diversification compounds
   significance, not just averages it. This is the strongest number the whole
   session produced.
9. **Trade-level inspection of AUDNZD+CADJPY** — also clean. Same
   annualization-artifact pattern confirmed generalizes (CADJPY 2023 shows a
   literal "422,066% CAGR" from 2 trades — ignore any single-year number for
   any of these pairs, always look at the aggregate/split-robust stats).
10. **Adding NZDUSD back (top3→top4)** — no reason it was excluded beyond
    literal "top 3" framing. Adding it: Sharpe drops slightly (4.15→3.75,
    NZDUSD is the weakest of the 4 individually) but sample grows 277→418
    days and deflated Sharpe edges up to 1.000. Recommended over top3.

## Methodology notes (read before extending this)

- **Walk-forward**: `run_finalist_trades(strategy_id, market, timeframe, processed_dir, tickers=...)`
  in `src/qresearch/optimize/finalist_selection.py`. `strategy_id` is just a
  label — the function always uses `jma_signals` + `SIGNAL_GRID` (fast∈{5,7,9},
  slow∈{21,28}) + `STOP_GRID` unless overridden. 5-way anchored split
  (`split_data_by_periods`, row-count based), grid-search best in-sample combo
  per fold, evaluate OOS on the next fold. `market`+`timeframe`+`processed_dir`
  alone determine the data source folder — no strategies.yaml entry needed for
  new timeframes/pairs.
- **Cost model**: forex trades get `cost_adjust()` → `realistic_net_ret()`
  (`src/qresearch/backtest/costs.py`) — real per-pair spread+swap, NOT the flat
  7bps engine fee (which overcharges forex ~10x). Applies on both fit and OOS
  legs of every fold.
- **Significance**: two tools from `src/qresearch/stats/risk.py`:
  - `paired_bootstrap_ci(returns, zeros_like(returns), statistic="sharpe")` —
    CI on Sharpe vs 0. Does NOT correct for multiple testing.
  - `deflated_sharpe(returns, n_trials=N)` — P(true Sharpe > 0), corrected for
    having tried N candidates. **Always set `n_trials` to the actual number of
    things you screened**, not 1, or you will fool yourself (see step 7 above).
    Conventional bar: ≥0.95 "survives multiple testing."
- **Annualization trap**: `portfolio_metrics()` annualizes via `(1+r)^bars_per_year`.
  With <15-20 data points this produces nonsense (seen twice: NZDUSD 2024,
  CADJPY 2023). Never trust a single-year or thin-sample Sharpe/CAGR in
  isolation — only the full-sample and split-robust numbers.
- **Data**: `data/processed_yf/forex/daily/*.parquet` — Yahoo Finance via
  `scripts/fetch_yahoo_daily.py`, 2000/2003-2026, kept SEPARATE from the
  validated `data/processed/` tree (which only has 2020-2025 forex). Schema:
  `Date, Ticker, open, high, low, close, volume` (same as the validated tree).
  Re-run `fetch_yahoo_daily.py` to refresh to "today."

## File inventory (this research thread)

- `scripts/fetch_yahoo_daily.py` — fetches all 28 forex pairs +
  commodities/indices to `data/processed_yf/`. Idempotent, re-run to refresh.
- `scripts/build_portfolio_subset.py` — 3-market (commodities+indices+forex)
  combiner, `--recompute --processed-dir data/processed_yf` for the deep-history
  run. Not directly relevant to the forex-only candidate but where this thread
  started.
- `src/qresearch/backtest/costs.py` — `FOREX_SPECS` extended from 8→28 pairs
  (20 new entries are rough estimates, flagged in-file).
- `src/qresearch/data/loaders.py` — `load_yfinance()` fixed for newer
  yfinance's MultiIndex column output (was silently broken before this
  session; fix is generally useful, not forex-specific).
- `data/campaign_summaries/tradebook_all28_forex_daily.csv` — every OOS trade,
  all 28 pairs, walk-forward, full precision.
- `data/campaign_summaries/tradebook_top3_audnzd_cadjpy_audusd.csv` — filtered
  to the 3-pair candidate, sorted, full precision.
- `data/campaign_summaries/forex_28pair_screen.csv` — per-pair summary stats,
  all 28, ranked by Sharpe, with the naive `significant` column (does NOT
  reflect the n_trials=28 correction — recompute deflated Sharpe if reusing).
- `tests/test_portfolio_subset.py` — covers the combine-API wiring used
  throughout (unaffected by anything forex-specific above).

## Open items / what would actually move the needle next

1. **Paper-trade AUDNZD/CADJPY/AUDUSD/NZDUSD forward, no capital.** This is
   the only thing left that isn't "more backtesting" — no live/broker infra
   exists in this repo (`src/qresearch/execution/broker.py` is unwired
   scaffolding; `KITE_API_KEY`/`BINANCE_API_KEY` are dead config; Kite can't
   trade spot forex anyway — India-only). Would need: a daily scheduled job,
   a live/near-live forex data source, and a signal log — no real broker
   needed for the paper-trade phase.
2. **Verify the 20 new `FOREX_SPECS` cost entries**, at least for CADJPY and
   AUDNZD specifically (the two new discoveries) — currently rough estimates,
   not sourced.
3. **Check AUDNZD/CADJPY at intraday timeframes** — only ever checked for
   NZDUSD+AUDUSD (failed). Unknown for the other two.
4. **Decide a paper-trade gate up front** (e.g. N months, bootstrap CI on the
   NEW data must still exclude 0) before considering any real capital —
   discussed but not formalized into a written rule yet.
5. Position sizing for eventual live use: reuse `risk/leverage_sim.py`'s
   `sizing_mode="disciplined"` formula (`risk_pct / stop_pct`, capped by
   leverage) — not `"aggressive"`.
