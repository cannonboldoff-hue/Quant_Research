# Data used by the ML-vs-rule-based trend-following study

All data is public and downloaded by `scripts/tfml_fetch_data.py`. The script reads the
universe from `configs/tfml/universe.yaml`. Raw files are never edited
(`data/raw/tfml/<source>/`). Cleaned bars are written to `data/processed/tfml/<freq>/`.
Every cleaning action and every raw-file SHA-256 is recorded in
`data/processed/tfml/manifest.json`. Coverage statistics in the paper (Table T1) are
generated from that manifest by `scripts/tfml_report.py`. The data directories are
git-ignored, so the manifest plus the fetch script is the reproducibility record. Vendors
revise history, so re-downloading can change the raw-file hashes; the registry stores the
dataset hash used by each run.

## Sources

| Source | Access | What | Frequency | Notes |
|---|---|---|---|---|
| Yahoo Finance (`yfinance`) | free, no key | 34 equity indices, 39 continuous futures (38 retrieved), 39 FX pairs (7 majors, 21 crosses, 11 EM), 15 crypto (USD), 49 US-listed ETFs, 30 US + 20 Indian large-cap stocks (19 retrieved) | daily | Unadjusted OHLC plus `Adj Close`. ETFs/stocks are dividend-adjusted by scaling OHLC with Adj Close / Close |
| Binance public archive (`data.binance.vision`) | free, no key | 12 spot USDT pairs | 1h (resampled to 4h and daily) | Monthly kline zips; timestamps switched from ms to µs in 2025 and are normalised |
| FRED (`fredgraph.csv`) | free, no key | `DTB3` 3-month T-bill | daily | Used only by Antonacci's absolute-momentum rule (T-bill hurdle) |
| Stooq | **CAPTCHA-gated download** | long histories, sugar futures | daily | Reached with a browser (Claude-in-Chrome). The CSV export requires solving a CAPTCHA, which the pipeline does not automate. `data.clean_stooq` imports files a user downloads manually into `data/raw/tfml/stooq/` |

## Cleaning rules (`data._clean_ohlc`)

1. Sort by date; drop duplicate timestamps (keep last).
2. Drop bars with missing or non-positive close. Example: WTI `CL=F` on 2020-04-20
   (−$37.63, a real print that breaks percentage returns).
3. Fill missing or non-positive open/high/low with the close.
4. Enforce `low ≤ min(open, close) ≤ max(open, close) ≤ high` by widening high/low (counted).
5. Remove isolated bad prints: a log return above 12 robust (MAD) standard deviations that
   is immediately reversed by an equally extreme move. Iterated up to 3 times.
6. **Gap trimming:** if a series has a hole longer than 31 calendar days (daily) or 3 days
   (intraday), only the segment after the last such hole is kept. Long holes come from
   vendor outages or sparse early history (e.g. USDBRL 2004–2006, USDSEK/NOK 2001–2003,
   pre-2010 platinum/palladium). They would otherwise turn months of missing data into a
   single fake bar.
7. Instruments with less than 3 years of history after cleaning are excluded.

## Known limitations (stated in the paper)

- **Continuous futures are not roll-adjusted** (Yahoo `=F` series): roll gaps remain in
  returns and signals. Commodity ETFs (USO, UNG, DBC, …) are included as an investable,
  roll-inclusive alternative.
- **Equity indices are price indices** without dividends.
- **FX returns exclude carry/swap.** This matters most for EM pairs.
- **Crypto (Yahoo, `-USD`)**: coins were chosen today among long-lived large caps (mild
  survivorship bias). Weekend bars are included.
- **Single stocks** are chosen from *today's* S&P 100 / NIFTY 50 membership. They are
  survivorship-biased by construction, so they are excluded from the benchmark and only
  used in a labelled robustness run. `TATAMOTORS.NS` no longer exists on Yahoo after the
  2025 demerger, a live example of the bias.
- **Missing series:** `DX=F` is not available on Yahoo. `SB=F` (sugar) returns a single bar
  on Yahoo, and Stooq's history requires a CAPTCHA.
- Transaction-cost levels are assumptions per asset class (per side, bps; see
  `universe.yaml`), not broker quotes. The paper reports 0×/1×/3× sensitivity.
