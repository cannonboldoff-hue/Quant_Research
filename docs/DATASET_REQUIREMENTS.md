# Dataset Requirements

## Canonical schema
All datasets use a **long format** with one row per (ticker, timestamp):

| Column | Type | Notes |
|---|---|---|
| `Date` | datetime | Bar timestamp (tz-naive, exchange local or UTC — be consistent). |
| `Ticker` | str | Symbol, e.g. `BTC/USDT`, `RELIANCE.NS`. |
| `open/high/low/close` | float | OHLC prices. |
| `volume` | float | Bar volume. |

Store processed data as **Parquet** in `data/processed/` (faster, typed). Raw pulls go in
`data/raw/`. All `data/` contents are git-ignored.

## Sources observed in the corpus
| Market | Source | Access |
|---|---|---|
| Crypto | Binance via `ccxt` / raw API | `BINANCE_API_KEY/SECRET` (public data needs no key). |
| Indian equities | NSE / Zerodha Kite | `KITE_API_KEY/SECRET`. |
| US equities | Yahoo Finance via `yfinance` | No key. |
| Forex/Options/Futures | Broker/vendor exports | Provide as CSV/Parquet in `data/raw/`. |

## Resampling
Base data is typically fetched at 1-minute or 1-hour resolution and resampled with
`qresearch.data.resample_ohlcv(df, timeframe)`. The JMA signal family generates on 1H and evaluates
exits on 1-minute bars, so retain both resolutions where possible.
