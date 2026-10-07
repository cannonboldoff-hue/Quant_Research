"""Data layer: universe config, downloaders (Yahoo, Binance public archive, FRED,
Stooq files), validation/cleaning, and a dataset manifest with checksums.

Raw downloads are stored untouched under ``data/raw/tfml/<source>/``; cleaned
bars go to ``data/processed/tfml/<frequency>/<instrument_id>.parquet`` with
columns ``Date, open, high, low, close, volume``. Every cleaning action is
counted per instrument and written to ``data/processed/tfml/manifest.json`` so
the paper's data section is generated from what actually happened.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "data" / "raw" / "tfml"
PROCESSED = REPO / "data" / "processed" / "tfml"
UNIVERSE_YAML = REPO / "configs" / "tfml" / "universe.yaml"


# --------------------------------------------------------------------------- universe
@dataclass(frozen=True)
class Instrument:
    id: str            # filesystem-safe id, e.g. "GSPC" or "EURUSD"
    ticker: str        # source ticker, e.g. "^GSPC", "EURUSD=X", "BTCUSDT"
    group: str         # universe group, e.g. "equity_index"
    asset_class: str
    source: str        # yahoo | binance
    exchange: str
    region: str
    sector: str
    name: str
    cost_bps: float
    slippage_bps: float
    survivorship: str
    benchmark: bool


def safe_id(ticker: str) -> str:
    t = ticker.replace("^", "").replace("=X", "").replace("=F", "_F").replace("-USD", "_USD")
    return t.replace(".", "_").replace("/", "_").replace("-", "_")


def load_universe(path: Path = UNIVERSE_YAML) -> list[Instrument]:
    cfg = yaml.safe_load(Path(path).read_text())
    out = []
    for group, g in cfg["groups"].items():
        for ticker, meta in g["instruments"].items():
            meta = meta or {}
            out.append(Instrument(
                id=safe_id(ticker), ticker=ticker, group=group, asset_class=g["asset_class"],
                source=g.get("source", cfg["defaults"]["daily_source"]),
                exchange=meta.get("exchange", ""), region=meta.get("region", ""),
                sector=meta.get("sector", g["asset_class"]), name=meta.get("name", ticker),
                cost_bps=float(g["cost_bps"]), slippage_bps=float(g["slippage_bps"]),
                survivorship=g.get("survivorship", "none"), benchmark=bool(g.get("benchmark", True)),
            ))
    ids = [i.id for i in out]
    dupes = {x for x in ids if ids.count(x) > 1}
    if dupes:
        raise ValueError(f"duplicate instrument ids: {dupes}")
    return out


def universe_frame(path: Path = UNIVERSE_YAML) -> pd.DataFrame:
    return pd.DataFrame([asdict(i) for i in load_universe(path)])


# --------------------------------------------------------------------------- downloads
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_yahoo(instruments: list[Instrument], start: str = "1985-01-01", end: str | None = None,
                batch: int = 20, pause: float = 2.0, force: bool = False) -> dict:
    """Download daily bars (unadjusted OHLC + Adj Close) for Yahoo instruments.
    Saves one raw CSV per instrument. Returns {id: status}."""
    import yfinance as yf

    out_dir = RAW / "yahoo"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in instruments if i.source == "yahoo" and (force or not (out_dir / f"{i.id}.csv").exists())]
    status = {}
    for k in range(0, len(todo), batch):
        chunk = todo[k:k + batch]
        tickers = [i.ticker for i in chunk]
        for attempt in range(4):
            try:
                df = yf.download(tickers, start=start, end=end, auto_adjust=False, actions=False,
                                 progress=False, group_by="ticker", threads=True)
                break
            except Exception as e:  # network / rate limit
                status["_error"] = str(e)
                time.sleep(10 * (attempt + 1))
        else:
            for i in chunk:
                status[i.id] = "download_failed"
            continue
        for i in chunk:
            try:
                sub = df[i.ticker] if isinstance(df.columns, pd.MultiIndex) else df
                sub = sub.dropna(how="all")
            except KeyError:
                sub = pd.DataFrame()
            if sub.empty:
                status[i.id] = "empty"
                continue
            sub.index.name = "Date"
            sub.to_csv(out_dir / f"{i.id}.csv")
            status[i.id] = f"ok:{len(sub)}"
        time.sleep(pause)
    return status


def fetch_binance_klines(instruments: list[Instrument], interval: str = "1h",
                         start: str = "2017-08", end: str | None = None, force: bool = False) -> dict:
    """Download Binance spot klines from the public archive data.binance.vision
    (monthly zip files), concatenated into one raw CSV per symbol."""
    import requests

    out_dir = RAW / "binance" / interval
    out_dir.mkdir(parents=True, exist_ok=True)
    end = end or pd.Timestamp.today().strftime("%Y-%m")
    months = pd.period_range(start, end, freq="M")[:-1]  # current month not yet archived
    status = {}
    sess = requests.Session()
    for inst in [i for i in instruments if i.source == "binance"]:
        target = out_dir / f"{inst.id}.csv"
        if target.exists() and not force:
            status[inst.id] = "cached"
            continue
        frames = []
        for m in months:
            url = (f"https://data.binance.vision/data/spot/monthly/klines/{inst.ticker}/{interval}/"
                   f"{inst.ticker}-{interval}-{m.strftime('%Y-%m')}.zip")
            for attempt in range(3):
                try:
                    r = sess.get(url, timeout=60)
                    break
                except Exception:
                    time.sleep(3)
            else:
                continue
            if r.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    part = pd.read_csv(f, header=None)
            if not str(part.iloc[0, 0]).isdigit():  # some months ship a header row
                part = part.iloc[1:]
            frames.append(part.iloc[:, :6])
        if not frames:
            status[inst.id] = "empty"
            continue
        raw = pd.concat(frames, ignore_index=True)
        raw.columns = ["open_time", "open", "high", "low", "close", "volume"]
        raw.to_csv(target, index=False)
        status[inst.id] = f"ok:{len(raw)}"
    return status


def fetch_fred(series: list[str] = ("DTB3",)) -> dict:
    import requests

    out_dir = RAW / "fred"
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {}
    for s in series:
        r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}", timeout=60)
        r.raise_for_status()
        (out_dir / f"{s}.csv").write_bytes(r.content)
        status[s] = "ok"
    return status


# --------------------------------------------------------------------------- cleaning
@dataclass
class CleanReport:
    id: str
    source: str
    raw_file: str
    raw_sha256: str
    raw_rows: int
    rows: int = 0
    start: str = ""
    end: str = ""
    dropped_nan_close: int = 0
    dropped_nonpositive: int = 0
    filled_missing_ohl: int = 0
    fixed_hl_inconsistency: int = 0
    removed_spikes: int = 0
    dropped_duplicate_dates: int = 0
    dropped_before_gap: int = 0
    max_gap_days: int = 0
    adjusted_for_dividends: bool = False
    status: str = "ok"
    notes: list = field(default_factory=list)


def _clean_ohlc(df: pd.DataFrame, rep: CleanReport, spike_k: float = 12.0,
                max_gap_days: int = 31) -> pd.DataFrame:
    df = df.sort_index()
    dup = df.index.duplicated(keep="last")
    rep.dropped_duplicate_dates = int(dup.sum())
    df = df[~dup]

    nan_close = df["close"].isna()
    rep.dropped_nan_close = int(nan_close.sum())
    df = df[~nan_close].copy()

    nonpos = df["close"] <= 0
    rep.dropped_nonpositive = int(nonpos.sum())
    if nonpos.any():
        rep.notes.append(f"non-positive closes on {list(df.index[nonpos].strftime('%Y-%m-%d'))[:5]}")
    df = df[~nonpos].copy()

    for c in ["open", "high", "low"]:
        bad = df[c].isna() | (df[c] <= 0)
        rep.filled_missing_ohl += int(bad.sum())
        df.loc[bad, c] = df.loc[bad, "close"]

    hi = df[["open", "close"]].max(axis=1)
    lo = df[["open", "close"]].min(axis=1)
    bad_hl = (df["high"] < hi - 1e-12) | (df["low"] > lo + 1e-12)
    rep.fixed_hl_inconsistency = int(bad_hl.sum())
    df["high"] = np.maximum(df["high"], hi)
    df["low"] = np.minimum(df["low"], lo)

    # Isolated bad prints: a huge move immediately reversed. Robust scale = MAD of log returns.
    removed = 0
    for _ in range(3):
        lr = np.log(df["close"]).diff()
        mad = (lr - lr.median()).abs().median() * 1.4826
        if not np.isfinite(mad) or mad == 0:
            break
        z = lr / mad
        spike = (z.abs() > spike_k) & (z.shift(-1).abs() > spike_k) & (np.sign(z) != np.sign(z.shift(-1)))
        if not spike.any():
            break
        removed += int(spike.sum())
        df = df[~spike]
    rep.removed_spikes = removed

    # Long data holes (vendor outages, sparse early history) break indicator state and
    # create one fake multi-month "bar"; keep only the segment after the last such hole.
    gaps = df.index.to_series().diff()
    big = gaps[gaps > pd.Timedelta(days=max_gap_days)]
    if len(big):
        cut = big.index[-1]
        rep.dropped_before_gap = int((df.index < cut).sum())
        rep.notes.append(f"trimmed {rep.dropped_before_gap} rows before {cut.date()} (gap {big.iloc[-1].days}d)")
        df = df[df.index >= cut]
    if "volume" not in df:
        df["volume"] = 0.0
    df["volume"] = df["volume"].fillna(0.0)
    gaps = df.index.to_series().diff().dt.days
    rep.max_gap_days = int(gaps.max()) if len(gaps) > 1 else 0   # after trimming
    return df[["open", "high", "low", "close", "volume"]]


def clean_yahoo(inst: Instrument, adjust: bool | None = None) -> tuple[pd.DataFrame, CleanReport]:
    path = RAW / "yahoo" / f"{inst.id}.csv"
    raw = pd.read_csv(path, index_col="Date", parse_dates=True)
    rep = CleanReport(inst.id, "yahoo", str(path.relative_to(REPO)), _sha256(path), len(raw))
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    df = raw.rename(columns=str.lower)
    adjust = inst.asset_class in ("etf", "equity_single") if adjust is None else adjust
    if adjust and "adj close" in df:
        factor = (df["adj close"] / df["close"]).replace([np.inf, -np.inf], np.nan)
        factor = factor.ffill().bfill()
        for c in ["open", "high", "low", "close"]:
            df[c] = df[c] * factor
        rep.adjusted_for_dividends = True
    df = _clean_ohlc(df, rep)
    return df, rep


def clean_stooq(inst: Instrument) -> tuple[pd.DataFrame, CleanReport]:
    """Import a manually downloaded Stooq daily CSV (Date,Open,High,Low,Close[,Volume]) placed at
    data/raw/tfml/stooq/<id>.csv. Stooq's download endpoint is CAPTCHA-gated, so the pipeline
    does not fetch it automatically; files are used only if the user supplies them."""
    path = RAW / "stooq" / f"{inst.id}.csv"
    raw = pd.read_csv(path, index_col=0, parse_dates=True)
    rep = CleanReport(inst.id, "stooq", str(path.relative_to(REPO)), _sha256(path), len(raw))
    df = raw.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index).normalize()
    return _clean_ohlc(df, rep), rep


def clean_binance(inst: Instrument, interval: str = "1h") -> tuple[pd.DataFrame, CleanReport]:
    path = RAW / "binance" / interval / f"{inst.id}.csv"
    raw = pd.read_csv(path)
    rep = CleanReport(inst.id, "binance", str(path.relative_to(REPO)), _sha256(path), len(raw))
    ts = raw["open_time"].astype("int64")
    # Binance switched spot archive timestamps from ms to microseconds in 2025.
    ts = np.where(ts > 10 ** 14, ts // 1000, ts)
    raw.index = pd.to_datetime(ts, unit="ms")
    raw.index.name = "Date"
    df = _clean_ohlc(raw[["open", "high", "low", "close", "volume"]].astype(float), rep, max_gap_days=3)
    return df, rep


def resample_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["close"])


def build_processed(instruments: list[Instrument], min_years: float = 3.0) -> dict:
    """Clean every downloaded instrument, write parquet + manifest. Returns manifest dict."""
    manifest = {"created": pd.Timestamp.utcnow().isoformat(), "instruments": {}}
    for inst in instruments:
        try:
            if inst.source == "yahoo":
                if not (RAW / "yahoo" / f"{inst.id}.csv").exists():
                    manifest["instruments"][inst.id] = {"status": "missing_raw"}
                    continue
                frames = {"daily": clean_yahoo(inst)}
            elif inst.source == "stooq":
                if not (RAW / "stooq" / f"{inst.id}.csv").exists():
                    manifest["instruments"][inst.id] = {"status": "missing_raw (manual Stooq download required)"}
                    continue
                frames = {"daily": clean_stooq(inst)}
            elif inst.source == "binance":
                if not (RAW / "binance" / "1h" / f"{inst.id}.csv").exists():
                    manifest["instruments"][inst.id] = {"status": "missing_raw"}
                    continue
                h1, rep = clean_binance(inst, "1h")
                frames = {"1h": (h1, rep), "4h": (resample_bars(h1, "4h"), rep),
                          "daily_binance": (resample_bars(h1, "1D"), rep)}
            else:
                continue
        except Exception as e:  # keep going; record failure
            manifest["instruments"][inst.id] = {"status": f"error: {e}"}
            continue
        for freq, (df, rep) in frames.items():
            years = (df.index[-1] - df.index[0]).days / 365.25 if len(df) else 0
            rep.rows, rep.start, rep.end = len(df), str(df.index[0].date()) if len(df) else "", \
                str(df.index[-1].date()) if len(df) else ""
            if years < min_years:
                rep.status = f"excluded_short_history({years:.1f}y)"
            else:
                out = PROCESSED / freq
                out.mkdir(parents=True, exist_ok=True)
                df.reset_index().to_parquet(out / f"{inst.id}.parquet", index=False)
            entry = asdict(rep)
            entry.update({"frequency": freq, "group": inst.group, "asset_class": inst.asset_class})
            manifest["instruments"][f"{inst.id}@{freq}"] = entry
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


def load_bars(instrument_id: str, frequency: str = "daily") -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / frequency / f"{instrument_id}.parquet")
    return df.set_index("Date")


def load_risk_free() -> pd.Series:
    """Daily 3-month T-bill rate (FRED DTB3), as a decimal annual rate, ffilled."""
    p = RAW / "fred" / "DTB3.csv"
    s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce") / 100.0
    return s.ffill()
