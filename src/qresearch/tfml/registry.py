"""Experiment registry (SQLite): every reported number is a row that links

    Strategy -> Indicators -> Market -> Instrument -> Dataset -> ML Stage -> ML Model
    -> Validation -> Backtest -> Metrics -> Result

Tables
- ``experiments``: one row per evaluated configuration (portfolio / market group /
  instrument / sub-period / sensitivity level), with metrics, ML-vs-baseline
  deltas and test statistics, and the path of the stored return series.
- ``ml_selection``: per fold x stage x model validation scores and which model
  was selected (model comparison + selection audit trail).
- ``runs``: run-level configuration (JSON) + git commit + timings.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "results" / "tfml"
DB = RESULTS / "registry.sqlite"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:
        return "unknown"


def code_hash() -> str:
    """Content hash of the framework source + configs (identifies the exact code that
    produced a result even when the working tree is not committed)."""
    h = hashlib.sha256()
    files = sorted((REPO / "src" / "qresearch" / "tfml").glob("*.py")) + sorted((REPO / "configs" / "tfml").glob("*.yaml"))         + [REPO / "scripts" / "tfml_run.py"]
    for f in files:
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()[:16]


def exp_id(row: dict) -> str:
    keys = ["run_id", "strategy", "market", "instrument", "ml_stage", "ml_model", "validation",
            "backtest", "level", "period"]
    return hashlib.sha1("|".join(str(row.get(k)) for k in keys).encode()).hexdigest()[:16]


def connect(db: Path = DB) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db, timeout=300)


def write(table: str, rows: list[dict] | pd.DataFrame, db: Path = DB, replace_keys: dict | None = None):
    """Append rows; if ``replace_keys`` given, first delete rows matching them
    (idempotent re-runs of one strategy/run)."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty:
        return
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v)
    with connect(db) as con:
        if replace_keys:
            try:
                where = " AND ".join(f"{k} = ?" for k in replace_keys)
                con.execute(f"DELETE FROM {table} WHERE {where}", list(replace_keys.values()))
            except sqlite3.OperationalError:
                pass
        existing = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        if existing:
            for c in df.columns:
                if c not in existing:
                    con.execute(f'ALTER TABLE {table} ADD COLUMN "{c}"')
        df.to_sql(table, con, if_exists="append", index=False)


def read(sql: str, db: Path = DB, params=()) -> pd.DataFrame:
    with connect(db) as con:
        return pd.read_sql_query(sql, con, params=params)
