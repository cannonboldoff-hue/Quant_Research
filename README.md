# Quant Research Repository

> **Main study: does machine learning improve trend following?** `paper/paper.md`
> (also `paper/paper.html`) compares 39 established trend-following strategies against
> ML-enhanced versions. ML enters at five pipeline stages, with five models per stage,
> under purged walk-forward validation, on a public multi-asset dataset. The code is the
> `qresearch.tfml` subpackage; see [`docs/TFML_FRAMEWORK.md`](docs/TFML_FRAMEWORK.md)
> (architecture, bias controls, registry schema) and [`docs/TFML_DATA.md`](docs/TFML_DATA.md)
> (sources, cleaning, limitations). To reproduce everything:
>
> ```bash
> pip install -e . && pip install -r requirements.txt
> python scripts/tfml_fetch_data.py          # download + clean (Yahoo, Binance, FRED)
> bash scripts/tfml_run_all.sh 6             # all experiment runs -> results/tfml/registry.sqlite
> python scripts/tfml_report.py              # tables / figures / results_summary.json
> python paper/build_paper.py                # paper.md + paper.html (numbers injected from results)
> pytest -q                                  # incl. look-ahead tests for every feature & strategy
> ```
>
> Results walkthrough notebook: `notebooks/22_ml_trend_following/tfml_results.ipynb`.

An institutional-grade quantitative trading research repository, rebuilt from 238 exploratory
Jupyter notebooks accumulated over 3+ years. The raw notebooks were classified, de-duplicated,
output-stripped, renamed, documented, and reorganized into logical sections, with all repeated
code consolidated into a reusable `qresearch` Python package.

## What this repository contains

- **`src/qresearch/`** — the shared library. All the logic that was previously copy-pasted across
  notebooks (data loaders, indicators, signal generation, backtesting, metrics, risk, optimization,
  visualization, execution) now lives here as clean, documented, importable modules.
- **`notebooks/`** — research notebooks organized into 22 numbered sections (data → strategies →
  execution → archive). Every notebook opens with an auto-generated documentation header and carries
  provenance metadata linking it back to its original filename.
- **`docs/`** — the strategy index, research index, code map, architecture, dataset requirements,
  environment setup, and roadmap.

## The research program (as reconstructed from the code)

The corpus centers on a **Jurik Moving Average (JMA) + ATR** signal family: fast/slow JMA crossovers
generate entries on a higher timeframe (mostly 1H), which are then mapped down to 1-minute bars for
realistic stop-loss / take-profit exit simulation. Strategies are backtested per ticker, scored on a
weighted composite of Sharpe / return / win-rate / drawdown, ranked, and validated with
walk-forward optimization. Markets covered: crypto (Binance), Indian equities, options, forex, and
US equities.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # installs the qresearch package
cp .env.example .env        # add broker keys here (never commit them)
```

```python
from qresearch.data import load_crypto_ohlcv, resample_ohlcv
from qresearch.signals import jma_signals
from qresearch.backtest import run_backtest

df   = load_crypto_ohlcv("BTC/USDT", timeframe="1h", limit=1000)
sig  = jma_signals(df, fast=7, slow=21)
res  = run_backtest(sig, sl_mult=1.5, tp_mult=3.0)
print(res["metrics"])
```

## Repository layout

```
quant_research_repo/
├── src/qresearch/          # shared, reusable library (the source of truth for logic)
│   ├── config/             # settings + constants (schema, timeframes, sessions)
│   ├── data/               # loaders + resampling (canonical long OHLCV schema)
│   ├── indicators/         # JMA, ATR, RSI/MACD (TA-Lib wrappers + fallbacks)
│   ├── features/           # feature-engineering pipeline
│   ├── signals/            # signal generators + timeframe mapping
│   ├── backtest/           # vectorized engine + performance metrics/ranking
│   ├── risk/               # ATR stops, trailing stops, take-profit
│   ├── portfolio/          # Kelly / fixed-fractional sizing
│   ├── optimize/           # grid search + walk-forward
│   ├── execution/          # broker abstraction (paper + live adapters)
│   ├── viz/                # equity curve, PnL distribution, trade summary
│   └── utils/              # logging, timing, IO
├── notebooks/              # 01_data_collection … 21_experiments, 99_archive
├── docs/                   # indexes + architecture + setup
├── data/                   # raw / processed / external (git-ignored)
├── tests/                  # import + smoke tests
└── requirements.txt, pyproject.toml, .env.example, .gitignore
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full folder guide,
[`docs/STRATEGY_INDEX.md`](docs/STRATEGY_INDEX.md) for every strategy,
[`docs/RESEARCH_INDEX.md`](docs/RESEARCH_INDEX.md) for supporting research, and
[`docs/CODE_MAP.md`](docs/CODE_MAP.md) for the legacy-function → module mapping.

## Provenance & safety

- **Nothing was deleted.** The original 238 notebooks remain untouched in your `Research_Papers`
  folder. Within this repo, exact/near-duplicate notebooks (74 of them) were moved to
  `notebooks/99_archive/duplicates/` rather than discarded.
- Every notebook's `metadata.qresearch` records its original filename and classification.
- `docs/notebook_mapping.csv` is the full original → new-path audit trail.
- Cell outputs (≈111 MB) were stripped for Git-friendliness; re-run notebooks to regenerate them.

## Caveats

Classification (section / market / style / timeframe) is heuristic, derived from static code
analysis. The strategy and research indexes are a strong first pass to review and correct — not a
substitute for your own judgment. The `qresearch` modules are clean reference implementations of the
consolidated patterns; wire your exact strategy parameters and data sources into them as you migrate
each notebook.
# RESEARCH_PAPERS
