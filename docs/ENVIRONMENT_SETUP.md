# Environment Setup

## 1. Python
Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .                     # editable install of qresearch
pip install -r requirements.txt      # full research stack
```

## 2. TA-Lib (native dependency)
TA-Lib needs the C library first:
- **macOS:** `brew install ta-lib`
- **Ubuntu:** `sudo apt-get install ta-lib` (or build from source)
- **Windows:** install a prebuilt wheel

Then `pip install TA-Lib`. The `qresearch.indicators` module falls back to pure-pandas
implementations if TA-Lib is absent, so the package works without it.

## 3. Secrets
```bash
cp .env.example .env
```
Fill broker keys in `.env` (loaded via `qresearch.config.settings`). Never commit `.env`.

## 4. Notebook hygiene
Enable output stripping so notebooks stay Git-friendly:
```bash
pip install nbstripout && nbstripout --install
```
(`.gitattributes` already wires the filter.)

## 5. Sanity check
```bash
pytest -q                            # runs the import + backtest smoke test
```
