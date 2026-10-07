"""Generate and execute notebooks/22_ml_trend_following/tfml_results.ipynb -- a walkthrough
that reads the experiment registry (no numbers are typed into the notebook)."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "22_ml_trend_following" / "tfml_results.ipynb"

cells = [
    ("md", "# ML-enhanced vs rule-based trend following — results walkthrough\n\n"
           "Everything below is read from the experiment registry (`results/tfml/registry.sqlite`) "
           "produced by `scripts/tfml_run.py`. See `paper/paper.md` for the write-up and "
           "`docs/TFML_FRAMEWORK.md` for the pipeline. Re-run `scripts/tfml_make_notebook.py` to refresh."),
    ("code", "import pandas as pd, numpy as np, json\n"
             "from qresearch.tfml import registry as R, strategies as S\n"
             "pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)\n"
             "runs = R.read('select run_id, started, n_instruments, n_rows, code_hash, dataset_hash from runs')\nruns"),
    ("md", "## 1. Lineage of one result (Strategy → … → Result)"),
    ("code", "R.read(\"\"\"select strategy, indicators, market, instrument, dataset, ml_stage, ml_model, selected_models,\n"
             "validation, backtest, sharpe, base_sharpe, d_sharpe, p_boot_two, result_path, code_hash, exp_id\n"
             "from experiments where run_id='benchmark' and level='portfolio' and ml_model='selected' limit 5\"\"\")"),
    ("md", "## 2. Rule-based baselines (benchmark, OOS portfolio level)"),
    ("code", "b = R.read(\"select strategy, strategy_family, cagr, sharpe, sortino, calmar, max_drawdown, turnover, cost_drag, \"\n"
             "           \"win_rate_trades, profit_factor_trades from experiments where run_id='benchmark' and level='portfolio' and ml_stage='none'\")\n"
             "b.sort_values('sharpe', ascending=False).round(3)"),
    ("md", "## 3. ML improvement by stage (selected model, bootstrap p-values, BH across 39×5 tests)"),
    ("code", "from qresearch.tfml import stats as ST\n"
             "pf = R.read(\"select strategy, ml_stage, base_sharpe, sharpe, d_sharpe, p_boot_two from experiments \"\n"
             "            \"where run_id='benchmark' and level='portfolio' and ml_model='selected'\")\n"
             "pf['p_bh'] = ST.bh(pf.p_boot_two.values)\n"
             "pf.groupby('ml_stage').agg(mean_dSharpe=('d_sharpe','mean'), share_improved=('d_sharpe', lambda x: (x>0).mean()),\n"
             "    sig_pos_bh=('p_bh', lambda p: int((p<0.05).sum()))).round(3)"),
    ("code", "pf.pivot(index='strategy', columns='ml_stage', values='d_sharpe').round(2)"),
    ("md", "## 4. Model comparison and selection frequency"),
    ("code", "m = R.read(\"select ml_stage, ml_model, avg(d_sharpe) mean_dSharpe, count(*) n from experiments where run_id='benchmark' \"\n"
             "           \"and level='portfolio' and ml_stage!='none' group by ml_stage, ml_model\")\n"
             "m.pivot(index='ml_model', columns='ml_stage', values='mean_dSharpe').round(3)"),
    ("code", "sel = R.read(\"select stage, model, sum(selected) times_selected from ml_selection where run_id='benchmark' group by stage, model\")\n"
             "sel.pivot(index='model', columns='stage', values='times_selected')"),
    ("md", "## 5. Composite multi-strategy portfolio (OOS)"),
    ("code", "comp = pd.read_csv('../../paper/tables/benchmark_composite_daily_returns.csv', index_col=0, parse_dates=True)\n"
             "ax = np.log((1+comp).cumprod()).plot(figsize=(9,4), title='Composite of all strategies: baseline vs ML stages')\n"
             "pd.read_csv('../../paper/tables/benchmark_T17_composite_portfolio.csv').round(3)"),
    ("md", "## 6. Robustness across runs (validation method, frequency, survivorship-biased universe)"),
    ("code", "pd.read_csv('../../paper/tables/T18_cross_run_robustness.csv').round(3)"),
]

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(OUT.parent)}}).execute()
nbf.write(nb, OUT)
print("wrote", OUT)
