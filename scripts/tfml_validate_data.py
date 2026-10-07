"""Cross-source data validation: pairs of series that track the same underlying should
have near-identical daily returns. Reports return correlation, tracking error, the share
of days with |return difference| > 2%, and the largest discrepancy (with its date) --
which flags vendor errors and, for unadjusted continuous futures, roll gaps.

    python scripts/tfml_validate_data.py   -> paper/tables/T0_data_validation.{md,csv}
"""
from pathlib import Path

import numpy as np
import pandas as pd

from qresearch.tfml import data as D

PAIRS = [
    ("BTC_USD", "daily", "BTCUSDT", "daily_binance", "BTC: Yahoo aggregate vs Binance spot"),
    ("ETH_USD", "daily", "ETHUSDT", "daily_binance", "ETH: Yahoo aggregate vs Binance spot"),
    ("XRP_USD", "daily", "XRPUSDT", "daily_binance", "XRP: Yahoo aggregate vs Binance spot"),
    ("GSPC", "daily", "SPY", "daily", "S&P 500 index vs SPY ETF (dividend-adjusted)"),
    ("GSPC", "daily", "ES_F", "daily", "S&P 500 index vs E-mini continuous future (unadjusted)"),
    ("IXIC", "daily", "QQQ", "daily", "NASDAQ Composite vs QQQ"),
    ("DJI", "daily", "DIA", "daily", "Dow index vs DIA"),
    ("RUT", "daily", "IWM", "daily", "Russell 2000 vs IWM"),
    ("EURUSD", "daily", "6E_F", "daily", "EURUSD spot vs Euro FX future"),
    ("USDJPY", "daily", "6J_F", "daily", "USDJPY spot vs Yen future (inverse quote)"),
    ("AUDUSD", "daily", "6A_F", "daily", "AUDUSD spot vs AUD future"),
    ("GC_F", "daily", "GLD", "daily", "Gold future vs GLD"),
    ("SI_F", "daily", "SLV", "daily", "Silver future vs SLV"),
    ("CL_F", "daily", "USO", "daily", "WTI future vs USO"),
    ("NG_F", "daily", "UNG", "daily", "Natural gas future vs UNG"),
    ("ZB_F", "daily", "TLT", "daily", "T-Bond future vs TLT"),
    ("N225", "daily", "EWJ", "daily", "Nikkei 225 vs EWJ (USD, different index)"),
    ("BSESN", "daily", "NSEI", "daily", "Sensex vs NIFTY 50"),
]


def main():
    rows = []
    for a, fa, b, fb, label in PAIRS:
        try:
            x = D.load_bars(a, fa)["close"]
            y = D.load_bars(b, fb)["close"]
        except FileNotFoundError:
            rows.append({"pair": label, "status": "missing"})
            continue
        x.index, y.index = x.index.normalize(), y.index.normalize()
        rx, ry = np.log(x).diff(), np.log(y).diff()
        if "inverse" in label:
            ry = -ry
        df = pd.concat([rx, ry], axis=1, keys=["a", "b"], sort=True).dropna()
        if len(df) < 250:
            rows.append({"pair": label, "status": "too little overlap"})
            continue
        diff = df.a - df.b
        worst = diff.abs().idxmax()
        rows.append({"pair": label, "series_a": f"{a}@{fa}", "series_b": f"{b}@{fb}",
                     "overlap_start": str(df.index[0].date()), "overlap_days": len(df),
                     "corr_daily": df.a.corr(df.b), "corr_weekly": df.resample("W").sum().corr().iloc[0, 1],
                     "tracking_err_ann": diff.std() * np.sqrt(252), "share_days_absdiff_gt_2pct": (diff.abs() > 0.02).mean(),
                     "max_abs_diff": diff.abs().max(), "max_diff_date": str(worst.date()), "status": "ok"})
    t = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "paper" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "T0_data_validation.csv", index=False)
    (out / "T0_data_validation.md").write_text(t.to_markdown(index=False, floatfmt=".3f"))
    print(t.to_string())


if __name__ == "__main__":
    main()
