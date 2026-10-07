"""Download + clean every instrument in configs/tfml/universe.yaml.

    python scripts/tfml_fetch_data.py            # download missing, then clean
    python scripts/tfml_fetch_data.py --force    # re-download everything
    python scripts/tfml_fetch_data.py --clean-only

Writes raw files to data/raw/tfml/, cleaned parquet to data/processed/tfml/,
and data/processed/tfml/manifest.json (checksums + cleaning counts).
"""
from __future__ import annotations

import argparse
import json

from qresearch.tfml import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--clean-only", action="store_true")
    ap.add_argument("--skip-binance", action="store_true")
    args = ap.parse_args()

    universe = data.load_universe()
    print(f"universe: {len(universe)} instruments")
    if not args.clean_only:
        st = data.fetch_yahoo(universe, force=args.force)
        print("yahoo:", json.dumps(st)[:2000])
        print("fred:", data.fetch_fred(["DTB3"]))
        if not args.skip_binance:
            st = data.fetch_binance_klines(universe, "1h", force=args.force)
            print("binance:", st)
    man = data.build_processed(universe)
    ok = [k for k, v in man["instruments"].items() if v.get("status") == "ok"]
    bad = {k: v.get("status") for k, v in man["instruments"].items() if v.get("status") != "ok"}
    print(f"processed ok: {len(ok)}; not ok: {len(bad)}")
    for k, v in bad.items():
        print("  ", k, v)


if __name__ == "__main__":
    main()
