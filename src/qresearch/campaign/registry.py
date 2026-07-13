"""Strategy registry: strategy_id -> {signal_fn, market, timeframe, params}.

Loaded from configs/strategies.yaml. docs/STRATEGY_INDEX.md + docs/CODE_MAP.md
+ docs/notebook_mapping.csv are the source of truth for what belongs in the
registry -- the 73 notebooks collapse into a handful of unique signal
patterns (JMA+ATR, SMC, KAMA/DSL, funding-rate arb, HalfTrend+session-reset),
so entries here point at one parameterized function per pattern, not one
function per notebook.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "configs" / "strategies.yaml"


class StrategyRegistry:
    def __init__(self, config_path: str | Path = _DEFAULT_PATH):
        self.config_path = Path(config_path)
        self._entries: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        with open(self.config_path) as f:
            spec = yaml.safe_load(f) or {}
        for entry in spec.get("strategies", []):
            if entry.get("status") == "unported":
                continue
            mod = importlib.import_module(f"qresearch.signals.{entry['signal_module']}")
            signal_fn = getattr(mod, entry["signal_func"])
            self._entries[entry["id"]] = {
                "signal_fn": signal_fn,
                "market": entry["market"],
                "timeframe": entry.get("timeframe", "1h"),
                "default_params": entry.get("default_params") or {},
                "param_grid": entry.get("param_grid") or {},
            }

    def get(self, strategy_id: str) -> dict:
        return self._entries[strategy_id]

    def for_market(self, market: str, timeframe: str | None = None) -> dict[str, dict]:
        return {sid: e for sid, e in self._entries.items()
                if e["market"] == market and (timeframe is None or e["timeframe"] == timeframe)}

    def __iter__(self):
        return iter(self._entries.items())

    def __len__(self) -> int:
        return len(self._entries)
