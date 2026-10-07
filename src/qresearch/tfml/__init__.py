"""qresearch.tfml -- ML-enhanced vs rule-based trend-following research framework.

Modular pipeline (each layer swappable via config):

    universe/data -> features (indicators) -> strategies (rule-based baselines)
    -> ML stages (filter / regime / sizing / exit / parameter adaptation)
    -> validation (walk-forward, purged) -> backtest (next-open execution, costs)
    -> metrics + statistics -> experiment registry -> paper tables/figures.

Every number reported in ``paper/`` is produced by ``scripts/tfml_run.py`` and
logged in the experiment registry (``results/tfml/registry.sqlite``).
"""
__all__ = ["data", "features", "strategies", "backtest", "metrics", "ml", "validation",
           "stats", "registry", "runner"]
