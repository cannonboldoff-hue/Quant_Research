# 22 — ML-enhanced vs rule-based trend following

`tfml_results.ipynb` walks through the results of the study in `paper/paper.md`. It is
generated and executed by `scripts/tfml_make_notebook.py`, and every table in it is read
from the experiment registry (`results/tfml/registry.sqlite`), never typed in.

The computation lives in the `qresearch.tfml` package (see `docs/TFML_FRAMEWORK.md`), not
in notebooks, so the paper, the notebook and the registry cannot diverge.
