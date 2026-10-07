"""Print the strategy ids of a run that are not yet in the registry (for resuming)."""
import sys

from qresearch.tfml import registry as R
from qresearch.tfml import strategies as S

run_id = sys.argv[1]
try:
    done = set(R.read("select distinct strategy from experiments where run_id = ?", params=(run_id,))["strategy"])
except Exception:
    done = set()
print(" ".join(s.id for s in S.STRATEGIES if s.id not in done))
