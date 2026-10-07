"""Keep Windows from sleeping while long experiment runs are in progress.

Uses SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) -- the same per-process
request media players use; no power settings are changed and the request ends when this
process exits. Exits automatically once the marker line appears in the watched log.

    python scripts/keep_awake.py results/tfml/phase2.log "phase2 done"
"""
import ctypes
import sys
import time
from pathlib import Path

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
log, marker = Path(sys.argv[1]), sys.argv[2]
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
try:
    while not (log.exists() and marker in log.read_text(errors="ignore")):
        time.sleep(60)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
print("released")
