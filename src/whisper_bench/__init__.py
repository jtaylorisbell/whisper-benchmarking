"""whisper_bench — shared core for the Whisper batch-inference benchmark.

Both benchmark arms (Databricks AI Runtime and Model Serving) import from this package.
All timing, metrics, accuracy, cost, and results-schema logic lives here exactly once; the
only per-arm code is a ``Runner`` subclass in :mod:`whisper_bench.runners`. Databricks
notebooks under ``notebooks/`` are thin entrypoints that parse job parameters and call in here.
"""

from __future__ import annotations

__version__ = "0.2.0"

# Arm identifiers used throughout (kept here so there is a single source of truth).
ARM_AI_RUNTIME = "ai_runtime"
ARM_SERVING = "serving"
ARMS = (ARM_AI_RUNTIME, ARM_SERVING)
