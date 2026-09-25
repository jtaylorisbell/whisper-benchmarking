"""Per-arm runners. The only place arm-specific logic lives."""

from __future__ import annotations

from ..config import ARM_AI_RUNTIME, ARM_SERVING
from .base import Runner


def make_runner(suite, run_config, rate_catalog, suite_id) -> Runner:
    """Factory: pick the Runner subclass for a run's arm (imports arm modules lazily)."""
    if run_config.arm == ARM_AI_RUNTIME:
        from .ai_runtime import AiRuntimeRunner

        return AiRuntimeRunner(suite, run_config, rate_catalog, suite_id)
    if run_config.arm == ARM_SERVING:
        from .serving import ServingRunner

        return ServingRunner(suite, run_config, rate_catalog, suite_id)
    raise ValueError(f"no runner for arm {run_config.arm!r}")


__all__ = ["Runner", "make_runner"]
