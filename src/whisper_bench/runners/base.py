"""Abstract Runner: the shared benchmark orchestration for both arms.

``run()`` is identical for every arm — it separates setup from inference timing, computes throughput
and accuracy, and turns measured inference seconds into cost via the rate catalog. Subclasses only
implement how they *transcribe* (and optionally how they set up / warm up). This is the DRY core:
add a new architecture by adding one ``transcribe`` method, nothing else.
"""

from __future__ import annotations

import abc
import uuid
from datetime import datetime, timezone
from typing import Optional

from ..config import RunConfig, Suite
from ..cost import RateCatalog, assemble_costs
from ..data import Clip
from ..metrics import PhaseTimer, compute_accuracy, percentile_ms, rtfx
from ..results import RunResult


class Runner(abc.ABC):
    def __init__(self, suite: Suite, run_config: RunConfig, rate_catalog: RateCatalog, suite_id: str):
        self.suite = suite
        self.run_config = run_config
        self.rate_catalog = rate_catalog
        self.suite_id = suite_id
        self.notes = ""

    # --- Hooks (override as needed). Everything except transcribe() defaults to a no-op. ---
    def setup(self) -> None:
        """Load the model / verify the endpoint is ready. Timed as setup (excluded from cost)."""

    def prepare(self, clips: list[Clip]) -> None:
        """Client-side prep, e.g. decode/encode audio. Timed as setup (excluded from cost)."""

    def warmup(self, clips: list[Clip]) -> None:
        """Warm the model/endpoint on a few clips. Timed as setup (excluded from cost)."""

    @abc.abstractmethod
    def transcribe(self, clips: list[Clip]) -> tuple[list[str], list[float]]:
        """Transcribe all clips. Returns (hypotheses in clip order, per-request latencies in ms).

        This — and ONLY this — is timed as inference and drives cost + throughput.
        """

    def teardown(self) -> None:
        """Release resources. Not timed."""

    def replicas(self) -> int:
        """Parallel billable units of the primary compute (endpoint replicas). Usually 1."""
        return 1

    # --- Shared orchestration ---
    def run(self, clips: list[Clip]) -> RunResult:
        timer = PhaseTimer()
        with timer.phase("setup"):
            self.setup()
            self.prepare(clips)
        warm = clips[: self.suite.warmup_clips]
        if warm:
            with timer.phase("setup"):
                self.warmup(warm)

        with timer.phase("inference"):
            hypotheses, latencies_ms = self.transcribe(clips)

        self.teardown()

        if len(hypotheses) != len(clips):
            raise RuntimeError(f"got {len(hypotheses)} hypotheses for {len(clips)} clips")

        infer_sec = timer.get("inference")
        setup_sec = timer.get("setup")
        total_audio = sum(c.duration_sec for c in clips)
        acc = compute_accuracy([c.reference_text for c in clips], hypotheses)

        rc = self.run_config
        cb = assemble_costs(
            self.rate_catalog,
            compute_primary=rc.compute,
            compute_secondary=rc.orchestration_compute,  # driver job runs over the same inference window
            inference_wall_sec=infer_sec,
            total_audio_sec=total_audio,
            replicas=self.replicas(),
        )
        run_id = uuid.uuid4().hex
        return RunResult(
            run_id=run_id,
            suite_id=self.suite_id,
            arm=rc.arm,
            label=rc.label or rc.arm,
            model=self.suite.model,
            gpu_type=rc.gpu_type,
            batch_size=rc.batch_size,
            concurrency=rc.concurrency,
            n_clips=len(clips),
            total_audio_sec=total_audio,
            inference_wall_sec=infer_sec,
            setup_wall_sec=setup_sec,
            throughput_rtfx=rtfx(total_audio, infer_sec),
            latency_p50_ms=percentile_ms(latencies_ms, 50),
            latency_p95_ms=percentile_ms(latencies_ms, 95),
            wer=acc.wer,
            cer=acc.cer,
            wer_normalized=acc.wer_normalized,
            compute_primary=rc.compute,
            rate_primary_usd_per_hr=cb.rate_primary_usd_per_hr,
            cost_primary_usd=cb.cost_primary_usd,
            compute_secondary=rc.orchestration_compute,
            rate_secondary_usd_per_hr=cb.rate_secondary_usd_per_hr,
            cost_secondary_usd=cb.cost_secondary_usd,
            total_cost_usd=cb.total_cost_usd,
            cost_per_audio_hour=cb.cost_per_audio_hour,
            replicas=self.replicas(),
            rate_catalog_version=cb.rate_catalog_version,
            notes=self.notes,
            started_at=datetime.now(timezone.utc),
            tags={"benchmark_run_id": run_id, "bench_arm": rc.arm, "bench_config": rc.label or rc.arm},
        )
