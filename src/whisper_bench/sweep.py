"""Suite orchestration.

``run_suite`` is **Spark-free**: it loads clips from the Volume Parquet, runs each config for an arm,
and emits one result JSON per run to the Volume. This lets the AI Runtime (serverless GPU) arm run as
a plain Python script with no Spark session. ``ingest_and_report`` then runs in a serverless notebook
(with Spark) to build the documented Delta tables from those JSONs.
"""

from __future__ import annotations

import uuid
from typing import Optional

from .config import Suite
from .cost import RateCatalog
from .data import load_clips
from .results import (
    RunResult,
    create_summary_view,
    ensure_rate_catalog_table,
    ingest_results,
    write_result_json,
)
from .runners import make_runner


def run_suite(
    suite: Suite,
    rate_catalog: RateCatalog,
    arm: Optional[str] = None,
    suite_id: Optional[str] = None,
    gpu: Optional[str] = None,
) -> list[RunResult]:
    """Run every config for ``arm`` (or all arms) and emit result JSONs to the Volume. No Spark.

    ``gpu`` optionally restricts to one GPU type — used by the AI Runtime arm, where each
    ``ai_runtime_task`` deployment runs on a single accelerator, so the script transcribes only the
    configs matching that accelerator.

    Both arms read the identical clip set (same Parquet, deterministic id order), which is what makes
    the cross-arm WER/CER parity check valid.
    """
    suite_id = suite_id or uuid.uuid4().hex
    clips = load_clips(suite, limit=suite.dataset.limit)
    if not clips:
        raise RuntimeError(f"no clips on the Volume for {suite.dataset.name}; run prep_data first")
    configs = [r for r in suite.runs_for(arm) if gpu is None or r.gpu_type == gpu]
    print(f"suite_id={suite_id} | {len(clips)} clips | arm={arm or 'ALL'} | gpu={gpu or 'ALL'} | {len(configs)} config(s)")

    results: list[RunResult] = []
    for rc in configs:
        print(f"--> running {rc.label}")
        runner = make_runner(suite, rc, rate_catalog, suite_id)
        result = runner.run(clips)
        path = write_result_json(suite, result)  # persist immediately (Spark-free)
        results.append(result)
        print(
            f"    rtfx={result.throughput_rtfx:.1f} | wer_norm={result.wer_normalized} | "
            f"infer={result.inference_wall_sec:.1f}s (setup {result.setup_wall_sec:.1f}s) | "
            f"$/audio-hr={result.cost_per_audio_hour:.4f} (total ${result.total_cost_usd:.4f}) | {path}"
        )
    return results


def ingest_and_report(spark, suite: Suite, rate_catalog: RateCatalog) -> int:
    """Build/refresh the documented Delta tables + summary view from the emitted result JSONs."""
    n = ingest_results(spark, suite)
    ensure_rate_catalog_table(spark, suite, rate_catalog)
    create_summary_view(spark, suite)
    print(f"Ingested results; {n} total rows in {suite.results_table}")
    return n
