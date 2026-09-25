"""Spark-free tests for the RunResult JSON contract (emit/ingest exchange format)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from whisper_bench.results import RESULT_FIELDS, RunResult


def _sample() -> RunResult:
    return RunResult(
        run_id="abc123", suite_id="s1", arm="serving", label="serving-A10-c4-bs1",
        model="openai/whisper-large-v3", gpu_type="A10", batch_size=1, concurrency=4,
        n_clips=100, total_audio_sec=600.0, inference_wall_sec=42.0, setup_wall_sec=9.0,
        throughput_rtfx=14.28, latency_p50_ms=120.0, latency_p95_ms=210.0,
        wer=0.05, cer=0.02, wer_normalized=0.03,
        compute_primary="serving_gpu_medium", rate_primary_usd_per_hr=1.40, cost_primary_usd=0.0163,
        compute_secondary="orchestration_serverless_jobs", rate_secondary_usd_per_hr=0.90,
        cost_secondary_usd=0.0105, total_cost_usd=0.0268, cost_per_audio_hour=0.1608,
        replicas=1, rate_catalog_version="2026-09-23.1", notes="endpoint=whisper_bench_a10",
        started_at=datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc),
        tags={"benchmark_run_id": "abc123", "bench_arm": "serving"},
    )


def test_json_round_trip():
    r = _sample()
    back = RunResult.from_dict(json.loads(r.to_json()))
    assert back.run_id == r.run_id
    assert back.total_cost_usd == r.total_cost_usd
    assert back.started_at == r.started_at
    assert back.tags == r.tags
    assert back.wer_normalized == 0.03


def test_result_fields_cover_dataclass():
    # Every documented column must be a real RunResult field (schema/doc drift guard).
    fields = set(RunResult.__dataclass_fields__)
    assert set(RESULT_FIELDS) <= fields
    # and the reverse: no dataclass field left undocumented
    assert fields <= set(RESULT_FIELDS)
