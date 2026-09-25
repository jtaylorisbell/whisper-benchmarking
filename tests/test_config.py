"""Unit tests for config loading — no workspace needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from whisper_bench import ARM_AI_RUNTIME, ARM_SERVING
from whisper_bench.config import RunConfig, Suite, load_suite

CONF = Path(__file__).resolve().parents[1] / "conf"


def test_smoke_suite_loads_and_resolves_names():
    suite = load_suite(CONF / "smoke.yml")
    assert isinstance(suite, Suite)
    assert suite.catalog == "classic_stable_h0vpq7_catalog"
    assert suite.results_table == "classic_stable_h0vpq7_catalog.whisper_bench.whisper_bench_results"
    assert suite.summary_view.endswith(".whisper_bench_summary")
    assert suite.volume_root == "/Volumes/classic_stable_h0vpq7_catalog/whisper_bench/audio"
    assert suite.audio_table.endswith(".librispeech_test_clean")

    arms = {r.arm for r in suite.runs}
    assert arms == {ARM_AI_RUNTIME, ARM_SERVING}
    assert len(suite.runs_for(ARM_AI_RUNTIME)) == 1
    assert len(suite.runs_for(ARM_SERVING)) == 1
    assert len(suite.runs_for(None)) == 2


def test_full_sweep_template_is_valid():
    suite = load_suite(CONF / "full_sweep.yml")
    # both arms present with at least one run each
    assert len(suite.runs_for("ai_runtime")) >= 1
    assert len(suite.runs_for(ARM_SERVING)) >= 1
    # every serving run must name both computes it pays for
    for r in suite.runs_for(ARM_SERVING):
        assert r.compute and r.orchestration_compute


def test_overrides_apply_and_are_validated():
    suite = load_suite(CONF / "smoke.yml", overrides={"catalog": "other_cat", "dataset_limit": "25", "warmup_clips": ""})
    assert suite.catalog == "other_cat"
    assert suite.dataset.limit == 25
    assert suite.warmup_clips == 3  # blank override ignored -> YAML default kept

    with pytest.raises(ValueError):
        load_suite(CONF / "smoke.yml", overrides={"not_allowed": "x"})


def test_labels_autoderive():
    ai = RunConfig(arm=ARM_AI_RUNTIME, gpu_type="A10", batch_size=16, compute="serverless_gpu_a10")
    assert ai.label == "ai_runtime-A10-bs16"
    fw = RunConfig(arm=ARM_AI_RUNTIME, gpu_type="A10", batch_size=1, engine="faster_whisper",
                   compute="serverless_gpu_a10")
    assert fw.label == "ai_runtime-fw-A10-bs1"
    sv = RunConfig(arm=ARM_SERVING, gpu_type="H100", concurrency=8, batch_size=1,
                   compute="serving_gpu_xlarge", orchestration_compute="orchestration_serverless_jobs")
    assert sv.label == "serving-H100-c8-bs1"


def test_faster_whisper_config_loads():
    suite = load_suite(CONF / "faster_whisper.yml")
    assert len(suite.runs) == 1
    r = suite.runs[0]
    assert r.arm == ARM_AI_RUNTIME and r.engine == "faster_whisper"


def test_bad_configs_raise():
    with pytest.raises(ValueError):
        RunConfig(arm="nonsense", gpu_type="A10", compute="x")
    with pytest.raises(ValueError):
        RunConfig(arm=ARM_AI_RUNTIME, gpu_type="A10", compute="serverless_gpu_a10", engine="bogus")
    with pytest.raises(ValueError):
        RunConfig(arm=ARM_AI_RUNTIME, gpu_type="A10", compute="")  # missing compute
    with pytest.raises(ValueError):
        RunConfig(arm=ARM_SERVING, gpu_type="A10", compute="serving_gpu_medium")  # missing orchestration
