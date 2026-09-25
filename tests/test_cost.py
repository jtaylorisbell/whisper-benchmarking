"""Unit tests for the cost engine and rate catalog. No workspace needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from whisper_bench.cost import Rate, RateCatalog, compute_cost, cost_per_audio_hour

CONF = Path(__file__).resolve().parents[1] / "conf"


def test_rate_effective_itemized_and_flat():
    itemized = Rate(key="x", dbus_per_hour=20.0, usd_per_dbu=0.07)
    assert itemized.effective_usd_per_hour() == pytest.approx(1.40)

    with_cloud = Rate(key="c", dbus_per_hour=1.0, usd_per_dbu=0.55, cloud_usd_per_hour=1.006)
    assert with_cloud.effective_usd_per_hour() == pytest.approx(1.556)

    flat = Rate(key="f", usd_per_hour=3.0)
    assert flat.effective_usd_per_hour() == 3.0

    with pytest.raises(ValueError):
        Rate(key="bad").effective_usd_per_hour()


def test_catalog_matches_known_rates():
    cat = RateCatalog.load(CONF / "compute_costs.yml")
    # serving A10 = 20 DBU/hr * $0.07 = $1.40/hr
    assert cat.usd_per_hour("serving_gpu_medium") == pytest.approx(1.40)
    # serverless GPU A10 = 3.6 DBU/hr * $0.65 = $2.34/hr (calibrated vs billing)
    assert cat.usd_per_hour("serverless_gpu_a10") == pytest.approx(2.34)
    # serverless GPU H100 = 10.0 * $0.65 = $6.50/hr (H100 DBU/hr still a placeholder)
    assert cat.usd_per_hour("serverless_gpu_h100") == pytest.approx(6.50)
    # orchestration (calibrated) = 0.75 DBU/hr * $0.45 = $0.3375/hr
    assert cat.usd_per_hour("orchestration_serverless_jobs") == pytest.approx(0.3375)
    with pytest.raises(KeyError):
        cat.usd_per_hour("does_not_exist")


def test_assemble_costs_ai_runtime_and_serving():
    from whisper_bench.cost import assemble_costs
    cat = RateCatalog.load(CONF / "compute_costs.yml")

    # AI Runtime: single compute, no secondary
    ai = assemble_costs(cat, "serverless_gpu_a10", None, inference_wall_sec=1034.0,
                        total_audio_sec=19440.0)  # 5.4h audio
    assert ai.rate_primary_usd_per_hr == pytest.approx(2.34)
    assert ai.cost_secondary_usd is None
    assert ai.total_cost_usd == pytest.approx(1034/3600 * 2.34)
    assert ai.cost_per_audio_hour == pytest.approx(ai.total_cost_usd / 5.4)

    # Serving: primary endpoint + secondary orchestration, both over the inference window
    sv = assemble_costs(cat, "serving_gpu_medium", "orchestration_serverless_jobs",
                        inference_wall_sec=2352.0, total_audio_sec=19440.0, replicas=1)
    assert sv.cost_secondary_usd == pytest.approx(2352/3600 * 0.3375)
    assert sv.total_cost_usd == pytest.approx(2352/3600 * 1.40 + 2352/3600 * 0.3375)


def test_compute_cost_and_cost_per_audio_hour():
    # 600s of inference at $1.40/hr = $0.2333...
    assert compute_cost(600.0, 1.40) == pytest.approx(1.40 * 600 / 3600)
    # replicas multiply
    assert compute_cost(600.0, 1.40, replicas=2) == pytest.approx(2 * 1.40 * 600 / 3600)
    # $0.10 to transcribe 360s (0.1hr) of audio -> $1.00 per audio-hour
    assert cost_per_audio_hour(0.10, 360.0) == pytest.approx(1.00)


def test_serving_replicas_and_cost_scale_with_workload_size():
    """GPU serving replicas = provisioned_concurrency/4; cost must scale with replica (GPU) count."""
    from whisper_bench import ARM_SERVING
    from whisper_bench.config import RunConfig
    from whisper_bench.runners.serving import ServingRunner

    def replicas_for(ws):
        rc = RunConfig(arm=ARM_SERVING, gpu_type="A10", workload_size=ws,
                       compute="serving_gpu_medium", orchestration_compute="orchestration_serverless_jobs")
        # replicas() only reads run_config; no workspace needed.
        r = ServingRunner.__new__(ServingRunner)
        r.run_config = rc
        return r.replicas()

    assert replicas_for("Small") == 1    # provisioned concurrency 4 -> 1 A10
    assert replicas_for("Medium") == 2
    assert replicas_for("Large") == 4
    # cost scales with replicas: 2 GPUs for the same inference window cost 2x
    assert compute_cost(600.0, 1.40, replicas=2) == pytest.approx(2 * compute_cost(600.0, 1.40))
