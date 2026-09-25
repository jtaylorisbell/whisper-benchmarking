"""Configuration model for a benchmark suite.

A *suite* = global settings (where data + tables live, which model, the dataset) plus a list of
*runs*. Each run is one point in the sweep (an arm + GPU type + batch size + concurrency + the
compute it executes on). Suites are declared in ``conf/*.yml`` and loaded here; Databricks job
parameters can override a small set of top-level fields at launch time (e.g. ``limit`` or the
target ``catalog``) so the same job runs any config without code changes.

Nothing in this module imports Spark, torch, or the Databricks SDK, so it is fully unit-testable
off-cluster.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from . import ARM_AI_RUNTIME, ARM_SERVING, ARMS

# Table / view base names created under <catalog>.<schema>. Centralised so notebooks, results
# writing, and the summary view never disagree on a name.
RESULTS_TABLE = "whisper_bench_results"
RATE_CATALOG_TABLE = "whisper_bench_rate_catalog"
SUMMARY_VIEW = "whisper_bench_summary"

# Model Serving GPU workload types → the GPU they map to (for labelling / rate lookup).
SERVING_WORKLOAD_GPU = {
    "GPU_SMALL": "T4",
    "GPU_MEDIUM": "A10",
    "GPU_LARGE": "A100",
    "GPU_XLARGE": "H100",
}


@dataclass
class DatasetConfig:
    """Which speech dataset to benchmark, and how much of it.

    Defaults target the full LibriSpeech ``test-clean`` split (~2,620 clips, ~5.4 hrs) — the same
    source as the reference notebook, but the full split so the GPUs and endpoint actually get
    saturated. ``limit`` caps the clip count (used by the smoke config); ``None`` means the whole
    split.
    """

    name: str = "librispeech_test_clean"
    hf_dataset: str = "openslr/librispeech_asr"
    hf_config: str = "clean"
    hf_split: str = "test"
    text_column: str = "text"
    id_column: str = "id"
    trust_remote_code: bool = True
    limit: Optional[int] = None


@dataclass
class RunConfig:
    """One point in the sweep.

    ``compute`` and (for serving) ``orchestration_compute`` are keys into the rate catalog
    (``conf/compute_costs.yml``) — that is the single link between a run and the $/hr used to cost
    it, so every dollar figure is traceable to a named, versioned rate.
    """

    arm: str
    gpu_type: str  # "A10" | "A100" | "H100" (labelling + sanity, not the source of truth)
    batch_size: int = 8  # ai_runtime: HF pipeline batch; serving: clips per request
    concurrency: int = 1  # serving only: parallel in-flight requests
    compute: str = ""  # rate-catalog key: ai_runtime GPU, or serving *endpoint* compute
    orchestration_compute: Optional[str] = None  # serving only: the driver job's rate key

    # --- AI Runtime compute detail (serverless GPU via ai_runtime_task) ---
    serverless_accelerator: Optional[str] = None  # e.g. "GPU_1xA10", "GPU_1xH100"
    engine: str = "hf"  # ai_runtime inference engine: "hf" (transformers) | "faster_whisper" (CTranslate2)
    fw_model: Optional[str] = None  # faster_whisper model id override; None = derived from suite.model

    # --- Serving detail ---
    endpoint_name: Optional[str] = None
    workload_type: Optional[str] = None  # GPU_MEDIUM | GPU_LARGE | GPU_XLARGE
    workload_size: str = "Small"  # Small | Medium | Large (concurrency provisioning)
    scale_to_zero: bool = False
    entity_version: Optional[str] = None  # system.ai.whisper_large_v3 version; None = latest READY

    label: Optional[str] = None  # human tag; auto-derived if omitted

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {self.arm!r}")
        if not self.compute:
            raise ValueError(f"run {self.label or self.arm!r} is missing a 'compute' rate key")
        if self.arm == ARM_SERVING and not self.orchestration_compute:
            raise ValueError("serving runs must set 'orchestration_compute' (the driver job pays too)")
        if self.engine not in ("hf", "faster_whisper"):
            raise ValueError(f"engine must be 'hf' or 'faster_whisper', got {self.engine!r}")
        if self.label is None:
            if self.arm == ARM_AI_RUNTIME:
                eng = "" if self.engine == "hf" else "-fw"
                self.label = f"ai_runtime{eng}-{self.gpu_type}-bs{self.batch_size}"
            else:
                self.label = f"serving-{self.gpu_type}-c{self.concurrency}-bs{self.batch_size}"


@dataclass
class Suite:
    """A full benchmark suite: where things live + the list of runs to execute."""

    catalog: str
    schema: str
    volume: str
    dataset: DatasetConfig
    runs: list[RunConfig]
    model: str = "openai/whisper-large-v3"
    warmup_clips: int = 5
    rate_catalog_version: str = "unset"

    # --- Resolved fully-qualified names (single source of truth) ---
    @property
    def results_table(self) -> str:
        return f"{self.catalog}.{self.schema}.{RESULTS_TABLE}"

    @property
    def rate_catalog_table(self) -> str:
        return f"{self.catalog}.{self.schema}.{RATE_CATALOG_TABLE}"

    @property
    def summary_view(self) -> str:
        return f"{self.catalog}.{self.schema}.{SUMMARY_VIEW}"

    @property
    def audio_table(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.dataset.name}"

    @property
    def volume_root(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"

    def runs_for(self, arm: Optional[str]) -> list[RunConfig]:
        """Runs for a given arm (or all runs when ``arm`` is None)."""
        return [r for r in self.runs if arm is None or r.arm == arm]


# Top-level keys that Databricks job parameters are allowed to override at launch time.
_OVERRIDABLE = {"catalog", "schema", "volume", "model", "warmup_clips", "rate_catalog_version"}


def load_suite(path: str | Path, overrides: Optional[dict[str, Any]] = None) -> Suite:
    """Load a suite from YAML, applying ``overrides`` (typically parsed Databricks job params).

    ``overrides`` may set any of :data:`_OVERRIDABLE`, plus ``dataset_limit`` (handy for shrinking
    a run to a smoke-sized subset without editing YAML). Empty-string / None override values are
    ignored so a job param left blank falls back to the YAML default.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config at {path} must be a YAML mapping")
    raw = copy.deepcopy(raw)
    overrides = {k: v for k, v in (overrides or {}).items() if v not in (None, "")}

    dataset_raw = raw.pop("dataset", {}) or {}
    if "dataset_limit" in overrides:
        dataset_raw["limit"] = int(overrides.pop("dataset_limit"))
    dataset = DatasetConfig(**dataset_raw)

    runs_raw = raw.pop("runs", []) or []
    if not runs_raw:
        raise ValueError(f"config at {path} declares no runs")
    runs = [RunConfig(**r) for r in runs_raw]

    for key, value in overrides.items():
        if key not in _OVERRIDABLE:
            raise ValueError(f"override {key!r} is not permitted (allowed: {sorted(_OVERRIDABLE)})")
        raw[key] = value

    if "warmup_clips" in raw:
        raw["warmup_clips"] = int(raw["warmup_clips"])

    return Suite(dataset=dataset, runs=runs, **raw)
