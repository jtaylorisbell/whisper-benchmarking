"""Results — the primary deliverable.

Data flow (so the Spark-less AI Runtime arm works too):
  1. Each run emits ONE JSON file to the Volume (``write_result_json``) — Spark-free.
  2. A Spark/SQL step (``ingest_results``) reads all JSONs and builds the documented Delta tables.

Results ARE the product, so the tables are self-documenting: every column carries a ``COMMENT``,
units are in the names (``_usd``, ``_sec``, ``_ms``, ``rtfx``), and the rate catalog used for the
cost math is persisted alongside so every dollar is reproducible from the tables alone.

Tables under ``<catalog>.<schema>``:
  * ``whisper_bench_results``       — one row per run/config (fact)
  * ``whisper_bench_rate_catalog``  — the $/hr rates used, versioned (dim)
  * ``whisper_bench_summary``       — human-readable leaderboard (view)

pyspark is imported lazily inside the ingest functions, so ``RunResult`` and the JSON helpers import
fine off-cluster and in the AI Runtime environment.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from . import volumes
from .config import Suite
from .cost import RateCatalog

# column name -> human COMMENT. Single source of truth for the table's documentation.
RESULT_COLUMN_DOCS: dict[str, str] = {
    "run_id": "Unique id for this single benchmarked config execution.",
    "suite_id": "Id grouping all runs launched together in one suite invocation.",
    "arm": "Architecture under test: 'ai_runtime' (serverless GPU job) or 'serving' (GPU endpoint).",
    "label": "Human-readable config label, e.g. serving-A10-c8-bs1.",
    "model": "HuggingFace model id transcribed. Identical across arms for apples-to-apples.",
    "gpu_type": "GPU label: A10 | A100 | H100.",
    "batch_size": "ai_runtime: HF pipeline batch size. serving: audio clips per request.",
    "concurrency": "serving: number of parallel in-flight requests. 1 for ai_runtime.",
    "n_clips": "Number of audio clips transcribed in this run.",
    "total_audio_sec": "Total seconds of audio across all clips (the work done).",
    "inference_wall_sec": "Wall time of JUST the transcription loop. EXCLUDES setup. Basis for cost + RTFx.",
    "setup_wall_sec": "Wall time of model load + data load + warmup + endpoint readiness. EXCLUDED from cost.",
    "throughput_rtfx": "Real-time factor = total_audio_sec / inference_wall_sec. Higher is faster.",
    "latency_p50_ms": "serving: median per-request latency (ms). NULL for ai_runtime.",
    "latency_p95_ms": "serving: p95 per-request latency (ms). NULL for ai_runtime.",
    "wer": "Raw word error rate (lower-cased/stripped) vs reference transcripts.",
    "cer": "Character error rate after the Whisper English text normalizer.",
    "wer_normalized": "Word error rate after the Whisper English normalizer. HEADLINE accuracy + parity check.",
    "compute_primary": "Rate-catalog key for the primary compute (ai_runtime GPU, or serving endpoint).",
    "rate_primary_usd_per_hr": "Effective $/hr used for the primary compute (from the rate catalog).",
    "cost_primary_usd": "Primary compute cost = inference_wall_sec/3600 * rate_primary * replicas.",
    "compute_secondary": "serving only: rate-catalog key for the orchestration/driver job. NULL for ai_runtime.",
    "rate_secondary_usd_per_hr": "serving only: effective $/hr for the orchestration job.",
    "cost_secondary_usd": "serving only: orchestration cost over the same inference window.",
    "total_cost_usd": "cost_primary_usd + cost_secondary_usd. What it costs to transcribe this dataset.",
    "cost_per_audio_hour": "total_cost_usd per hour of audio transcribed. HEADLINE price/performance.",
    "replicas": "Number of endpoint replicas / GPU units billed in parallel (usually 1).",
    "rate_catalog_version": "Version of conf/compute_costs.yml used for the cost math.",
    "notes": "Free text: fallbacks used, errors, caveats.",
    "started_at": "Run start time (UTC).",
    "tags": "Custom tags (benchmark_run_id, bench_arm, bench_config) for optional billing cross-check.",
}

_TABLE_COMMENT = (
    "Whisper batch-inference benchmark results. One row per run/config. Cost is self-computed "
    "from measured inference-only time x documented rates (see whisper_bench_rate_catalog); setup "
    "time is recorded but excluded from cost."
)

# Field order == results table column order.
RESULT_FIELDS = list(RESULT_COLUMN_DOCS.keys())


@dataclass
class RunResult:
    run_id: str
    suite_id: str
    arm: str
    label: str
    model: str
    gpu_type: str
    batch_size: int
    concurrency: int
    n_clips: int
    total_audio_sec: float
    inference_wall_sec: float
    setup_wall_sec: float
    throughput_rtfx: float
    latency_p50_ms: Optional[float]
    latency_p95_ms: Optional[float]
    wer: Optional[float]
    cer: Optional[float]
    wer_normalized: Optional[float]
    compute_primary: str
    rate_primary_usd_per_hr: float
    cost_primary_usd: float
    compute_secondary: Optional[str]
    rate_secondary_usd_per_hr: Optional[float]
    cost_secondary_usd: Optional[float]
    total_cost_usd: float
    cost_per_audio_hour: float
    replicas: int = 1
    rate_catalog_version: str = "unset"
    notes: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        d["started_at"] = self.started_at.astimezone(timezone.utc).isoformat()
        return json.dumps(d)

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        d = dict(d)
        if isinstance(d.get("started_at"), str):
            d["started_at"] = datetime.fromisoformat(d["started_at"])
        return cls(**d)


# --------------------------------------------------------------------------- #
# Spark-free emit / read (used by both arms, incl. AI Runtime)
# --------------------------------------------------------------------------- #
def results_dir(suite: Suite) -> str:
    return f"{suite.volume_root}/results"


def write_result_json(suite: Suite, result: RunResult) -> str:
    """Persist one result as ``<volume>/results/<run_id>.json``. Returns the path."""
    path = f"{results_dir(suite)}/{result.run_id}.json"
    volumes.write_bytes(path, result.to_json().encode("utf-8"))
    return path


def read_all_results(suite: Suite) -> list[RunResult]:
    """Read every result JSON currently on the Volume (Spark-free)."""
    out = []
    for p in volumes.list_files(results_dir(suite), suffix=".json"):
        out.append(RunResult.from_dict(json.loads(volumes.read_bytes(p).decode("utf-8"))))
    return out


# --------------------------------------------------------------------------- #
# Spark ingest -> documented Delta tables (run in a serverless notebook)
# --------------------------------------------------------------------------- #
def _results_schema():
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        MapType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    S, D, I, T = StringType(), DoubleType(), IntegerType(), TimestampType()
    M = MapType(StringType(), StringType())
    spec = [
        ("run_id", S), ("suite_id", S), ("arm", S), ("label", S), ("model", S), ("gpu_type", S),
        ("batch_size", I), ("concurrency", I), ("n_clips", I),
        ("total_audio_sec", D), ("inference_wall_sec", D), ("setup_wall_sec", D), ("throughput_rtfx", D),
        ("latency_p50_ms", D), ("latency_p95_ms", D),
        ("wer", D), ("cer", D), ("wer_normalized", D),
        ("compute_primary", S), ("rate_primary_usd_per_hr", D), ("cost_primary_usd", D),
        ("compute_secondary", S), ("rate_secondary_usd_per_hr", D), ("cost_secondary_usd", D),
        ("total_cost_usd", D), ("cost_per_audio_hour", D), ("replicas", I),
        ("rate_catalog_version", S), ("notes", S), ("started_at", T), ("tags", M),
    ]
    return StructType([StructField(n, t, True) for n, t in spec])


def ingest_results(spark, suite: Suite) -> int:
    """Read all result JSONs from the Volume and MERGE them into the documented fact table.

    Reads the JSONs **Spark-natively** (``spark.read.text`` + ``from_json``) rather than via the
    Python Files API: Spark reliably lists+reads the Volume path, whereas the Files API / FUSE
    ``os.listdir`` can return a stale-empty listing inside a serverless notebook. Idempotent
    (dedupes on run_id). Returns the number of rows now in the table.
    """
    from delta.tables import DeltaTable
    from pyspark.sql.functions import col, from_json, to_timestamp
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        MapType,
        StringType,
        StructField,
        StructType,
    )

    # Parse schema mirrors the table schema, but started_at is read as a string then cast to a
    # timestamp (JSON has it as an ISO8601 string).
    tbl_schema = _results_schema()
    parse_fields = []
    for f in tbl_schema.fields:
        t = StringType() if f.name == "started_at" else f.dataType
        parse_fields.append(StructField(f.name, t, True))
    parse_schema = StructType(parse_fields)

    glob = f"{results_dir(suite)}/*.json"
    try:
        raw = spark.read.text(glob)
    except Exception:
        return spark.table(suite.results_table).count() if spark.catalog.tableExists(suite.results_table) else 0
    if raw.limit(1).count() == 0:
        return spark.table(suite.results_table).count() if spark.catalog.tableExists(suite.results_table) else 0

    df = (raw.select(from_json(col("value"), parse_schema).alias("r")).select("r.*")
             .withColumn("started_at", to_timestamp(col("started_at"))))

    if not spark.catalog.tableExists(suite.results_table):
        df.write.format("delta").saveAsTable(suite.results_table)
    else:
        tgt = DeltaTable.forName(spark, suite.results_table)
        (tgt.alias("t").merge(df.alias("s"), "t.run_id = s.run_id")
            .whenNotMatchedInsertAll().whenMatchedUpdateAll().execute())

    document_results_table(spark, suite)
    return spark.table(suite.results_table).count()


def document_results_table(spark, suite: Suite) -> None:
    """Apply the table + column COMMENTs to the fact table (idempotent)."""
    _apply_table_docs(spark, suite.results_table, _TABLE_COMMENT, RESULT_COLUMN_DOCS)


def _apply_table_docs(spark, table: str, table_comment: str, column_docs: dict[str, str]) -> None:
    spark.sql(f"COMMENT ON TABLE {table} IS {_sql_str(table_comment)}")
    existing = {f.name for f in spark.table(table).schema.fields}
    for col, doc in column_docs.items():
        if col in existing:
            spark.sql(f"ALTER TABLE {table} ALTER COLUMN {col} COMMENT {_sql_str(doc)}")


def ensure_rate_catalog_table(spark, suite: Suite, catalog: RateCatalog) -> None:
    """Persist the rate catalog used, so every cost figure is reproducible from the tables alone."""
    from pyspark.sql.types import DoubleType, StringType, StructField, StructType

    schema = StructType([
        StructField("catalog_version", StringType()),
        StructField("compute_key", StringType()),
        StructField("description", StringType()),
        StructField("databricks_sku", StringType()),
        StructField("usd_per_dbu", DoubleType()),
        StructField("dbus_per_hour", DoubleType()),
        StructField("cloud_instance", StringType()),
        StructField("cloud_usd_per_hour", DoubleType()),
        StructField("effective_usd_per_hour", DoubleType()),
    ])
    rows = [[
        catalog.version, r.key, r.description, r.databricks_sku, r.usd_per_dbu, r.dbus_per_hour,
        r.cloud_instance, r.cloud_usd_per_hour, r.effective_usd_per_hour(),
    ] for r in catalog.rates.values()]
    df = spark.createDataFrame(rows, schema=schema)
    (df.write.format("delta").mode("overwrite")
        .option("replaceWhere", f"catalog_version = {_sql_str(catalog.version)}")
        .option("mergeSchema", "true").saveAsTable(suite.rate_catalog_table))
    spark.sql(f"COMMENT ON TABLE {suite.rate_catalog_table} IS "
              f"'Versioned compute rates behind every cost figure. effective_usd_per_hour = "
              f"dbus_per_hour*usd_per_dbu (+cloud_usd_per_hour). Source: {catalog.source}'")


def create_summary_view(spark, suite: Suite) -> None:
    """Human-readable leaderboard: best price/performance per arm & GPU, accuracy shown for parity."""
    spark.sql(f"""
        CREATE OR REPLACE VIEW {suite.summary_view} AS
        SELECT
          arm, gpu_type, batch_size, concurrency, n_clips,
          round(total_audio_sec/3600.0, 2) AS audio_hours,
          round(inference_wall_sec, 1)      AS inference_sec,
          round(throughput_rtfx, 1)         AS rtfx,
          round(wer_normalized, 4)          AS wer_normalized,
          round(cer, 4)                     AS cer,
          round(total_cost_usd, 4)          AS total_cost_usd,
          round(cost_per_audio_hour, 4)     AS usd_per_audio_hour,
          label, started_at
        FROM {suite.results_table}
        QUALIFY row_number() OVER (
            PARTITION BY arm, gpu_type, batch_size, concurrency ORDER BY started_at DESC
        ) = 1
        ORDER BY usd_per_audio_hour ASC
    """)
    spark.sql(f"COMMENT ON VIEW {suite.summary_view} IS "
              f"'Leaderboard: latest run per (arm,gpu,batch,concurrency), cheapest $/audio-hour first. "
              f"WER/CER shown so cross-arm accuracy parity is visible at a glance.'")


def _sql_str(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"
