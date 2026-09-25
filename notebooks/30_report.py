# Databricks notebook source
# MAGIC %md
# MAGIC # Report — price/performance leaderboard
# MAGIC Ingests the emitted result JSONs into the documented Delta tables and prints the winner. Cost
# MAGIC is self-computed from measured inference-only time × documented rates (see
# MAGIC `whisper_bench_rate_catalog`). System billing tables are only an optional cross-check (they
# MAGIC include excluded setup and lag ~12h).
# MAGIC
# MAGIC NOTE: in Databricks `.py` notebooks a cell that starts with `# MAGIC %md` is a *markdown* cell —
# MAGIC keep every markdown header in its own cell, separate from the code cells below it.

# COMMAND ----------
dbutils.widgets.text("config", "", "Absolute workspace path to suite YAML")
dbutils.widgets.text("rate_catalog", "", "Absolute workspace path to compute_costs.yml")

# COMMAND ----------
from whisper_bench.config import load_suite
from whisper_bench.cost import RateCatalog
from whisper_bench.sweep import ingest_and_report

suite = load_suite(dbutils.widgets.get("config"))
rate_catalog = RateCatalog.load(dbutils.widgets.get("rate_catalog"))

# COMMAND ----------
# MAGIC %md ## Ingest result JSONs -> documented Delta tables (idempotent)

# COMMAND ----------
n = ingest_and_report(spark, suite, rate_catalog)
print(f"{n} rows in {suite.results_table}")

# COMMAND ----------
# MAGIC %md ## Leaderboard (cheapest $/audio-hour first). WER/CER shown for cross-arm parity.

# COMMAND ----------
display(spark.table(suite.summary_view))

# COMMAND ----------
# MAGIC %md ## Best config per arm + overall winner

# COMMAND ----------
display(spark.sql(f"""
    WITH ranked AS (
      SELECT *, row_number() OVER (PARTITION BY arm ORDER BY usd_per_audio_hour ASC) AS rk
      FROM {suite.summary_view}
    )
    SELECT arm, gpu_type, batch_size, concurrency, rtfx, wer_normalized,
           total_cost_usd, usd_per_audio_hour, label
    FROM ranked WHERE rk = 1 ORDER BY usd_per_audio_hour ASC
"""))

# COMMAND ----------
# MAGIC %md ## Accuracy parity check — both arms must agree on WER/CER (same model, same audio)

# COMMAND ----------
display(spark.sql(f"""
    SELECT arm,
           round(min(wer_normalized), 4) AS min_wer_norm,
           round(max(wer_normalized), 4) AS max_wer_norm,
           round(avg(cer), 4)            AS avg_cer
    FROM {suite.results_table} GROUP BY arm ORDER BY arm
"""))

# COMMAND ----------
# MAGIC %md ## Rate catalog used (every $ figure traces here)

# COMMAND ----------
display(spark.table(suite.rate_catalog_table))

# COMMAND ----------
# MAGIC %md ## Optional cross-check vs system billing tables
# MAGIC Expected to read **higher** than self-computed cost — it includes setup (model load, spin-up,
# MAGIC warmup) and lands ~12h late. Sanity check, not the source of truth.

# COMMAND ----------
try:
    display(spark.sql(f"""
        SELECT u.custom_tags['bench_arm']    AS arm,
               u.custom_tags['bench_config'] AS config,
               u.sku_name,
               round(sum(u.usage_quantity), 2)                             AS dbus,
               round(sum(u.usage_quantity * lp.pricing.effective_list), 2) AS billed_usd
        FROM system.billing.usage u
        LEFT JOIN system.billing.list_prices lp
          ON u.sku_name = lp.sku_name AND lp.price_end_time IS NULL
        WHERE u.custom_tags['benchmark_run_id'] IS NOT NULL
        GROUP BY 1, 2, 3 ORDER BY 1, 2
    """))
except Exception as e:  # noqa: BLE001
    print("Billing cross-check unavailable yet (data lags ~12h) or no tagged usage:", repr(e))
