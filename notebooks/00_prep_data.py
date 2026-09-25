# Databricks notebook source
# MAGIC %md
# MAGIC # Prep — materialize the benchmark dataset
# MAGIC Downloads LibriSpeech (per the suite config) once into a single Parquet on the UC Volume:
# MAGIC raw audio bytes + reference transcripts + durations. Both arms read this identical Parquet, so
# MAGIC they transcribe byte-identical inputs. Thin: all logic is in `whisper_bench`.

# COMMAND ----------
dbutils.widgets.text("config", "", "Absolute workspace path to suite YAML")
dbutils.widgets.text("catalog", "", "Override: catalog")
dbutils.widgets.text("schema", "", "Override: schema")
dbutils.widgets.text("volume", "", "Override: volume")
dbutils.widgets.text("dataset_limit", "", "Override: clip limit (blank = as configured)")
dbutils.widgets.dropdown("overwrite", "false", ["false", "true"], "Rebuild Parquet if it exists")

# COMMAND ----------
from whisper_bench.config import load_suite
from whisper_bench.data import prepare_dataset

overrides = {k: dbutils.widgets.get(k) for k in ("catalog", "schema", "volume", "dataset_limit")}
suite = load_suite(dbutils.widgets.get("config"), overrides)

# UC scaffolding (this notebook has Spark/SQL; the data write itself is Spark-free).
# The catalog must already exist (creating one needs a managed location under Default Storage);
# the schema + volume inherit the catalog's storage root, so no location is required for them.
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {suite.catalog}.{suite.schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {suite.catalog}.{suite.schema}.{suite.volume}")

# COMMAND ----------
n = prepare_dataset(suite, overwrite=dbutils.widgets.get("overwrite") == "true")
print(f"Materialized {n} clips into {suite.volume_root}/dataset/{suite.dataset.name}.parquet")

# COMMAND ----------
# Quick peek (read the Parquet back, Spark-free).
from whisper_bench.data import load_clips

for c in load_clips(suite, limit=5):
    print(f"  {c.id}: {c.duration_sec:.2f}s @ {c.sampling_rate}Hz | {c.reference_text[:60]}...")
