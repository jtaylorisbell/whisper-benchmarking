# Databricks notebook source
# MAGIC %md
# MAGIC # Serving arm — concurrent benchmark against the endpoint(s)
# MAGIC Runs every `serving` config (GPU type × concurrency × batch) against its deployed endpoint and
# MAGIC emits result JSONs to the Volume. Accounts for BOTH computes (endpoint + this orchestration job).
# MAGIC Requires `20_deploy_serving` first. Thin: logic lives in `whisper_bench`.

# COMMAND ----------
dbutils.widgets.text("config", "", "Absolute workspace path to suite YAML")
dbutils.widgets.text("rate_catalog", "", "Absolute workspace path to compute_costs.yml")
dbutils.widgets.text("catalog", "", "Override: catalog")
dbutils.widgets.text("schema", "", "Override: schema")
dbutils.widgets.text("dataset_limit", "", "Override: clip limit")

# COMMAND ----------
from whisper_bench import ARM_SERVING
from whisper_bench.config import load_suite
from whisper_bench.cost import RateCatalog
from whisper_bench.sweep import run_suite

overrides = {k: dbutils.widgets.get(k) for k in ("catalog", "schema", "dataset_limit")}
suite = load_suite(dbutils.widgets.get("config"), overrides)
rate_catalog = RateCatalog.load(dbutils.widgets.get("rate_catalog"))

# COMMAND ----------
results = run_suite(suite, rate_catalog, arm=ARM_SERVING)
print(f"Completed {len(results)} serving run(s); result JSONs emitted to {suite.volume_root}/results")
for r in results:
    print(f"  {r.label}: rtfx={r.throughput_rtfx:.1f} p95={r.latency_p95_ms}ms "
          f"wer_norm={r.wer_normalized} ${r.total_cost_usd:.4f}")
