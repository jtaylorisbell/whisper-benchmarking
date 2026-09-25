# Databricks notebook source
# MAGIC %md
# MAGIC # Serving arm — tear down endpoint(s)
# MAGIC Deletes the GPU serving endpoints created for the benchmark so they stop billing. Run this
# MAGIC when you're done with the serving sweep. Idempotent (skips endpoints that don't exist).

# COMMAND ----------
dbutils.widgets.text("config", "", "Absolute workspace path to suite YAML")

# COMMAND ----------
from databricks.sdk import WorkspaceClient

from whisper_bench import ARM_SERVING
from whisper_bench.config import load_suite

w = WorkspaceClient()
suite = load_suite(dbutils.widgets.get("config"))
names = sorted({r.endpoint_name for r in suite.runs_for(ARM_SERVING) if r.endpoint_name})
print("Endpoints to delete:", names)

# COMMAND ----------
for name in names:
    try:
        w.serving_endpoints.delete(name)
        print(f"  deleted {name}")
    except Exception as e:  # noqa: BLE001
        print(f"  skip {name}: {e!r}")
