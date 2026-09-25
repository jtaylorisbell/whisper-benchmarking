# Databricks notebook source
# MAGIC %md
# MAGIC # Serving arm — deploy GPU endpoint(s) from `system.ai.whisper_large_v3`
# MAGIC Creates/updates one endpoint per distinct (endpoint, workload) in the suite's serving runs, waits
# MAGIC until ready, then **probes the live request schema** so we can confirm the payload format used by
# MAGIC `whisper_bench.runners.serving` before benchmarking.

# COMMAND ----------
dbutils.widgets.text("config", "", "Absolute workspace path to suite YAML")
dbutils.widgets.text("model", "system.ai.whisper_large_v3", "UC model to serve")
dbutils.widgets.text("default_version", "3", "Model version if a run doesn't pin one")

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

from whisper_bench import ARM_SERVING
from whisper_bench.config import load_suite

w = WorkspaceClient()
suite = load_suite(dbutils.widgets.get("config"))
model = dbutils.widgets.get("model")
default_version = dbutils.widgets.get("default_version")


def _workload_type(value: str):
    """Coerce to the SDK enum when available; fall back to the raw string."""
    try:
        from databricks.sdk.service.serving import ServingModelWorkloadType

        return ServingModelWorkloadType(value)
    except Exception:
        return value


# Distinct endpoints to stand up (a run's endpoint_name + its workload settings).
endpoints = {}
for r in suite.runs_for(ARM_SERVING):
    endpoints[r.endpoint_name] = r
print("Endpoints to ensure:", list(endpoints))

# COMMAND ----------
from datetime import timedelta

from databricks.sdk.service.serving import EndpointStateConfigUpdate

# GPU endpoint cold-start (container + model download) can exceed the SDK's 20-min default wait,
# so we use a generous timeout and never issue a conflicting update while one is in progress.
WAIT = timedelta(minutes=50)


def _matches(ep, r) -> bool:
    """True if the endpoint already serves the requested model/version/workload."""
    for se in (ep.config.served_entities if ep.config else []) or []:
        if (se.entity_name == model and str(se.entity_version) == str(r.entity_version or default_version)
                and str(getattr(se, "workload_type", "")).endswith(str(r.workload_type))):
            return True
    return False


for name, r in endpoints.items():
    served = ServedEntityInput(
        name="whisper",
        entity_name=model,
        entity_version=r.entity_version or default_version,
        workload_type=_workload_type(r.workload_type),
        workload_size=r.workload_size,
        scale_to_zero_enabled=r.scale_to_zero,
    )
    try:
        ep = w.serving_endpoints.get(name)
        exists = True
    except Exception:
        exists = False

    if not exists:
        print(f"Creating endpoint {name} ({r.workload_type}/{r.workload_size}) ...")
        w.serving_endpoints.create(name=name, config=EndpointCoreConfigInput(served_entities=[served]))
    elif ep.state and ep.state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
        print(f"Endpoint {name} already provisioning; waiting for it to finish ...")
    elif not _matches(ep, r):
        print(f"Updating endpoint {name} to the requested config ...")
        w.serving_endpoints.update_config(name=name, served_entities=[served])
    else:
        print(f"Endpoint {name} already serves the requested config.")

    # Single generous wait for readiness, regardless of which path above ran.
    w.serving_endpoints.wait_get_serving_endpoint_not_updating(name, timeout=WAIT)
    print(f"  ready: {w.serving_endpoints.get(name).state}")

# COMMAND ----------
# MAGIC %md ## Probe the live request schema
# MAGIC Confirms the payload the model expects. If it differs from the default (`{"inputs": [...]}`),
# MAGIC set `ServingRunner.payload_format` / `audio_column` accordingly in `runners/serving.py`.

# COMMAND ----------
import json

import requests

probe_name = next(iter(endpoints))
host = w.config.host
for suffix in (f"/api/2.0/serving-endpoints/{probe_name}/openapi", f"/serving-endpoints/{probe_name}/openapi"):
    try:
        resp = requests.get(host + suffix, headers=w.config.authenticate(), timeout=30)
        if resp.status_code == 200:
            print(f"OpenAPI @ {suffix}:")
            print(json.dumps(resp.json(), indent=2)[:4000])
            break
        print(suffix, "->", resp.status_code)
    except Exception as e:  # noqa: BLE001
        print(suffix, "->", repr(e))

# Also show the registered model's MLflow signature if present.
try:
    mv = w.model_versions.get(full_name=model, version=int(default_version))
    print("\nModel version:", mv.version, "| status:", mv.status)
except Exception as e:  # noqa: BLE001
    print("model_versions.get:", repr(e))
