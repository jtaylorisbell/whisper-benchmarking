"""Model Serving arm: concurrent client against a GPU endpoint serving system.ai.whisper_large_v3.

We drive the endpoint with an explicit thread pool (rather than ``ai_query``) so we can set the
number of parallel in-flight requests precisely and measure per-request latency. Each request
carries ``batch_size`` clips.

REPLICAS / COST: for GPU endpoints Databricks provisions ``replicas = provisioned_concurrency / 4``,
and each replica is a full GPU (one A10 for GPU_MEDIUM). This benchmark's price/performance study
runs at **one replica** (``workload_size: Small`` = provisioned concurrency 4), so client
concurrency is kept <= 4 (beyond that would only queue on the single GPU, not add hardware). Cost is
charged as ``replicas * per-GPU-rate`` via :meth:`replicas`, so it stays correct if a larger
workload size (more replicas / GPUs) is ever used. The throughput lever we sweep here is therefore
client ``batch_size`` (clips per request), the direct analog of the AI Runtime batch size.

REQUEST/RESPONSE SCHEMA (confirmed from the model's MLflow signature, system.ai.whisper_large_v3 v3):
  inputs  = a single unnamed binary column (raw audio bytes)
  outputs = string (the transcript)
Over REST/JSON, MLflow base64-encodes binary column values, so the request is
``{"inputs": ["<base64 audio>", ...]}`` and the response is ``{"predictions": ["<text>", ...]}``.
Multiple clips per request (batch_size > 1) map to multiple rows of that column. Payload
construction and response parsing are still isolated in ``_build_payload`` / ``_parse_response`` so
any endpoint-specific quirk is a one-method change.
"""

from __future__ import annotations

import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..data import Clip
from .base import Runner

# How the audio is packed into the request body. Confirmed default for whisper_large_v3 is "inputs"
# (single unnamed binary column -> base64 string list). Fallbacks kept for other packagings.
#   "inputs"            -> {"inputs": [b64, ...]}
#   "dataframe_split"   -> {"dataframe_split": {"columns": [0], "data": [[b64], ...]}}
#   "dataframe_records" -> {"dataframe_records": [{<audio_column>: b64}, ...]}
_DEFAULT_PAYLOAD_FORMAT = "inputs"
_AUDIO_COLUMN = 0  # the signature's column is unnamed (index 0)

# GPU endpoint replicas = provisioned_concurrency / 4, and each replica is one GPU. workload_size
# maps to a provisioned concurrency, hence a replica (GPU) count used for cost. Small is confirmed
# (1 replica = 1 A10); Medium/Large are best-effort and only matter for a future scale-out study.
_WORKLOAD_SIZE_REPLICAS = {"Small": 1, "Medium": 2, "Large": 4}


class ServingRunner(Runner):
    payload_format = _DEFAULT_PAYLOAD_FORMAT
    audio_column = _AUDIO_COLUMN

    def replicas(self) -> int:
        """GPU replicas billed in parallel = provisioned_concurrency / 4 (derived from workload_size)."""
        return _WORKLOAD_SIZE_REPLICAS.get(self.run_config.workload_size, 1)

    def setup(self) -> None:
        import requests
        from databricks.sdk import WorkspaceClient

        self._session = requests.Session()
        self._w = WorkspaceClient()
        name = self.run_config.endpoint_name
        if not name:
            raise ValueError("serving run requires endpoint_name")
        ep = self._w.serving_endpoints.get(name)
        state = getattr(ep.state, "ready", None)
        if state is not None and str(state).upper().find("READY") < 0:
            # Give a just-created endpoint time to finish provisioning.
            self._w.serving_endpoints.wait_get_serving_endpoint_not_updating(name)
        self._url = f"{self._w.config.host}/serving-endpoints/{name}/invocations"
        self.notes = f"endpoint={name}; workload={self.run_config.workload_type}/{self.run_config.workload_size}"

    def prepare(self, clips: list[Clip]) -> None:
        # base64-encode once (client-side prep, excluded from inference timing) and group into
        # request-sized batches, preserving global clip order for reassembly.
        b64 = [base64.b64encode(c.audio_bytes).decode("ascii") for c in clips]
        bs = max(1, self.run_config.batch_size)
        self._batches = [list(range(i, min(i + bs, len(clips)))) for i in range(0, len(clips), bs)]
        self._b64 = b64

    def warmup(self, clips: list[Clip]) -> None:
        payload = self._build_payload([base64.b64encode(c.audio_bytes).decode("ascii") for c in clips])
        self._post(payload)

    def transcribe(self, clips: list[Clip]) -> tuple[list[str], list[float]]:
        hypotheses: list[str | None] = [None] * len(clips)
        latencies_ms: list[float] = []
        concurrency = max(1, self.run_config.concurrency)

        def do_batch(indices: list[int]):
            payload = self._build_payload([self._b64[i] for i in indices])
            start = time.perf_counter()
            resp = self._post(payload)
            latency = (time.perf_counter() - start) * 1000.0
            texts = self._parse_response(resp, len(indices))
            return indices, texts, latency

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(do_batch, b) for b in self._batches]
            for fut in as_completed(futures):
                indices, texts, latency = fut.result()
                latencies_ms.append(latency)
                for idx, text in zip(indices, texts):
                    hypotheses[idx] = text

        missing = [i for i, h in enumerate(hypotheses) if h is None]
        if missing:
            raise RuntimeError(f"{len(missing)} clips got no transcription from the endpoint")
        return [h for h in hypotheses], latencies_ms  # type: ignore[misc]

    def teardown(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass

    # --- Schema-specific bits (verify against the deployed endpoint) ---
    def _build_payload(self, b64_batch: list[str]) -> dict:
        if self.payload_format == "inputs":
            return {"inputs": b64_batch}
        if self.payload_format == "dataframe_records":
            return {"dataframe_records": [{self.audio_column: b} for b in b64_batch]}
        if self.payload_format == "dataframe_split":
            return {"dataframe_split": {"columns": [self.audio_column], "data": [[b] for b in b64_batch]}}
        raise ValueError(f"unknown payload_format {self.payload_format!r}")

    def _parse_response(self, resp: dict, n: int) -> list[str]:
        preds = resp.get("predictions", resp)
        if isinstance(preds, dict):  # some models nest, e.g. {"predictions": {"text": [...]}}
            preds = preds.get("text") or preds.get("transcription") or list(preds.values())[0]
        texts = []
        for p in preds:
            if isinstance(p, dict):
                texts.append(str(p.get("text") or p.get("transcription") or p.get("prediction") or "").strip())
            else:
                texts.append(str(p).strip())
        if len(texts) != n:
            raise RuntimeError(f"expected {n} transcriptions in response, got {len(texts)}: {str(resp)[:200]}")
        return texts

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json", **self._w.config.authenticate()}
        r = self._session.post(self._url, headers=headers, json=payload, timeout=300)
        if r.status_code != 200:
            raise RuntimeError(f"serving request failed {r.status_code}: {r.text[:300]}")
        return r.json()
