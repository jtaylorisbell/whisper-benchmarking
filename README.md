# whisper-benchmarking

Price/performance benchmarking of **Whisper batch inference** on Databricks, comparing two
architectures head-to-head:

- **AI Runtime arm** — serverless GPU compute running the HuggingFace Transformers ASR pipeline
  (the pattern from the [`sgc-whisper-batch-inference`](https://docs.databricks.com/aws/en/notebooks/source/sgc-examples/sgc-whisper-batch-inference.html)
  reference notebook), driven as a DABs `ai_runtime_task`.
- **Model Serving arm** — a GPU serving endpoint deployed from `system.ai.whisper_large_v3`, hit by
  a concurrent client.

Both arms transcribe the **same model** (`whisper-large-v3`) on the **same audio** (full LibriSpeech
`test-clean`), so the comparison is apples-to-apples. We explore three axes — **batch size**, **GPU
type**, and **serving concurrency** — to find the best price/performance.

## What gets measured

For every run we record **runtime**, **accuracy**, and **cost**:

- **Throughput (RTFx)** — seconds of audio transcribed per second of *inference* wall time.
- **Accuracy (WER / CER)** — computed against LibriSpeech reference transcripts after applying
  OpenAI's Whisper English text normalizer. Accuracy is first-class: both arms must reach the *same*
  WER/CER, or the speed/cost comparison isn't valid.
- **Cost ($)** — **self-computed**, not read from system billing tables (see below).

### Why cost is self-computed

The benchmark jobs embed non-production setup — model download, cluster/endpoint spin-up, data
load, warmup — that `system.billing.usage` bills indiscriminately and reports ~12h late. So instead:

1. We time **only the transcription loop** (setup is timed separately and **excluded** from cost).
2. Cost = `inference_wall_sec × documented $/hr`, where the rate comes from a **versioned rate
   catalog** ([`conf/compute_costs.yml`](conf/compute_costs.yml)) whose `$/DBU` figures are pulled
   live from `system.billing.list_prices`.
3. The **serving arm counts both computes**: the GPU endpoint *and* the orchestration job that
   drives it. Higher concurrency shrinks the endpoint's active window — the core price/performance
   lever.

System billing tables are used only as an **optional cross-check** in the report (expected to read
higher, since they include the excluded setup).

## Results tables (the deliverable)

Created under `<catalog>.<schema>` (default `classic_stable_h0vpq7_catalog.whisper_bench`). Every
table and column carries a `COMMENT`, and units are in the names (`_usd`, `_sec`, `_ms`, `rtfx`).

| Object | Type | What it holds |
|---|---|---|
| `whisper_bench_results` | table | One row per run/config: runtime, accuracy, and self-computed cost, split into `cost_primary_usd` (GPU/endpoint) + `cost_secondary_usd` (orchestration). |
| `whisper_bench_rate_catalog` | table | The versioned `$/hr` rates behind every cost figure — so every dollar is reproducible from the tables alone. |
| `whisper_bench_summary` | view | Leaderboard: latest run per (arm, gpu, batch, concurrency), cheapest `$/audio-hour` first, WER/CER shown for parity. |

Headline columns: **`cost_per_audio_hour`** (price/performance) and **`throughput_rtfx`** (speed).

## Architecture

```
conf/                     Suite configs (smoke.yml, full_sweep.yml) + rate catalog (compute_costs.yml)
src/whisper_bench/        Shared library — ALL timing/metrics/accuracy/cost/results logic lives here once
  ├── config.py           Suite / RunConfig dataclasses + YAML loading
  ├── data.py             LibriSpeech -> one Parquet on a UC Volume; identical loading for both arms
  ├── metrics.py          Inference-only timing, RTFx, WER/CER (Whisper normalizer)
  ├── cost.py             Rate catalog + self-computed cost
  ├── results.py          RunResult + documented Delta tables/view (Spark-free emit, Spark ingest)
  ├── volumes.py          UC Volume I/O (FUSE with Files-API fallback — works without Spark)
  ├── runners/            The ONLY per-arm code: ai_runtime.py, serving.py (both subclass base.Runner)
  └── sweep.py            Orchestration (Spark-free run + Spark ingest/report)
airuntime/                AI Runtime entrypoint: run.py + command.sh (plain script, no Spark/notebook)
notebooks/                Thin serverless entrypoints (prep, deploy serving, run serving, report)
resources/                One DABs job per concern
databricks.yml            Bundle root (targets dev/prod; OAuth via fevm-aws profile)
```

**DRY guarantee:** adding a new architecture = adding one `Runner` subclass with a `transcribe`
method. Everything else is shared.

Data flows through the UC Volume so the Spark-less AI Runtime environment participates: clips in
(one Parquet), one result JSON per run out; the report job ingests those JSONs into the Delta tables.

## Usage

Prereqs: [`uv`](https://docs.astral.sh/uv/), Databricks CLI ≥ 1.18 (for `ai_runtime_task`), and the
`fevm-aws` OAuth profile.

```bash
# Test the pure logic locally (no workspace needed)
uv run pytest

# Deploy the bundle (builds the wheel + code.tgz, uploads, creates jobs)
databricks bundle deploy -t dev -p fevm-aws

# 1) Materialize the dataset (smoke = 100 clips; drop dataset.limit for the full ~2,620)
databricks bundle run whisper_prep_data -t dev -p fevm-aws

# 2a) AI Runtime arm
databricks bundle run whisper_airuntime_bench -t dev -p fevm-aws
# 2b) Model Serving arm (deploys the endpoint, then benchmarks it)
databricks bundle run whisper_serving_bench -t dev -p fevm-aws

# 3) Build/refresh the documented results tables + leaderboard
databricks bundle run whisper_report -t dev -p fevm-aws
```

Run a bigger sweep by deploying with `--var config_name=full_sweep.yml` (edit
[`conf/full_sweep.yml`](conf/full_sweep.yml) to pick the grid).

## Calibrating rates

`$/DBU` in the rate catalog is live from `system.billing.list_prices`; `dbus_per_hour` per compute
size is Databricks-published (marked `verify: docs`). Before quoting **absolute** dollars
externally, calibrate `dbus_per_hour` against a real tagged billing window — but note that relative
comparisons across batch size / concurrency on the *same* compute are exact regardless.

## Status

Smoke-test-first: the harness runs one cheap config per arm to validate the full pipeline
(data → inference → metrics → accuracy → cost → tables) before scaling to the full sweep. AI Runtime
serverless GPU is Public Preview; if this workspace isn't entitled, deploy/run surfaces it.
