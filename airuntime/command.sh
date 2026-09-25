#!/usr/bin/env bash
# Launcher for the AI Runtime (serverless GPU) arm. AI Runtime unpacks the code tarball and exposes
# it via $CODE_SOURCE_PATH (older builds: $CODE_SOURCE); we cd there so relative paths (conf/, src/)
# resolve, put the bundled package on PYTHONPATH, and run the entrypoint. Config/GPU are taken from
# env vars with smoke-friendly defaults (ai_runtime_task passes no job parameters).
set -euo pipefail
# AI Runtime sets CODE_SOURCE_PATH to the unpacked archive root. Our tarball has a single top-level
# `code/` dir, so descend into it whether CODE_SOURCE_PATH points at the parent or already at code/.
SRC="${CODE_SOURCE_PATH:-${CODE_SOURCE:-.}}"
[ -d "$SRC/code" ] && SRC="$SRC/code"
cd "$SRC"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# Default to the full sweep config; the AI Runtime task takes no job params, so the config is chosen
# here (override with a WHISPER_CONFIG env var if needed). The capped-vs-full dataset size is set by
# the `dataset.limit` line inside the config, which travels in this code tarball.
echo "whisper-bench AI Runtime launcher | config=${WHISPER_CONFIG:-conf/full_sweep.yml} | gpu=${WHISPER_GPU:-A10}"
python airuntime/run.py \
  --config "${WHISPER_CONFIG:-conf/full_sweep.yml}" \
  --rate-catalog "${WHISPER_RATE_CATALOG:-conf/compute_costs.yml}" \
  --gpu "${WHISPER_GPU:-A10}"
