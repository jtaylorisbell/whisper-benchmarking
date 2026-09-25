#!/usr/bin/env bash
# Launcher for the faster-whisper (CTranslate2) AI Runtime arm. Same entry script as the HF arm
# (run.py), but first puts the pip-installed CUDA libs on the loader path — ctranslate2 doesn't
# bundle them and the base env doesn't expose cuBLAS/cuDNN, so without this the encoder fails with
# "libcublas.so.12 not found". nvidia.*.lib are namespace packages, so read __path__ (not __file__).
set -euo pipefail
SRC="${CODE_SOURCE_PATH:-${CODE_SOURCE:-.}}"
[ -d "$SRC/code" ] && SRC="$SRC/code"
cd "$SRC"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

NVIDIA_LIBS=$(python -c "
import nvidia.cublas.lib, nvidia.cudnn.lib
print(list(nvidia.cublas.lib.__path__)[0] + ':' + list(nvidia.cudnn.lib.__path__)[0])
")
export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
echo "faster-whisper launcher | LD_LIBRARY_PATH += $NVIDIA_LIBS"

python airuntime/run.py \
  --config "${WHISPER_CONFIG:-conf/faster_whisper.yml}" \
  --rate-catalog "${WHISPER_RATE_CATALOG:-conf/compute_costs.yml}" \
  --gpu "${WHISPER_GPU:-A10}"
