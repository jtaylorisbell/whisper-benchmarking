"""AI Runtime arm, alternate engine: faster-whisper (CTranslate2) on a Databricks serverless GPU.

Same weights as the HF pipeline arm (whisper-large-v3), different inference engine — CTranslate2,
which is typically far more efficient than the transformers pipeline. This isolates "how much of the
throughput ceiling was the engine, not the hardware?" while holding model + dataset + GPU constant.

Scope: **sequential** transcription (one clip per ``model.transcribe`` call), the path proven by the
probe. It maps 1:1 to clips (no segment-reattribution), so accuracy parity with the other arms is
preserved. faster-whisper's cross-clip batching (concatenate + clip_timestamps) is a further lever
but carries timestamp/attribution edge cases, so it's intentionally left as future headroom rather
than risk corrupting WER. Because it's sequential, per-clip latency is captured for free.

Requires the CUDA libs on the loader path (see airuntime/command_fw.sh); ctranslate2 is imported
lazily so this module imports fine off-cluster.
"""

from __future__ import annotations

import time

from ..data import Clip, decode_to_array
from .base import Runner


class FasterWhisperRunner(Runner):
    def _model_id(self) -> str:
        """faster-whisper model id — explicit override, else derived from the HF name.

        'openai/whisper-large-v3' -> 'large-v3' (faster-whisper downloads the Systran CT2 weights).
        """
        if self.run_config.fw_model:
            return self.run_config.fw_model
        return self.suite.model.split("/")[-1].replace("whisper-", "")

    def setup(self) -> None:
        import ctranslate2
        from faster_whisper import WhisperModel

        if ctranslate2.get_cuda_device_count() < 1:
            raise RuntimeError("faster-whisper arm requires a GPU but ctranslate2 sees no CUDA device")
        model_id = self._model_id()
        self._model = WhisperModel(model_id, device="cuda", compute_type="float16")
        self.notes = f"engine=faster_whisper({ctranslate2.__version__}); model={model_id}; accelerator={self.run_config.serverless_accelerator}"

    def prepare(self, clips: list[Clip]) -> None:
        # Decode to float32 waveforms up front — excluded from inference timing (setup phase).
        self._arrays = [decode_to_array(c) for c in clips]

    def warmup(self, clips: list[Clip]) -> None:
        for arr in self._arrays[: len(clips)]:
            segments, _ = self._model.transcribe(arr, language="en", beam_size=1)
            _ = " ".join(s.text for s in segments)  # consume the generator to force compute

    def transcribe(self, clips: list[Clip]) -> tuple[list[str], list[float]]:
        hypotheses: list[str] = []
        latencies_ms: list[float] = []
        for arr in self._arrays:
            t0 = time.perf_counter()
            segments, _info = self._model.transcribe(arr, language="en", beam_size=1)
            text = " ".join(s.text for s in segments).strip()  # generator; consume to force compute
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            hypotheses.append(text)
        return hypotheses, latencies_ms

    def teardown(self) -> None:
        try:
            del self._model
        except Exception:
            pass
