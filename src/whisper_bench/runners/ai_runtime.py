"""AI Runtime arm: HuggingFace Transformers ASR pipeline on a Databricks GPU.

Mirrors the reference notebook's inference shape (``pipeline("automatic-speech-recognition",
torch_dtype=fp16, device="cuda")`` with a ``batch_size``), but standardized on the full
``whisper-large-v3`` model. Audio decoding happens in ``prepare`` (setup phase) so only the pipeline
call itself is timed as inference. torch/transformers come from the runtime and are imported lazily.
"""

from __future__ import annotations

from ..data import Clip, decode_to_array
from .base import Runner


class AiRuntimeRunner(Runner):
    def setup(self) -> None:
        import torch
        from transformers import pipeline

        if not torch.cuda.is_available():
            raise RuntimeError("AI Runtime arm requires a GPU but torch.cuda.is_available() is False")
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.suite.model,
            torch_dtype=torch.float16,
            device="cuda",
        )
        self.notes = f"device={torch.cuda.get_device_name(0)}; accelerator={self.run_config.serverless_accelerator}"

    def prepare(self, clips: list[Clip]) -> None:
        # Decode all clips to waveforms up front — excluded from inference timing.
        self._arrays = [decode_to_array(c) for c in clips]

    def warmup(self, clips: list[Clip]) -> None:
        arrays = self._arrays[: len(clips)]
        self._pipe(list(arrays), batch_size=self.run_config.batch_size, generate_kwargs={"language": "en"})

    def transcribe(self, clips: list[Clip]) -> tuple[list[str], list[float]]:
        results = self._pipe(
            list(self._arrays),
            batch_size=self.run_config.batch_size,
            chunk_length_s=30,  # robust to any clip > 30s (Whisper's frame window)
            generate_kwargs={"language": "en"},
        )
        hypotheses = [r["text"].strip() for r in results]
        # Batched pipeline has no meaningful per-clip latency; leave latency empty (NULL in results).
        return hypotheses, []

    def teardown(self) -> None:
        try:
            import gc

            import torch

            del self._pipe
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
