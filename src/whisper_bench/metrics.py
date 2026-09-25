"""Timing, throughput, latency, and accuracy metrics — shared by both arms.

Accuracy is a first-class result, not a sanity check: because both arms run the *same* model on
the *same* audio, they must land at the *same* WER/CER, and if they don't the speed/cost
comparison is meaningless. We report raw WER plus a **normalized** WER/CER computed after applying
OpenAI's Whisper English text normalizer (the standard way Whisper WER is reported — otherwise
casing and punctuation inflate the error).

No Spark / torch imports here, so this is unit-testable off-cluster.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
from jiwer import cer as _jiwer_cer
from jiwer import wer as _jiwer_wer
from whisper_normalizer.english import EnglishTextNormalizer

_normalizer = EnglishTextNormalizer()


@dataclass
class PhaseTimer:
    """Accumulates wall time per named phase so we can separate *setup* from *inference*.

    Only inference time feeds cost and throughput; setup (model load, endpoint warmup, data load)
    is recorded separately and explicitly excluded — that is the whole point of measuring ourselves
    rather than reading system billing tables.
    """

    _phases: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._phases[name] = self._phases.get(name, 0.0) + (time.perf_counter() - start)

    def get(self, name: str) -> float:
        return self._phases.get(name, 0.0)


def rtfx(total_audio_sec: float, inference_wall_sec: float) -> float:
    """Real-time factor: seconds of audio transcribed per second of inference wall time.

    RTFx of 30 means the system transcribes 30s of audio every 1s. Higher is faster.
    """
    if inference_wall_sec <= 0:
        return float("nan")
    return total_audio_sec / inference_wall_sec


def percentile_ms(latencies_ms: list[float], p: float) -> Optional[float]:
    """The p-th percentile of a list of per-request latencies (ms); None if empty."""
    if not latencies_ms:
        return None
    return float(np.percentile(np.asarray(latencies_ms, dtype=float), p))


@dataclass
class Accuracy:
    wer: Optional[float]
    cer: Optional[float]
    wer_normalized: Optional[float]


def compute_accuracy(references: list[str], hypotheses: list[str]) -> Accuracy:
    """WER/CER of ``hypotheses`` against ``references``.

    - ``wer`` — raw jiwer WER (lightly cased/stripped only), for reference.
    - ``wer_normalized`` / ``cer`` — computed after the Whisper English normalizer; these are the
      headline accuracy numbers and the apples-to-apples parity check.

    Pairs whose reference normalizes to empty are dropped (jiwer rejects empty references); returns
    ``None`` fields if nothing remains.
    """
    if len(references) != len(hypotheses):
        raise ValueError(f"references ({len(references)}) and hypotheses ({len(hypotheses)}) differ in length")
    if not references:
        return Accuracy(None, None, None)

    raw_refs = [r.strip().lower() for r in references]
    raw_hyps = [h.strip().lower() for h in hypotheses]
    raw_wer = _safe(lambda: _jiwer_wer(raw_refs, raw_hyps))

    norm_refs, norm_hyps = [], []
    for ref, hyp in zip(references, hypotheses):
        nref = _normalizer(ref)
        if not nref.strip():
            continue
        norm_refs.append(nref)
        norm_hyps.append(_normalizer(hyp))

    norm_wer = _safe(lambda: _jiwer_wer(norm_refs, norm_hyps)) if norm_refs else None
    norm_cer = _safe(lambda: _jiwer_cer(norm_refs, norm_hyps)) if norm_refs else None
    return Accuracy(wer=raw_wer, cer=norm_cer, wer_normalized=norm_wer)


def _safe(fn):
    try:
        return float(fn())
    except Exception:
        return None
