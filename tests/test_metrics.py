"""Unit tests for metrics — timing, throughput, accuracy. No workspace needed."""

from __future__ import annotations

import math
import time

from whisper_bench.metrics import PhaseTimer, compute_accuracy, percentile_ms, rtfx


def test_rtfx():
    assert rtfx(300.0, 10.0) == 30.0
    assert math.isnan(rtfx(300.0, 0.0))


def test_percentile():
    assert percentile_ms([], 50) is None
    vals = [10, 20, 30, 40, 50]
    assert percentile_ms(vals, 50) == 30.0
    assert percentile_ms(vals, 100) == 50.0


def test_phase_timer_separates_phases():
    t = PhaseTimer()
    with t.phase("setup"):
        time.sleep(0.02)
    with t.phase("inference"):
        time.sleep(0.01)
    with t.phase("inference"):
        time.sleep(0.01)
    assert t.get("setup") >= 0.019
    assert t.get("inference") >= 0.019  # accumulated across two blocks
    assert t.get("missing") == 0.0


def test_accuracy_identical_is_zero():
    refs = ["the quick brown fox", "hello there world"]
    acc = compute_accuracy(refs, refs)
    assert acc.wer == 0.0
    assert acc.wer_normalized == 0.0
    assert acc.cer == 0.0


def test_accuracy_normalizer_ignores_case_and_punctuation():
    # Raw differs only by case/punctuation; the Whisper normalizer should zero it out.
    refs = ["Hello, World!"]
    hyps = ["hello world"]
    acc = compute_accuracy(refs, hyps)
    assert acc.wer_normalized == 0.0


def test_accuracy_counts_real_errors():
    refs = ["the quick brown fox"]  # 4 words
    hyps = ["the quick red fox"]  # 1 substitution
    acc = compute_accuracy(refs, hyps)
    assert acc.wer_normalized == 0.25
