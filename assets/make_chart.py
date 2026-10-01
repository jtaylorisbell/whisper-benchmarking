"""Render the README results chart from the landed leaderboard.

Regenerate with:  uv run --with matplotlib python assets/make_chart.py
Numbers mirror whisper_bench_summary (full LibriSpeech test-clean, 2,620 clips / 5.4h, single A10).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.unicode_minus"] = False

# (label, $/audio-hr, RTFx, WER%, color)
ROWS = [
    ("Model Serving · A10 endpoint",          0.2100,  8.3, 1.89, "#E45756"),
    ("AI Runtime · HF pipeline (batch 8)",     0.1823, 12.8, 1.79, "#F58518"),
    ("AI Runtime · faster-whisper large-v3",   0.1226, 19.1, 2.19, "#4C78A8"),
    ("AI Runtime · faster-whisper turbo",      0.0539, 43.4, 1.90, "#54A24B"),
]

labels = [r[0] for r in ROWS]
costs = [r[1] for r in ROWS]
y = list(range(len(ROWS)))[::-1]  # first row at top

fig, ax = plt.subplots(figsize=(9.6, 3.9), dpi=150)
for yi, (label, cost, rtfx, wer, color) in zip(y, ROWS):
    ax.barh(yi, cost, height=0.6, color=color, alpha=0.95, edgecolor=color, linewidth=1.5)
    ax.text(
        cost + 0.004, yi,
        f"${cost:.3f}   ·   RTFx {rtfx:.0f}   ·   WER {wer:.2f}%",
        va="center", ha="left", fontsize=9.5, color="#33373D",
    )

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlim(0, 0.30)
ax.set_xlabel("Dollars per audio-hour   (self-computed: inference-only time × documented hourly rate)", fontsize=10)
ax.set_title(
    "Cost to transcribe one hour of audio — Whisper on a single A10",
    fontsize=13, fontweight="bold", pad=26, loc="left",
)
ax.text(
    0, 1.055, "LibriSpeech test-clean · 2,620 clips / 5.4 h · lower is better · same weights except turbo (distilled)",
    transform=ax.transAxes, fontsize=9.5, color="#6B7280",
)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#C9CDD4")
ax.tick_params(length=0)
ax.set_axisbelow(True)
ax.xaxis.grid(True, color="#EDEFF2", linewidth=1)

fig.tight_layout()
out = "assets/cost_per_audio_hour.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
