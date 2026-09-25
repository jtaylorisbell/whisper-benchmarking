"""Dataset preparation and loading — Spark-free, via a single Parquet on the UC Volume.

We materialize LibriSpeech once into one Parquet file on the Volume holding raw (undecoded) audio
bytes + reference transcript + duration. Both arms then read the *same* Parquet, so they transcribe
byte-identical inputs — the precondition for the accuracy-parity check. A single file (not
thousands) keeps I/O trivial and works from the AI Runtime environment, which has no Spark session.

``datasets`` is provided by the runtime and imported lazily; ``pandas``/``pyarrow``/``soundfile`` are
light wheel dependencies.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

from .config import Suite
from . import volumes


@dataclass
class Clip:
    """One audio clip: raw encoded bytes + metadata. Identical for both arms."""

    id: str
    audio_bytes: bytes
    sampling_rate: int
    duration_sec: float
    reference_text: str


def _dataset_parquet_path(suite: Suite) -> str:
    return f"{suite.volume_root}/dataset/{suite.dataset.name}.parquet"


def prepare_dataset(suite: Suite, overwrite: bool = False) -> int:
    """Materialize the benchmark dataset into one Parquet on the Volume. Returns the clip count.

    Idempotent: skips if the Parquet already exists (unless ``overwrite``). Stores the original
    encoded audio bytes (FLAC for LibriSpeech) so nothing is re-encoded and both arms get identical
    inputs.
    """
    import itertools

    import pandas as pd
    import soundfile as sf
    from datasets import Audio, load_dataset

    path = _dataset_parquet_path(suite)
    if not overwrite:
        try:
            existing = _read_parquet(path)
            if len(existing) > 0:
                return len(existing)
        except Exception:
            pass  # not present yet -> build it

    d = suite.dataset
    # Stream so we download only the clips we consume: cheap for the smoke subset, and for the full
    # run it pulls just the requested split's shards (not LibriSpeech's ~30GB of training splits).
    # decode=False keeps the original encoded bytes and avoids the torchcodec/FFmpeg dependency.
    ds = load_dataset(d.hf_dataset, d.hf_config, split=d.hf_split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    stream = ds if d.limit is None else itertools.islice(ds, d.limit)

    records = []
    for i, ex in enumerate(stream):
        raw = ex["audio"].get("bytes")
        if raw is None:  # streaming sometimes yields a path/URL instead of inline bytes
            with open(ex["audio"]["path"], "rb") as fh:
                raw = fh.read()
        info = sf.info(io.BytesIO(raw))
        records.append({
            "id": str(ex.get(d.id_column, i)),
            "audio_bytes": raw,
            "sampling_rate": int(info.samplerate),
            "duration_sec": float(info.frames) / info.samplerate,
            "reference_text": ex[d.text_column],
        })

    buf = io.BytesIO()
    pd.DataFrame.from_records(records).to_parquet(buf, index=False)
    volumes.write_bytes(path, buf.getvalue())
    return len(records)


def load_clips(suite: Suite, limit: Optional[int] = None) -> list[Clip]:
    """Read clips from the Volume Parquet into memory, ordered by id (deterministic)."""
    df = _read_parquet(_dataset_parquet_path(suite)).sort_values("id").reset_index(drop=True)
    if limit is not None:
        df = df.head(limit)
    return [
        Clip(
            id=str(r["id"]),
            audio_bytes=bytes(r["audio_bytes"]),
            sampling_rate=int(r["sampling_rate"]),
            duration_sec=float(r["duration_sec"]),
            reference_text=str(r["reference_text"]),
        )
        for _, r in df.iterrows()
    ]


def decode_to_array(clip: Clip):
    """Decode a clip's bytes to a float32 mono waveform (for the HF pipeline)."""
    import numpy as np
    import soundfile as sf

    array, _sr = sf.read(io.BytesIO(clip.audio_bytes))
    if getattr(array, "ndim", 1) > 1:  # downmix stereo -> mono
        array = array.mean(axis=1)
    return np.asarray(array, dtype="float32")


def _read_parquet(volume_path: str):
    import pandas as pd

    return pd.read_parquet(io.BytesIO(volumes.read_bytes(volume_path)))
