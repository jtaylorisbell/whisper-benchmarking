"""Tiny UC Volume I/O that works from any Databricks compute.

Both arms exchange data through a UC Volume: clips in (one Parquet), results out (one JSON per run).
The AI Runtime (serverless GPU) environment runs a plain Python script and may not have the
``/Volumes`` FUSE mount, so every helper tries the FUSE path first and falls back to the
``databricks-sdk`` Files API (which only needs the ambient job credentials). No Spark required.
"""

from __future__ import annotations

import io
import os
from typing import Optional


def _files_api():
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient().files


def read_bytes(volume_path: str) -> bytes:
    """Read a file from a ``/Volumes/...`` path (FUSE first, Files API fallback)."""
    try:
        with open(volume_path, "rb") as fh:
            return fh.read()
    except OSError:
        resp = _files_api().download(volume_path)
        return resp.contents.read()


def write_bytes(volume_path: str, data: bytes) -> None:
    """Write a file to a ``/Volumes/...`` path (FUSE first, Files API fallback)."""
    try:
        os.makedirs(os.path.dirname(volume_path), exist_ok=True)
        with open(volume_path, "wb") as fh:
            fh.write(data)
    except OSError:
        _files_api().upload(volume_path, io.BytesIO(data), overwrite=True)


def list_files(volume_dir: str, suffix: Optional[str] = None) -> list[str]:
    """List file paths under a Volume directory.

    Uses the Files API first because it is authoritative: the ``/Volumes`` FUSE mount caches
    directory metadata and can return a stale (empty) listing for files another job just wrote,
    even though reading a known path works. Falls back to FUSE ``os.listdir`` if the API is
    unavailable.
    """
    try:
        names = [f.path for f in _files_api().list_directory_contents(volume_dir) if not f.is_directory]
    except Exception:
        try:
            names = [os.path.join(volume_dir, n) for n in os.listdir(volume_dir)]
        except OSError:
            names = []
    return [n for n in names if suffix is None or n.endswith(suffix)]
