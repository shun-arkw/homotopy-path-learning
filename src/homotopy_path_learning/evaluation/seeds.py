"""Seed-list files: one target seed per line."""

from __future__ import annotations

import hashlib
from pathlib import Path


def read_seed_list(path: str | Path) -> list[int]:
    """Read target seeds from a text file.

    The file has one integer per line. Blank lines and lines starting with
    ``#`` are ignored. The order of the file is preserved. Duplicate and
    negative seeds, and lines that are not integers, raise ``ValueError``.
    Returns a list of Python ``int``.
    """

    file_path = Path(path)
    seeds: list[int] = []
    seen: set[int] = set()
    for number, raw in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            seed = int(line)
        except ValueError:
            raise ValueError(f"{file_path}:{number}: not an integer seed: {line!r}.") from None
        if seed < 0:
            raise ValueError(f"{file_path}:{number}: negative seed {seed}.")
        if seed in seen:
            raise ValueError(f"{file_path}:{number}: duplicate seed {seed}.")
        seen.add(seed)
        seeds.append(seed)
    if not seeds:
        raise ValueError(f"{file_path}: contains no seeds.")
    return seeds


def file_sha256(path: str | Path) -> str:
    """Return the hex SHA-256 digest of the file's bytes."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_file_record(path: str | Path) -> dict[str, str]:
    """Return ``{"path": ..., "sha256": ...}`` for metadata."""

    return {"path": str(path), "sha256": file_sha256(path)}


__all__ = ["file_sha256", "read_seed_list", "seed_file_record"]
