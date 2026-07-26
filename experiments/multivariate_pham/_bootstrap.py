"""CLI bootstrap helpers for the reference Julia runtime."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def preload_julia() -> None:
    """Set juliacall environment variables and import Julia before torch."""

    repository_root = Path(__file__).resolve().parents[2]
    cli_dir = str(Path(__file__).resolve().parent)
    sys.path[:] = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != cli_dir]
    os.environ.setdefault("PYTHON_JULIACALL_EXE", "/usr/local/bin/julia")
    os.environ.setdefault("PYTHON_JULIACALL_PROJECT", str(repository_root / "julia"))
    os.environ.setdefault("JULIAPKG_OFF", "1")
    os.environ.setdefault("JULIA_CONDAPKG_OFF", "1")
    os.environ.setdefault("JULIA_PYTHONCALL_INSTALL", "never")
    from juliacall import Main as _jl  # noqa: F401


__all__ = ["preload_julia"]
