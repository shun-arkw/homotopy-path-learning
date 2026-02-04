from __future__ import annotations


def _maybe_import_julia(enable: bool):
    """
    Import juliacall optionally.

    NOTE:
      Entry scripts should call this before importing torch when Julia is enabled,
      to avoid potential segfaults (see comments in task1/task2 scripts).
    """

    if not enable:
        return None
    try:
        from juliacall import Main as jl  # type: ignore

        return jl
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to import juliacall. Install/configure Julia+juliacall, or run with --no-julia."
        ) from exc


