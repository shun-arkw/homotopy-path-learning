from __future__ import annotations


def parse_int_list(s: str) -> list[int]:
    xs: list[int] = []
    for part in str(s).split(","):
        p = part.strip()
        if not p:
            continue
        xs.append(int(p))
    if not xs:
        raise ValueError("Empty list.")
    return xs


