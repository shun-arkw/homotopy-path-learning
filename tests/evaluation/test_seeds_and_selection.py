from __future__ import annotations

import hashlib

import numpy as np
import pytest

from homotopy_path_learning.evaluation.seeds import file_sha256, read_seed_list, seed_file_record
from homotopy_path_learning.evaluation.selection import parse_split, select_hard_seeds


def test_read_seed_list_order_comments_blank(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("# header\n30\n\n  10  \n# x\n20\n", encoding="utf-8")
    assert read_seed_list(path) == [30, 10, 20]


@pytest.mark.parametrize("text", ["1\n2\n1\n", "-3\n", "abc\n", "1.5\n", "# only comment\n\n"])
def test_read_seed_list_rejects(tmp_path, text):
    path = tmp_path / "s.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        read_seed_list(path)


def test_seed_file_record_sha256(tmp_path):
    path = tmp_path / "s.txt"
    path.write_bytes(b"1\n2\n")
    assert file_sha256(path) == hashlib.sha256(b"1\n2\n").hexdigest()
    assert seed_file_record(path) == {"path": str(path), "sha256": file_sha256(path)}


def rows(costs, successes=None, start=100):
    successes = successes if successes is not None else [1] * len(costs)
    return [
        {"target_seed": str(start + i), "linear_cost": str(c), "linear_success": str(s)}
        for i, (c, s) in enumerate(zip(costs, successes))
    ]


def test_parse_split():
    assert parse_split("test:500") == ("test", 500)
    for bad in ("test", "test:0", "test:x", ":3", "a/b:3"):
        with pytest.raises(ValueError):
            parse_split(bad)


def test_selection_threshold_and_assignment():
    costs = [1, 2, 3, 4, 10, 1, 5, 20, 3, 6, 7, 2, 8]
    successes = [1] * len(costs)
    successes[6] = 0  # cost 5 but failed
    data = rows(costs, successes)
    shuffled = list(reversed(data))  # input order must not matter
    sel = select_hard_seeds(shuffled, calibration=4, quantile=0.5, splits=[("a", 2), ("b", 3)])
    assert sel.threshold == pytest.approx(float(np.quantile([1, 2, 3, 4], 0.5)))  # 2.5
    assert sel.calibration_seed_range == (100, 103)
    # remaining seeds 104..112 costs 10,1,5(fail),20,3,6,7,2,8 -> eligible >= 2.5
    assert sel.splits["a"] == [104, 107]
    assert sel.splits["b"] == [108, 109, 110]
    assert sel.n_eligible == 6
    assert sel.n_candidates == 13


def test_selection_errors():
    data = rows([1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError, match="eligible"):
        select_hard_seeds(data, calibration=3, quantile=0.0, splits=[("a", 10)])
    with pytest.raises(ValueError):
        select_hard_seeds(data, calibration=6, quantile=0.5, splits=[("a", 1)])
    with pytest.raises(ValueError, match="unique"):
        select_hard_seeds(data + data[:1], calibration=2, quantile=0.5, splits=[("a", 1)])
    with pytest.raises(ValueError):
        select_hard_seeds(data, calibration=2, quantile=0.5, splits=[("a", 1), ("a", 1)])
