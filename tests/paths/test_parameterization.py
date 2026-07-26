from __future__ import annotations

import numpy as np
import pytest

from homotopy_path_learning.paths import (
    complex_coefficients_to_real,
    real_to_complex_coefficients,
)


def test_complex_real_roundtrip_uses_re_concat_im_layout() -> None:
    coefficients = np.array([1 + 2j, -3 + 0.5j, -4j], dtype=np.complex128)

    real_vector = complex_coefficients_to_real(coefficients)

    assert real_vector.dtype == np.float64
    np.testing.assert_array_equal(real_vector, np.array([1, -3, 0, 2, 0.5, -4]))
    np.testing.assert_array_equal(real_to_complex_coefficients(real_vector), coefficients)


def test_real_to_complex_rejects_odd_length_vector() -> None:
    with pytest.raises(ValueError, match="even"):
        real_to_complex_coefficients(np.array([1.0, 2.0, 3.0]))
