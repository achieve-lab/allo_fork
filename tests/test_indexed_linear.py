# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import allo
from allo.ir.types import float32, int32
from allo.library import indexed_linear, schedule_indexed_linear


def indexed_linear_reference(X, W, indices, bias):
    """NumPy reference for a fixed-fan-in indexed linear layer."""
    return bias + np.sum(W * X[indices], axis=1)


@pytest.mark.parametrize(
    "dtype,allo_type,n_in,n_out,n_active,output_parallelism",
    [
        (np.float32, float32, 7, 4, 3, 2),
        (np.int32, int32, 9, 3, 4, 1),
    ],
)
def test_indexed_linear_llvm(
    dtype, allo_type, n_in, n_out, n_active, output_parallelism
):
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.floating):
        X = rng.standard_normal(n_in).astype(dtype)
        W = rng.standard_normal((n_out, n_active)).astype(dtype)
        bias = rng.standard_normal(n_out).astype(dtype)
    else:
        X = rng.integers(-4, 5, size=n_in, dtype=dtype)
        W = rng.integers(-3, 4, size=(n_out, n_active), dtype=dtype)
        bias = rng.integers(-5, 6, size=n_out, dtype=dtype)
    indices = np.stack(
        [rng.choice(n_in, size=n_active, replace=False) for _ in range(n_out)]
    ).astype(np.int32)

    s = allo.customize(
        indexed_linear,
        instantiate=[allo_type, allo_type, allo_type, n_in, n_out, n_active],
    )
    schedule_indexed_linear(s, output_parallelism=output_parallelism)
    mod = s.build(target="llvm")

    actual = mod(X, W, indices, bias)
    expected = indexed_linear_reference(X, W, indices, bias)
    if np.issubdtype(dtype, np.floating):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    else:
        np.testing.assert_array_equal(actual, expected)


def test_indexed_linear_vhls_codegen():
    s = allo.customize(
        indexed_linear,
        instantiate=[float32, float32, float32, 8, 4, 3],
    )
    schedule_indexed_linear(s, output_parallelism=2)
    hls_code = s.build(target="vhls").hls_code

    assert "void indexed_linear" in hls_code
    assert "#pragma HLS pipeline II=1 rewind" in hls_code
    assert "#pragma HLS unroll factor=2" in hls_code
    assert hls_code.count("cyclic dim=1 factor=2") == 4
    assert "complete dim=1" in hls_code


def test_indexed_linear_schedule_rejects_nonpositive_parallelism():
    s = allo.customize(
        indexed_linear,
        instantiate=[float32, float32, float32, 4, 2, 2],
    )
    with pytest.raises(ValueError, match="output_parallelism must be positive"):
        schedule_indexed_linear(s, output_parallelism=0)
