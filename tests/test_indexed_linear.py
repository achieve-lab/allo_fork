# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import allo
from allo.ir.types import Fixed, float16, float32, int8, int16, int32
from allo.library import KERNEL2SCHEDULE, indexed_linear, schedule_indexed_linear


def indexed_linear_reference(X, W, indices, bias):
    """Follow the kernel's active-slot-major accumulation order exactly."""
    output = bias.copy()
    for k in range(W.shape[1]):
        for o in range(W.shape[0]):
            output[o] += W[o, k] * X[indices[o, k]]
    return output


def build_indexed_linear(
    input_type,
    n_in,
    n_out,
    n_active,
    output_parallelism=1,
    *,
    weight_type=None,
    output_type=None,
):
    """Build one scheduled indexed-linear specialization for LLVM."""
    weight_type = input_type if weight_type is None else weight_type
    output_type = input_type if output_type is None else output_type
    s = allo.customize(
        indexed_linear,
        instantiate=[input_type, weight_type, output_type, n_in, n_out, n_active],
    )
    schedule_indexed_linear(s, output_parallelism=output_parallelism)
    return s.build(target="llvm")


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

    mod = build_indexed_linear(allo_type, n_in, n_out, n_active, output_parallelism)

    actual = mod(X, W, indices, bias)
    expected = indexed_linear_reference(X, W, indices, bias)
    if np.issubdtype(dtype, np.floating):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    else:
        np.testing.assert_array_equal(actual, expected)


def test_indexed_linear_hand_computed_llvm():
    X = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float32)
    W = np.array([[1.5, -2.0], [4.0, 0.25], [0.5, -1.0]], dtype=np.float32)
    indices = np.array([[0, 2], [1, 3], [3, 0]], dtype=np.int32)
    bias = np.array([0.5, -0.5, 1.0], dtype=np.float32)

    mod = build_indexed_linear(float32, 4, 3, 2)

    np.testing.assert_allclose(
        mod(X, W, indices, bias),
        np.array([2.5, -3.75, 0.5], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "X,W,indices,bias,expected",
    [
        pytest.param(
            np.array([7], dtype=np.int32),
            np.array([[3]], dtype=np.int32),
            np.array([[0]], dtype=np.int32),
            np.array([-2], dtype=np.int32),
            np.array([19], dtype=np.int32),
            id="minimum-shape",
        ),
        pytest.param(
            np.array([2, -1, 4], dtype=np.int32),
            np.array([[1, 2, -3], [4, -2, 1]], dtype=np.int32),
            np.array([[0, 0, 2], [1, 1, 1]], dtype=np.int32),
            np.array([5, -3], dtype=np.int32),
            np.array([-1, -6], dtype=np.int32),
            id="repeated-indices",
        ),
        pytest.param(
            np.array([-3, 2, 8], dtype=np.int32),
            np.zeros((2, 2), dtype=np.int32),
            np.array([[0, 2], [2, 1]], dtype=np.int32),
            np.array([4, -7], dtype=np.int32),
            np.array([4, -7], dtype=np.int32),
            id="zero-weights-return-bias",
        ),
    ],
)
def test_indexed_linear_edge_cases_llvm(X, W, indices, bias, expected):
    n_out, n_active = W.shape
    mod = build_indexed_linear(int32, X.shape[0], n_out, n_active)

    np.testing.assert_array_equal(mod(X, W, indices, bias), expected)


def test_indexed_linear_matches_reconstructed_dense_llvm():
    rng = np.random.default_rng(1)
    n_in, n_out, n_active = 6, 5, 4
    X = rng.standard_normal(n_in).astype(np.float32)
    W = rng.standard_normal((n_out, n_active)).astype(np.float32)
    # Repeated indices are intentional: their weights must add in the dense form.
    indices = rng.integers(0, n_in, size=(n_out, n_active), dtype=np.int32)
    bias = rng.standard_normal(n_out).astype(np.float32)
    dense_W = np.zeros((n_out, n_in), dtype=np.float32)
    for o in range(n_out):
        for k in range(n_active):
            dense_W[o, indices[o, k]] += W[o, k]

    mod = build_indexed_linear(float32, n_in, n_out, n_active)

    np.testing.assert_allclose(
        mod(X, W, indices, bias), dense_W @ X + bias, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize(
    "output_parallelism",
    [
        pytest.param(1, id="serial"),
        pytest.param(2, id="partial-nondivisible"),
        pytest.param(5, id="fully-unrolled"),
    ],
)
def test_indexed_linear_schedule_variants_llvm(output_parallelism):
    n_in, n_out, n_active = 7, 5, 3
    X = np.array([1.0, -2.0, 3.0, 0.5, -1.5, 4.0, 2.5], dtype=np.float32)
    W = np.array(
        [
            [0.5, -1.0, 2.0],
            [1.5, 0.25, -0.5],
            [-2.0, 1.0, 0.75],
            [0.0, -1.5, 2.5],
            [1.25, -0.75, 0.5],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [[0, 3, 6], [1, 4, 2], [5, 0, 3], [6, 2, 1], [4, 5, 0]],
        dtype=np.int32,
    )
    bias = np.array([0.25, -0.5, 1.0, 2.0, -1.25], dtype=np.float32)
    mod = build_indexed_linear(float32, n_in, n_out, n_active, output_parallelism)

    np.testing.assert_allclose(
        mod(X, W, indices, bias),
        indexed_linear_reference(X, W, indices, bias),
        rtol=1e-6,
        atol=1e-6,
    )


def test_indexed_linear_default_vhls_codegen():
    s = allo.customize(
        indexed_linear,
        instantiate=[float32, float32, float32, 8, 5, 3],
    )
    schedule_indexed_linear(s)
    hls_code = s.build(target="vhls").hls_code

    assert "void indexed_linear" in hls_code
    assert "#pragma HLS pipeline II=1 rewind" in hls_code
    assert "#pragma HLS unroll" not in hls_code
    assert "#pragma HLS array_partition" not in hls_code


def test_indexed_linear_vhls_codegen():
    s = allo.customize(
        indexed_linear,
        instantiate=[float32, float32, float32, 8, 5, 3],
    )
    schedule_indexed_linear(s, output_parallelism=2)
    hls_code = s.build(target="vhls").hls_code

    assert "void indexed_linear" in hls_code
    assert "#pragma HLS pipeline II=1 rewind" in hls_code
    assert "#pragma HLS unroll factor=2" in hls_code
    assert hls_code.count("cyclic dim=1 factor=2") == 4
    assert "complete dim=1" in hls_code


@pytest.mark.parametrize("output_parallelism", [0, -1])
def test_indexed_linear_schedule_rejects_nonpositive_parallelism(
    output_parallelism,
):
    s = allo.customize(
        indexed_linear,
        instantiate=[float32, float32, float32, 4, 2, 2],
    )
    with pytest.raises(ValueError, match="output_parallelism must be positive"):
        schedule_indexed_linear(s, output_parallelism=output_parallelism)


def test_indexed_linear_schedule_rejects_noninteger_parallelism():
    s = allo.customize(
        indexed_linear,
        instantiate=[float32, float32, float32, 4, 2, 2],
    )
    with pytest.raises(TypeError, match="output_parallelism must be an integer"):
        schedule_indexed_linear(s, output_parallelism=1.5)


def test_indexed_linear_int8_inputs_widen_to_int32_llvm():
    X = np.array([100, -120, 75, -64, 31], dtype=np.int8)
    W = np.array([[100, -100, 80], [-90, 110, -120]], dtype=np.int8)
    indices = np.array([[0, 1, 2], [1, 3, 4]], dtype=np.int32)
    bias = np.array([100_000, -50_000], dtype=np.int32)

    mod = build_indexed_linear(int8, 5, 2, 3, weight_type=int8, output_type=int32)

    # Products exceed int8 range. Correct results therefore require the
    # int8-by-int8 product and the accumulation to use their widened types.
    expected = np.array([128_000, -49_960], dtype=np.int32)
    np.testing.assert_array_equal(mod(X, W, indices, bias), expected)


def test_indexed_linear_distinct_integer_template_types_llvm():
    X = np.array([1_000, -2_000, 300, -400], dtype=np.int16)
    W = np.array([[100, -50], [-120, 75]], dtype=np.int8)
    indices = np.array([[0, 1], [2, 3]], dtype=np.int32)
    bias = np.array([7, -9], dtype=np.int32)

    mod = build_indexed_linear(int16, 4, 2, 2, weight_type=int8, output_type=int32)

    # This exercises three distinct template types and values whose products
    # exceed int16 range before they are accumulated into int32 outputs.
    expected = np.array([200_007, -66_009], dtype=np.int32)
    np.testing.assert_array_equal(mod(X, W, indices, bias), expected)


def test_indexed_linear_float16_inputs_accumulate_in_float32_llvm():
    X = np.array([1.5, -2.0, 0.25, 3.0, -0.5], dtype=np.float16)
    W = np.array([[0.5, -1.25, 2.0], [-2.0, 0.75, 1.5]], dtype=np.float16)
    indices = np.array([[0, 1, 4], [3, 2, 0]], dtype=np.int32)
    bias = np.array([0.125, -1.5], dtype=np.float32)

    mod = build_indexed_linear(
        float16, 5, 2, 3, weight_type=float16, output_type=float32
    )

    expected = indexed_linear_reference(X, W, indices, bias)
    np.testing.assert_allclose(mod(X, W, indices, bias), expected, rtol=1e-6, atol=1e-6)


def test_indexed_linear_fixed_point_llvm():
    input_type = Fixed(8, 4)
    output_type = Fixed(24, 8)
    X = np.array([1.5, -2.0, 0.25, 3.0], dtype=np.float32)
    W = np.array([[0.5, -1.25], [-2.0, 0.75]], dtype=np.float32)
    indices = np.array([[0, 3], [1, 2]], dtype=np.int32)
    bias = np.array([0.125, -1.5], dtype=np.float32)

    mod = build_indexed_linear(
        input_type, 4, 2, 2, weight_type=input_type, output_type=output_type
    )

    np.testing.assert_allclose(
        mod(X, W, indices, bias),
        np.array([-2.875, 2.6875], dtype=np.float32),
        rtol=0,
        atol=1 / 256,
    )


def test_indexed_linear_composes_into_parent_llvm():
    n_in, n_out, n_active = 6, 3, 3

    def parent(
        X: float32[n_in],
        W: float32[n_out, n_active],
        indices: int32[n_out, n_active],
        bias: float32[n_out],
    ) -> float32[n_out]:
        return indexed_linear[float32, float32, float32, n_in, n_out, n_active](
            X, W, indices, bias
        )

    X = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 4.0], dtype=np.float32)
    W = np.array(
        [[0.5, -1.0, 2.0], [1.25, 0.75, -0.5], [-2.0, 1.5, 0.25]],
        dtype=np.float32,
    )
    indices = np.array([[0, 3, 5], [1, 4, 2], [5, 0, 3]], dtype=np.int32)
    bias = np.array([0.25, -0.75, 1.0], dtype=np.float32)

    assert KERNEL2SCHEDULE[indexed_linear] is schedule_indexed_linear
    s = allo.customize(parent)
    s.compose(
        indexed_linear,
        instantiate=[float32, float32, float32, n_in, n_out, n_active],
    )
    mod = s.build(target="llvm")

    np.testing.assert_allclose(
        mod(X, W, indices, bias),
        indexed_linear_reference(X, W, indices, bias),
        rtol=1e-6,
        atol=1e-6,
    )


def test_two_indexed_linear_layers_compose_by_id_llvm():
    n_in, n_hidden, n_out = 6, 4, 3
    n_active_0, n_active_1 = 3, 2

    def two_layers(
        X: float32[n_in],
        W0: float32[n_hidden, n_active_0],
        indices0: int32[n_hidden, n_active_0],
        bias0: float32[n_hidden],
        W1: float32[n_out, n_active_1],
        indices1: int32[n_out, n_active_1],
        bias1: float32[n_out],
    ) -> float32[n_out]:
        hidden = indexed_linear[float32, float32, float32, n_in, n_hidden, n_active_0](
            X, W0, indices0, bias0
        )
        return indexed_linear[float32, float32, float32, n_hidden, n_out, n_active_1](
            hidden, W1, indices1, bias1
        )

    X = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 4.0], dtype=np.float32)
    W0 = np.array(
        [[0.5, -1.0, 2.0], [1.25, 0.75, -0.5], [-2.0, 1.5, 0.25], [1, 2, -1]],
        dtype=np.float32,
    )
    indices0 = np.array([[0, 3, 5], [1, 4, 2], [5, 0, 3], [2, 4, 1]], dtype=np.int32)
    bias0 = np.array([0.25, -0.75, 1.0, 0.5], dtype=np.float32)
    W1 = np.array([[1.0, -0.5], [0.25, 2.0], [-1.5, 0.75]], dtype=np.float32)
    indices1 = np.array([[0, 3], [2, 1], [3, 0]], dtype=np.int32)
    bias1 = np.array([-1.0, 0.5, 2.0], dtype=np.float32)

    s = allo.customize(two_layers)
    s.compose(
        indexed_linear,
        instantiate=[
            float32,
            float32,
            float32,
            n_in,
            n_hidden,
            n_active_0,
        ],
    )
    s.compose(
        indexed_linear,
        id="1",
        instantiate=[
            float32,
            float32,
            float32,
            n_hidden,
            n_out,
            n_active_1,
        ],
    )
    mod = s.build(target="llvm")

    hidden = indexed_linear_reference(X, W0, indices0, bias0)
    expected = indexed_linear_reference(hidden, W1, indices1, bias1)
    np.testing.assert_allclose(
        mod(X, W0, indices0, bias0, W1, indices1, bias1), expected
    )


def test_two_composed_indexed_linear_layers_vhls_codegen():
    n_in, n_hidden, n_out = 6, 4, 3
    n_active_0, n_active_1 = 3, 2

    def two_layers(
        X: float32[n_in],
        W0: float32[n_hidden, n_active_0],
        indices0: int32[n_hidden, n_active_0],
        bias0: float32[n_hidden],
        W1: float32[n_out, n_active_1],
        indices1: int32[n_out, n_active_1],
        bias1: float32[n_out],
    ) -> float32[n_out]:
        hidden = indexed_linear[float32, float32, float32, n_in, n_hidden, n_active_0](
            X, W0, indices0, bias0
        )
        return indexed_linear[float32, float32, float32, n_hidden, n_out, n_active_1](
            hidden, W1, indices1, bias1
        )

    s = allo.customize(two_layers)
    s.compose(
        indexed_linear,
        instantiate=[
            float32,
            float32,
            float32,
            n_in,
            n_hidden,
            n_active_0,
        ],
    )
    s.compose(
        indexed_linear,
        id="1",
        instantiate=[
            float32,
            float32,
            float32,
            n_hidden,
            n_out,
            n_active_1,
        ],
    )
    hls_code = s.build(target="vhls").hls_code

    assert "void indexed_linear(" in hls_code
    assert "void indexed_linear_1(" in hls_code
    assert hls_code.count("#pragma HLS pipeline II=1") == 4
    assert hls_code.count("#pragma HLS pipeline II=1 rewind") == 2
