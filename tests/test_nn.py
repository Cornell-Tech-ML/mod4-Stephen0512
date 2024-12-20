import pytest
from hypothesis import given

import minitorch
from minitorch import Tensor

from .strategies import assert_close
from .tensor_strategies import tensors


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t: Tensor) -> None:
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t: Tensor) -> None:
    ### Test max reduction calculated by max reduction forward method along different dimensions

    # Max along last dimension
    out = minitorch.max(t, 2)

    # Check that the shape of the output is correct
    assert out.shape == (2, 3, 1)

    # Check that the output values are correct
    for i in range(2):
        for j in range(3):
            assert_close(out[i, j, 0], max([t[i, j, k] for k in range(4)]))

    # Max along middle dimension
    out = minitorch.max(t, 1)

    # Check that the shape of the output is correct
    assert out.shape == (2, 1, 4)

    # Check that the output values are correct
    for i in range(2):
        for j in range(4):
            assert_close(out[i, 0, j], max([t[i, k, j] for k in range(3)]))

    # Max along first dimension
    out = minitorch.max(t, 0)

    # Check that the shape of the output is correct
    assert out.shape == (1, 3, 4)

    # Check that the output values are correct
    for i in range(3):
        for j in range(4):
            assert_close(out[0, i, j], max([t[k, i, j] for k in range(2)]))

    ### Test gradients calcualted by max reduction backward method along different dimensions

    # As max's gradient is undefined when there are duplicate maximum values, we need to ensure unique elements for gradient checking.
    # We do this by adding small random perturbations to create three variants of the input tensor.
    unique_t0 = t + minitorch.rand(t.shape) * 1e-5  # For testing max along dim 0
    unique_t1 = t + minitorch.rand(t.shape) * 1e-5  # For testing max along dim 1
    unique_t2 = t + minitorch.rand(t.shape) * 1e-5  # For testing max along dim 2

    # Check gradients for max reduction along each dimension
    # grad_check verifies that our analytical gradients match numerical approximations
    minitorch.grad_check(
        lambda t: minitorch.max(t, 0), unique_t0
    )  # Max along first dimension
    minitorch.grad_check(
        lambda t: minitorch.max(t, 1), unique_t1
    )  # Max along second dimension
    minitorch.grad_check(
        lambda t: minitorch.max(t, 2), unique_t2
    )  # Max along third dimension


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t: Tensor) -> None:
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)
