import numpy as np

from jsopt.callbacks import ValueGradient, normalize_value_gradient


def test_normalize_value_gradient_from_tuple():
    gradient = np.array([1.0, -2.0])

    result = normalize_value_gradient((3.5, gradient))

    assert isinstance(result, ValueGradient)
    assert result.value == 3.5
    assert np.array_equal(result.gradient, gradient)


def test_normalize_value_gradient_keeps_dataclass():
    expected = ValueGradient(1.0, np.ones((2, 2)))

    result = normalize_value_gradient(expected)

    assert result is expected
