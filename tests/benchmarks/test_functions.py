import dataclasses

import numpy as np
import pytest
from scipy.optimize import minimize

from benchmarks.functions import FUNCTIONS, TestFunction, get_function

NAMES = sorted(FUNCTIONS)


def grid(fn: TestFunction, n: int = 401) -> np.ndarray:
    """An n x n grid over the box domain of a 2-D function."""
    axes = [np.linspace(lo, hi, n) for lo, hi in zip(fn.lower_bound, fn.upper_bound)]
    return np.stack([g.ravel() for g in np.meshgrid(*axes)], axis=1)


# --- Registry -------------------------------------------------------------------------


def test_registry_contains_the_benchmark_functions():
    assert set(FUNCTIONS) == {"peaks", "ackley", "himmelblau"}


@pytest.mark.parametrize("name", NAMES)
def test_registry_keys_match_names(name):
    assert FUNCTIONS[name].name == name
    assert get_function(name) is FUNCTIONS[name]


def test_unknown_function_lists_available_names():
    with pytest.raises(KeyError, match="available"):
        get_function("rosenbrock")


# --- Definitions: domain and known minima --------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_domain_is_a_valid_2d_box(name):
    fn = get_function(name)
    assert len(fn.lower_bound) == len(fn.upper_bound) == 2
    assert all(lo < hi for lo, hi in zip(fn.lower_bound, fn.upper_bound))


@pytest.mark.parametrize("name", NAMES)
def test_known_minimizers_lie_inside_the_domain(name):
    fn = get_function(name)
    for x_min in fn.x_min:
        assert len(x_min) == len(fn.lower_bound)
        assert np.all(np.array(x_min) >= fn.lower_bound)
        assert np.all(np.array(x_min) <= fn.upper_bound)


@pytest.mark.parametrize(
    "name, domain",
    [("peaks", (-2.0, 2.0)), ("ackley", (-3.5, 3.5)), ("himmelblau", (-5.0, 5.0))],
)
def test_domains_follow_plate_et_al(name, domain):
    fn = get_function(name)
    assert fn.lower_bound == (domain[0], domain[0])
    assert fn.upper_bound == (domain[1], domain[1])


@pytest.mark.parametrize("name", NAMES)
def test_published_minima(name):
    fn = get_function(name)
    for x_min in fn.x_min:
        assert fn(np.array(x_min))[0] == pytest.approx(fn.function_min, abs=1e-2)


@pytest.mark.parametrize("name", NAMES)
def test_published_minima_are_local_minima(name):
    fn = get_function(name)
    for x_min in fn.x_min:
        res = minimize(
            lambda x: fn(x)[0],
            x_min,
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-10},
        )
        assert np.allclose(res.x, x_min, atol=2e-3)
        assert res.fun == pytest.approx(fn.function_min, abs=1e-3)


@pytest.mark.parametrize("name", NAMES)
def test_published_minimum_is_global_on_the_domain(name):
    # nothing on a fine grid over the whole domain is below the known minimum
    fn = get_function(name)
    values = fn(grid(fn))
    assert values.min() >= fn.function_min - 1e-3
    assert values.min() == pytest.approx(fn.function_min, abs=0.05)


# --- Values at hand-computed points ---------------------------------------------------


@pytest.mark.parametrize(
    "name, x, expected",
    [
        ("peaks", (0.0, 0.0), 8 / (3 * np.e)),  # 3/e - 1/(3e)
        ("ackley", (0.0, 0.0), 0.0),
        ("ackley", (1.0, 0.0), 20 - 20 * np.exp(-0.2 * np.sqrt(0.5))),  # cos terms give e
        ("himmelblau", (0.0, 0.0), 170.0),  # 11**2 + 7**2
        ("himmelblau", (3.0, 2.0), 0.0),
    ],
)
def test_values_at_known_points(name, x, expected):
    assert get_function(name)(np.array(x))[0] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("name", ["ackley", "himmelblau"])
def test_nonnegative_functions(name):
    fn = get_function(name)
    assert np.all(fn(grid(fn, 101)) >= -1e-12)


def test_ackley_is_symmetric():
    fn = get_function("ackley")
    x = np.random.default_rng(0).uniform(-3.5, 3.5, (100, 2))
    assert np.allclose(fn(x), fn(-x))  # even in each coordinate
    assert np.allclose(fn(x), fn(x[:, ::-1]))  # swapping x1 and x2


# --- Calling convention ---------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_vectorized_matches_row_by_row(name):
    fn = get_function(name)
    x = np.random.default_rng(0).uniform(fn.lower_bound, fn.upper_bound, (50, 2))
    batch = fn(x)
    assert batch.shape == (50,)
    assert np.allclose(batch, [fn(xi)[0] for xi in x])


@pytest.mark.parametrize("name", NAMES)
def test_accepts_1d_arrays_lists_and_ints(name):
    fn = get_function(name)
    expected = fn(np.array([[1.0, -1.0]]))
    assert fn(np.array([1.0, -1.0])).shape == (1,)
    assert np.array_equal(fn([1.0, -1.0]), expected)
    assert np.array_equal(fn([[1, -1]]), expected)  # ints are converted to float


@pytest.mark.parametrize("name", NAMES)
def test_empty_batch(name):
    assert get_function(name)(np.zeros((0, 2))).shape == (0,)


@pytest.mark.parametrize("name", NAMES)
def test_does_not_modify_input(name):
    x = np.array([[0.5, -0.5], [1.0, 1.0]])
    before = x.copy()
    get_function(name)(x)
    assert np.array_equal(x, before)


@pytest.mark.parametrize("shape", [(3, 3), (3, 1), (3,)])
def test_rejects_wrong_dimension(shape):
    with pytest.raises(ValueError, match="expects 2 inputs"):
        get_function("peaks")(np.zeros(shape))


def test_test_function_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        get_function("peaks").name = "other"  # type: ignore[misc]
