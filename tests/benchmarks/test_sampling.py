import numpy as np
import pytest

from benchmarks import FUNCTIONS, get_function, load_samples, sample_dataset, sample_function, save_samples
from hybopt.data import Dataset


def multi_output_data(n=20, seed=0):
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(300, 400, n), rng.uniform(0, 1e-3, n)])
    return x, np.column_stack([np.sin(x[:, 0]), 1e4 * x[:, 1] + 5.0])


# --- sample_function ------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FUNCTIONS))
@pytest.mark.parametrize("design", ["lhs", "uniform"])
def test_samples_shapes_domain_and_labels(name, design):
    fn = get_function(name)
    x, y = sample_function(fn, 500, seed=0, design=design)
    assert x.shape == (500, 2) and y.shape == (500,)
    assert x.dtype == np.float64 and y.dtype == np.float64
    assert np.all(x >= fn.lower_bound) and np.all(x <= fn.upper_bound)
    assert np.array_equal(y, fn(x))


@pytest.mark.parametrize("design", ["lhs", "uniform"])
def test_same_seed_same_samples(design):
    fn = get_function("peaks")
    x1, y1 = sample_function(fn, 200, seed=3, design=design)
    x2, y2 = sample_function(fn, 200, seed=3, design=design)
    x3, _ = sample_function(fn, 200, seed=4, design=design)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
    assert not np.array_equal(x1, x3)


def test_lhs_and_uniform_differ():
    fn = get_function("peaks")
    x_lhs, _ = sample_function(fn, 200, seed=0, design="lhs")
    x_uni, _ = sample_function(fn, 200, seed=0, design="uniform")
    assert not np.array_equal(x_lhs, x_uni)


@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_lhs_stratifies_each_axis(name):
    # Latin hypercube: exactly one sample in each of the n equal-width bins per axis
    fn = get_function(name)
    n = 100
    x, _ = sample_function(fn, n, seed=0)
    unit = (x - np.array(fn.lower_bound)) / (
        np.array(fn.upper_bound) - np.array(fn.lower_bound)
    )
    for axis in range(len(fn.lower_bound)):
        assert sorted(np.floor(unit[:, axis] * n).astype(int)) == list(range(n))


def test_samples_cover_the_domain():
    fn = get_function("himmelblau")
    x, _ = sample_function(fn, 2000, seed=0)
    assert np.allclose(x.mean(axis=0), 0.0, atol=0.2)  # centre of [-5, 5]^2
    assert np.all(x.min(axis=0) < -4.9) and np.all(x.max(axis=0) > 4.9)


def test_single_sample():
    x, y = sample_function(get_function("ackley"), 1, seed=0)
    assert x.shape == (1, 2) and y.shape == (1,)


def test_unknown_design():
    with pytest.raises(ValueError, match="grid"):
        sample_function(get_function("peaks"), 10, seed=0, design="grid")


# --- save_samples / load_samples ------------------------------------------------------


def test_csv_round_trip_is_exact(tmp_path):
    x, y = sample_function(get_function("peaks"), 100, seed=0)
    path = tmp_path / "peaks__d0.csv"
    save_samples(x, y, path)
    x_back, y_back = load_samples(path)
    assert np.array_equal(x_back, x) and np.array_equal(y_back, y)


def test_csv_header_and_layout(tmp_path):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([5.0, 6.0])
    path = tmp_path / "s.csv"
    save_samples(x, y, path)
    assert path.read_text().splitlines() == ["x1,x2,y", "1,2,5", "3,4,6"]


def test_csv_with_several_outputs(tmp_path):
    x, y = multi_output_data(20)
    path = tmp_path / "s.csv"
    save_samples(x, y, path)
    assert path.read_text().splitlines()[0] == "x1,x2,x3,y1,y2"
    x_back, y_back = load_samples(path)
    assert x_back.shape == (20, 3) and y_back.shape == (20, 2)
    assert np.array_equal(x_back, x) and np.array_equal(y_back, y)


def test_csv_with_a_single_output_column_returns_1d_y(tmp_path):
    save_samples(np.zeros((4, 2)), np.ones((4, 1)), tmp_path / "s.csv")
    _, y_back = load_samples(tmp_path / "s.csv")
    assert y_back.shape == (4,)


def test_csv_round_trip_extreme_values(tmp_path):
    x = np.array([[1e-300, -1e300], [-0.0, np.pi], [1 / 3, -2 / 7]])
    y = np.array([np.e, -1e-17, 123456789.123456789])
    save_samples(x, y, tmp_path / "s.csv")
    x_back, y_back = load_samples(tmp_path / "s.csv")
    assert np.array_equal(x_back, x) and np.array_equal(y_back, y)


def test_csv_round_trip_more_inputs(tmp_path):
    x = np.random.default_rng(0).normal(size=(10, 3))
    y = x.sum(axis=1)
    save_samples(x, y, tmp_path / "s.csv")
    x_back, y_back = load_samples(tmp_path / "s.csv")
    assert x_back.shape == (10, 3) and y_back.shape == (10,)
    assert np.array_equal(x_back, x) and np.array_equal(y_back, y)


def test_csv_single_row_keeps_2d_shape(tmp_path):
    save_samples(np.array([[0.5, -0.5]]), np.array([1.5]), tmp_path / "s.csv")
    x_back, y_back = load_samples(tmp_path / "s.csv")
    assert x_back.shape == (1, 2) and y_back.shape == (1,)


def test_save_creates_folders_and_overwrites(tmp_path):
    path = tmp_path / "a" / "b" / "s.csv"
    save_samples(np.zeros((3, 2)), np.zeros(3), path)
    save_samples(np.ones((2, 2)), np.ones(2), path)
    x_back, y_back = load_samples(path)
    assert np.array_equal(x_back, np.ones((2, 2))) and np.array_equal(y_back, np.ones(2))


def test_load_rejects_old_files_with_a_split_column(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("x1,x2,y,split\n0.1,0.2,0.3,train\n")
    with pytest.raises(ValueError, match="unexpected column 'split'"):
        load_samples(path)


# --- sample_dataset -------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_sample_dataset_is_a_named_dataset_on_the_domain(name):
    fn = get_function(name)
    data = sample_dataset(fn, 300, seed=2)
    x, y = sample_function(fn, 300, seed=2)
    assert isinstance(data, Dataset) and data.name == name
    assert np.array_equal(data.x, x) and np.array_equal(data.y[:, 0], y)
    assert data.lower_bound.tolist() == list(fn.lower_bound)
    assert data.upper_bound.tolist() == list(fn.upper_bound)
    assert data.input_names == ["x1", "x2"] and data.output_names == ["y"]


def test_sample_dataset_uses_the_design():
    fn = get_function("peaks")
    lhs, uniform = sample_dataset(fn, 100, 0), sample_dataset(fn, 100, 0, design="uniform")
    assert not np.array_equal(lhs.x, uniform.x)
