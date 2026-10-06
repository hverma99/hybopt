import numpy as np
import pandas as pd
import pytest

from hybopt.data import Dataset
from hybopt.data.dataset import as_arrays, column_names, input_box

SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}


def reactor_frame(n=40, seed=0):
    """Process-style data with named columns, plus a column that is not used for fitting."""
    rng = np.random.default_rng(seed)
    T, tau = rng.uniform(300, 400, n), rng.uniform(1, 100, n)
    conversion = 1 - np.exp(-1e7 * np.exp(-6000 / T) * tau)
    return pd.DataFrame(
        {"run": np.arange(n), "T": T, "tau": tau, "conversion": conversion, "duty": -100 * conversion}
    )


# --- as_arrays / column_names ---------------------------------------------------------


def test_as_arrays_shapes_and_default_names():
    x, y, input_names, output_names = as_arrays(np.zeros((5, 3)), np.arange(5.0))
    assert x.shape == (5, 3) and y.shape == (5, 1)
    assert input_names == ["x1", "x2", "x3"] and output_names == ["y"]


def test_as_arrays_several_outputs():
    _, y, _, output_names = as_arrays(np.zeros((5, 2)), np.ones((5, 3)))
    assert y.shape == (5, 3) and output_names == ["y1", "y2", "y3"]


def test_as_arrays_single_input_as_1d():
    x, _, input_names, _ = as_arrays(np.arange(6.0), np.arange(6.0))
    assert x.shape == (6, 1) and input_names == ["x1"]


def test_as_arrays_accepts_lists_and_ints():
    x, y, _, _ = as_arrays([[1, 2], [3, 4]], [5, 6])
    assert x.dtype == np.float64 and y.dtype == np.float64
    assert np.array_equal(x, [[1.0, 2.0], [3.0, 4.0]]) and np.array_equal(y, [[5.0], [6.0]])


def test_as_arrays_uses_dataframe_and_series_names():
    x = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=["T", "tau"])
    x_out, y_out, input_names, output_names = as_arrays(x, pd.Series([7, 8, 9], name="conversion"))
    assert input_names == ["T", "tau"] and output_names == ["conversion"]
    assert np.array_equal(x_out, [[1, 2], [3, 4], [5, 6]]) and y_out.shape == (3, 1)

    y = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=["conversion", "duty"])
    assert as_arrays(x, y)[3] == ["conversion", "duty"]


def test_as_arrays_copies_the_data():
    x = np.zeros((4, 2))
    x_out, _, _, _ = as_arrays(x, np.arange(4.0))
    x[0, 0] = 99.0
    assert x_out[0, 0] == 0.0


def test_column_names_rules():
    assert column_names(pd.DataFrame([[1, 2]], columns=["a", "b"]), "x", 2) == ["a", "b"]
    assert column_names(pd.Series([1, 2], name="out"), "y", 1) == ["out"]
    assert column_names(pd.Series([1, 2], name=None), "y", 1) == ["y"]
    assert column_names(np.zeros((2, 1)), "x", 1) == ["x1"]
    assert column_names(np.zeros((2, 2)), "y", 2) == ["y1", "y2"]


def test_as_arrays_rejects_mismatched_rows():
    with pytest.raises(ValueError, match="same number of rows, got 5 and 4"):
        as_arrays(np.zeros((5, 2)), np.zeros(4))


@pytest.mark.parametrize("x, y", [(np.zeros((2, 2, 2)), np.zeros(2)), (np.zeros((3, 2)), np.zeros((3, 1, 1)))])
def test_as_arrays_rejects_3d_data(x, y):
    with pytest.raises(ValueError, match="must be 2-D"):
        as_arrays(x, y)


def test_as_arrays_rejects_data_without_columns():
    with pytest.raises(ValueError, match="must be 2-D"):
        as_arrays(np.zeros((3, 0)), np.zeros(3))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("where", ["x", "y"])
def test_as_arrays_rejects_non_finite_values(bad, where):
    x, y = np.zeros((4, 2)), np.zeros(4)
    (x if where == "x" else y).flat[1] = bad
    with pytest.raises(ValueError, match=f"{where} contains NaN or infinite values"):
        as_arrays(x, y)


@pytest.mark.parametrize("x", [[["a", "b"], ["c", "d"]], [[1.0, 2.0], [3.0]]])
def test_as_arrays_rejects_non_numeric_or_ragged_data(x):
    with pytest.raises(ValueError, match="x must be numeric"):
        as_arrays(x, [1.0, 2.0])


def test_as_arrays_rejects_empty_data():
    with pytest.raises(ValueError, match="no samples"):
        as_arrays(np.zeros((0, 2)), np.zeros(0))


def test_as_arrays_rejects_duplicate_column_names():
    with pytest.raises(ValueError, match="unique names"):
        as_arrays(pd.DataFrame([[1, 2], [3, 4]], columns=["T", "T"]), [1.0, 2.0])


# --- input_box ------------------------------------------------------------------------


def test_input_box_defaults_to_the_data_range():
    x = np.array([[0.0, 10.0], [2.0, 30.0], [1.0, 20.0]])
    lower, upper = input_box(x, None, None, ["a", "b"])
    assert np.array_equal(lower, [0.0, 10.0]) and np.array_equal(upper, [2.0, 30.0])


def test_input_box_uses_given_bounds():
    x = np.array([[0.5, 15.0], [1.5, 25.0]])
    lower, upper = input_box(x, (0, 10), [2, 30], ["a", "b"])
    assert lower.dtype == float and np.array_equal(lower, [0.0, 10.0])
    assert np.array_equal(upper, [2.0, 30.0])


def test_input_box_needs_both_bounds():
    with pytest.raises(ValueError, match="both lower_bound and upper_bound"):
        input_box(np.zeros((2, 2)), (0, 0), None, ["a", "b"])


def test_input_box_needs_one_bound_per_input():
    with pytest.raises(ValueError, match=r"one value per input \(2\), got 3 and 2"):
        input_box(np.zeros((2, 2)), (0, 0, 0), (1, 1), ["a", "b"])


def test_input_box_lower_must_be_below_upper():
    with pytest.raises(ValueError, match="input 'b': lower_bound 1 must be below upper_bound 1"):
        input_box(np.zeros((2, 2)) + 1, (0, 1), (2, 1), ["a", "b"])


def test_input_box_rejects_data_outside_the_bounds():
    x = np.array([[0.5, 15.0], [1.5, 35.0]])
    with pytest.raises(ValueError, match=r"input 'b' has values in \[15, 35\], outside its bounds \[10, 30\]"):
        input_box(x, (0, 10), (2, 30), ["a", "b"])


def test_input_box_rejects_a_constant_input_without_bounds():
    x = np.array([[0.0, 5.0], [1.0, 5.0]])
    with pytest.raises(ValueError, match="input 'b' is constant"):
        input_box(x, None, None, ["a", "b"])


def test_input_box_allows_a_constant_input_with_bounds():
    x = np.array([[0.0, 5.0], [1.0, 5.0]])
    lower, upper = input_box(x, (0, 4), (1, 6), ["a", "b"])
    assert np.array_equal(lower, [0, 4]) and np.array_equal(upper, [1, 6])


# --- Dataset --------------------------------------------------------------------------


def test_dataset_from_arrays():
    x = np.random.default_rng(0).uniform(-1, 1, (50, 3))
    data = Dataset(x, x[:, :2] ** 2)
    assert data.n_samples == 50 and data.n_inputs == 3 and data.n_outputs == 2
    assert data.input_names == ["x1", "x2", "x3"] and data.output_names == ["y1", "y2"]
    assert np.array_equal(data.lower_bound, x.min(axis=0)) and data.name == "data"
    assert np.array_equal(data.x, x) and data.y.shape == (50, 2)


def test_dataset_keeps_given_bounds_and_name():
    x = np.random.default_rng(0).uniform(0, 1, (20, 2))
    data = Dataset(x, x.sum(axis=1), lower_bound=[0, 0], upper_bound=[1, 1], name="toy")
    assert data.lower_bound.tolist() == [0, 0] and data.upper_bound.tolist() == [1, 1]
    assert data.name == "toy" and data.output_names == ["y"]


def test_dataset_validates_like_as_arrays():
    with pytest.raises(ValueError, match="same number of rows"):
        Dataset(np.zeros((5, 2)), np.zeros(4))
    with pytest.raises(ValueError, match="outside its bounds"):
        Dataset(np.ones((5, 1)) * 3, np.arange(5.0), lower_bound=[0], upper_bound=[2])


def test_dataset_copies_the_data():
    x = np.random.default_rng(0).uniform(size=(10, 2))
    data = Dataset(x, x[:, 0])
    x[0, 0] = 99.0
    assert data.x[0, 0] != 99.0


def test_inputs_and_outputs_need_different_names():
    frame = pd.DataFrame({"T": [1.0, 2.0, 3.0], "X": [0.1, 0.2, 0.4]})
    with pytest.raises(ValueError, match=r"different names, both have \['T'\]"):
        Dataset(frame[["T"]], frame["T"])


def test_from_frame_selects_columns_in_the_given_order():
    frame = reactor_frame()
    data = Dataset.from_frame(frame, inputs=["tau", "T"], outputs=["duty", "conversion"], name="reactor")
    assert data.input_names == ["tau", "T"] and data.output_names == ["duty", "conversion"]
    assert np.array_equal(data.x, frame[["tau", "T"]].to_numpy())  # the "run" column is ignored
    assert data.name == "reactor"


def test_from_frame_with_bounds():
    data = Dataset.from_frame(reactor_frame(), ["T", "tau"], ["conversion"], [300, 1], [400, 100])
    assert data.lower_bound.tolist() == [300, 1] and data.upper_bound.tolist() == [400, 100]


def test_from_frame_reports_missing_columns():
    with pytest.raises(ValueError, match=r"columns \['pressure'\] are not in the data"):
        Dataset.from_frame(reactor_frame(), ["T", "pressure"], ["conversion"])


def test_from_frame_rejects_a_column_used_twice():
    with pytest.raises(ValueError, match="both inputs and outputs"):
        Dataset.from_frame(reactor_frame(), ["T", "tau"], ["tau"])


def test_from_csv_reads_exactly_and_names_the_dataset_after_the_file(tmp_path):
    frame = reactor_frame()
    path = tmp_path / "plant_runs.csv"
    frame.to_csv(path, index=False, float_format="%.17g")
    data = Dataset.from_csv(path, ["T", "tau"], ["conversion", "duty"])
    assert data.name == "plant_runs"
    assert np.array_equal(data.x, frame[["T", "tau"]].to_numpy())
    assert np.array_equal(data.y, frame[["conversion", "duty"]].to_numpy())
    assert Dataset.from_csv(path, ["T"], ["duty"], name="other").name == "other"


def test_split_matches_split_indices():
    data = Dataset.from_frame(reactor_frame(200), ["T", "tau"], ["conversion"])
    from hybopt.data import split_indices

    expected = split_indices(200, SPLIT, seed=5)
    idx = data.split(SPLIT, seed=5)
    assert all(np.array_equal(idx[s], expected[s]) for s in expected)


def test_subset_keeps_rows_names_and_box():
    data = Dataset.from_frame(reactor_frame(200), ["T", "tau"], ["conversion", "duty"], [300, 1], [400, 100])
    train = data.subset(data.split(SPLIT)["train"])
    assert train.n_samples == 140 and train.input_names == ["T", "tau"]
    assert train.output_names == ["conversion", "duty"] and train.name == data.name
    assert np.array_equal(train.lower_bound, data.lower_bound)
    assert np.array_equal(train.x, data.x[data.split(SPLIT)["train"]])


def test_to_frame():
    data = Dataset.from_frame(reactor_frame(), ["T", "tau"], ["conversion"])
    frame = data.to_frame()
    assert list(frame.columns) == ["T", "tau", "conversion"]
    assert np.array_equal(frame.to_numpy(), np.hstack([data.x, data.y]))


def test_repr():
    data = Dataset.from_frame(reactor_frame(), ["T", "tau"], ["conversion"], name="reactor")
    assert repr(data) == "Dataset('reactor': 40 samples, inputs ['T', 'tau'], outputs ['conversion'])"
