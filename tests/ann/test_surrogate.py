import csv
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import hybopt.ann as ann
from hybopt.ann import ANNSurrogate, TrainSettings, fit_ann
from hybopt.ann.data import DataModule
from hybopt.ann.network import build_model, fold_normalization, load_model
from hybopt.ann.train import train_network
from benchmarks import get_function, sample_dataset, sample_function
from hybopt.data import Dataset

SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}
FAST = TrainSettings(epochs=3)  # for tests about plumbing rather than accuracy


def synthetic(n, seed=0):
    """A non-benchmark function: 3 inputs, 3 outputs on very different scales and offsets."""
    x = np.random.default_rng(seed).uniform(-1, 1, (n, 3))
    y = np.column_stack(
        [
            np.sin(np.pi * x[:, 0]) * x[:, 1],
            50 * (x[:, 0] ** 2 + 0.5 * x[:, 2]),
            np.exp(-x[:, 1] ** 2) * np.cos(2 * x[:, 2]) + 1000,
        ]
    )
    return x, y


def reactor(n, seed=0):
    """Process-style data: conversion and heat duty of a first-order reaction vs T and tau."""
    rng = np.random.default_rng(seed)
    T, tau = rng.uniform(300, 400, n), rng.uniform(1, 100, n)
    conversion = 1 - np.exp(-1e7 * np.exp(-6000 / T) * tau)
    duty = -100 * conversion + 0.5 * (T - 300)
    return np.column_stack([T, tau]), np.column_stack([conversion, duty])


def small_fit(**kwargs):
    x, y = synthetic(300)
    args = dict(split=SPLIT, activation="tanh", depth=2, width=8, settings=FAST)
    return fit_ann(x, y, **{**args, **kwargs})


def relative_errors(surrogate: ANNSurrogate, x, y) -> np.ndarray:
    """RMSE of each output divided by its std, on data the model has not seen."""
    return np.sqrt(np.mean((surrogate.predict(x) - y) ** 2, axis=0)) / y.std(axis=0)


# --- Fitting non-benchmark data (multi-output) ----------------------------------------


@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_fits_a_synthetic_function_with_several_outputs(activation):
    x, y = synthetic(2000)
    s = fit_ann(x, y, split=SPLIT, activation=activation, depth=2, width=32, settings=TrainSettings(epochs=150, patience=150))
    assert s.output_names == ["y1", "y2", "y3"]
    for split in ("train", "val", "test"):
        for name, values in s.metrics[split]["per_output"].items():
            assert values["rel_rmse"] < 0.1, (split, name, values)
    x_new, y_new = synthetic(500, seed=1)  # generalizes to fresh points
    assert np.all(relative_errors(s, x_new, y_new) < 0.1)


@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_fits_process_data_with_named_columns(activation):
    x, y = reactor(2000)
    s = fit_ann(pd.DataFrame(x, columns=["T", "tau"]), pd.DataFrame(y, columns=["conversion", "duty"]), split=(0.7, 0.15, 0.15), activation=activation, depth=2, width=32, settings=TrainSettings(epochs=150, patience=150))
    assert s.input_names == ["T", "tau"] and s.output_names == ["conversion", "duty"]
    assert list(s.metrics["test"]["per_output"]) == ["conversion", "duty"]
    assert s.metrics["test"]["rel_rmse"] < 0.1
    x_new, y_new = reactor(500, seed=1)
    assert np.all(relative_errors(s, pd.DataFrame(x_new, columns=["T", "tau"]), y_new) < 0.1)


def test_fits_a_function_of_a_single_input():
    x = np.linspace(-2, 2, 400)
    settings = TrainSettings(batch_size=16, epochs=300)  # 280 training points: use small batches
    s = fit_ann(x, np.tanh(3 * x), split=SPLIT, activation="tanh", depth=1, width=16, settings=settings)
    assert s.input_names == ["x1"] and s.output_names == ["y"]
    assert s.predict(np.array([0.5, -0.5])).shape == (2, 1)
    assert s.metrics["test"]["rel_rmse"] < 0.1


# --- Benchmark functions are just a data source ---------------------------------------


def test_benchmark_function_through_the_general_api():
    fn = get_function("peaks")
    x, y = sample_function(fn, 2000, seed=0)
    s = fit_ann(x, y, split=SPLIT, activation="relu", depth=2, width=20, lower_bound=fn.lower_bound, upper_bound=fn.upper_bound, settings=TrainSettings(epochs=100))
    assert s.lower_bound == [-2.0, -2.0] and s.upper_bound == [2.0, 2.0]
    assert s.metrics["test"]["rel_rmse"] < 0.2
    x_new, y_new = sample_function(fn, 500, seed=1)
    assert relative_errors(s, x_new, y_new[:, None])[0] < 0.2


def test_fit_ann_is_exactly_the_building_blocks():
    # no separate code path: the entry point composes DataModule, build_model,
    # train_network and fold_normalization, giving bit-identical weights
    fn = get_function("himmelblau")
    x, y = sample_function(fn, 500, seed=0)
    settings = TrainSettings(epochs=5)
    s = fit_ann(x, y, split=SPLIT, activation="tanh", depth=2, width=6, lower_bound=fn.lower_bound, upper_bound=fn.upper_bound, seed=3, split_seed=4, settings=settings)

    dm = DataModule(Dataset(x, y, fn.lower_bound, fn.upper_bound), settings.batch_size, SPLIT, random_state=4)
    model = build_model(2, [6, 6], 1, "tanh", seed=3)
    train_network(model, *dm.labeled_data_loader(shuffle_seed=3)[:2], settings)
    manual = fold_normalization(model, dm.normalization)
    for p, q in zip(s.model.parameters(), manual.parameters()):
        assert torch.equal(p, q)


# --- Dataset input --------------------------------------------------------------------


def test_fit_ann_accepts_a_dataset():
    x, y = reactor(400)
    frame = pd.DataFrame(np.column_stack([x, y]), columns=["T", "tau", "conversion", "duty"])
    data = Dataset.from_frame(frame, ["T", "tau"], ["conversion", "duty"], [300, 1], [400, 100], name="reactor")
    s = fit_ann(data, split=SPLIT, activation="tanh", depth=1, width=6, settings=FAST)
    assert s.input_names == ["T", "tau"] and s.output_names == ["conversion", "duty"]
    assert s.lower_bound == [300.0, 1.0] and s.upper_bound == [400.0, 100.0]
    assert s.training["data"] == "reactor"


def test_dataset_and_arrays_give_the_same_model():
    fn = get_function("peaks")
    x, y = sample_function(fn, 400, seed=0)
    args = dict(split=SPLIT, activation="relu", depth=2, width=6, seed=1, split_seed=2, settings=FAST)
    from_arrays = fit_ann(x, y, lower_bound=fn.lower_bound, upper_bound=fn.upper_bound, **args)
    from_dataset = fit_ann(sample_dataset(fn, 400, seed=0), **args)
    for p, q in zip(from_arrays.model.parameters(), from_dataset.model.parameters()):
        assert torch.equal(p, q)
    assert from_arrays.metrics == from_dataset.metrics
    assert from_arrays.training["data"] == "data" and from_dataset.training["data"] == "peaks"


def test_dataset_input_rejects_extra_arguments():
    data = Dataset(*synthetic(300))
    with pytest.raises(ValueError, match="leave out y, lower_bound and upper_bound"):
        fit_ann(data, data.y, split=SPLIT, activation="relu", depth=1, width=4)
    with pytest.raises(ValueError, match="leave out y, lower_bound and upper_bound"):
        fit_ann(data, split=SPLIT, activation="relu", depth=1, width=4, lower_bound=[0, 0, 0], upper_bound=[1, 1, 1])


def test_arrays_need_outputs():
    x, _ = synthetic(300)
    with pytest.raises(ValueError, match="give the outputs y"):
        fit_ann(x, split=SPLIT, activation="relu", depth=1, width=4)


def test_model_settings_are_keyword_only():
    x, y = synthetic(300)
    with pytest.raises(TypeError):
        fit_ann(x, y, SPLIT, "relu", 1, 4)  # type: ignore[misc]


# --- The ANNSurrogate result -------------------------------------------------------------


def test_result_fields():
    x, y = synthetic(300)
    s = fit_ann(x, y, split=SPLIT, activation="relu", depth=3, width=7, seed=5, split_seed=6, settings=FAST)
    dm = DataModule(Dataset(x, y), FAST.batch_size, SPLIT, random_state=6)
    assert s.activation == "relu"
    assert s.input_names == ["x1", "x2", "x3"] and s.output_names == ["y1", "y2", "y3"]
    assert s.normalization == dm.normalization
    assert s.lower_bound == x.min(axis=0).tolist() and s.upper_bound == x.max(axis=0).tolist()
    assert s.training["depth"] == 3 and s.training["width"] == 7
    assert s.training["split"] == SPLIT
    assert s.training["seed"] == 5 and s.training["split_seed"] == 6
    assert s.training["settings"] == {"lr": 3e-3, "batch_size": 64, "epochs": 3, "patience": 100}
    assert s.training["n_samples"] == {"train": 210, "val": 45, "test": 45}
    assert s.training["epochs_run"] == len(s.history) == 3
    assert s.training["best_epoch"] == min(s.history, key=lambda h: h["val_mse"])["epoch"]
    assert s.training["train_seconds"] >= 0


@pytest.mark.parametrize("activation, module", [("relu", nn.ReLU), ("tanh", nn.Tanh)])
@pytest.mark.parametrize("depth, width", [(1, 4), (2, 10), (4, 3)])
def test_model_architecture_follows_depth_and_width(activation, module, depth, width):
    s = small_fit(activation=activation, depth=depth, width=width)
    linears = [m for m in s.model if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == (
        [(3, width)] + [(width, width)] * (depth - 1) + [(width, 3)]
    )
    assert sum(isinstance(m, module) for m in s.model) == depth
    assert all(p.dtype == torch.float64 for p in s.model.parameters())


def test_metrics_report_every_split_and_output():
    s = small_fit()
    assert set(s.metrics) == {"train", "val", "test"}
    for values in s.metrics.values():
        assert {"rmse", "max_abs_err", "rel_rmse", "per_output"} <= set(values)
        assert list(values["per_output"]) == ["y1", "y2", "y3"]


def test_single_output_from_1d_y():
    x, y = synthetic(300)
    s = fit_ann(x, y[:, 0], split=SPLIT, activation="relu", depth=1, width=8, settings=FAST)
    assert s.output_names == ["y"]
    assert s.predict(x[:4]).shape == (4, 1)
    test = s.metrics["test"]
    assert test["per_output"]["y"] == {k: test[k] for k in ("rmse", "max_abs_err", "rel_rmse")}


def test_default_input_box_is_the_data_range_and_given_box_is_kept():
    x, y = synthetic(300)
    s = small_fit()
    assert s.lower_bound == x.min(axis=0).tolist() and s.upper_bound == x.max(axis=0).tolist()
    s = small_fit(lower_bound=[-1, -1, -1], upper_bound=[1, 1, 1])
    assert s.lower_bound == [-1.0, -1.0, -1.0] and s.upper_bound == [1.0, 1.0, 1.0]
    assert s.normalization["x_shift"] == [0.0, 0.0, 0.0]


# --- predict --------------------------------------------------------------------------


def test_predict_matches_the_model():
    s = small_fit()
    x, _ = synthetic(20, seed=2)
    with torch.no_grad():
        expected = s.model(torch.tensor(x, dtype=torch.float64)).numpy()
    assert s.predict(x).shape == (20, 3)
    assert np.array_equal(s.predict(x), expected)
    assert np.array_equal(s.predict(x[0]), expected[:1])  # one sample as a 1-D row
    assert np.array_equal(s.predict(x.tolist()), expected)


@pytest.mark.parametrize("x", [np.zeros((5, 2)), np.zeros((5, 4)), np.zeros(2), np.zeros((2, 3, 1))])
def test_predict_rejects_the_wrong_number_of_inputs(x):
    with pytest.raises(ValueError, match=r"x must have 3 columns \['x1', 'x2', 'x3'\]"):
        small_fit().predict(x)


# --- Reproducibility ------------------------------------------------------------------


def test_same_seeds_give_identical_surrogates():
    a, b = small_fit(seed=1, split_seed=2), small_fit(seed=1, split_seed=2)
    for p, q in zip(a.model.parameters(), b.model.parameters()):
        assert torch.equal(p, q)
    assert a.metrics == b.metrics and a.history == b.history and a.normalization == b.normalization


def test_seed_changes_initialization_and_shuffling():
    a, b = small_fit(seed=1), small_fit(seed=2)
    assert not torch.equal(a.model[0].weight, b.model[0].weight)
    assert a.normalization == b.normalization  # same split


def test_split_seed_changes_the_split():
    a, b = small_fit(split_seed=1), small_fit(split_seed=2)
    assert a.normalization["y_shift"] != b.normalization["y_shift"]  # different training rows
    assert a.training["n_samples"] == b.training["n_samples"]


# --- Input validation (fails before any training) -------------------------------------


@pytest.mark.parametrize(
    "split, message",
    [
        ({"train": 0.6, "val": 0.15, "test": 0.15}, "must sum to 1"),
        ((0.8, 0.1, 0.2), "must sum to 1"),
        ({"train": 0.7, "val": 0.3}, "exactly the keys"),
        ((0.7, 0.3), "split must be"),
        ((1.0, 0.0, 0.0), "between 0 and 1"),
        ((0.9, -0.05, 0.15), "between 0 and 1"),
    ],
)
def test_rejects_bad_split(split, message):
    with pytest.raises(ValueError, match=message):
        small_fit(split=split)


@pytest.mark.parametrize("activation", ["sigmoid", "ReLU", "", None])
def test_rejects_bad_activation(activation):
    with pytest.raises(ValueError, match=r"activation must be one of \['relu', 'tanh'\]"):
        small_fit(activation=activation)


@pytest.mark.parametrize("field", ["depth", "width"])
@pytest.mark.parametrize("value", [0, -2, 2.5, "3", True, None])
def test_rejects_non_positive_integer_depth_or_width(field, value):
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        small_fit(**{field: value})


def test_rejects_mismatched_rows():
    x, y = synthetic(300)
    with pytest.raises(ValueError, match="same number of rows, got 300 and 299"):
        fit_ann(x, y[:-1], split=SPLIT, activation="relu", depth=1, width=4)


@pytest.mark.parametrize("where", ["x", "y"])
def test_rejects_nan(where):
    x, y = synthetic(300)
    (x if where == "x" else y)[7, 1] = np.nan
    with pytest.raises(ValueError, match=f"{where} contains NaN"):
        fit_ann(x, y, split=SPLIT, activation="relu", depth=1, width=4)


@pytest.mark.parametrize(
    "bounds, message",
    [
        (dict(lower_bound=[-1, -1, -1]), "both lower_bound and upper_bound"),
        (dict(lower_bound=[-1, -1], upper_bound=[1, 1]), "one value per input"),
        (dict(lower_bound=[-0.5, -1, -1], upper_bound=[1, 1, 1]), "outside its bounds"),
        (dict(lower_bound=[1, -1, -1], upper_bound=[-1, 1, 1]), "must be below"),
    ],
)
def test_rejects_bad_bounds(bounds, message):
    with pytest.raises(ValueError, match=message):
        small_fit(**bounds)


def test_rejects_constant_input_and_output():
    x, y = synthetic(300)
    x_const = x.copy()
    x_const[:, 2] = 4.0
    with pytest.raises(ValueError, match="input 'x3' is constant"):
        fit_ann(x_const, y, split=SPLIT, activation="relu", depth=1, width=4)
    y_const = y.copy()
    y_const[:, 1] = 7.0
    with pytest.raises(ValueError, match="output 'y2' is constant"):
        fit_ann(x, y_const, split=SPLIT, activation="relu", depth=1, width=4)


def test_rejects_too_little_data():
    x, y = synthetic(8)
    with pytest.raises(ValueError, match="needs at least 2"):
        fit_ann(x, y, split=SPLIT, activation="relu", depth=1, width=4)


def test_rejects_bad_settings():
    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        small_fit(settings=TrainSettings(epochs=0))


# --- Saving and loading ---------------------------------------------------------------


def test_save_to_writes_model_meta_and_history(tmp_path):
    folder = tmp_path / "models" / "synthetic__tanh__h2x8"
    s = small_fit(save_to=folder, meta={"config": {"source": "test"}})
    assert sorted(p.name for p in folder.iterdir()) == ["history.csv", "meta.json", "model.pt"]

    meta = json.loads((folder / "meta.json").read_text())
    assert meta["activation"] == "tanh" and meta["layer_sizes"] == [3, 8, 8, 3]
    assert meta["input_names"] == ["x1", "x2", "x3"] and meta["output_names"] == ["y1", "y2", "y3"]
    assert meta["domain"] == {"lower_bound": s.lower_bound, "upper_bound": s.upper_bound}
    assert meta["normalization"] == s.normalization  # input/output scaling stored with the model
    assert meta["metrics"] == s.metrics and meta["training"] == s.training
    assert meta["config"] == {"source": "test"}

    with open(folder / "history.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["epoch"]) for r in rows] == [h["epoch"] for h in s.history]


def test_load_round_trip(tmp_path):
    s = small_fit(save_to=tmp_path / "m")
    loaded = ANNSurrogate.load(tmp_path / "m")
    x, _ = synthetic(50, seed=3)
    assert np.array_equal(loaded.predict(x), s.predict(x))
    for field in ("activation", "input_names", "output_names", "lower_bound", "upper_bound",
                  "normalization", "metrics", "history", "training"):
        assert getattr(loaded, field) == getattr(s, field), field
    assert not loaded.model.training


def test_saved_model_loads_for_embedding(tmp_path):
    # the optimization side reads model.pt + meta.json with load_model; it must give the
    # surrogate's raw-unit outputs and the input box
    s = small_fit(activation="relu", save_to=tmp_path / "m")
    model, meta = load_model(tmp_path / "m")
    x, _ = synthetic(30, seed=4)
    with torch.no_grad():
        assert np.array_equal(model(torch.tensor(x, dtype=torch.float64)).numpy(), s.predict(x))
    assert meta["domain"]["lower_bound"] == s.lower_bound


def test_refuses_to_overwrite_a_saved_model(tmp_path):
    small_fit(save_to=tmp_path / "m", seed=0)
    before = (tmp_path / "m" / "model.pt").read_bytes()
    with pytest.raises(FileExistsError, match="already holds a model"):
        small_fit(save_to=tmp_path / "m", seed=1)
    with pytest.raises(FileExistsError, match="already holds a model"):
        small_fit(seed=1).save(tmp_path / "m")
    assert (tmp_path / "m" / "model.pt").read_bytes() == before


def test_overwrite_replaces_a_saved_model(tmp_path):
    small_fit(save_to=tmp_path / "m", seed=0)
    s = small_fit(save_to=tmp_path / "m", seed=1, overwrite=True)
    assert np.array_equal(ANNSurrogate.load(tmp_path / "m").predict(np.zeros((1, 3))), s.predict(np.zeros((1, 3))))


def test_extra_meta_cannot_override_model_information(tmp_path):
    s = small_fit(save_to=tmp_path / "m", meta={"activation": "relu", "domain": None, "note": 1})
    meta = json.loads((tmp_path / "m" / "meta.json").read_text())
    assert meta["activation"] == "tanh" and meta["domain"]["lower_bound"] == s.lower_bound
    assert meta["note"] == 1


def test_save_and_load_with_names(tmp_path):
    x, y = reactor(300)
    fit_ann(pd.DataFrame(x, columns=["T", "tau"]), pd.DataFrame(y, columns=["conversion", "duty"]), split=SPLIT, activation="relu", depth=1, width=4, settings=FAST, save_to=tmp_path / "m")
    loaded = ANNSurrogate.load(tmp_path / "m")
    assert loaded.input_names == ["T", "tau"] and loaded.output_names == ["conversion", "duty"]


# --- pandas ---------------------------------------------------------------------------


def test_dataframe_and_series_inputs():
    x, y = reactor(300)
    s = fit_ann(pd.DataFrame(x, columns=["T", "tau"]), pd.Series(y[:, 0], name="conversion"), split=SPLIT, activation="tanh", depth=1, width=4, settings=FAST)
    assert s.input_names == ["T", "tau"] and s.output_names == ["conversion"]


def test_predict_with_a_dataframe_returns_a_dataframe():
    x, y = reactor(300)
    s = fit_ann(pd.DataFrame(x, columns=["T", "tau"]), pd.DataFrame(y, columns=["conversion", "duty"]), split=SPLIT, activation="relu", depth=1, width=4, settings=FAST)
    new = pd.DataFrame(x[:5], columns=["T", "tau"], index=list("abcde"))
    result = s.predict(new)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["conversion", "duty"] and list(result.index) == list("abcde")
    assert np.array_equal(result.to_numpy(), s.predict(x[:5]))


def test_predict_matches_dataframe_columns_by_name():
    x, y = reactor(300)
    s = fit_ann(pd.DataFrame(x, columns=["T", "tau"]), y, split=SPLIT, activation="relu", depth=1, width=4, settings=FAST)
    expected = s.predict(x[:5])
    reordered = pd.DataFrame({"tau": x[:5, 1], "pressure": 1.0, "T": x[:5, 0]})  # extra column ignored
    assert np.array_equal(s.predict(reordered).to_numpy(), expected)


def test_predict_accepts_array_views():
    s = small_fit()
    x, _ = synthetic(10, seed=2)
    assert x[::-1].strides[0] < 0  # a view with negative strides
    assert np.array_equal(s.predict(x[::-1])[::-1], s.predict(x))
    assert np.array_equal(s.predict(x[:, [2, 1, 0]][:, [2, 1, 0]]), s.predict(x))


def test_predict_names_missing_dataframe_columns():
    x, y = reactor(300)
    s = fit_ann(pd.DataFrame(x, columns=["T", "tau"]), y, split=SPLIT, activation="relu", depth=1, width=4, settings=FAST)
    with pytest.raises(ValueError, match=r"missing the input columns \['tau'\]"):
        s.predict(pd.DataFrame({"T": x[:5, 0]}))


def test_dataframe_with_missing_values_is_rejected():
    x, y = reactor(300)
    frame = pd.DataFrame(x, columns=["T", "tau"])
    frame.loc[3, "tau"] = None
    with pytest.raises(ValueError, match="x contains NaN"):
        fit_ann(frame, y, split=SPLIT, activation="relu", depth=1, width=4, settings=FAST)


def test_dataframe_with_text_is_rejected():
    x, y = reactor(300)
    frame = pd.DataFrame(x, columns=["T", "tau"]).assign(unit="K")
    with pytest.raises(ValueError, match="x must be numeric"):
        fit_ann(frame, y, split=SPLIT, activation="relu", depth=1, width=4, settings=FAST)


# --- Package interface ----------------------------------------------------------------


def test_package_exports_the_general_api():
    assert set(ann.__all__) == {"ANNSurrogate", "TrainSettings", "fit_ann"}
