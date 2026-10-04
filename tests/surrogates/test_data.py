import json
import math
from collections.abc import Sized
from typing import cast

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from hybopt.surrogates.data import DataModule, load_samples, sample_function, save_samples
from hybopt.surrogates.functions import FUNCTIONS, get_function

SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}
SPLITS = ("train", "val", "test")


def make_dm(n=1000, batch_size=64, random_state=0, name="peaks"):
    fn = get_function(name)
    x, y = sample_function(fn, n, seed=0)
    return DataModule(
        x, y, fn.lower_bound, fn.upper_bound, batch_size, SPLIT, random_state
    )


def n_samples(loader: DataLoader) -> int:
    return len(cast(Sized, loader.dataset))


def stack(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """All (X, Y) of a loader, concatenated in iteration order."""
    xs, ys = zip(*loader)
    return torch.cat(xs), torch.cat(ys)


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


# --- DataModule: split ----------------------------------------------------------------


def test_split_sizes_disjoint_and_complete():
    dm = make_dm(n=1000)
    assert [len(dm.idx[s]) for s in SPLITS] == [700, 150, 150]
    all_idx = np.concatenate([dm.idx[s] for s in SPLITS])
    assert sorted(all_idx) == list(range(1000))


@pytest.mark.parametrize("n", [7, 101, 1001, 2999])
def test_split_sizes_when_fractions_do_not_divide_n(n):
    dm = make_dm(n=n)
    sizes = [len(dm.idx[s]) for s in SPLITS]
    assert sum(sizes) == n
    for size, name in zip(sizes, SPLITS):
        assert abs(size - SPLIT[name] * n) <= 1


def test_split_is_seeded():
    a, b, c = make_dm(random_state=0), make_dm(random_state=0), make_dm(random_state=1)
    for s in SPLITS:
        assert np.array_equal(a.idx[s], b.idx[s])
    assert not np.array_equal(a.idx["test"], c.idx["test"])


def test_split_is_random_not_sequential():
    dm = make_dm(n=1000)
    assert not np.array_equal(dm.idx["train"], np.arange(700))


def test_other_split_fractions():
    fn = get_function("peaks")
    x, y = sample_function(fn, 200, seed=0)
    dm = DataModule(
        x, y, fn.lower_bound, fn.upper_bound, 32, {"train": 0.5, "val": 0.25, "test": 0.25}
    )
    assert [len(dm.idx[s]) for s in SPLITS] == [100, 50, 50]


def test_fractions_must_sum_to_one():
    fn = get_function("peaks")
    x, y = sample_function(fn, 100, seed=0)
    with pytest.raises(ValueError):
        DataModule(
            x, y, fn.lower_bound, fn.upper_bound, 32, {"train": 0.5, "val": 0.3, "test": 0.1}
        )


def test_missing_split_key():
    fn = get_function("peaks")
    x, y = sample_function(fn, 100, seed=0)
    with pytest.raises(KeyError):
        DataModule(x, y, fn.lower_bound, fn.upper_bound, 32, {"train": 0.7, "val": 0.3})


# --- DataModule: normalization --------------------------------------------------------


def test_normalization_constants():
    dm = make_dm(name="himmelblau")
    y_train = dm.y[dm.idx["train"]]
    assert dm.normalization["x_shift"] == [0.0, 0.0]  # centre of [-5, 5]^2
    assert dm.normalization["x_scale"] == [5.0, 5.0]  # half-width
    assert dm.normalization["y_shift"] == pytest.approx([y_train.mean()])
    assert dm.normalization["y_scale"] == pytest.approx([y_train.std()])  # population std


def test_normalization_for_asymmetric_domain():
    x = np.array([[0.0, 10.0], [2.0, 30.0], [1.0, 20.0], [0.5, 15.0]])
    dm = DataModule(x, x.sum(axis=1), (0.0, 10.0), (2.0, 30.0), 2, {"train": 0.5, "val": 0.25, "test": 0.25})
    assert dm.normalization["x_shift"] == [1.0, 20.0]
    assert dm.normalization["x_scale"] == [1.0, 10.0]


def test_output_normalization_uses_training_split_only():
    dm = make_dm(n=500)
    before = dict(dm.normalization)
    y = dm.y.copy()
    y[dm.idx["val"]] += 1e6
    y[dm.idx["test"]] -= 1e6
    fn = get_function("peaks")
    dm2 = DataModule(dm.x, y, fn.lower_bound, fn.upper_bound, 64, SPLIT, random_state=0)
    assert dm2.normalization == before


def test_normalization_is_json_serializable():
    dm = make_dm()
    restored = json.loads(json.dumps(dm.normalization))
    assert restored == dm.normalization
    assert all(isinstance(v, list) for v in restored.values())


def test_bounds_stored_as_float_arrays():
    dm = make_dm()
    assert isinstance(dm.lower_bound, np.ndarray) and dm.lower_bound.dtype == float
    assert np.array_equal(dm.upper_bound, [2.0, 2.0])


# --- DataModule: loaders --------------------------------------------------------------


def test_loader_sizes_shapes_and_dtypes():
    dm = make_dm(n=1000, batch_size=64)
    loaders = dm.labeled_data_loader(shuffle_seed=0)
    assert [n_samples(dl) for dl in loaders] == [700, 150, 150]
    for dl in loaders:
        xb, yb = next(iter(dl))
        assert xb.shape[1] == 2 and yb.shape[1] == 1
        assert xb.dtype == torch.float32 and yb.dtype == torch.float32


def test_batch_counts_and_last_partial_batch():
    dm = make_dm(n=1000, batch_size=64)
    train_loader, val_loader, test_loader = dm.labeled_data_loader()
    assert len(train_loader) == math.ceil(700 / 64)  # last batch kept, not dropped
    sizes = [len(xb) for xb, _ in train_loader]
    assert sizes[:-1] == [64] * (len(sizes) - 1) and sizes[-1] == 700 - 64 * 10
    assert len(val_loader) == len(test_loader) == math.ceil(150 / 64)


def test_batch_larger_than_split_gives_one_batch():
    dm = make_dm(n=100, batch_size=1000)
    for dl in dm.labeled_data_loader():
        assert len(dl) == 1


def test_only_train_loader_is_shuffled():
    train_loader, val_loader, test_loader = make_dm().labeled_data_loader()
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)


@pytest.mark.parametrize("split, position", [("val", 1), ("test", 2)])
def test_val_and_test_loaders_hold_their_split_in_order(split, position):
    dm = make_dm(n=500, name="ackley")
    X, Y = stack(dm.labeled_data_loader()[position])
    norm = {k: np.array(v) for k, v in dm.normalization.items()}
    x_raw = X.double().numpy() * norm["x_scale"] + norm["x_shift"]
    y_raw = Y[:, 0].double().numpy() * norm["y_scale"] + norm["y_shift"]
    assert np.allclose(x_raw, dm.x[dm.idx[split]], atol=1e-5)
    assert np.allclose(y_raw, dm.y[dm.idx[split]], rtol=1e-5, atol=1e-5)


def test_train_loader_holds_exactly_the_training_split():
    dm = make_dm(n=500)
    X, _ = stack(dm.labeled_data_loader()[0])
    norm = {k: np.array(v) for k, v in dm.normalization.items()}
    x_raw = X.double().numpy() * norm["x_scale"] + norm["x_shift"]
    expected = dm.x[dm.idx["train"]]

    def sort_rows(a):  # the loader is shuffled, so compare as sets of rows
        return a[np.lexsort(a.T)]

    assert np.allclose(sort_rows(x_raw), sort_rows(expected), atol=1e-5)


def test_normalized_data_ranges():
    dm = make_dm(name="himmelblau")
    X, Y = stack(dm.labeled_data_loader()[0])
    assert X.min() >= -1 and X.max() <= 1  # inputs scaled from [-5, 5] to [-1, 1]
    assert abs(Y.mean().item()) < 1e-5
    assert abs(Y.std(unbiased=False).item() - 1) < 1e-5


def test_loaders_shuffling():
    dm = make_dm(n=1000, batch_size=64)
    _, _, test_loader = dm.labeled_data_loader(shuffle_seed=0)

    # train batches are shuffled with a seed: a fresh loader repeats the same order,
    # and each new epoch of the same loader gets a new order; val/test keep a fixed order
    def first(loader):
        return next(iter(loader))[0]

    def fresh(seed):
        return dm.labeled_data_loader(shuffle_seed=seed)[0]

    assert torch.equal(first(fresh(0)), first(fresh(0)))
    assert not torch.equal(first(fresh(0)), first(fresh(1)))
    loader = fresh(0)
    assert not torch.equal(first(loader), first(loader))
    assert torch.equal(first(test_loader), first(test_loader))


def test_building_loaders_does_not_change_the_data_module():
    dm = make_dm()
    before = (dm.x.copy(), dm.y.copy(), dict(dm.normalization), {k: v.copy() for k, v in dm.idx.items()})
    dm.labeled_data_loader(shuffle_seed=0)
    dm.labeled_data_loader(shuffle_seed=1)
    assert np.array_equal(dm.x, before[0]) and np.array_equal(dm.y, before[1])
    assert dm.normalization == before[2]
    assert all(np.array_equal(dm.idx[k], before[3][k]) for k in SPLITS)
