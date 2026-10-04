import json
import math
from collections.abc import Sized
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from benchmarks import get_function, sample_function
from hybopt.ann.data import DataModule
from hybopt.data import Dataset

SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}
SPLITS = ("train", "val", "test")


def make_dm(n=1000, batch_size=64, random_state=0, name="peaks"):
    fn = get_function(name)
    x, y = sample_function(fn, n, seed=0)
    return DataModule(Dataset(x, y, fn.lower_bound, fn.upper_bound), batch_size, SPLIT, random_state)


def multi_output_data(n=600, seed=0):
    """Three inputs on different scales and two outputs on very different scales."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(300, 400, n), rng.uniform(0, 1e-3, n)])
    y = np.column_stack([np.sin(x[:, 0]) + x[:, 2] * 1e3, 1e4 * x[:, 1] + 5.0])
    return x, y


def n_samples(loader: DataLoader) -> int:
    return len(cast(Sized, loader.dataset))


def stack(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """All (X, Y) of a loader, concatenated in iteration order."""
    xs, ys = zip(*loader)
    return torch.cat(xs), torch.cat(ys)


def denormalize(dm: DataModule, X: torch.Tensor, Y: torch.Tensor):
    norm = {k: np.array(v) for k, v in dm.normalization.items()}
    x_raw = X.double().numpy() * norm["x_scale"] + norm["x_shift"]
    y_raw = Y.double().numpy() * norm["y_scale"] + norm["y_shift"]
    return x_raw, y_raw


# --- DataModule: split ----------------------------------------------------------------


def test_split_sizes_disjoint_and_complete():
    dm = make_dm(n=1000)
    assert [len(dm.idx[s]) for s in SPLITS] == [700, 150, 150]
    all_idx = np.concatenate([dm.idx[s] for s in SPLITS])
    assert sorted(all_idx) == list(range(1000))


@pytest.mark.parametrize("n", [20, 101, 1001, 2999])
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


def test_other_split_fractions_and_tuple_form():
    fn = get_function("peaks")
    x, y = sample_function(fn, 200, seed=0)
    dm = DataModule(Dataset(x, y, fn.lower_bound, fn.upper_bound), 32, (0.5, 0.25, 0.25))
    assert [len(dm.idx[s]) for s in SPLITS] == [100, 50, 50]
    assert dm.split == {"train": 0.5, "val": 0.25, "test": 0.25}


def test_fractions_must_sum_to_one():
    fn = get_function("peaks")
    x, y = sample_function(fn, 100, seed=0)
    with pytest.raises(ValueError, match="must sum to 1"):
        DataModule(Dataset(x, y, fn.lower_bound, fn.upper_bound), 32, {"train": 0.5, "val": 0.3, "test": 0.1})


def test_missing_split_key():
    fn = get_function("peaks")
    x, y = sample_function(fn, 100, seed=0)
    with pytest.raises(ValueError, match="exactly the keys"):
        DataModule(Dataset(x, y, fn.lower_bound, fn.upper_bound), 32, {"train": 0.7, "val": 0.3})


@pytest.mark.parametrize("n", [3, 7, 12])
def test_every_split_needs_at_least_two_samples(n):
    fn = get_function("peaks")
    x, y = sample_function(fn, n, seed=0)
    with pytest.raises(ValueError, match="needs at least 2"):
        DataModule(Dataset(x, y, fn.lower_bound, fn.upper_bound), 32, SPLIT)


@pytest.mark.parametrize("batch_size", [0, -4, 2.5, "64"])
def test_batch_size_must_be_a_positive_integer(batch_size):
    fn = get_function("peaks")
    x, y = sample_function(fn, 100, seed=0)
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        DataModule(Dataset(x, y, fn.lower_bound, fn.upper_bound), batch_size, SPLIT)


def test_mismatched_rows_are_rejected():
    with pytest.raises(ValueError, match="same number of rows"):
        DataModule(Dataset(np.zeros((10, 2)), np.zeros(9)), 4, SPLIT)


# --- DataModule: data and names -------------------------------------------------------


def test_data_is_stored_as_2d_arrays():
    dm = make_dm(n=100)
    assert dm.x.shape == (100, 2) and dm.y.shape == (100, 1)
    assert dm.n_inputs == 2 and dm.n_outputs == 1
    assert dm.input_names == ["x1", "x2"] and dm.output_names == ["y"]


def test_several_outputs_and_names():
    x, y = multi_output_data()
    dm = DataModule(Dataset(pd.DataFrame(x, columns=["a", "T", "c"]), pd.DataFrame(y, columns=["conv", "duty"])), 32, SPLIT)
    assert dm.n_inputs == 3 and dm.n_outputs == 2
    assert dm.input_names == ["a", "T", "c"] and dm.output_names == ["conv", "duty"]


def test_constant_output_is_rejected():
    x = np.random.default_rng(0).uniform(size=(50, 2))
    y = np.column_stack([x[:, 0], np.full(50, 3.0)])
    with pytest.raises(ValueError, match="output 'y2' is constant"):
        DataModule(Dataset(x, y), 8, SPLIT)


# --- DataModule: normalization --------------------------------------------------------


def test_normalization_constants():
    dm = make_dm(name="himmelblau")
    y_train = dm.y[dm.idx["train"]]
    assert dm.normalization["x_shift"] == [0.0, 0.0]  # centre of [-5, 5]^2
    assert dm.normalization["x_scale"] == [5.0, 5.0]  # half-width
    assert dm.normalization["y_shift"] == pytest.approx(y_train.mean(axis=0))
    assert dm.normalization["y_scale"] == pytest.approx(y_train.std(axis=0))  # population std


def test_normalization_for_asymmetric_domain():
    rng = np.random.default_rng(0)
    x = np.column_stack([rng.uniform(0, 2, 40), rng.uniform(10, 30, 40)])
    dm = DataModule(Dataset(x, x.sum(axis=1), (0.0, 10.0), (2.0, 30.0)), 8, SPLIT)
    assert dm.normalization["x_shift"] == [1.0, 20.0]
    assert dm.normalization["x_scale"] == [1.0, 10.0]


def test_default_input_box_is_the_data_range():
    x, y = multi_output_data()
    dm = DataModule(Dataset(x, y), 32, SPLIT)
    assert np.array_equal(dm.lower_bound, x.min(axis=0))
    assert np.array_equal(dm.upper_bound, x.max(axis=0))


def test_normalization_per_output():
    x, y = multi_output_data()
    dm = DataModule(Dataset(x, y), 32, SPLIT)
    y_train = y[dm.idx["train"]]
    assert len(dm.normalization["y_shift"]) == len(dm.normalization["y_scale"]) == 2
    assert dm.normalization["y_shift"] == pytest.approx(y_train.mean(axis=0))
    assert dm.normalization["y_scale"] == pytest.approx(y_train.std(axis=0))


def test_output_normalization_uses_training_split_only():
    dm = make_dm(n=500)
    before = dict(dm.normalization)
    y = dm.y.copy()
    y[dm.idx["val"]] += 1e6
    y[dm.idx["test"]] -= 1e6
    fn = get_function("peaks")
    dm2 = DataModule(Dataset(dm.x, y, fn.lower_bound, fn.upper_bound), 64, SPLIT, random_state=0)
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


def test_loaders_with_several_outputs():
    x, y = multi_output_data()
    dm = DataModule(Dataset(x, y), 32, SPLIT)
    X, Y = stack(dm.labeled_data_loader()[0])
    assert X.shape == (420, 3) and Y.shape == (420, 2)
    assert X.min() >= -1 and X.max() <= 1
    assert torch.allclose(Y.mean(dim=0), torch.zeros(2), atol=1e-4)
    assert torch.allclose(Y.std(dim=0, unbiased=False), torch.ones(2), atol=1e-4)


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
    x_raw, y_raw = denormalize(dm, *stack(dm.labeled_data_loader()[position]))
    assert np.allclose(x_raw, dm.x[dm.idx[split]], atol=1e-5)
    assert np.allclose(y_raw, dm.y[dm.idx[split]], rtol=1e-5, atol=1e-5)


def test_train_loader_holds_exactly_the_training_split():
    dm = make_dm(n=500)
    x_raw, _ = denormalize(dm, *stack(dm.labeled_data_loader()[0]))
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


def test_data_module_needs_a_dataset():
    x, y = multi_output_data()
    with pytest.raises(ValueError, match="must be a hybopt.data.Dataset"):
        DataModule(x, 32, SPLIT)  # type: ignore[arg-type]


def test_data_module_uses_the_shared_split():
    x, y = multi_output_data()
    data = Dataset(x, y)
    dm = DataModule(data, 32, SPLIT, random_state=3)
    expected = data.split(SPLIT, seed=3)
    assert all(np.array_equal(dm.idx[s], expected[s]) for s in SPLITS)
    assert dm.data is data
