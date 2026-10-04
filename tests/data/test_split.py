import numpy as np
import pytest

from hybopt.data import SPLITS, split_fractions, split_indices


# --- split_fractions ------------------------------------------------------------------


def test_split_keys():
    assert SPLITS == ("train", "val", "test")


@pytest.mark.parametrize(
    "split",
    [
        {"train": 0.7, "val": 0.15, "test": 0.15},
        {"test": 0.15, "train": 0.7, "val": 0.15},  # key order does not matter
        (0.7, 0.15, 0.15),
        [0.7, 0.15, 0.15],
    ],
)
def test_split_fractions_accepts_dicts_and_sequences(split):
    assert split_fractions(split) == {"train": 0.7, "val": 0.15, "test": 0.15}


def test_split_fractions_returns_floats_in_fixed_order():
    result = split_fractions((np.float32(0.5), 0.25, 0.25))
    assert list(result) == list(SPLITS)
    assert all(type(v) is float for v in result.values())


def test_split_fractions_tolerates_rounding():
    # 0.7 + 0.2 + 0.1 is 0.9999999999999999 in floating point
    assert sum(split_fractions((0.7, 0.2, 0.1)).values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "split",
    [
        {"train": 0.5, "val": 0.3, "test": 0.1},
        (0.7, 0.2, 0.2),
        (0.33, 0.33, 0.33),
    ],
)
def test_split_fractions_must_sum_to_one(split):
    with pytest.raises(ValueError, match="must sum to 1"):
        split_fractions(split)


@pytest.mark.parametrize(
    "split",
    [
        {"train": 0.7, "val": 0.3},
        {"train": 0.7, "val": 0.15, "test": 0.15, "extra": 0.0},
        {"training": 0.7, "val": 0.15, "test": 0.15},
    ],
)
def test_split_fractions_needs_exactly_the_three_keys(split):
    with pytest.raises(ValueError, match="exactly the keys"):
        split_fractions(split)


@pytest.mark.parametrize("split", [(0.7, 0.3), (0.5, 0.2, 0.2, 0.1), 0.7, "0.7,0.15,0.15", None])
def test_split_fractions_rejects_other_shapes(split):
    with pytest.raises(ValueError, match="split must be"):
        split_fractions(split)


@pytest.mark.parametrize(
    "split",
    [(1.0, 0.0, 0.0), (0.0, 0.5, 0.5), (1.2, -0.1, -0.1), (0.7, "0.15", 0.15), (0.7, None, 0.3), (True, 0.0, 0.0)],
)
def test_split_fractions_must_be_strictly_between_zero_and_one(split):
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_fractions(split)


# --- split_indices --------------------------------------------------------------------


def test_split_indices_sizes_disjoint_and_complete():
    idx = split_indices(1000, (0.7, 0.15, 0.15), seed=0)
    assert list(idx) == list(SPLITS)
    assert [len(idx[s]) for s in SPLITS] == [700, 150, 150]
    assert sorted(np.concatenate(list(idx.values()))) == list(range(1000))
    assert all(rows.dtype.kind == "i" for rows in idx.values())


@pytest.mark.parametrize("n", [20, 101, 1001, 2999])
def test_split_indices_when_fractions_do_not_divide_n(n):
    idx = split_indices(n, {"train": 0.7, "val": 0.15, "test": 0.15})
    sizes = [len(idx[s]) for s in SPLITS]
    assert sum(sizes) == n
    assert all(abs(size - f * n) <= 1 for size, f in zip(sizes, (0.7, 0.15, 0.15)))


def test_split_indices_is_seeded():
    a, b, c = (split_indices(500, (0.7, 0.15, 0.15), seed=s) for s in (3, 3, 4))
    assert all(np.array_equal(a[s], b[s]) for s in SPLITS)
    assert not np.array_equal(a["test"], c["test"])


def test_split_indices_are_unchanged():
    # pinned: models already trained must keep seeing the same split when training resumes
    idx = split_indices(20, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    assert idx["train"].tolist() == [4, 5, 13, 19, 7, 14, 3, 6, 9, 17, 11, 2, 16, 18]
    assert idx["val"].tolist() == [10, 12, 15] and idx["test"].tolist() == [8, 1, 0]


@pytest.mark.parametrize("n", [3, 7, 12])
def test_split_indices_needs_two_rows_per_split(n):
    with pytest.raises(ValueError, match="needs at least 2"):
        split_indices(n, (0.7, 0.15, 0.15))


def test_split_indices_checks_the_fractions():
    with pytest.raises(ValueError, match="must sum to 1"):
        split_indices(100, (0.7, 0.2, 0.2))
