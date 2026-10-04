import copy
import json

import pytest
import torch
from torch import nn

from hybopt.surrogates.network import (
    ACTIVATIONS,
    build_model,
    fold_normalization,
    load_model,
    save_model,
)

NORM = {"x_shift": [1.0, -2.0], "x_scale": [3.0, 0.5], "y_shift": [4.0], "y_scale": [2.5]}


def linears(model: nn.Sequential) -> list[nn.Linear]:
    return [m for m in model if isinstance(m, nn.Linear)]


def unfolded(model: nn.Sequential, x: torch.Tensor, norm: dict) -> torch.Tensor:
    """What a folded model must compute: normalize x, run the model, map y back."""
    t = {k: torch.tensor(v, dtype=torch.float64) for k, v in norm.items()}
    return model((x - t["x_shift"]) / t["x_scale"]) * t["y_scale"] + t["y_shift"]


# --- build_model ----------------------------------------------------------------------


def test_activations_registry():
    assert ACTIVATIONS == {"relu": nn.ReLU, "tanh": nn.Tanh}


@pytest.mark.parametrize("activation, module", [("relu", nn.ReLU), ("tanh", nn.Tanh)])
def test_build_model_architecture(activation, module):
    model = build_model(2, [5, 4], 1, activation, seed=0)
    assert [type(m) for m in model] == [nn.Linear, module, nn.Linear, module, nn.Linear]
    assert [m.out_features for m in linears(model)] == [5, 4, 1]


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@pytest.mark.parametrize("width", [10, 50])
def test_build_model_depth_and_width(depth, width):
    model = build_model(2, [width] * depth, 1, "relu", seed=0)
    layers = linears(model)
    assert len(layers) == depth + 1
    assert [(m.in_features, m.out_features) for m in layers] == (
        [(2, width)] + [(width, width)] * (depth - 1) + [(width, 1)]
    )
    assert len(model) == 2 * depth + 1  # an activation after every hidden layer only


def test_build_model_uneven_widths_and_sizes():
    model = build_model(3, [7, 2, 5], 4, "tanh", seed=0)
    assert [(m.in_features, m.out_features) for m in linears(model)] == [
        (3, 7),
        (7, 2),
        (2, 5),
        (5, 4),
    ]
    assert model(torch.zeros(6, 3)).shape == (6, 4)


def test_build_model_parameter_count():
    model = build_model(2, [10, 10], 1, "relu", seed=0)
    expected = (2 + 1) * 10 + (10 + 1) * 10 + (10 + 1) * 1
    assert sum(p.numel() for p in model.parameters()) == expected


def test_build_model_without_hidden_layers_is_linear():
    model = build_model(2, [], 1, "relu", seed=0)
    assert [type(m) for m in model] == [nn.Linear]


def test_build_model_is_float32_and_trainable():
    model = build_model(2, [8], 1, "relu", seed=0)
    assert all(p.dtype == torch.float32 and p.requires_grad for p in model.parameters())


def test_build_model_is_seeded():
    a, b, c = (build_model(2, [8, 8], 1, "relu", seed=s) for s in (1, 1, 2))
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.equal(p, q)
    assert not torch.equal(a[0].weight, c[0].weight)


def test_unknown_activation():
    with pytest.raises(KeyError):
        build_model(2, [4], 1, "sigmoid", seed=0)


@pytest.mark.parametrize("activation, fn", [("relu", torch.relu), ("tanh", torch.tanh)])
def test_build_model_applies_the_activation(activation, fn):
    # the hidden layer computes activation(W x + b); the output layer adds no activation
    model = build_model(2, [16], 1, activation, seed=0)
    first, last = linears(model)
    x = torch.randn(100, 2)
    with torch.no_grad():
        assert torch.equal(model[:-1](x), fn(first(x)))
        assert torch.equal(model(x), last(fn(first(x))))


# --- fold_normalization ---------------------------------------------------------------


@pytest.mark.parametrize("activation", ["relu", "tanh"])
@pytest.mark.parametrize("widths", [[6], [6, 5], [8, 8, 8, 8]])
def test_fold_normalization_preserves_function(activation, widths):
    model = build_model(2, widths, 1, activation, seed=0).double()
    raw_model = fold_normalization(model, NORM)
    x = torch.rand(50, 2, dtype=torch.float64) * 10 - 5
    with torch.no_grad():
        assert torch.allclose(raw_model(x), unfolded(model, x, NORM), atol=1e-12)


def test_fold_normalization_with_several_outputs():
    model = build_model(3, [5], 2, "tanh", seed=0).double()
    norm = {
        "x_shift": [0.5, -1.0, 2.0],
        "x_scale": [2.0, 0.1, 4.0],
        "y_shift": [1.0, -3.0],
        "y_scale": [3.0, 0.2],
    }
    raw_model = fold_normalization(model, norm)
    x = torch.rand(20, 3, dtype=torch.float64)
    with torch.no_grad():
        assert torch.allclose(raw_model(x), unfolded(model, x, norm), atol=1e-12)


def test_fold_normalization_accepts_a_float32_model():
    model = build_model(2, [6], 1, "relu", seed=0)  # float32, as after training
    raw_model = fold_normalization(model, NORM)
    assert all(p.dtype == torch.float64 for p in raw_model.parameters())
    x = torch.rand(20, 2, dtype=torch.float64)
    with torch.no_grad():
        assert torch.allclose(raw_model(x), unfolded(model.double(), x, NORM), atol=1e-12)


def test_fold_normalization_leaves_the_model_unchanged():
    model = build_model(2, [6, 5], 1, "relu", seed=0)
    before = copy.deepcopy(model.state_dict())
    raw_model = fold_normalization(model, NORM)
    assert raw_model is not model
    for name, value in model.state_dict().items():
        assert value.dtype == torch.float32
        assert torch.equal(value, before[name])


def test_fold_normalization_changes_only_first_and_last_layers():
    model = build_model(2, [6, 5, 4], 1, "tanh", seed=0).double()
    raw_model = fold_normalization(model, NORM)
    old, new = linears(model), linears(raw_model)
    assert not torch.equal(old[0].weight, new[0].weight)
    assert not torch.equal(old[-1].weight, new[-1].weight)
    for a, b in zip(old[1:-1], new[1:-1]):
        assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias)


def test_identity_normalization_changes_nothing():
    model = build_model(2, [6], 1, "relu", seed=0).double()
    identity = {"x_shift": [0.0, 0.0], "x_scale": [1.0, 1.0], "y_shift": [0.0], "y_scale": [1.0]}
    raw_model = fold_normalization(model, identity)
    for p, q in zip(model.parameters(), raw_model.parameters()):
        assert torch.equal(p, q)


def test_fold_normalization_formula():
    # first layer: W / x_scale and b - W @ (x_shift / x_scale); last: y_scale * W, y_scale * b + y_shift
    model = build_model(2, [3], 1, "relu", seed=0).double()
    raw_model = fold_normalization(model, NORM)
    t = {k: torch.tensor(v, dtype=torch.float64) for k, v in NORM.items()}
    (w0, b0), (w1, b1) = [(m.weight, m.bias) for m in linears(model)]
    (v0, c0), (v1, c1) = [(m.weight, m.bias) for m in linears(raw_model)]
    assert torch.allclose(v0, w0 / t["x_scale"])
    assert torch.allclose(c0, b0 - w0 @ (t["x_shift"] / t["x_scale"]))
    assert torch.allclose(v1, t["y_scale"] * w1)
    assert torch.allclose(c1, t["y_scale"] * b1 + t["y_shift"])


# --- save_model / load_model ----------------------------------------------------------


@pytest.mark.parametrize("activation", ["relu", "tanh"])
@pytest.mark.parametrize("widths", [[4], [4, 3], [10, 10, 10, 10]])
def test_save_load_round_trip(tmp_path, activation, widths):
    model = build_model(2, widths, 1, activation, seed=0).double()
    save_model(model, activation, tmp_path / "m", meta={"note": "x"})
    loaded, meta = load_model(tmp_path / "m")
    assert meta["layer_sizes"] == [2, *widths, 1] and meta["activation"] == activation
    assert meta["note"] == "x"
    assert [type(m) for m in loaded] == [type(m) for m in model]
    x = torch.rand(20, 2, dtype=torch.float64)
    with torch.no_grad():
        assert torch.equal(loaded(x), model(x))


def test_save_writes_model_and_meta_files(tmp_path):
    folder = tmp_path / "nested" / "peaks__relu__h1x4__t0"
    save_model(build_model(2, [4], 1, "relu", seed=0), "relu", folder, meta={})
    assert sorted(p.name for p in folder.iterdir()) == ["meta.json", "model.pt"]
    meta = json.loads((folder / "meta.json").read_text())
    assert meta == {"activation": "relu", "layer_sizes": [2, 4, 1]}


def test_save_keeps_nested_meta(tmp_path):
    meta = {"config": {"seed": 3, "hidden_widths": [4]}, "metrics": {"test": {"rmse": 0.1}}}
    save_model(build_model(2, [4], 1, "tanh", seed=0), "tanh", tmp_path / "m", meta=meta)
    _, loaded_meta = load_model(tmp_path / "m")
    assert loaded_meta["config"] == meta["config"]
    assert loaded_meta["metrics"] == meta["metrics"]


def test_meta_cannot_override_the_saved_architecture(tmp_path):
    model = build_model(2, [4], 1, "relu", seed=0).double()
    bad_meta = {"activation": "tanh", "layer_sizes": [2, 9, 1], "note": "kept"}
    save_model(model, "relu", tmp_path / "m", meta=bad_meta)
    loaded, meta = load_model(tmp_path / "m")
    assert meta["activation"] == "relu" and meta["layer_sizes"] == [2, 4, 1]
    assert meta["note"] == "kept"
    assert isinstance(loaded[1], nn.ReLU)
    x = torch.randn(10, 2, dtype=torch.float64)
    with torch.no_grad():
        assert torch.equal(loaded(x), model(x))


def test_loaded_model_is_float64_and_in_eval_mode(tmp_path):
    model = build_model(2, [4], 1, "relu", seed=0)  # saved as float32
    save_model(model, "relu", tmp_path / "m", meta={})
    loaded, _ = load_model(tmp_path / "m")
    assert not loaded.training
    assert all(p.dtype == torch.float64 for p in loaded.parameters())
    for p, q in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(p.double(), q)  # float32 values are cast exactly


def test_loaded_model_does_not_depend_on_the_global_seed(tmp_path):
    model = build_model(2, [4], 1, "tanh", seed=7).double()
    save_model(model, "tanh", tmp_path / "m", meta={})
    torch.manual_seed(123)
    loaded, _ = load_model(tmp_path / "m")
    for p, q in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(p, q)


def test_save_overwrites_an_existing_model(tmp_path):
    save_model(build_model(2, [4], 1, "relu", seed=0), "relu", tmp_path / "m", meta={})
    second = build_model(2, [6, 6], 1, "tanh", seed=1).double()
    save_model(second, "tanh", tmp_path / "m", meta={})
    loaded, meta = load_model(tmp_path / "m")
    assert meta["layer_sizes"] == [2, 6, 6, 1] and meta["activation"] == "tanh"
    x = torch.rand(5, 2, dtype=torch.float64)
    with torch.no_grad():
        assert torch.equal(loaded(x), second(x))


def test_folded_model_survives_save_and_load(tmp_path):
    model = build_model(2, [8, 8], 1, "relu", seed=0)
    raw_model = fold_normalization(model, NORM)
    save_model(raw_model, "relu", tmp_path / "m", meta={"normalization": NORM})
    loaded, meta = load_model(tmp_path / "m")
    x = torch.rand(30, 2, dtype=torch.float64) * 10 - 5
    with torch.no_grad():
        assert torch.equal(loaded(x), raw_model(x))
        assert torch.allclose(loaded(x), unfolded(model.double(), x, meta["normalization"]))
