import math

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from benchmarks import get_function, sample_function
from hybopt.ann.data import DataModule
from hybopt.data import Dataset
from hybopt.ann.network import build_model, fold_normalization
from hybopt.ann.train import TrainSettings, error_metrics, evaluate, train_network

SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}
METRIC_KEYS = {"rmse", "max_abs_err", "rel_rmse"}


def make_dm(name, n, batch_size=64):
    fn = get_function(name)
    x, y = sample_function(fn, n, seed=0)
    return DataModule(Dataset(x, y, fn.lower_bound, fn.upper_bound), batch_size, SPLIT)


def linear_targets(n=300, seed=0):
    """Two inputs and two linear outputs on different scales: y = x @ W.T + b."""
    x = np.random.default_rng(seed).uniform(-1, 1, (n, 2))
    W, b = np.array([[3.0, -2.0], [100.0, 50.0]]), np.array([0.5, -1000.0])
    return x, x @ W.T + b, W, b


def full_batch_mse(model: nn.Module, loader: DataLoader) -> float:
    X, Y = (torch.cat(t) for t in zip(*loader))
    with torch.no_grad():
        return nn.MSELoss()(model(X), Y).item()


def linear_model(weight, bias) -> nn.Sequential:
    """A float64 model computing weight @ x + bias, used as an exact 'raw' model."""
    weight = np.atleast_2d(np.asarray(weight, dtype=float))
    layer = nn.Linear(weight.shape[1], weight.shape[0]).double()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weight, dtype=torch.float64))
        layer.bias.copy_(torch.tensor(np.atleast_1d(bias), dtype=torch.float64))
    return nn.Sequential(layer)


def predict(raw_model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return raw_model(torch.tensor(x, dtype=torch.float64)).numpy()


# --- TrainSettings --------------------------------------------------------------------


def test_train_settings_defaults():
    s = TrainSettings()
    assert (s.lr, s.batch_size, s.epochs, s.patience) == (3e-3, 64, 1000, 100)


def test_train_settings_from_config_dict():
    s = TrainSettings(**{"lr": 1e-2, "batch_size": 32, "epochs": 5, "patience": 2})
    assert (s.lr, s.batch_size, s.epochs, s.patience) == (1e-2, 32, 5, 2)


def test_train_settings_allows_zero_learning_rate():
    assert TrainSettings(lr=0.0).lr == 0.0
    assert TrainSettings(lr=1).lr == 1


@pytest.mark.parametrize("lr", [-1e-3, float("nan"), float("inf"), "0.01", None, True])
def test_train_settings_rejects_bad_learning_rates(lr):
    with pytest.raises(ValueError, match="lr must be a non-negative number"):
        TrainSettings(lr=lr)


@pytest.mark.parametrize("field", ["batch_size", "epochs", "patience"])
@pytest.mark.parametrize("value", [0, -5, 2.5, "10", None])
def test_train_settings_rejects_non_positive_integers(field, value):
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        TrainSettings(**{field: value})


# --- evaluate -------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 7, 64, 1000])
def test_evaluate_equals_full_batch_mse(batch_size):
    # sample-weighted averaging must be exact even when the last batch is smaller
    dm = make_dm("peaks", 500, batch_size=batch_size)
    model = build_model(2, [8], 1, "tanh", seed=0)
    _, val_loader, _ = dm.labeled_data_loader()
    assert evaluate(model, val_loader, nn.MSELoss()) == pytest.approx(
        full_batch_mse(model, val_loader), rel=1e-5
    )


def test_evaluate_sets_eval_mode_and_does_not_change_the_model():
    dm = make_dm("peaks", 300)
    model = build_model(2, [8], 1, "relu", seed=0)
    before = [p.detach().clone() for p in model.parameters()]
    evaluate(model, dm.labeled_data_loader()[1], nn.MSELoss())
    assert not model.training
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before))
    assert all(p.grad is None for p in model.parameters())


def test_evaluate_is_zero_for_a_perfect_model():
    x = torch.randn(50, 2)
    y = (x @ torch.tensor([[2.0], [-1.0]])) + 0.5
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -1.0]]))
        model.bias.fill_(0.5)
    assert evaluate(model, loader, nn.MSELoss()) == pytest.approx(0.0, abs=1e-12)


def test_evaluate_with_several_outputs():
    x, y, _, _ = linear_targets()
    dm = DataModule(Dataset(x, y), 16, SPLIT)
    model = build_model(2, [4], 2, "tanh", seed=0)
    _, val_loader, _ = dm.labeled_data_loader()
    assert evaluate(model, val_loader, nn.MSELoss()) == pytest.approx(
        full_batch_mse(model, val_loader), rel=1e-5
    )


# --- train_network --------------------------------------------------------------------


@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_training_is_deterministic_and_learns(activation):
    dm = make_dm("himmelblau", 2000)
    settings = TrainSettings(lr=3e-3, epochs=150, patience=150)

    def run():
        model = build_model(2, [16, 16], 1, activation, seed=0)
        train_loader, val_loader, _ = dm.labeled_data_loader(shuffle_seed=0)
        return model, train_network(model, train_loader, val_loader, settings)

    model_a, history = run()
    model_b, _ = run()
    assert all(torch.equal(p, q) for p, q in zip(model_a.parameters(), model_b.parameters()))
    assert history[-1]["train_mse"] < 0.2 * history[0]["train_mse"]
    raw_model = fold_normalization(model_a, dm.normalization)
    assert error_metrics(raw_model, dm)["test"]["rel_rmse"] < 0.3


@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_training_with_several_outputs_learns_every_output(activation):
    x, y, _, _ = linear_targets(n=1500)
    dm = DataModule(Dataset(x, y), 32, SPLIT)
    model = build_model(2, [16], 2, activation, seed=0)
    history = train_network(model, *dm.labeled_data_loader()[:2], TrainSettings(epochs=60))
    assert history[-1]["val_mse"] < 0.05 * history[0]["val_mse"]
    raw_model = fold_normalization(model, dm.normalization)
    per_output = error_metrics(raw_model, dm)["test"]["per_output"]
    assert all(m["rel_rmse"] < 0.1 for m in per_output.values())


def test_different_seeds_give_different_models():
    dm = make_dm("peaks", 500)
    settings = TrainSettings(epochs=3)
    models = []
    for seed in (0, 1):
        model = build_model(2, [8], 1, "relu", seed=seed)
        train_network(model, *dm.labeled_data_loader(shuffle_seed=seed)[:2], settings)
        models.append(model)
    assert not torch.equal(models[0][0].weight, models[1][0].weight)


def test_history_format():
    dm = make_dm("peaks", 500)
    model = build_model(2, [8], 1, "tanh", seed=0)
    history = train_network(model, *dm.labeled_data_loader()[:2], TrainSettings(epochs=5))
    assert [h["epoch"] for h in history] == [1, 2, 3, 4, 5]
    for h in history:
        assert set(h) == {"epoch", "train_mse", "val_mse"}
        assert math.isfinite(h["train_mse"]) and h["train_mse"] > 0
        assert math.isfinite(h["val_mse"]) and h["val_mse"] > 0


def test_runs_all_epochs_when_validation_keeps_improving():
    dm = make_dm("peaks", 500)
    model = build_model(2, [8], 1, "tanh", seed=0)
    history = train_network(
        model, *dm.labeled_data_loader()[:2], TrainSettings(epochs=10, patience=100)
    )
    assert len(history) == 10


def test_stops_after_patience_epochs_without_improvement():
    # lr = 0 freezes the weights, so validation never improves after epoch 1
    dm = make_dm("peaks", 500)
    model = build_model(2, [8], 1, "relu", seed=0)
    history = train_network(
        model, *dm.labeled_data_loader()[:2], TrainSettings(lr=0.0, epochs=1000, patience=7)
    )
    assert len(history) == 1 + 7
    assert len({h["val_mse"] for h in history}) == 1


def test_recorded_losses_match_the_loaders():
    # with frozen weights, the recorded losses are exactly the losses on each split
    dm = make_dm("ackley", 600, batch_size=50)
    model = build_model(2, [8], 1, "tanh", seed=0)
    train_loader, val_loader, _ = dm.labeled_data_loader()
    history = train_network(model, train_loader, val_loader, TrainSettings(lr=0.0, epochs=2))
    assert history[0]["train_mse"] == pytest.approx(full_batch_mse(model, train_loader), rel=1e-5)
    assert history[0]["val_mse"] == pytest.approx(full_batch_mse(model, val_loader), rel=1e-5)


def test_early_stopping_restores_best_epoch():
    # a huge learning rate makes validation loss jump around; the best epoch must be kept
    dm = make_dm("peaks", 500)
    model = build_model(2, [8], 1, "tanh", seed=0)
    train_loader, val_loader, _ = dm.labeled_data_loader()
    history = train_network(
        model, train_loader, val_loader, TrainSettings(lr=0.5, epochs=40, patience=5)
    )
    best = min(history, key=lambda h: h["val_mse"])
    assert len(history) <= 40
    assert evaluate(model, val_loader, nn.MSELoss()) == pytest.approx(best["val_mse"], rel=1e-5)
    assert len(history) - best["epoch"] <= 5  # stopped within `patience` of the best epoch


def test_frozen_weights_are_returned_unchanged():
    dm = make_dm("peaks", 300)
    model = build_model(2, [8], 1, "relu", seed=0)
    before = [p.detach().clone() for p in model.parameters()]
    train_network(model, *dm.labeled_data_loader()[:2], TrainSettings(lr=0.0, epochs=3))
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before))


def test_training_updates_the_model_in_place():
    dm = make_dm("peaks", 300)
    model = build_model(2, [8], 1, "relu", seed=0)
    before = model[0].weight.detach().clone()
    result = train_network(model, *dm.labeled_data_loader()[:2], TrainSettings(epochs=3))
    assert isinstance(result, list)
    assert not torch.equal(model[0].weight, before)


# --- error_metrics --------------------------------------------------------------------


def test_error_metrics_structure():
    dm = make_dm("peaks", 500)
    raw_model = fold_normalization(build_model(2, [8], 1, "relu", seed=0), dm.normalization)
    metrics = error_metrics(raw_model, dm)
    assert set(metrics) == {"train", "val", "test"}
    for values in metrics.values():
        assert set(values) == METRIC_KEYS | {"per_output"}
        assert list(values["per_output"]) == ["y"]
        for key in METRIC_KEYS:
            assert isinstance(values[key], float) and values[key] >= 0
        assert values["rmse"] <= values["max_abs_err"]


def test_error_metrics_with_one_output_equal_the_per_output_values():
    dm = make_dm("ackley", 400)
    raw_model = fold_normalization(build_model(2, [8], 1, "tanh", seed=0), dm.normalization)
    for values in error_metrics(raw_model, dm).values():
        assert {k: values[k] for k in METRIC_KEYS} == values["per_output"]["y"]


def test_error_metrics_uses_raw_units():
    dm = make_dm("peaks", 500)
    raw_model = fold_normalization(build_model(2, [8], 1, "relu", seed=0), dm.normalization)
    metrics = error_metrics(raw_model, dm)
    for split in ("train", "val", "test"):
        x, y = dm.x[dm.idx[split]], dm.y[dm.idx[split]]
        err = predict(raw_model, x) - y
        assert metrics[split]["rmse"] == pytest.approx(np.sqrt(np.mean(err**2)))
        assert metrics[split]["max_abs_err"] == pytest.approx(np.max(np.abs(err)))
        assert metrics[split]["rel_rmse"] == pytest.approx(np.sqrt(np.mean(err**2)) / np.std(y))


def test_error_metrics_per_output_values():
    x, y, _, _ = linear_targets()
    dm = DataModule(Dataset(x, y), 16, SPLIT)
    raw_model = fold_normalization(build_model(2, [4], 2, "relu", seed=0), dm.normalization)
    metrics = error_metrics(raw_model, dm)
    for split in ("train", "val", "test"):
        x_s, y_s = dm.x[dm.idx[split]], dm.y[dm.idx[split]]
        err = predict(raw_model, x_s) - y_s
        per_output = metrics[split]["per_output"]
        assert list(per_output) == ["y1", "y2"]
        for k, name in enumerate(["y1", "y2"]):
            rmse = np.sqrt(np.mean(err[:, k] ** 2))
            assert per_output[name]["rmse"] == pytest.approx(rmse)
            assert per_output[name]["max_abs_err"] == pytest.approx(np.max(np.abs(err[:, k])))
            assert per_output[name]["rel_rmse"] == pytest.approx(rmse / np.std(y_s[:, k]))


def test_error_metrics_combines_outputs():
    x, y, _, _ = linear_targets()
    dm = DataModule(Dataset(x, y), 16, SPLIT)
    raw_model = fold_normalization(build_model(2, [4], 2, "tanh", seed=0), dm.normalization)
    test = error_metrics(raw_model, dm)["test"]
    per_output = list(test["per_output"].values())
    err = predict(raw_model, dm.x[dm.idx["test"]]) - dm.y[dm.idx["test"]]
    assert test["rmse"] == pytest.approx(np.sqrt(np.mean(err**2)))  # pooled over outputs
    assert test["max_abs_err"] == max(m["max_abs_err"] for m in per_output)
    rms_of_rel = np.sqrt(np.mean([m["rel_rmse"] ** 2 for m in per_output]))
    assert test["rel_rmse"] == pytest.approx(rms_of_rel)


def test_error_metrics_use_output_names():
    x, y, _, _ = linear_targets()
    dm = DataModule(Dataset(x, pd.DataFrame(y, columns=["conversion", "duty"])), 16, SPLIT)
    raw_model = fold_normalization(build_model(2, [4], 2, "relu", seed=0), dm.normalization)
    assert list(error_metrics(raw_model, dm)["val"]["per_output"]) == ["conversion", "duty"]


def test_error_metrics_are_zero_for_an_exact_model():
    # data from a linear function, and a model that is exactly that function
    x, y, W, b = linear_targets()
    dm = DataModule(Dataset(x, y, (-1, -1), (1, 1)), 32, SPLIT)
    metrics = error_metrics(linear_model(W, b), dm)
    for values in metrics.values():
        assert all(values[k] == pytest.approx(0.0, abs=1e-9) for k in METRIC_KEYS)
        for per_output in values["per_output"].values():
            assert all(v == pytest.approx(0.0, abs=1e-9) for v in per_output.values())


def test_error_metrics_for_a_zero_model():
    # predicting 0 everywhere: rmse = sqrt(mean(y^2)), max error = max |y|
    dm = make_dm("himmelblau", 400)
    metrics = error_metrics(linear_model([0.0, 0.0], 0.0), dm)
    for split in ("train", "val", "test"):
        y = dm.y[dm.idx[split]]
        assert metrics[split]["rmse"] == pytest.approx(np.sqrt(np.mean(y**2)))
        assert metrics[split]["max_abs_err"] == pytest.approx(np.max(np.abs(y)))


def test_error_metrics_for_the_mean_predictor():
    # predicting the mean of a split gives rel_rmse = 1 on that split
    dm = make_dm("ackley", 400)
    y_test = dm.y[dm.idx["test"]]
    metrics = error_metrics(linear_model([0.0, 0.0], float(y_test.mean())), dm)
    assert metrics["test"]["rel_rmse"] == pytest.approx(1.0)


def test_error_metrics_rejects_the_normalized_model():
    # the trained (float32, normalized) model must be folded first; using it directly
    # would report errors in the wrong units, so it fails loudly instead
    dm = make_dm("peaks", 300)
    model = build_model(2, [8], 1, "relu", seed=0)
    with pytest.raises(RuntimeError, match="dtype"):
        error_metrics(model, dm)


def test_error_metrics_match_training_loss_scale():
    # val MSE on normalized outputs times y_scale^2 is the raw val MSE
    dm = make_dm("peaks", 600)
    model = build_model(2, [8], 1, "tanh", seed=0)
    _, val_loader, _ = dm.labeled_data_loader()
    normalized_mse = evaluate(model, val_loader, nn.MSELoss())
    raw_rmse = error_metrics(fold_normalization(model, dm.normalization), dm)["val"]["rmse"]
    y_scale = dm.normalization["y_scale"][0]
    assert raw_rmse**2 == pytest.approx(normalized_mse * y_scale**2, rel=1e-4)
