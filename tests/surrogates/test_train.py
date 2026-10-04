import math
from collections.abc import Sized
from typing import cast

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hybopt.surrogates.data import DataModule, sample_function
from hybopt.surrogates.functions import get_function
from hybopt.surrogates.network import build_model, fold_normalization
from hybopt.surrogates.train import TrainSettings, error_metrics, evaluate, train_network

SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)  # same setting as scripts/train.py, so results are reproducible


def make_dm(name, n, batch_size=64):
    fn = get_function(name)
    x, y = sample_function(fn, n, seed=0)
    return DataModule(x, y, fn.lower_bound, fn.upper_bound, batch_size, SPLIT)


def full_batch_mse(model: nn.Module, loader: DataLoader) -> float:
    X, Y = (torch.cat(t) for t in zip(*loader))
    with torch.no_grad():
        return nn.MSELoss()(model(X), Y).item()


def linear_model(weight, bias) -> nn.Sequential:
    """A float64 model computing weight @ x + bias, used as an exact 'raw' model."""
    layer = nn.Linear(len(weight), 1).double()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([weight], dtype=torch.float64))
        layer.bias.fill_(bias)
    return nn.Sequential(layer)


# --- TrainSettings --------------------------------------------------------------------


def test_train_settings_defaults():
    s = TrainSettings()
    assert (s.lr, s.batch_size, s.epochs, s.patience) == (3e-3, 64, 1000, 100)


def test_train_settings_from_config_dict():
    s = TrainSettings(**{"lr": 1e-2, "batch_size": 32, "epochs": 5, "patience": 2})
    assert (s.lr, s.batch_size, s.epochs, s.patience) == (1e-2, 32, 5, 2)


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
        assert set(values) == {"rmse", "max_abs_err", "rel_rmse"}
        assert all(isinstance(v, float) and v >= 0 for v in values.values())
        assert values["rmse"] <= values["max_abs_err"]


def test_error_metrics_uses_raw_units():
    dm = make_dm("peaks", 500)
    raw_model = fold_normalization(build_model(2, [8], 1, "relu", seed=0), dm.normalization)
    metrics = error_metrics(raw_model, dm)
    for split in ("train", "val", "test"):
        x, y = dm.x[dm.idx[split]], dm.y[dm.idx[split]]
        with torch.no_grad():
            err = raw_model(torch.tensor(x, dtype=torch.float64))[:, 0].numpy() - y
        assert metrics[split]["rmse"] == pytest.approx(np.sqrt(np.mean(err**2)))
        assert metrics[split]["max_abs_err"] == pytest.approx(np.max(np.abs(err)))
        assert metrics[split]["rel_rmse"] == pytest.approx(np.sqrt(np.mean(err**2)) / np.std(y))


def test_error_metrics_are_zero_for_an_exact_model():
    # data from a linear function, and a model that is exactly that function
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (200, 2))
    y = 3.0 * x[:, 0] - 2.0 * x[:, 1] + 0.5
    dm = DataModule(x, y, (-1, -1), (1, 1), 32, SPLIT)
    metrics = error_metrics(linear_model([3.0, -2.0], 0.5), dm)
    for values in metrics.values():
        assert all(v == pytest.approx(0.0, abs=1e-12) for v in values.values())


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
    assert len(cast(Sized, val_loader.dataset)) == len(dm.idx["val"])
