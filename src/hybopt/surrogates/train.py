"""
Train dense ReLU/tanh regression networks with PyTorch.
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from hybopt.surrogates.data import DataModule


@dataclass
class TrainSettings:
    lr: float = 3e-3
    batch_size: int = 64
    epochs: int = 1000
    patience: int = 100


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, loss_fn) -> float:
    """
    Average loss of ``model`` over all batches of ``loader``.
    """
    model.eval()

    total, n = 0.0, 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            total += loss_fn(model(x_batch), y_batch).item() * len(x_batch)
            n += len(x_batch)

    return total / n


def train_network(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    settings: TrainSettings,
) -> list[dict]:
    """
    Train ``model`` with Adam on MSE, with early stopping on the validation loss.

    Inputs
    ------
    model                       : ANN
    train_loader, val_loader    : Training and validation batches.
    settings                    : TrainSettings

    Returns
    -------
    history : Per epoch: {"epoch", "train_mse", "val_mse"} on normalized outputs.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)
    loss_fn = nn.MSELoss()

    best_val, best_epoch = np.inf, 0
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(1, settings.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x_batch)
            n += len(x_batch)

        val_loss = evaluate(model, val_loader, loss_fn)
        history.append({"epoch": epoch, "train_mse": total / n, "val_mse": val_loss})

        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= settings.patience:
            break

    model.load_state_dict(best_state)

    return history


def error_metrics(raw_model: nn.Module, dm: DataModule) -> dict:
    metrics = {}
    for name in ("train", "val", "test"):
        x, y = dm.x[dm.idx[name]], dm.y[dm.idx[name]]
        with torch.no_grad():
            y_pred = raw_model(torch.tensor(x, dtype=torch.float64))[:, 0].numpy()
        err = y_pred - y
        rmse = float(np.sqrt(np.mean(err**2)))
        metrics[name] = {
            "rmse": rmse,
            "max_abs_err": float(np.max(np.abs(err))),
            "rel_rmse": rmse / float(np.std(y)),
        }
    return metrics
