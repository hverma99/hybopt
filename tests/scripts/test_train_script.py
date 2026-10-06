"""End-to-end runs of scripts/generate_data.py and scripts/train.py on tiny configs."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from hybopt.ann import ANNSurrogate

ROOT = Path(__file__).resolve().parents[2]
TINY = {
    "activation": "relu",
    "functions": ["peaks", "himmelblau"],
    "data_seed": 0,
    "hidden_layers": [1],
    "widths": [3, 4],
    "seeds": [0],
    "settings": {"lr": 3.0e-3, "batch_size": 64, "epochs": 2, "patience": 100},
}


def run(*args, check=True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stderr[-3000:])
    return result


def write_yaml(path: Path, content: dict) -> Path:
    path.write_text(yaml.safe_dump(content))
    return path


@pytest.fixture
def train_config(tmp_path) -> Path:
    return write_yaml(tmp_path / "tiny.yaml", TINY)


@pytest.fixture
def reactor_csv(tmp_path) -> Path:
    rng = np.random.default_rng(0)
    T, tau = rng.uniform(300, 400, 200), rng.uniform(1, 100, 200)
    conversion = 1 - np.exp(-1e7 * np.exp(-6000 / T) * tau)
    frame = pd.DataFrame({"run": range(200), "T": T, "tau": tau, "conversion": conversion})
    path = tmp_path / "runs.csv"
    frame.to_csv(path, index=False)
    return path


def data_config(tmp_path, csv, **extra) -> Path:
    content = {
        "name": "reactor",
        "csv": str(csv),
        "inputs": ["T", "tau"],
        "outputs": ["conversion"],
        "split": {"train": 0.7, "val": 0.15, "test": 0.15},
        "split_seed": 0,
        **extra,
    }
    return write_yaml(tmp_path / "reactor.yaml", content)


def test_benchmark_mode(tmp_path, train_config):
    out = tmp_path / "out"
    run("scripts/generate_data.py", "configs/data.yaml", "--output-root", out)
    run("scripts/train.py", train_config, "--output-root", out, "--functions", "peaks")

    folder = out / "models" / "peaks__relu__h1x3__t0"
    assert sorted(p.name for p in folder.iterdir()) == ["history.csv", "meta.json", "model.pt"]
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["config"]["function"] == "peaks" and meta["domain"]["lower_bound"] == [-2.0, -2.0]
    summary = pd.read_csv(out / "models" / "summary_relu.csv")
    assert summary["model"].tolist() == ["peaks__relu__h1x3__t0", "peaks__relu__h1x4__t0"]
    assert summary["data"].tolist() == ["peaks", "peaks"]

    rerun = run("scripts/train.py", train_config, "--output-root", out, "--functions", "peaks")
    assert "training 0 networks" in rerun.stderr and rerun.stderr.count("skip ") == 2


def test_csv_mode_with_bounds(tmp_path, train_config, reactor_csv):
    config = data_config(tmp_path, reactor_csv, bounds={"T": [300, 400], "tau": [1, 100]})
    out = tmp_path / "out"
    run("scripts/train.py", train_config, "--data", config, "--output-root", out)

    ann = ANNSurrogate.load(out / "models" / "reactor__relu__h1x3__t0")
    assert ann.input_names == ["T", "tau"] and ann.output_names == ["conversion"]
    assert ann.lower_bound == [300.0, 1.0] and ann.upper_bound == [400.0, 100.0]
    assert ann.training["data"] == "reactor" and ann.training["n_samples"]["train"] == 140
    meta = json.loads((out / "models" / "reactor__relu__h1x3__t0" / "meta.json").read_text())
    assert meta["config"]["data"] == "reactor" and meta["config"]["inputs"] == ["T", "tau"]
    assert pd.read_csv(out / "models" / "summary_relu.csv")["data"].tolist() == ["reactor"] * 2
    assert (out / "logs" / "train_relu_reactor.log").exists()


def test_csv_mode_defaults_to_the_data_range(tmp_path, train_config, reactor_csv):
    config = data_config(tmp_path, reactor_csv)
    out = tmp_path / "out"
    run("scripts/train.py", train_config, "--data", config, "--output-root", out)
    ann = ANNSurrogate.load(out / "models" / "reactor__relu__h1x4__t0")
    frame = pd.read_csv(reactor_csv)
    assert ann.lower_bound == frame[["T", "tau"]].min().tolist()


def test_csv_mode_bounds_must_cover_every_input(tmp_path, train_config, reactor_csv):
    config = data_config(tmp_path, reactor_csv, bounds={"T": [300, 400]})
    result = run("scripts/train.py", train_config, "--data", config, "--output-root", tmp_path / "o", check=False)
    assert result.returncode != 0 and "bounds must list every input" in result.stderr


def test_csv_mode_needs_the_required_keys(tmp_path, train_config):
    config = write_yaml(tmp_path / "bad.yaml", {"csv": "x.csv", "inputs": ["a"]})
    result = run("scripts/train.py", train_config, "--data", config, check=False)
    assert result.returncode != 0 and "is missing ['outputs', 'split']" in result.stderr


def test_functions_cannot_be_combined_with_data(tmp_path, train_config, reactor_csv):
    config = data_config(tmp_path, reactor_csv)
    result = run("scripts/train.py", train_config, "--data", config, "--functions", "peaks", check=False)
    assert result.returncode != 0 and "--functions only applies" in result.stderr


def test_parallel_jobs_give_identical_models(tmp_path, train_config, reactor_csv):
    config = data_config(tmp_path, reactor_csv)
    for jobs in (1, 2):
        run("scripts/train.py", train_config, "--data", config, "--output-root", tmp_path / f"j{jobs}", "--jobs", jobs)
    one, two = (pd.read_csv(tmp_path / f"j{j}" / "models" / "summary_relu.csv") for j in (1, 2))
    assert one.drop(columns="train_seconds").equals(two.drop(columns="train_seconds"))
    for name in one["model"]:
        a = ANNSurrogate.load(tmp_path / "j1" / "models" / name)
        b = ANNSurrogate.load(tmp_path / "j2" / "models" / name)
        assert np.array_equal(a.predict(np.array([[350.0, 50.0]])), b.predict(np.array([[350.0, 50.0]])))
