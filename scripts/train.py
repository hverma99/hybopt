"""Stage 2: train one network per (data set, depth, width, seed) in the config.

The data sets are either the benchmark functions in the config (sampled by generate_data.py into
outputs/data/<function>__d<seed>.csv, with the function's domain as the input box), or one labeled
CSV file described by a data set config (--data configs/datasets/<name>.yaml). Every network is
fitted with the general ANN API ``fit_ann``.

Writes outputs/models/<data>__<activation>__h<depth>x<width>__t<seed>/ with model.pt (PyTorch
weights of the network on raw inputs/outputs, used for embedding), meta.json and history.csv, and
outputs/models/summary_<activation>.csv with the error metrics of every model.

Usage: python scripts/train.py configs/relu.yaml [--data configs/datasets/x.yaml]
                               [--output-root outputs] [--jobs 8] [--functions peaks] [--force]
"""

import argparse
import json
import logging
from itertools import product
from pathlib import Path

import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed

from benchmarks import get_function, load_samples
from hybopt.ann import TrainSettings, fit_ann
from hybopt.data import Dataset

log = logging.getLogger("train")


def model_name(data: str, activation: str, depth: int, width: int, seed: int) -> str:
    return f"{data}__{activation}__h{depth}x{width}__t{seed}"


def benchmark_sources(
    functions: list[str], data_seed: int, output_root: Path
) -> list[dict]:
    """
    The sampled benchmark functions, each with the split recorded in its dataset's meta.json.
    """
    sources = []
    for name in functions:
        fn = get_function(name)
        path = output_root / "data" / f"{name}__d{data_seed}.csv"
        if not path.exists():
            raise SystemExit(f"{path} not found; run scripts/generate_data.py first")
        data_cfg = json.loads(path.with_suffix(".meta.json").read_text())["config"]
        x, y = load_samples(path)
        sources.append(
            {
                "data": Dataset(x, y, fn.lower_bound, fn.upper_bound, name=name),
                "split": data_cfg["split"],
                "split_seed": data_cfg["seed"],
                "config": {
                    "function": name,
                    "dataset": path.relative_to(output_root).as_posix(),
                    "split": data_cfg["split"],
                },
            }
        )
    return sources


def csv_source(config_path: Path) -> dict:
    """
    A labeled CSV data set described by a data set config with the keys csv, inputs, outputs,
    split and optionally bounds (for every input), split_seed and name.
    """
    cfg = yaml.safe_load(config_path.read_text())
    if missing := [
        key for key in ("csv", "inputs", "outputs", "split") if key not in cfg
    ]:
        raise SystemExit(f"{config_path} is missing {missing}")

    lower = upper = None
    if (bounds := cfg.get("bounds")) is not None:
        if set(bounds) != set(cfg["inputs"]):
            raise SystemExit(
                f"{config_path}: bounds must list every input {cfg['inputs']} (or be left out "
                f"to use the data min/max), got {sorted(bounds)}"
            )
        lower = [bounds[name][0] for name in cfg["inputs"]]
        upper = [bounds[name][1] for name in cfg["inputs"]]

    csv = Path(cfg["csv"])
    data = Dataset.from_csv(
        csv,
        cfg["inputs"],
        cfg["outputs"],
        lower,
        upper,
        name=cfg.get("name", config_path.stem),
    )
    split_seed = cfg.get("split_seed", 0)
    return {
        "data": data,
        "split": cfg["split"],
        "split_seed": split_seed,
        "config": {
            "data": data.name,
            "dataset": csv.as_posix(),
            "inputs": data.input_names,
            "outputs": data.output_names,
            "bounds": bounds,
            "split": cfg["split"],
            "split_seed": split_seed,
        },
    }


def write_summary(models_dir: Path, activation: str) -> Path:
    rows = []
    for meta_path in sorted(models_dir.glob(f"*__{activation}__*/meta.json")):
        meta = json.loads(meta_path.read_text())
        config = meta["config"]
        row = {
            "model": meta_path.parent.name,
            "data": config.get("function", config.get("data")),
            "depth": len(config["hidden_widths"]),
            "width": config["hidden_widths"][0],
            "seed": config["seed"],
            "epochs_run": meta["training"]["epochs_run"],
            "best_epoch": meta["training"]["best_epoch"],
            "train_seconds": meta["training"]["train_seconds"],
        }
        for split, values in meta["metrics"].items():
            for key in ("rmse", "max_abs_err", "rel_rmse"):
                row[f"{split}_{key}"] = values[key]
        rows.append(row)
    path = models_dir / f"summary_{activation}.csv"
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    return path


def train_one(
    data, split, split_seed, model_cfg, settings, out_dir, meta, overwrite
) -> dict:
    """
    Fit and save one network; runs in a worker process when --jobs > 1.
    """
    # One thread is faster for these small networks, and results then do not depend on the core count.
    torch.set_num_threads(1)
    ann = fit_ann(
        data,
        split=split,
        activation=model_cfg["activation"],
        depth=len(model_cfg["hidden_widths"]),
        width=model_cfg["hidden_widths"][0],
        seed=model_cfg["seed"],
        split_seed=split_seed,
        settings=settings,
        save_to=out_dir,
        meta={**meta, "torch_threads": torch.get_num_threads()},
        overwrite=overwrite,
    )
    return {"name": out_dir.name, **ann.training, "test": ann.metrics["test"]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config", type=Path, help="training config, e.g. configs/relu.yaml"
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="train on this labeled CSV data set config instead of the benchmark functions",
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="retrain even if a model with another config exists",
    )
    parser.add_argument(
        "--functions",
        nargs="+",
        help="train only these benchmark functions from the config",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="networks to train in parallel, one CPU thread each (-1: one per core)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    activation = cfg["activation"]
    settings = TrainSettings(**cfg["settings"])
    torch.set_num_threads(1)

    if args.data is not None:
        if args.functions:
            raise SystemExit(
                "--functions only applies to the benchmark functions, not to --data"
            )
        sources = [csv_source(args.data)]
        log_suffix = f"_{sources[0]['data'].name}"
    else:
        functions = args.functions or cfg["functions"]
        if unknown := set(functions) - set(cfg["functions"]):
            raise SystemExit(f"functions {sorted(unknown)} are not in {args.config}")
        sources = benchmark_sources(functions, cfg["data_seed"], args.output_root)
        log_suffix = f"_{'-'.join(args.functions)}" if args.functions else ""

    models_dir = args.output_root / "models"
    log_dir = args.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f"train_{activation}{log_suffix}.log"),
        ],
    )

    # Collect the networks still to train; finished ones are skipped
    tasks = []
    for source in sources:
        for depth, width, seed in product(
            cfg["hidden_layers"], cfg["widths"], cfg["seeds"]
        ):
            out_dir = models_dir / model_name(
                source["data"].name, activation, depth, width, seed
            )
            model_cfg = {
                **source["config"],
                "activation": activation,
                "hidden_widths": [width] * depth,
                "seed": seed,
                "settings": cfg["settings"],
            }
            meta_path = out_dir / "meta.json"
            if meta_path.exists() and not args.force:
                if json.loads(meta_path.read_text())["config"] == model_cfg:
                    log.info("skip %s (exists, same config)", out_dir.name)
                    continue
                raise SystemExit(
                    f"{out_dir} exists with a different config; rerun with --force to overwrite"
                )
            tasks.append(
                dict(
                    data=source["data"],
                    split=source["split"],
                    split_seed=source["split_seed"],
                    model_cfg=model_cfg,
                    settings=settings,
                    out_dir=out_dir,
                    meta={"config": model_cfg, "source_config": args.config.as_posix()},
                    overwrite=args.force,
                )
            )

    log.info("training %d networks with %d parallel jobs", len(tasks), args.jobs)
    results = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(train_one)(**task) for task in tasks
    )
    for result in results:
        if result is None:
            log.warning("training task returned no result")
            continue
        log.info(
            "trained %s in %.1fs (%d epochs, best %d): test rmse %.4g, rel %.3g",
            result["name"],
            result["train_seconds"],
            result["epochs_run"],
            result["best_epoch"],
            result["test"]["rmse"],
            result["test"]["rel_rmse"],
        )

    log.info("summary: %s", write_summary(models_dir, activation))


if __name__ == "__main__":
    main()
