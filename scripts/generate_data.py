"""
Stage 1: sample every test function in the config into outputs/data/<function>__d<seed>.csv.

The CSV holds the raw samples (x1, ..., xn, y). The train/val/test split in the config is recorded in
the dataset's meta.json and applied later by ``hybopt.ann.data.DataModule`` when training.

Usage: python scripts/generate_data.py configs/data.yaml [--output-root outputs] [--force]
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

from benchmarks import get_function, sample_function, save_samples

log = logging.getLogger("generate_data")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate even if a dataset with another config exists",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data_dir = args.output_root / "data"
    log_dir = args.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "generate_data.log"),
        ],
    )

    for name in cfg["functions"]:
        fn = get_function(name)
        dataset_cfg = {
            "function": name,
            "n_samples": cfg["n_samples"][name],
            "design": cfg["design"],
            "seed": cfg["seed"],
            "split": cfg["split"],
        }
        csv_path = data_dir / f"{name}__d{cfg['seed']}.csv"
        meta_path = csv_path.with_suffix(".meta.json")
        if meta_path.exists() and not args.force:
            if json.loads(meta_path.read_text())["config"] == dataset_cfg:
                log.info("skip %s (exists, same config)", csv_path.name)
                continue
            raise SystemExit(
                f"{csv_path} exists with a different config; rerun with --force to overwrite"
            )

        x, y = sample_function(
            fn, dataset_cfg["n_samples"], dataset_cfg["seed"], dataset_cfg["design"]
        )
        save_samples(x, y, csv_path)
        meta = {
            "config": dataset_cfg,
            "domain": {
                "lower_bound": list(fn.lower_bound),
                "upper_bound": list(fn.upper_bound),
            },
            "y_range": [float(y.min()), float(y.max())],
            "source_config": args.config.as_posix(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        log.info("wrote %s (%d samples)", csv_path.name, len(y))


if __name__ == "__main__":
    main()
