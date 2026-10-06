# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

**hybopt** is a CSE/APC 524 group project (Brooke Soobrian, Harshit Verma, Zhe Li; Fall 2026). Stages 1–2 of the pipeline (data generation, surrogate training) are implemented; embedding/solution methods (`methods/`, `solve.py`, `benchmark.py`, scripts 3–4) are not yet.

Commands (a project `.venv` reuses the system NumPy/SciPy via `--system-site-packages`):
- `python -m venv --system-site-packages .venv && .venv/Scripts/python -m pip install -e ".[dev]"` (rerun the editable install after adding a top-level package under `src/`)
- `.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`
- `.venv/Scripts/python -m pytest` — all tests; `-k <name>` for one
- `.venv/Scripts/python scripts/generate_data.py configs/data.yaml` → `outputs/data/<fn>__d<seed>.csv`
- `.venv/Scripts/python scripts/train.py configs/relu.yaml --jobs 8` (or `tanh.yaml`) → `outputs/models/<data>__<act>__h<depth>x<width>__t<seed>/` for the benchmark functions; add `--data configs/datasets/<name>.yaml` to train on a labeled CSV instead. `--jobs` trains networks in parallel (joblib, one torch thread each, so results do not depend on it), `--functions` limits the benchmark run, `--output-root` picks the folder.

Layout of `src/` (tests mirror it in `tests/<package>/`; `tests/scripts/` runs the scripts end to end):
- `hybopt/data/` — labeled data shared by every surrogate type. `Dataset` (`dataset.py`): checked float arrays x (n, n_inputs) and y (n, n_outputs), column names, input box (default: data min/max; it becomes the input domain in the optimization model) and a name; built from arrays/DataFrames, `Dataset.from_frame(frame, inputs, outputs, ...)` or `Dataset.from_csv(path, inputs, outputs, ...)`; `split()`, `subset()`, `to_frame()`. `split.py`: `split_fractions` and the seeded `split_indices` (torch `random_split`; identical for every surrogate type so they are compared on the same rows; pinned by a test because trained models depend on it).
- `hybopt/ann/` — ANN surrogates (other surrogate types get their own subpackages beside it, fitted to the same `Dataset`). Entry point `fit_ann(x, y=None, *, split, activation, depth, width, lower_bound=None, upper_bound=None, seed=0, split_seed=0, settings, save_to, meta, overwrite)` in `surrogate.py`: `x` is a `Dataset`, or arrays/DataFrames with `y`; it trains, reports train/val/test errors (combined and per output), and returns an `ANNSurrogate` (`predict` — a DataFrame is matched to the inputs by column name and gives a DataFrame — `save`, `load`). `data.py`: `DataModule(dataset, batch_size, split, random_state)` scales inputs to [-1, 1], standardizes outputs on the training split, and returns train/val/test `DataLoader`s (modeled on the user's pinnse package). `network.py`: `build_model` (`nn.Sequential`; He-normal init for ReLU, PyTorch default for tanh), `fold_normalization` (float64 copy with the scaling folded into the first/last Linear layers, so it maps raw inputs to raw outputs), `save_model`/`load_model` (`model.pt` + `meta.json` with `layer_sizes`, `activation`, domain, normalization, metrics; `meta.json` is written last). `train.py`: `TrainSettings`, Adam with early stopping, `error_metrics`. `checks.py`: `positive_int`.
- `benchmarks/` — a separate top-level package (not part of the `hybopt` library) with the benchmark problems: `functions.py` (Peaks, Ackley, Himmelblau as `TestFunction` with domains and known minima, registry `get_function`; domains follow Plate et al. 2026: Peaks [-2,2]², Ackley [-3.5,3.5]², Himmelblau [-5,5]²) and `sampling.py` (`sample_function` → arrays, `sample_dataset` → `Dataset` with the domain as input box; `save_samples`/`load_samples` CSV with exact float round-trip). It depends on `hybopt.data`; `hybopt` never imports it.
- The optimization side (methods, solve, benchmark) will sit in `src/hybopt/`.

Scripts skip models whose `meta.json` config matches and refuse to overwrite a differing one without `--force`. Benchmark data CSVs hold raw samples only; the split from `configs/data.yaml` is recorded in the dataset's `meta.json`. A data set config (`configs/datasets/*.yaml`) gives `csv`, `inputs`, `outputs`, optional `bounds` (for every input), `split`, `split_seed` and `name`.

## Working conventions

- **Python packages:** when a well-established package makes the code clearly more concise or more effective, install it in `.venv` (`.venv/Scripts/python -m pip install <pkg>`) and add it to `pyproject.toml`, then say why in the summary. Prefer small, standard packages (currently numpy, scipy, torch, pandas, joblib, pyyaml, pytest). Do not switch the ML framework (PyTorch) without asking.
- Use pandas for tabular data and CSV files; keep NumPy/torch for numerics.
- Never overwrite saved models or results in `outputs/`; write new runs to a new `--output-root`.

Other files:
- `docs/proposal.tex` (+ compiled `proposal.pdf`) — the submitted one-page proposal. Built with the VS Code LaTeX Workshop extension (`.vscode/settings.json` auto-cleans aux files).
- `docs/description.md` — the latest project description (most current statement of scope).
- `literature/` — the two reference papers (`s11081-026-10075-8.pdf` = Plate et al. 2026, ReLU; `s10957-018-1396-0.pdf` = Schweidtmann & Mitsos 2019, tanh) and `Project Design.docx`, the detailed design plan with the method menu and API.

`.gitignore` excludes `/.venv/`, `*.pdf`, `*.docx`, `.DS_Store`, `/literature`, `/.vscode`, and `/.claude`. `outputs/` (benchmark data, trained models, logs) is committed so everyone works from the same networks.

## What the package will do

Convert an already-trained dense feed-forward network (bounded box inputs, one hidden activation type for all layers, linear output layer; no skip connections/normalization/dropout) into a Pyomo optimization model. Training and writing a new solver are out of scope as package features (`hybopt.ann` only produces the benchmark networks).

Data flow: load net → (optional) preprocessing: bound computation / OBBT / scaling → formulation builder looked up in a **string-keyed method registry** → solve via Pyomo (Gurobi/HiGHS for MILP, SCIP for nonlinear tanh, IPOPT possible) → benchmark row.

## Method registry (keys are used across code, tests, and benchmark tables)

ReLU (Plate et al. 2026) — implement first. All share the single-neuron big-M encoding (Paper 1 eq. 5); each method changes exactly one of {bounds, network parameters, encoding}:
- `relu.bigm_ia` (R1) — big-M with interval-arithmetic bounds (eqs. 6–7). Baseline.
- `relu.bigm_obbt` (R2) — LP-based OBBT in fixed forward order over the LP relaxation (eq. 8); stably active/inactive neurons have their binaries fixed/removed.
- `relu.bigm_scaled` (R3) — a-posteriori scaling: solve convex problem (12) for log-scaling factors with SciPy, rebuild the functionally identical net (eq. 10), then R1.
- `relu.partition` (R4) — partition-based formulation with P partitions per neuron (extension; P=1 must reproduce R1, which doubles as a correctness test).
- `relu.bigm_scaled_obbt` (R5) — R3 then R2.

tanh (Schweidtmann & Mitsos 2019) — after ReLU is done:
- `tanh.fs_f3`, `tanh.fs_f1`, `tanh.fs_f4` (T1–T3) — full-space formulations using algebraic tanh rewrites F3 = 1 − 2/(e^{2x}+1), F1 = (e^x − e^{−x})/(e^x + e^{−x}), F4 = (1 − e^{−2x})/(1 + e^{−2x}); one builder with a `form` argument (F2 also exposed but not benchmarked).
- `tanh.rs_f3` (T4) — reduced-space via a single nested Pyomo expression in the inputs (no intermediate variables).

Note: `docs/description.md` (newer than the design doc) additionally adds tanh **approximation** extensions — piecewise-linear (SOS2 or binary encoding → MILP) and piecewise-quadratic on the convex/concave branches (→ MIQCP), with user control over breakpoints or fit tolerance.

## Benchmarks and validation

- Test functions: Peaks, Ackley, Himmelblau (trained NN approximations, direct output minimization). Trained networks (`configs/relu.yaml`, `configs/tanh.yaml`): 1–4 hidden layers × 10–50 neurons for both ReLU and tanh, in `outputs/relu/` and `outputs/tanh/` (the design doc capped the tanh campaign at 1–2 hidden layers).
- Correctness checks planned: formulation outputs vs. direct network evaluation, small problems with known solutions, bound validity, scaling preserving the network function, and cross-solver consistency.