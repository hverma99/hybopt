# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

**hybopt** is a CSE/APC 524 group project (Brooke Soobrian, Harshit Verma, Zhe Li; Fall 2026). Stages 1–2 of the pipeline (data generation, training) are implemented; embedding/solution methods (`methods/`, `solve.py`, `benchmark.py`, scripts 3–4) are not yet. The target layout is `docs/repo-outline.md`.

Commands (a project `.venv` reuses the system NumPy/SciPy via `--system-site-packages`):
- `python -m venv --system-site-packages .venv && .venv/Scripts/python -m pip install -e ".[dev]"`
- `.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`
- `.venv/Scripts/python -m pytest` — all tests; `-k <name>` for one
- `.venv/Scripts/python scripts/generate_data.py configs/data.yaml` → `outputs/data/<fn>__d<seed>.csv`
- `.venv/Scripts/python scripts/train.py configs/relu.yaml` (or `tanh.yaml`) → `outputs/models/<fn>__<act>__h<depth>x<width>__t<seed>/`

Scripts skip artifacts whose `meta.json` config matches and refuse to overwrite a differing one without `--force`. The code that produces the benchmark networks lives in the `hybopt.surrogates` subpackage (`src/hybopt/surrogates/`: `functions.py`, `data.py`, `network.py`, `train.py`; tests in `tests/surrogates/`); the optimization side (methods, solve, benchmark) will sit beside it in `src/hybopt/`. The data CSV holds raw samples only; `DataModule` (`data.py`, modeled on the user's pinnse package) applies the seeded split from `configs/data.yaml` (recorded in the dataset's `meta.json`), normalizes, and returns train/val/test `DataLoader`s. `network.py` builds the model (`nn.Sequential`), and `fold_normalization` returns a float64 copy with the input/output normalization folded into its first/last Linear layers, so it maps raw inputs to raw outputs; that copy is saved as `model.pt` (+ `meta.json` with `layer_sizes`, `activation`, domain) and reloaded with `load_model`. Training (`train.py`: Adam, early stopping, one CPU thread for reproducibility) uses PyTorch. Test-function domains follow Plate et al. (2026): Peaks [-2,2]², Ackley [-3.5,3.5]², Himmelblau [-5,5]².

Other files:
- `docs/proposal.tex` (+ compiled `proposal.pdf`) — the submitted one-page proposal. Built with the VS Code LaTeX Workshop extension (`.vscode/settings.json` auto-cleans aux files).
- `docs/description.md` — the latest project description (most current statement of scope).
- `literature/` — the two reference papers (`s11081-026-10075-8.pdf` = Plate et al. 2026, ReLU; `s10957-018-1396-0.pdf` = Schweidtmann & Mitsos 2019, tanh) and `Project Design.docx`, the detailed design plan with the method menu and API.

`.gitignore` excludes `/outputs/`, `/.venv/`, `*.pdf`, `*.docx`, `/literature`, and `/.vscode`.

## What the package will do

Convert an already-trained dense feed-forward network (bounded box inputs, one hidden activation type for all layers, linear output layer; no skip connections/normalization/dropout) into a Pyomo optimization model. Training and writing a new solver are out of scope as package features (`hybopt.surrogates` only produces the benchmark networks).

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

- Test functions: Peaks, Ackley, Himmelblau (trained NN approximations, direct output minimization). Campaign size: ReLU 1–4 hidden layers × 10–50 neurons; tanh 1–2 hidden layers × 10–50 neurons.
- Correctness checks planned: formulation outputs vs. direct network evaluation, small problems with known solutions, bound validity, scaling preserving the network function, and cross-solver consistency.