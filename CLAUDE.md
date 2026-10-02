# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

**hybopt** is a CSE/APC 524 group project (Brooke Soobrian, Harshit Verma, Zhe Li; Fall 2026). The repository is currently at the **proposal stage — there is no Python code, build system, or test suite yet.** Do not assume commands such as `pytest` or `pip install -e .` work until a `pyproject.toml` and `tests/` exist.

What exists:
- `docs/proposal.tex` (+ compiled `proposal.pdf`) — the submitted one-page proposal. Built with the VS Code LaTeX Workshop extension (`.vscode/settings.json` auto-cleans aux files).
- `docs/description.md` — the latest project description (most current statement of scope).
- `literature/` — the two reference papers (`s11081-026-10075-8.pdf` = Plate et al. 2026, ReLU; `s10957-018-1396-0.pdf` = Schweidtmann & Mitsos 2019, tanh) and `Project Design.docx`, the detailed design plan with the method menu and API.

**`.gitignore` excludes `*.md`, `*.pdf`, `*.docx`, `*.json`, `/literature`, and `/.vscode`.** This means `CLAUDE.md`, `docs/description.md`, and future `README.md`/mkdocs pages are untracked unless the ignore rules are changed (or files are force-added). Flag this when adding documentation or JSON config.

## What the package will do

Convert an already-trained dense feed-forward network (bounded box inputs, one hidden activation type for all layers, linear output layer; no skip connections/normalization/dropout) into a Pyomo optimization model. Training and writing a new solver are out of scope (a thin seeded `examples/train.py` in PyTorch only produces benchmark networks).

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