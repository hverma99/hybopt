# Embedding Trained Neural Networks in Mixed-Integer Optimization: A Benchmarkable Python Package

*CSE 524 group project. The package name is **hybopt**.*

Artificial neural networks (ANNs) are widely used as surrogate models in engineering optimization. However, ANN-embedded optimization models are often computationally intractable because of the nonconvexity and mathematically complex nature of ANNs. The two widely utilized ANNs in engineering applications are ReLU-based and tanh-based ANNs. ReLU networks are often formulated as a mixed-integer linear program (MILP), where integer variables are used to represent active and inactive neurons, using big-M reformulation; however, these big-M coefficients (bounds) can grow exponentially with increasing ANN depth. On the other hand, tanh networks are formulated as nonconvex mixed-integer non linear programs (MINLP), for which computational tractabiltiy depends strongly on how tanh is formulated in the MINLP model. Existing literature offers several reformulations and methods, often called solution methods, for both cases, yet researchers have no common tool for implementing and comparing these solution methods.

We propose to build a Python package, hybopt, based on Pyomo, a Python-based algebraic modeling language for formulating and solving optimization problems, that embeds a trained ANN in an optimization model. hybopt will support two use cases: optimizing the ANN output directly - useful for applications where the ANN acts as a surrogate for the entire problem, or embedding the ANN as one constraint block inside a larger user-defined optimization problem. The package allows flexibility to a user to choose a solution method for each activation function.

**For ReLU networks**, we implement the following solution methods from Plate et al. (2026): big-M with interval-arithmetic bounds, big-M with LP-based bound tightening (OBBT), a-posteriori weight scaling, and a scaling-then-tightening method.

**For tanh networks**, hybopt will offer two families of solution methods. The first is the exact formulations of Schweidtmann and Mitsos (2019): full-space formulations with alternative algebraic rewrites of tanh, and a reduced-space formulation, solved to global optimality with SCIP. The second family is our extension beyond the source paper: approximation-based formulations of the tanh activation function.

- **Piecewise-linear (PWL) approximation.** Each tanh is approximated over its pre-activation bounds with an SOS2 or binary encoding, reformulating the ANN as an MILP formulation.
- **Piecewise-quadratic (PWQ) approximation.** Quadratic pieces are fitted separately on the convex and concave branch of the tanh function, reformulating the ANN as a mixed-integer quadratically constrained program (MIQCP) formulation.

In both cases the user can control the number of breakpoints or a target fit tolerance.

The project is also an exercise in research-software engineering. We will use a GitHub repository with a pull-request and review workflow, pytest suites that check formulation correctness, continuous integration through GitHub Actions, pre-commit quality checks, and documentation built with mkdocs alongside example notebooks.

**References**

- Plate, C., Hahn, M., Klimek, A., Ganzer, C., Sundmacher, K., and Sager, S. (2026). An analysis of optimization problems involving ReLU neural networks. *Optimization and Engineering*. https://doi.org/10.1007/s11081-026-10075-8

- Schweidtmann, A. M., and Mitsos, A. (2019). Deterministic global optimization with artificial neural networks embedded. *Journal of Optimization Theory and Applications*, 180, 925–948. https://doi.org/10.1007/s10957-018-1396-0