r"""
tanh embedding methods.

``tanh.pq`` relaxes every hidden neuron :math:`y = \tanh(x)`, :math:`x \in [l, u]`, by a two-sided
piecewise-quadratic enclosure whose pieces are chosen by binaries. Every constraint is convex, so
the model is a *convex* MIQCP: ``gurobi_direct`` solves it without the ``NonConvex`` option and
without ``exp``. The model is a **relaxation** of the network, not an exact encoding. Its optimum,
or better the solver's dual bound, is a valid bound on the true network optimum, and the bound
tightens as ``tol`` decreases.

.. plot:: tanh_pq_schematic.py
   :caption: How ``tanh.pq`` encloses each neuron's tanh in convex quadratics (Sections 2-4).

Notation: :math:`f = \tanh` throughout. Its derivatives are

.. math::

    f' = 1 - f^2, \qquad f'' = -2 f f', \qquad f''' = -2 f' (1 - 3 f^2).

Layers are :math:`k = 0` (inputs), :math:`1, \dots, K` (hidden) and :math:`K + 1` (outputs), as in
``hybopt.methods.relu``.

.. rubric:: 1. What the result guarantees

Let :math:`F` be the network and :math:`X` its input box. Section 4 shows that for every
:math:`x \in X` the true layer values, with each neuron's binary set to the piece that contains its
pre-activation, satisfy all constraints. So the feasible set :math:`R` of the model satisfies

.. math::

    \{(x, F(x)) : x \in X\} \subseteq \operatorname{proj}_{(x,\,y)} R
    \quad\Longrightarrow\quad
    \min_R y \le \min_{x \in X} F(x).

Use the solver's **dual bound** (``Solution.bound``), not the incumbent objective. The dual bound is
a valid bound on :math:`\min_R y` even if the search stops early, while the incumbent is only a
feasible point of :math:`R`. The relaxed optimizer :math:`x^*` is generally not the true one;
:math:`\text{bound} \le \min_X F \le F(x^*)`, so ``net.predict(x*)`` minus the bound brackets the
true optimum.

.. rubric:: 2. The quadratic enclosure of one piece

On a piece :math:`[a, b]` let :math:`L(x) = f(a) + m (x - a)`, with :math:`m = (f(b) - f(a))/(b - a)`,
be the chord. For a curvature :math:`\kappa` let

.. math::

    q_\kappa(x) = L(x) + \frac{\kappa}{2} (x - a)(x - b)
    = \frac{\kappa}{2} x^2 + \Big(m - \frac{\kappa}{2}(a + b)\Big) x
      + \Big(f(a) - m a + \frac{\kappa}{2} a b\Big)

(the monomial form is what Pyomo receives). Both terms are fixed by the endpoints, so
:math:`q_\kappa(a) = f(a)` and :math:`q_\kappa(b) = f(b)` for every :math:`\kappa`, and
:math:`q_\kappa'' \equiv \kappa`. This is **not** a Taylor expansion. A Taylor polynomial is exact at
one point and does not enclose :math:`f`. :math:`q_\kappa` is the chord plus a term that vanishes at
both endpoints, with :math:`\kappa` an *extreme* of :math:`f''` over the whole piece. With

.. math::

    \underline c = \min_{[a, b]} f'', \qquad \overline c = \max_{[a, b]} f'',

the two bounding quadratics are

.. math::

    q_{\mathrm{up}} = q_{\kappa_{\mathrm{up}}},\ \kappa_{\mathrm{up}} = \min(\underline c, 0) \le 0
    \ \text{(concave)}, \qquad
    q_{\mathrm{lo}} = q_{\kappa_{\mathrm{lo}}},\ \kappa_{\mathrm{lo}} = \max(\overline c, 0) \ge 0
    \ \text{(convex)}.

*Exact curvature range.* :math:`f''` is smooth, so its extremes on :math:`[a, b]` are attained at
:math:`a`, :math:`b`, or at interior zeros of :math:`f'''`. Since :math:`f' > 0`,
:math:`f'''(t) = 0 \iff f(t)^2 = 1/3 \iff t = \pm x^\star` with
:math:`x^\star = \operatorname{arctanh}(1/\sqrt 3) \approx 0.6585`, where
:math:`f''(\mp x^\star) = \pm 4/(3\sqrt 3) \approx \pm 0.7698` (the global extremes of :math:`f''`).
So :math:`\underline c` and :math:`\overline c` are the min and max of :math:`f''` over
:math:`\{a, b\} \cup (\{\pm x^\star\} \cap [a, b])`. No sampling is involved, and validity is exact
(``curvature_range``).

**Theorem (enclosure).** Let :math:`f \in C^2([a, b])`.

1. If :math:`\kappa \le \underline c`, then :math:`q_\kappa \ge f` on :math:`[a, b]`.
2. If :math:`\kappa \ge \overline c`, then :math:`q_\kappa \le f` on :math:`[a, b]`.

In particular :math:`q_{\mathrm{lo}} \le f \le q_{\mathrm{up}}` on :math:`[a, b]`, because
:math:`\kappa_{\mathrm{up}} \le \underline c` and :math:`\kappa_{\mathrm{lo}} \ge \overline c`.

*Proof A (concavity).* Case 1: let :math:`h = q_\kappa - f`. Then :math:`h(a) = h(b) = 0`, since both
functions pass through :math:`(a, f(a))` and :math:`(b, f(b))`. Also
:math:`h'' = \kappa - f'' \le \kappa - \underline c \le 0`, so :math:`h` is concave. Write
:math:`x = (1 - \lambda) a + \lambda b` with :math:`\lambda \in [0, 1]`; concavity gives
:math:`h(x) \ge (1 - \lambda) h(a) + \lambda h(b) = 0`. Case 2 is the same with
:math:`h = f - q_\kappa`, for which :math:`h'' = f'' - \kappa \le \overline c - \kappa \le 0`.
:math:`\blacksquare`

*Proof B (interpolation error).* **Lemma.** For every :math:`x \in [a, b]` there is
:math:`\xi \in [a, b]` with :math:`f(x) - L(x) = \tfrac12 f''(\xi)(x - a)(x - b)`.
At :math:`x = a` or :math:`x = b` both sides are zero. For :math:`x \in (a, b)` set
:math:`c = (f(x) - L(x)) / ((x - a)(x - b))` and :math:`g(t) = f(t) - L(t) - c (t - a)(t - b)`. Then
:math:`g(a) = g(x) = g(b) = 0`. Rolle's theorem on :math:`[a, x]` and :math:`[x, b]` gives zeros
:math:`\eta_1 < \eta_2` of :math:`g'`, and Rolle on :math:`[\eta_1, \eta_2]` gives a zero
:math:`\xi \in (a, b)` of :math:`g''`. Since :math:`L'' = 0`, :math:`g'' = f'' - 2c`, so
:math:`c = f''(\xi)/2`. :math:`\square`
With the lemma,

.. math::

    q_\kappa(x) - f(x) = \frac{\kappa - f''(\xi)}{2}\,\underbrace{(x - a)(x - b)}_{\le 0}.

In case 1 the first factor is :math:`\le 0`, so the product is :math:`\ge 0`. In case 2 it is
:math:`\ge 0`, so the product is :math:`\le 0`. :math:`\blacksquare` Because :math:`\xi` depends on
:math:`x` and is unknown, :math:`\kappa` must bound :math:`f''` over the whole piece.

*Clipping.* :math:`\kappa_{\mathrm{up}} \le 0` makes :math:`y \le q_{\mathrm{up}}(x)` convex, and
:math:`\kappa_{\mathrm{lo}} \ge 0` makes :math:`y \ge q_{\mathrm{lo}}(x)` convex. Clipping only moves
:math:`\kappa` further to the safe side, so the theorem still applies. Clipping costs tightness:
on a piece with :math:`b \le 0` (:math:`f'' \ge 0`, :math:`f` convex) :math:`\kappa_{\mathrm{up}} = 0`,
which is the chord. On a piece with :math:`a \ge 0` (:math:`f` concave) :math:`\kappa_{\mathrm{lo}} = 0`.
The chord is the best a convex constraint can do on that side. This is why 0 is a breakpoint
whenever :math:`l < 0 < u`. A piece that crossed 0 would still be valid, only looser, because both
clipped curvatures would then be nonzero.

*Band width.* :math:`q_{\mathrm{up}} - q_{\mathrm{lo}} = \frac{\kappa_{\mathrm{lo}} -
\kappa_{\mathrm{up}}}{2}(a - x)(x - b)` is a concave parabola vanishing at :math:`a` and :math:`b`.
Its maximum, at the midpoint, is

.. math::

    w = \frac{\kappa_{\mathrm{lo}} - \kappa_{\mathrm{up}}}{8} (b - a)^2
    \le \frac{(b - a)^2}{6 \sqrt 3},

because on a one-signed piece :math:`\kappa_{\mathrm{lo}} - \kappa_{\mathrm{up}}` is
:math:`\overline c` or :math:`-\underline c`, both :math:`\le 4/(3\sqrt3)`. Any feasible
:math:`(x, y)` on the piece has :math:`|y - f(x)| \le w` (``Piece.width``).

.. plot::
   :caption: The enclosure of :math:`\tanh` on :math:`[-3, 3]`, with the breakpoints dotted.
             Left: a coarse ``tanh_pieces(-3, 3, tol=0.2)``, so that the band between
             :math:`q_{\mathrm{lo}}` and :math:`q_{\mathrm{up}}` is visible. On :math:`x < 0` the
             upper bound is the chord, on :math:`x > 0` the lower bound is. Right:
             ``tanh_pieces(-3, 3, tol=0.02, max_pieces=50)``, drawn as the distance of each
             quadratic from :math:`\tanh`. The band is zero at every breakpoint, and its height
             (the gap between the two curves) is at most ``tol`` on every piece.

   import matplotlib.pyplot as plt
   import numpy as np

   from hybopt.methods.tanh import evaluate, tanh_pieces

   def draw(axis, pieces, relative):
       for piece in pieces:
           x = np.linspace(piece.a, piece.b, 200)
           shift = np.tanh(x) if relative else 0.0
           lower, upper = evaluate(piece.lower, x) - shift, evaluate(piece.upper, x) - shift
           axis.fill_between(x, lower, upper, color="tab:blue", alpha=0.3, linewidth=0)
           axis.plot(x, upper, color="tab:blue", linewidth=1)
           axis.plot(x, lower, color="tab:orange", linewidth=1)
       for point in [piece.a for piece in pieces] + [pieces[-1].b]:
           axis.axvline(point, color="gray", linestyle=":", linewidth=0.8)
       axis.set_xlabel("x")

   fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
   draw(left, tanh_pieces(-3.0, 3.0, tol=0.2), relative=False)
   x = np.linspace(-3.0, 3.0, 600)
   left.plot(x, np.tanh(x), color="black", linewidth=1.2, label="tanh")
   left.plot([], [], color="tab:blue", label=r"$q_{\mathrm{up}}$")
   left.plot([], [], color="tab:orange", label=r"$q_{\mathrm{lo}}$")
   left.legend(loc="upper left")
   left.set_title("tol = 0.2", fontsize=10)
   draw(right, tanh_pieces(-3.0, 3.0, tol=0.02, max_pieces=50), relative=True)
   right.axhline(0.0, color="black", linewidth=1.2)
   right.plot([], [], color="tab:blue", label=r"$q_{\mathrm{up}} - \tanh$")
   right.plot([], [], color="tab:orange", label=r"$q_{\mathrm{lo}} - \tanh$")
   right.legend(loc="upper left")
   right.set_title("tol = 0.02", fontsize=10)

.. rubric:: 3. Constraints per neuron

Binaries :math:`z_s`, one per piece :math:`s = 1, \dots, S` with endpoints :math:`[a_s, b_s]`:

.. math::

    \textstyle\sum_s z_s = 1, \qquad \sum_s a_s z_s \le x \le \sum_s b_s z_s,

    y \ge q_{\mathrm{lo}, s}(x) - M_{\mathrm{lo}, s}(1 - z_s), \qquad
    y \le q_{\mathrm{up}, s}(x) + M_{\mathrm{up}, s}(1 - z_s) \qquad \forall s,

with :math:`x \in [l, u]` and :math:`y \in [f(l), f(u)]`. With :math:`z = e_t` the second constraint
reads :math:`x \in [a_t, b_t]`, the constraints of piece :math:`t` read
:math:`q_{\mathrm{lo}, t} \le y \le q_{\mathrm{up}, t}`, and those of the other pieces must be
inactive. For :math:`s \ne t` the lower one never cuts off a point with :math:`y \ge f(l)` iff
:math:`M_{\mathrm{lo}, s} \ge \max_{[l, u]} q_{\mathrm{lo}, s} - f(l)`. A convex function attains its
maximum over an interval at an endpoint, and a concave one its minimum, so the smallest valid values
are

.. math::

    M_{\mathrm{lo}, s} = \max\big(0, \max\{q_{\mathrm{lo}, s}(l), q_{\mathrm{lo}, s}(u)\} - f(l)\big),
    \qquad
    M_{\mathrm{up}, s} = \max\big(0, f(u) - \min\{q_{\mathrm{up}, s}(l), q_{\mathrm{up}, s}(u)\}\big)

(``big_m``). Each constraint has the form (convex in :math:`x`) + (affine in :math:`z`)
:math:`\le` (affine in :math:`y`). So the feasible set is convex even for the continuous relaxation
:math:`z \in [0, 1]^S`, and every node of branch and bound is a convex QCP.

The pre-activation is a variable (``pre_activation[k, i]``) tied to the previous layer by a linear
constraint, so each quadratic term is :math:`\alpha x^2` in one variable. Squaring the affine
expression :math:`W_i x^{(k-1)} + b_i` instead would give a dense rank-one matrix, which Gurobi's
numerical PSD check can reject.

*Validity (graph containment).* For :math:`x \in [l, u]` pick :math:`t` with
:math:`x \in [a_t, b_t]`, and set :math:`z = e_t` and :math:`y = f(x)`. The piece constraints hold,
:math:`q_{\mathrm{lo}, t}(x) \le f(x) \le q_{\mathrm{up}, t}(x)` by the theorem, the other big-M
constraints hold by the choice of :math:`M`, and :math:`f(x) \in [f(l), f(u)]` because :math:`f` is
increasing. Applied layer by layer, with pre-activation bounds from interval arithmetic
(``hybopt.bounds``), this gives Section 1.

.. rubric:: 4. Choosing the pieces by bisection

``tanh_pieces`` starts from the breakpoints :math:`\{l, 0, u\}` (0 only if :math:`l < 0 < u`). It then
repeatedly bisects the piece with the largest band width :math:`w`, and stops when
:math:`\max_s w_s \le` ``tol`` or there are ``max_pieces`` pieces. The procedure is justified as
follows.

* **Validity does not depend on the breakpoints.** The theorem holds on every piece :math:`[a, b]`,
  so any breakpoints give a valid relaxation. Refinement only changes how tight it is.
* **Pieces stay one-signed.** The midpoint of a piece with :math:`b \le 0` or :math:`a \ge 0` stays
  on the same side of 0. Every piece therefore keeps the tight clipping of Section 2.
* **Each bisection shrinks the band by at least 4.** A half-width child :math:`[a', b'] \subset
  [a, b]` has a curvature range :math:`[\underline c', \overline c'] \subseteq [\underline c,
  \overline c]`, so :math:`\kappa_{\mathrm{lo}}' - \kappa_{\mathrm{up}}' \le \kappa_{\mathrm{lo}} -
  \kappa_{\mathrm{up}}` (min and max with 0 preserve the order). Also :math:`(b' - a')^2 = (b -
  a)^2/4`, so :math:`w' \le w/4`.
* **Greedy choice.** :math:`\max_s w_s` can only drop if the widest piece is split, since splitting
  any other piece leaves it in place. Splitting the widest piece is therefore the step that reduces
  the error bound :math:`\max_s w_s` the most. Bisecting at the midpoint, rather than at an
  optimized point, keeps the breakpoints nested (next point) and the closed form exact.
* **Termination and error bound.** :math:`w \le (b - a)^2 / (6\sqrt3)`, so every piece shorter than
  :math:`\sqrt{6\sqrt3\,\mathrm{tol}}` meets the tolerance and the loop ends after finitely many
  steps. When it stops on ``tol``, every feasible point has :math:`|y - \tanh x| \le` ``tol`` for
  each neuron. When it stops on ``max_pieces`` the error is :math:`\max_s w_s`.
* **Monotone tightening.** The choice of the next piece depends only on the current pieces, never on
  ``tol``. With the same ``max_pieces``, a smaller ``tol`` continues the same sequence of bisections,
  so its breakpoints are a superset. Nested pieces give nested enclosures: for :math:`[p, r] \subseteq
  [a, b]` let :math:`h = q_{\mathrm{up}}^{[a, b]} - q_{\mathrm{up}}^{[p, r]}`. Then :math:`h'' =
  \kappa_{\mathrm{up}}^{[a, b]} - \kappa_{\mathrm{up}}^{[p, r]} \le 0` (a minimum over a larger set
  is smaller), and :math:`h(p) = q_{\mathrm{up}}^{[a, b]}(p) - f(p) \ge 0` and :math:`h(r) \ge 0` by
  the theorem. So the concave :math:`h` is :math:`\ge 0` on :math:`[p, r]`, i.e. the sub-piece's upper
  bound is lower. The lower side is symmetric. Each neuron's set of feasible :math:`(x, y)`, which
  is the union of its pieces' bands, can only shrink. The interval bounds do not depend on ``tol``,
  so the network's feasible set shrinks too, and the bound can only improve as ``tol`` falls.

All of this is exact arithmetic. In floating point, the enclosure holds up to rounding
(:math:`\sim 10^{-16}`), well below the solver's feasibility tolerance.
"""

from dataclasses import dataclass

import numpy as np
import pyomo.environ as pyo

from hybopt.bounds import interval_bounds
from hybopt.net import FeedForwardNet

XSTAR = float(np.arctanh(1 / np.sqrt(3)))  # where tanh'' has its extremes (Section 2)
MAX_CURVATURE = 4 / (3 * np.sqrt(3))  # max |tanh''|, attained at -XSTAR and XSTAR
MIN_PIECE = 1e-6  # shorter intervals are padded, shorter pieces next to 0 are not split off


def tanh_second_derivative(t):
    r""":math:`\tanh''(t) = -2 \tanh t\,(1 - \tanh^2 t)`."""
    f = np.tanh(t)
    return -2 * f * (1 - f**2)


def curvature_range(a: float, b: float) -> tuple[float, float]:
    r"""
    Exact :math:`(\min, \max)` of :math:`\tanh''` over :math:`[a, b]`, from the endpoints and the
    critical points :math:`\pm x^\star` inside (Section 2).
    """
    points = [a, b] + [t for t in (-XSTAR, XSTAR) if a < t < b]
    values = tanh_second_derivative(np.array(points))
    return float(values.min()), float(values.max())


@dataclass(frozen=True)
class Piece:
    r"""
    The enclosure :math:`q_{\mathrm{lo}} \le \tanh \le q_{\mathrm{up}}` on :math:`[a, b]`, with
    curvatures kappa_up :math:`\le 0` (concave upper quadratic) and kappa_lo :math:`\ge 0` (convex
    lower quadratic).
    """

    a: float
    b: float
    kappa_up: float
    kappa_lo: float

    @classmethod
    def make(cls, a: float, b: float) -> "Piece":
        """The enclosure on [a, b] with the clipped extreme curvatures."""
        c_min, c_max = curvature_range(a, b)
        return cls(float(a), float(b), min(c_min, 0.0), max(c_max, 0.0))

    def coefficients(self, kappa: float) -> tuple[float, float, float]:
        r"""(alpha, beta, gamma) of :math:`q_\kappa(x) = \alpha x^2 + \beta x + \gamma`."""
        a, b = self.a, self.b
        fa = float(np.tanh(a))
        slope = (float(np.tanh(b)) - fa) / (b - a)
        alpha = kappa / 2
        return alpha, slope - alpha * (a + b), fa - slope * a + alpha * a * b

    @property
    def upper(self) -> tuple[float, float, float]:
        """Coefficients of the concave upper quadratic."""
        return self.coefficients(self.kappa_up)

    @property
    def lower(self) -> tuple[float, float, float]:
        """Coefficients of the convex lower quadratic."""
        return self.coefficients(self.kappa_lo)

    @property
    def width(self) -> float:
        """Largest gap between the upper and lower quadratic, at the midpoint (Section 2)."""
        return (self.kappa_lo - self.kappa_up) * (self.b - self.a) ** 2 / 8


def evaluate(coefficients, x):
    """The quadratic with (alpha, beta, gamma) = coefficients at x."""
    alpha, beta, gamma = coefficients
    return alpha * x**2 + beta * x + gamma


def tanh_pieces(lower: float, upper: float, tol: float = 0.02, max_pieces: int = 12) -> list[Piece]:
    """
    Pieces covering [lower, upper]: split at 0, then bisect the piece with the widest band until
    every band is at most ``tol`` or there are ``max_pieces`` pieces (Section 4).

    Intervals shorter than MIN_PIECE are padded, which keeps them valid (they only get wider).
    """
    if not tol > 0:
        raise ValueError(f"tol must be positive, got {tol}")
    if int(max_pieces) != max_pieces or max_pieces < 1:
        raise ValueError(f"max_pieces must be a positive integer, got {max_pieces}")
    lower, upper = float(lower), float(upper)
    if upper - lower < MIN_PIECE:
        middle = (lower + upper) / 2
        lower, upper = middle - MIN_PIECE / 2, middle + MIN_PIECE / 2
    breakpoints = [lower, upper]
    if max_pieces >= 2 and lower < -MIN_PIECE and upper > MIN_PIECE:
        breakpoints.insert(1, 0.0)
    pieces = [Piece.make(a, b) for a, b in zip(breakpoints[:-1], breakpoints[1:])]
    while len(pieces) < max_pieces:
        widths = [piece.width for piece in pieces]
        s = int(np.argmax(widths))
        if widths[s] <= tol:
            break
        a, b = pieces[s].a, pieces[s].b
        middle = (a + b) / 2
        pieces[s : s + 1] = [Piece.make(a, middle), Piece.make(middle, b)]
    return pieces


def big_m(piece: Piece, lower: float, upper: float) -> tuple[float, float]:
    """
    (M_lo, M_up) for ``piece``: the smallest values that make its constraints inactive for every
    x in [lower, upper] and y in [tanh(lower), tanh(upper)] (Section 3).
    """
    ends = np.array([lower, upper])
    m_lo = max(0.0, float(evaluate(piece.lower, ends).max() - np.tanh(lower)))
    m_up = max(0.0, float(np.tanh(upper) - evaluate(piece.upper, ends).min()))
    return m_lo, m_up


def quadratic(coefficients, x):
    """The Pyomo expression of the quadratic, linear when alpha is 0 (a chord)."""
    alpha, beta, gamma = coefficients
    if alpha == 0.0:
        return beta * x + gamma
    return alpha * x**2 + beta * x + gamma


def add_pq(
    block: pyo.Block, net: FeedForwardNet, bounds, tol: float = 0.02, max_pieces: int = 12
) -> None:
    r"""
    Add the piecewise-quadratic relaxation of ``net`` to ``block``, using the pre-activation
    ``bounds`` (one (L, U) pair per layer, see ``hybopt.bounds``) for the pieces and big-Ms.

    x[k, i]                          : inputs (k = 0, within the box) and hidden-layer outputs
                                       :math:`\approx \tanh` (pre-activation), within
                                       :math:`[\tanh L, \tanh U]`.
    pre_activation[k, i]             : variable, :math:`W^{(k)}_i x^{(k-1)} + b^{(k)}_i` (constraint
                                       linear[k, i]) within [L, U], for hidden layers.
    y[i]                             : outputs, :math:`W^{(K+1)} x^{(K)} + b^{(K+1)}` (constraint
                                       output[i]), within the output-layer bounds.
    z[k, i, s]                       : binary, 1 if the pre-activation lies in piece s.
    one_piece[k, i]                  : exactly one piece is chosen.
    in_piece_lower/upper[k, i]       : the pre-activation lies in the chosen piece.
    above_lower_quadratic[k, i, s]   : :math:`x \ge q_{\mathrm{lo}, s}(\text{pre}) - M (1 - z)`.
    below_upper_quadratic[k, i, s]   : :math:`x \le q_{\mathrm{up}, s}(\text{pre}) + M (1 - z)`.

    block.tanh_pieces maps each hidden neuron (k, i) to its list of ``Piece``.
    """
    sizes = net.layer_sizes
    n_hidden = len(sizes) - 2
    hidden = [(k, i) for k in range(1, n_hidden + 1) for i in range(sizes[k])]
    lower = {(k, i): float(bounds[k - 1][0][i]) for k, i in hidden}
    upper = {(k, i): float(bounds[k - 1][1][i]) for k, i in hidden}
    pieces = {n: tanh_pieces(lower[n], upper[n], tol, max_pieces) for n in hidden}
    # big-Ms over the interval the pieces cover (padded if degenerate), which contains [L, U]
    big_ms = {
        (*n, s): big_m(piece, pieces[n][0].a, pieces[n][-1].b)
        for n in hidden
        for s, piece in enumerate(pieces[n])
    }

    def x_bounds(b, k, i):
        if k == 0:
            return float(net.lower_bound[i]), float(net.upper_bound[i])
        return float(np.tanh(lower[k, i])), float(np.tanh(upper[k, i]))

    def affine(b, k, i):
        w, bias = net.weights[k - 1][i], net.biases[k - 1][i]
        return sum(w[j] * b.x[k - 1, j] for j in range(sizes[k - 1])) + bias

    output_lower, output_upper = bounds[-1]
    block.tanh_pieces = pieces
    block.x = pyo.Var(
        [(k, i) for k in range(n_hidden + 1) for i in range(sizes[k])], bounds=x_bounds
    )
    block.pre_activation = pyo.Var(hidden, bounds=lambda b, k, i: (lower[k, i], upper[k, i]))
    block.y = pyo.Var(
        range(net.n_outputs),
        bounds=lambda b, i: (float(output_lower[i]), float(output_upper[i])),
    )
    block.pieces = pyo.Set(initialize=list(big_ms), dimen=3, ordered=True)
    block.z = pyo.Var(block.pieces, within=pyo.Binary)

    block.linear = pyo.Constraint(
        hidden, rule=lambda b, k, i: b.pre_activation[k, i] == affine(b, k, i)
    )
    block.output = pyo.Constraint(
        range(net.n_outputs), rule=lambda b, i: b.y[i] == affine(b, n_hidden + 1, i)
    )
    block.one_piece = pyo.Constraint(
        hidden,
        rule=lambda b, k, i: sum(b.z[k, i, s] for s in range(len(pieces[k, i]))) == 1,
    )
    block.in_piece_lower = pyo.Constraint(
        hidden,
        rule=lambda b, k, i: b.pre_activation[k, i]
        >= sum(piece.a * b.z[k, i, s] for s, piece in enumerate(pieces[k, i])),
    )
    block.in_piece_upper = pyo.Constraint(
        hidden,
        rule=lambda b, k, i: b.pre_activation[k, i]
        <= sum(piece.b * b.z[k, i, s] for s, piece in enumerate(pieces[k, i])),
    )
    block.above_lower_quadratic = pyo.Constraint(
        block.pieces,
        rule=lambda b, k, i, s: b.x[k, i]
        >= quadratic(pieces[k, i][s].lower, b.pre_activation[k, i])
        - big_ms[k, i, s][0] * (1 - b.z[k, i, s]),
    )
    block.below_upper_quadratic = pyo.Constraint(
        block.pieces,
        rule=lambda b, k, i, s: b.x[k, i]
        <= quadratic(pieces[k, i][s].upper, b.pre_activation[k, i])
        + big_ms[k, i, s][1] * (1 - b.z[k, i, s]),
    )


def pq(block: pyo.Block, net: FeedForwardNet, tol: float = 0.02, max_pieces: int = 12) -> None:
    """
    tanh.pq: convex piecewise-quadratic MIQCP relaxation with interval-arithmetic bounds. Solve with
    ``gurobi_direct``; the dual bound bounds the true optimum (see the module docstring).

    tol        : Largest band width per neuron; smaller is tighter but needs more binaries.
    max_pieces : Most pieces (binaries) per neuron.
    """
    add_pq(block, net, interval_bounds(net), tol, max_pieces)
