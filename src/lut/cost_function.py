import math
import numpy as np

from numba import njit
from scipy.special import xlogy # x ln x with 0 ln 0 = 0

# Cost of the static optimization (notes):
#     J_p(u) = sum_i J_i(u_i),    J_i(u) = (1 - W) u^p + W u ln u
# J_i is the same for every muscle and depends only on (p, W).
#
# To change the cost, change ONLY this file and keep these functions consistent with each other:
#   - cost_terms       J_i split in its terms (numpy): check_solver, plots
#   - cost_derivative  J_i'(u)                (numpy): jacobian of check_solver, KKT checks
#   - u_from_y         argmin_u J_i(u) - y u  (numba): the solver, called with y = lam * K_i
# The solver in lut_generator works for every cost that is separable and strictly convex.


# ----- numpy: plots and checks -----
def cost_terms(u, W: float, p: float) -> dict:
    """The terms of J_i with their label for the plots: {label: values}."""
    u = np.asarray(u, dtype=float)
    return {rf"$(1-W)\,u^{{{p:g}}}$": (1.0 - W) * u ** p,
            r"$W\,u\ln u$": W * xlogy(u, u)}


def cost_i(u, W: float, p: float):
    """J_i(u) for every element of u."""
    return sum(cost_terms(u, W, p).values())


def cost_total(u, W: float, p: float):
    """J_p = sum_i J_i(u_i): sum over the last axis (the muscles). u (..., n_muscles) -> (...)."""
    return np.sum(cost_i(u, W, p), axis=-1)


def cost_derivative(u, W: float, p: float):
    """J_i'(u) = p (1 - W) u^(p-1) + W (ln u + 1): left side of the 6 equations of the notes.
    -inf at u = 0 when W > 0: that is why no muscle is ever fully off."""
    u = np.asarray(u, dtype=float)
    with np.errstate(divide="ignore"):
        entropy = W * (np.log(u) + 1.0) if W > 0 else np.zeros_like(u)
    return p * (1.0 - W) * u ** (p - 1) + entropy


# ----- numba: the solver -----
@njit
def u_from_y(y, W, p):
    """argmin over u in [0, 1] of J_i(u) - y u: the u that solves J_i'(u) = y, clipped to [0, 1].
    The solver calls it with y = lam * K_i: the cost needs only the product, not lam and K_i."""
    a = p * (1 - W)

    if y >= a + W: # J_i'(1) = p(1-W) + W: the muscle saturates
        return 1.0
    if W == 0: # W = 0:  p u^(p-1) = y  ->  u = (y/p)^(1/(p-1));  if y <= 0 the muscle is off
        return (y / p) ** (1 / (p - 1)) if y > 0 else 0.0
    if W == 1: # W = 1:  ln u + 1 = y  ->  u = e^(y-1)
        return math.exp(y - 1)

    # 0 < W < 1: no closed form, Newton on s = ln u (so u = e^s, u^(p-1) = e^((p-1)s)):
    #   g(s) = a e^((p-1)s) + W (s + 1) - y   is increasing and convex in s,
    #   so Newton started where g > 0 moves only to the left and never jumps past the zero
    s = min(0.0, (y - W) / W) # g > 0 at both points: at 0 because y < a + W, at (y-W)/W because a e^(..) > 0
    for _ in range(100):
        e = a * math.exp((p - 1) * s)
        step = (e + W * (s + 1) - y) / ((p - 1) * e + W) # g / g'
        s -= step
        if step < 1e-14 * (1.0 + abs(s)): # relative: works for any size of s
            break
    return math.exp(s)


def u_rest(W: float, p: float) -> float:
    """Minimum of J_i (y = 0, i.e. lam = 0): baseline co-contraction, the same for every muscle."""
    return u_from_y(0.0, float(W), float(p))