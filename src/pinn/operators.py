"""Pointwise derivative helpers for writing PDE residuals.

A residual acts on a single coordinate vector ``coords`` (shape ``(dim,)``);
the optimiser vmaps it over a batch. ``u`` is a scalar field written as a
plain function of that vector, e.g. ``u = lambda c: model(c)[0]``. The helpers
below then read like the maths:

    u_t  = grad(u, coords, t)
    u_x  = grad(u, coords, x)
    u_xx = grad2(u, coords, x)
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array


def diff(f: Callable) -> Callable:
    """Derivative of a scalar-to-scalar function (forward mode)."""
    return lambda x: jax.jvp(f, (x,), (jnp.ones_like(x),))[1]


def _along(u: Callable, coords: Array, axis: int) -> Callable:
    """u restricted to varying only ``coords[axis]``."""
    return lambda xi: u(coords.at[axis].set(xi))


def grad(u: Callable, coords: Array, axis: int) -> Array:
    """First derivative ∂u/∂x_axis at ``coords``."""
    return diff(_along(u, coords, axis))(coords[axis])


def grad2(u: Callable, coords: Array, axis: int) -> Array:
    """Second derivative ∂²u/∂x_axis² at ``coords``."""
    return diff(diff(_along(u, coords, axis)))(coords[axis])


def grad_n(u: Callable, coords: Array, axis: int, order: int) -> Array:
    """n-th derivative ∂ⁿu/∂x_axisⁿ at ``coords``."""
    g = _along(u, coords, axis)
    for _ in range(order):
        g = diff(g)
    return g(coords[axis])


def laplacian(u: Callable, coords: Array, axes: tuple[int, ...] | None = None) -> Array:
    """Sum of second derivatives over ``axes`` (all axes by default)."""
    if axes is None:
        axes = range(coords.shape[0])
    return sum(grad2(u, coords, a) for a in axes)
