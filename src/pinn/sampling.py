"""Collocation-point samplers for the common domain shapes.

Each factory returns a sampler with the signature the trainer expects:

    PDE / BC samplers:  (key, n, t0, t1) -> (n, coord_dim)
    IC samplers:        (key, n, t0)     -> (n, coord_dim)

For steady problems the trainer passes ``t0 = t1 = 0``; the space-only
samplers accept and ignore those trailing arguments.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey


def interior(x_min: float, x_max: float):
    """Uniform points over a 1D interval × time. Returns [x, t]."""
    def sample(key: PRNGKey, n: int, t0: float, t1: float) -> Array:
        kx, kt = jax.random.split(key)
        x = jax.random.uniform(kx, (n,), minval=x_min, maxval=x_max)
        t = jax.random.uniform(kt, (n,), minval=t0, maxval=t1)
        return jnp.stack([x, t], axis=1)
    return sample


def initial_line(x_min: float, x_max: float):
    """Evenly spaced spatial points at the slab start time. Returns [x, t0]."""
    def sample(key: PRNGKey, n: int, t0: float) -> Array:
        x = jnp.linspace(x_min, x_max, n, endpoint=False)
        t = jnp.full((n,), t0)
        return jnp.stack([x, t], axis=1)
    return sample


def dirichlet_walls(x_min: float, x_max: float):
    """Points on the two spatial ends x=x_min and x=x_max across time."""
    def sample(key: PRNGKey, n: int, t0: float, t1: float) -> Array:
        half = n // 2
        t = jnp.linspace(t0, t1, half)
        left  = jnp.stack([jnp.full((half,), x_min), t], axis=1)
        right = jnp.stack([jnp.full((half,), x_max), t], axis=1)
        return jnp.concatenate([left, right], axis=0)
    return sample


def box(x_min: float, x_max: float, dim: int):
    """Uniform interior points on the cube [x_min, x_max]^dim (steady)."""
    def sample(key: PRNGKey, n: int, *_) -> Array:
        return jax.random.uniform(key, (n, dim), minval=x_min, maxval=x_max)
    return sample


def boundary_faces(x_min: float, x_max: float, dim: int):
    """Random points on ∂[x_min, x_max]^dim: pin one coordinate to a face."""
    def sample(key: PRNGKey, n: int, *_) -> Array:
        k_pt, k_dim, k_face = jax.random.split(key, 3)
        x = jax.random.uniform(k_pt, (n, dim), minval=x_min, maxval=x_max)
        face_dim = jax.random.randint(k_dim, (n,), minval=0, maxval=dim)
        faces    = jnp.array([x_min, x_max])
        face_val = faces[jax.random.randint(k_face, (n,), minval=0, maxval=2)]
        return x.at[jnp.arange(n), face_dim].set(face_val)
    return sample
