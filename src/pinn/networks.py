"""Network architectures for PINN training.

Four families, all sharing the same call convention — construct with
``(key, problem, ...)``, call with a single coordinate vector ``(n_inputs,)``
and get back ``(n_outputs,)``:

    MLP       tanh multilayer perceptron with per-layer activation scales
    SIREN     sinusoidal network, optional learnable ω and linear skip
    GaborNet  learnable Gabor wavelet features (one GEMM-safe layer)
    SPINN     separable PINN — one small branch network per input dimension

Every architecture takes ``time_dependent`` (default True).  When True the
last input coordinate is treated as time: it is passed through the embedding
unchanged while the spatial coordinates get the periodic / Fourier features.
Set it to False for steady problems so all inputs are embedded as spatial.

The two embedding options (mutually exclusive, Fourier wins):
    periodic_bc=True   [cos(2πx̃), sin(2πx̃)] per spatial dim — exact
                       periodicity across the domain
    n_fourier=M        M cos/sin pairs per spatial dim — linear modes
                       1..M for time-dependent problems, geometric modes
                       1,2,4,..,2^(M-1) for steady ones
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray as PRNGKey

_TANH_GAIN = 5.0 / 3.0


# ── Initialisers ──────────────────────────────────────────────────────────────

def _ortho_init(key, shape, gain: float = 1.0):
    return jax.nn.initializers.orthogonal(scale=gain)(key, shape)

def _lecun_init(key, shape):
    fan_in = shape[1]
    return jax.random.normal(key, shape) * (1.0 / fan_in) ** 0.5

def _lecun_uniform(key, shape):
    fan_in = shape[1]
    limit  = (3.0 / fan_in) ** 0.5
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)

def _siren_first(key, shape):
    fan_in = shape[1]
    limit  = 1.0 / fan_in
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)

def _siren_hidden(key, shape, omega):
    fan_in = shape[1]
    limit  = (6.0 / fan_in) ** 0.5 / omega
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)

def _count_params(model) -> int:
    return sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )


# ── Shared input embedding ────────────────────────────────────────────────────

def _domain_bounds(problem, n_inputs: int) -> tuple[tuple, tuple]:
    """Per-dimension (x_mins, x_maxs) from the problem's scalar or vector bounds."""
    def _get(attr):
        val = getattr(problem, attr)
        if hasattr(val, "__len__"):
            return tuple(float(v) for v in val)
        return tuple(float(val) for _ in range(n_inputs))
    return _get("x_min"), _get("x_max")


def _embed_dim(n_inputs, n_fourier, periodic_bc, time_dependent) -> tuple[int, str]:
    """Embedding width and a short label for the construction log."""
    n_spatial = n_inputs - 1 if time_dependent else n_inputs
    n_time    = 1 if time_dependent else 0
    if n_fourier > 0 and n_spatial > 0:
        return 2 * n_fourier * n_spatial + n_time, f"Fourier [M={n_fourier}]"
    if periodic_bc and n_spatial > 0:
        label = "periodic [cos/sin + t]" if time_dependent else "periodic [cos/sin]"
        return 2 * n_spatial + n_time, label
    return n_inputs, f"raw {n_inputs}D"


def _embed(coords, x_mins, x_maxs, n_inputs, n_fourier, periodic_bc,
           time_dependent) -> Array:
    """Map raw coordinates to the chosen feature embedding."""
    n_spatial = n_inputs - 1 if time_dependent else n_inputs
    if n_spatial <= 0 or (n_fourier == 0 and not periodic_bc):
        return coords

    spatial = coords[:n_spatial]
    x_mins  = jnp.array(x_mins[:n_spatial])
    x_maxs  = jnp.array(x_maxs[:n_spatial])
    x_norm  = (spatial - x_mins) / (x_maxs - x_mins)

    if n_fourier > 0:
        if time_dependent:
            modes = jnp.arange(1, n_fourier + 1, dtype=coords.dtype)
        else:
            modes = 2 ** jnp.arange(0, n_fourier, dtype=coords.dtype)
        theta   = 2.0 * jnp.pi * x_norm[:, None] * modes[None, :]
        fourier = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1).reshape(-1)
        if time_dependent:
            return jnp.concatenate([coords[-1:], fourier])
        return fourier

    theta = 2.0 * jnp.pi * x_norm
    if time_dependent:
        return jnp.concatenate([jnp.cos(theta), jnp.sin(theta), coords[-1:]])
    return jnp.concatenate([jnp.cos(theta), jnp.sin(theta)])


# ──────────────────────────────────────────────────────────────────────────────
# MLP
# ──────────────────────────────────────────────────────────────────────────────

class MLP(eqx.Module):
    """Tanh MLP with a trainable activation scale per hidden layer."""

    # ── Trainable ─────────────────────────────────────────────────────────
    weights:    tuple   # (n_layers + 1) × (out, in)
    biases:     tuple   # (n_layers + 1) × (out,)
    act_scales: tuple   # n_layers × ()   — scalar per hidden layer

    # ── Static ────────────────────────────────────────────────────────────
    x_mins:         tuple = eqx.field(static=True)
    x_maxs:         tuple = eqx.field(static=True)
    periodic_bc:    bool  = eqx.field(static=True)
    n_inputs:       int   = eqx.field(static=True)
    n_fourier:      int   = eqx.field(static=True)
    time_dependent: bool  = eqx.field(static=True)

    def __init__(
        self,
        key:            PRNGKey,
        problem:        type,
        hidden_dims:    tuple[int, ...],
        n_inputs:       int  = 2,
        n_outputs:      int  = 1,
        periodic_bc:    bool = False,
        n_fourier:      int  = 0,
        time_dependent: bool = True,
        **kwargs,
    ):
        self.n_inputs       = n_inputs
        self.n_fourier      = n_fourier
        self.periodic_bc    = periodic_bc
        self.time_dependent = time_dependent
        self.x_mins, self.x_maxs = _domain_bounds(problem, n_inputs)

        embed_dim, embed_label = _embed_dim(
            n_inputs, n_fourier, periodic_bc, time_dependent)

        all_dims = (embed_dim, *hidden_dims, n_outputs)
        n_layers = len(hidden_dims)
        gain       = _TANH_GAIN if time_dependent else 1.0
        bias_scale = 0.01 if time_dependent else 0.05

        k_w, k_b = jax.random.split(key)
        w_keys   = jax.random.split(k_w, len(all_dims) - 1)
        b_keys   = jax.random.split(k_b, len(all_dims) - 1)

        weights = []
        for i, (wk, in_d, out_d) in enumerate(
            zip(w_keys, all_dims[:-1], all_dims[1:])
        ):
            is_last = i == n_layers
            W = (
                _lecun_init(wk, (out_d, in_d))
                if is_last
                else _ortho_init(wk, (out_d, in_d), gain=gain)
            )
            weights.append(W)
        self.weights = tuple(weights)

        # Tiny bias noise for symmetry breaking
        self.biases = tuple(
            jax.random.normal(bk, (out_d,)) * bias_scale
            for bk, out_d in zip(b_keys, all_dims[1:])
        )

        self.act_scales = tuple(jnp.ones(()) for _ in range(n_layers))

        n_params = _count_params(self)
        arch_str = "→".join(str(d) for d in hidden_dims)
        print(f"[MLP] {embed_label} | {arch_str}→{n_outputs} | {n_params:,} params")

    def _embed(self, coords: Array) -> Array:
        return _embed(coords, self.x_mins, self.x_maxs, self.n_inputs,
                      self.n_fourier, self.periodic_bc, self.time_dependent)

    def __call__(self, coords: Array) -> Array:
        """coords : (n_inputs,)  →  (n_outputs,)"""
        h = self._embed(coords)
        for W, b, a in zip(self.weights[:-1], self.biases[:-1], self.act_scales):
            h = jax.nn.tanh(a * (W @ h + b))
        return self.weights[-1] @ h + self.biases[-1]


# ──────────────────────────────────────────────────────────────────────────────
# SIREN
# ──────────────────────────────────────────────────────────────────────────────
#
# Sitzmann et al., "Implicit Neural Representations with Periodic Activation
# Functions", NeurIPS 2020.  Two additions over the vanilla network, each
# independently toggleable:
#
#   learnable_omega — ω per layer is trained as exp(log ω), so the network
#       tunes its own frequency rather than relying on a hand-picked ω₀.
#   skip_connection — a direct linear path from the embedding to the output,
#       which helps when the solution has a significant low-frequency
#       component the sin layers struggle with at initialisation.

class SIRENLayer(eqx.Module):
    W:         Array
    b:         Array
    log_omega: Array              # scalar — exp(log_omega) = ω > 0
    fixed:     bool = eqx.field(static=True)   # if True, log_omega is not trained

    def __init__(self, key, in_dim, out_dim, omega, is_first, learnable_omega):
        wk, bk     = jax.random.split(key)
        self.W     = _siren_first(wk, (out_dim, in_dim)) if is_first \
                     else _siren_hidden(wk, (out_dim, in_dim), omega)
        self.b     = jax.random.uniform(bk, (out_dim,), minval=-1., maxval=1.) \
                     if is_first else jax.random.normal(bk, (out_dim,)) * 0.01
        self.log_omega = jnp.log(jnp.array(omega))
        self.fixed     = not learnable_omega

    def __call__(self, h: Array) -> Array:
        omega = jax.lax.stop_gradient(jnp.exp(self.log_omega)) \
                if self.fixed else jnp.exp(self.log_omega)
        return jnp.sin(omega * (self.W @ h + self.b))


class SIREN(eqx.Module):
    """SIREN with optional learnable ω and optional linear skip."""

    # ── Trainable ─────────────────────────────────────────────────────────
    layers: tuple
    W_out:  Array
    W_skip: Array | None    # (n_outputs, embed_dim) or None
    b_out:  Array

    # ── Static ────────────────────────────────────────────────────────────
    x_mins:          tuple = eqx.field(static=True)
    x_maxs:          tuple = eqx.field(static=True)
    periodic_bc:     bool  = eqx.field(static=True)
    n_inputs:        int   = eqx.field(static=True)
    n_fourier:       int   = eqx.field(static=True)
    skip_connection: bool  = eqx.field(static=True)
    time_dependent:  bool  = eqx.field(static=True)

    def __init__(
        self,
        key:              PRNGKey,
        problem:          type,
        hidden_dims:      tuple[int, ...],
        n_inputs:         int   = 2,
        n_outputs:        int   = 1,
        periodic_bc:      bool  = False,
        omega_0:          float = 1.0,
        omega_hidden:     float = 1.0,
        n_fourier:        int   = 0,
        learnable_omega:  bool  = True,
        skip_connection:  bool  = False,
        time_dependent:   bool  = True,
        **kwargs,
    ):
        self.n_inputs        = n_inputs
        self.n_fourier       = n_fourier
        self.periodic_bc     = periodic_bc
        self.skip_connection = skip_connection
        self.time_dependent  = time_dependent
        self.x_mins, self.x_maxs = _domain_bounds(problem, n_inputs)

        embed_dim, embed_label = _embed_dim(
            n_inputs, n_fourier, periodic_bc, time_dependent)

        n_layers  = len(hidden_dims)
        k_layers  = jax.random.split(key, n_layers + 2)
        k_out     = k_layers[-2]
        k_skip    = k_layers[-1]
        k_layers  = k_layers[:-2]

        all_in_dims = (embed_dim, *hidden_dims[:-1])
        self.layers = tuple(
            SIRENLayer(
                k_layers[i],
                in_dim  = all_in_dims[i],
                out_dim = hidden_dims[i],
                omega   = omega_0 if i == 0 else omega_hidden,
                is_first    = (i == 0),
                learnable_omega = learnable_omega,
            )
            for i in range(n_layers)
        )

        self.W_out  = _lecun_uniform(k_out, (n_outputs, hidden_dims[-1]))
        self.b_out  = jnp.zeros(n_outputs)
        self.W_skip = _lecun_uniform(k_skip, (n_outputs, embed_dim)) \
                      if skip_connection else None

        n_params = _count_params(self)
        arch_str = "→".join(str(d) for d in hidden_dims)
        flags    = []
        if learnable_omega:  flags.append("learnable ω")
        if skip_connection:  flags.append("linear skip")
        flag_str = ", ".join(flags) if flags else "vanilla"
        print(
            f"[SIREN] {embed_label} | "
            f"ω₀={omega_0} ω_h={omega_hidden} | "
            f"{flag_str} | "
            f"embed({embed_dim})→{arch_str}→{n_outputs} | "
            f"{n_params:,} params"
        )

    def _embed(self, coords: Array) -> Array:
        return _embed(coords, self.x_mins, self.x_maxs, self.n_inputs,
                      self.n_fourier, self.periodic_bc, self.time_dependent)

    def __call__(self, coords: Array) -> Array:
        e = self._embed(coords)
        h = e
        for layer in self.layers:
            h = layer(h)
        out = self.W_out @ h + self.b_out
        if self.skip_connection:
            out = out + self.W_skip @ e
        return out


# ──────────────────────────────────────────────────────────────────────────────
# GaborNet
# ──────────────────────────────────────────────────────────────────────────────
#
# A Gabor filter is a sinusoid modulated by a Gaussian envelope:
#
#     g(e; μ, ω, σ) = exp(−‖e − μ‖² / 2σ²) · cos(ωᵀe + φ)
#
# with the centres μ, frequencies ω, bandwidths σ and phases φ all trained.
# Unlike random Fourier features (frequencies drawn once and frozen), the
# network concentrates its filters on the frequencies and regions that
# matter for the specific PDE.
#
# The envelope is expanded as ‖e‖² − 2μᵀe + ‖μ‖² so the whole forward pass
# is matrix-vector products — no (n_gabor × embed_dim) broadcast appears,
# which keeps memory flat in high dimension.  ``anisotropic=True`` gives
# each filter a per-dimension bandwidth via the same factorisation.

class GaborNet(eqx.Module):
    """Learnable Gabor wavelet feature network.

    GaborNet-specific options (the rest follow MLP / SIREN):
        n_gabor         — number of Gabor filters (feature dimension)
        omega_scale     — std of the initial frequency vectors
        sigma_init      — initial envelope width, in normalised embed space
        anisotropic     — per-filter per-dimension bandwidth
        learnable_mu    — train the filter centres (almost always helps)
        skip_connection — add a linear W_skip @ e term
        drop_rate       — feature dropout during training (0 = off)
    """

    # ── Trainable ─────────────────────────────────────────────────────────
    mu:        Array        # (n_gabor, embed_dim)  — filter centres
    log_sigma: Array        # (n_gabor,) isotropic | (n_gabor, embed_dim) anisotropic
    Omega:     Array        # (n_gabor, embed_dim)  — frequency vectors
    phi:       Array        # (n_gabor,)            — phase offsets
    W_out:     Array        # (n_outputs, n_gabor)
    b_out:     Array        # (n_outputs,)
    W_skip:    Array | None # (n_outputs, embed_dim) or None

    # ── Static ────────────────────────────────────────────────────────────
    x_mins:         tuple = eqx.field(static=True)
    x_maxs:         tuple = eqx.field(static=True)
    periodic_bc:    bool  = eqx.field(static=True)
    n_inputs:       int   = eqx.field(static=True)
    n_fourier:      int   = eqx.field(static=True)
    n_gabor:        int   = eqx.field(static=True)
    anisotropic:    bool  = eqx.field(static=True)
    learnable_mu:   bool  = eqx.field(static=True)
    skip_connection:bool  = eqx.field(static=True)
    drop_rate:      float = eqx.field(static=True)
    time_dependent: bool  = eqx.field(static=True)

    def __init__(
        self,
        key:            PRNGKey,
        problem:        type,
        n_gabor:        int   = 512,
        n_inputs:       int   = 2,
        n_outputs:      int   = 1,
        periodic_bc:    bool  = False,
        n_fourier:      int   = 0,
        omega_scale:    float = 1.0,
        sigma_init:     float = 0.5,
        anisotropic:    bool  = False,
        learnable_mu:   bool  = True,
        skip_connection:bool  = False,
        drop_rate:      float = 0.0,
        time_dependent: bool  = True,
        **kwargs,
    ):
        self.n_gabor         = n_gabor
        self.n_inputs        = n_inputs
        self.n_fourier       = n_fourier
        self.periodic_bc     = periodic_bc
        self.anisotropic     = anisotropic
        self.learnable_mu    = learnable_mu
        self.skip_connection = skip_connection
        self.drop_rate       = drop_rate
        self.time_dependent  = time_dependent
        self.x_mins, self.x_maxs = _domain_bounds(problem, n_inputs)

        embed_dim, embed_label = _embed_dim(
            n_inputs, n_fourier, periodic_bc, time_dependent)

        k_mu, k_omega, k_phi, k_out, k_skip = jax.random.split(key, 5)

        # Filter centres, uniform over the normalised embedding cube
        self.mu = jax.random.uniform(
            k_mu, (n_gabor, embed_dim), minval=-1.0, maxval=1.0
        )

        # All filters start at the same width
        log_s0 = jnp.log(jnp.array(sigma_init))
        if anisotropic:
            self.log_sigma = jnp.full((n_gabor, embed_dim), log_s0)
        else:
            self.log_sigma = jnp.full((n_gabor,), log_s0)

        # Frequencies ω ~ N(0, omega_scale² I) — trained, unlike RFF
        self.Omega = jax.random.normal(k_omega, (n_gabor, embed_dim)) * omega_scale

        self.phi = jax.random.uniform(
            k_phi, (n_gabor,), minval=0.0, maxval=2.0 * jnp.pi
        )

        self.W_out = _lecun_uniform(k_out, (n_outputs, n_gabor))
        self.b_out = jnp.zeros(n_outputs)
        self.W_skip = _lecun_uniform(k_skip, (n_outputs, embed_dim)) \
                      if skip_connection else None

        n_params = _count_params(self)
        flags = []
        if learnable_mu:    flags.append("learnable μ")
        if anisotropic:     flags.append("anisotropic σ")
        if skip_connection: flags.append("linear skip")
        if drop_rate > 0:   flags.append(f"drop={drop_rate}")
        flag_str = ", ".join(flags) if flags else "vanilla"
        print(
            f"[GaborNet] {embed_label} | "
            f"ω_scale={omega_scale} σ_init={sigma_init} | "
            f"{flag_str} | "
            f"embed({embed_dim})→{n_gabor} filters→{n_outputs} | "
            f"{n_params:,} params"
        )

    def _embed(self, coords: Array) -> Array:
        return _embed(coords, self.x_mins, self.x_maxs, self.n_inputs,
                      self.n_fourier, self.periodic_bc, self.time_dependent)

    def _gabor_features(self, e: Array) -> Array:
        """e : (embed_dim,) → features : (n_gabor,)"""
        z = self.Omega @ e + self.phi                      # oscillation, GEMV

        if self.anisotropic:
            inv_s2 = jnp.exp(-2.0 * self.log_sigma)        # (n_gabor, embed_dim)
            e2     = e ** 2
            mu_s2  = self.mu * inv_s2
            q      = inv_s2 @ e2 - 2.0 * (mu_s2 @ e) \
                     + jnp.sum(self.mu * mu_s2, axis=-1)
        else:
            inv_s2 = jnp.exp(-2.0 * self.log_sigma)        # (n_gabor,)
            q      = inv_s2 * (jnp.dot(e, e)
                               - 2.0 * (self.mu @ e)
                               + jnp.sum(self.mu ** 2, axis=-1))

        return jnp.exp(-0.5 * q) * jnp.cos(z)

    def __call__(self, coords: Array, *, key: PRNGKey | None = None) -> Array:
        """coords : (n_inputs,) → (n_outputs,).  key enables dropout."""
        e   = self._embed(coords)
        phi = self._gabor_features(e)

        if self.drop_rate > 0.0 and key is not None:
            keep = jax.random.bernoulli(key, 1.0 - self.drop_rate, (self.n_gabor,))
            phi  = phi * keep / (1.0 - self.drop_rate)

        out = self.W_out @ phi + self.b_out
        if self.skip_connection:
            out = out + self.W_skip @ e
        return out


def freeze_centres(model: GaborNet) -> GaborNet:
    """Stop gradients through the filter centres μ.

    Useful early in training when the centres haven't settled and large μ
    gradients destabilise ω and σ.
    """
    return eqx.tree_at(
        lambda m: m.mu,
        model,
        replace_fn=jax.lax.stop_gradient,
    )


def gabor_frequency_stats(model: GaborNet) -> dict:
    """Summary statistics of the learned frequencies — a training diagnostic."""
    omega_norms = jnp.linalg.norm(model.Omega, axis=-1)
    sigma       = jnp.exp(model.log_sigma)
    return {
        "omega_norms": omega_norms,
        "sigma":       sigma,
        "mean_freq":   jnp.mean(omega_norms),
        "max_freq":    jnp.max(omega_norms),
        "mean_sigma":  jnp.mean(sigma),
    }


# ──────────────────────────────────────────────────────────────────────────────
# SPINN
# ──────────────────────────────────────────────────────────────────────────────

class SPINNBranch(eqx.Module):
    """A single 1D branch network for SPINN."""
    matrices: list
    biases: list
    activation: Callable = eqx.field(static=True)

    def __init__(self, key, in_dim, hidden_dims, out_dim, activation):
        keys = jax.random.split(key, len(hidden_dims) + 1)
        dims = [in_dim] + list(hidden_dims) + [out_dim]

        self.matrices = []
        self.biases = []
        for i in range(len(dims) - 1):
            self.matrices.append(_lecun_uniform(keys[i], (dims[i+1], dims[i])))
            self.biases.append(jnp.zeros(dims[i+1]))

        self.activation = activation


    def __call__(self, x: Array) -> Array:
        for w, b in zip(self.matrices[:-1], self.biases[:-1]):
            x = self.activation(w @ x + b)
        w, b = self.matrices[-1], self.biases[-1]
        return w @ x + b


class SPINN(eqx.Module):
    """Separable PINN: a small network per input dimension, combined by a
    rank-R product — so a full tensor grid can be evaluated separably."""

    # ── Trainable ─────────────────────────────────────────────────────────
    branches: tuple        # (n_inputs,) tuple of SPINNBranch networks
    W_out:    Array        # (n_outputs, rank)
    b_out:    Array        # (n_outputs,)

    # ── Static ────────────────────────────────────────────────────────────
    x_mins:         tuple = eqx.field(static=True)
    x_maxs:         tuple = eqx.field(static=True)
    periodic_bc:    bool  = eqx.field(static=True)
    n_inputs:       int   = eqx.field(static=True)
    n_fourier:      int   = eqx.field(static=True)
    rank:           int   = eqx.field(static=True)
    time_dependent: bool  = eqx.field(static=True)

    def __init__(
        self,
        key:            PRNGKey,
        problem:        type,
        rank:           int   = 256,
        n_inputs:       int   = 2,
        n_outputs:      int   = 1,
        periodic_bc:    bool  = False,
        n_fourier:      int   = 0,
        activation:     str   = "tanh",
        hidden_dims:    tuple = (128, 128),
        time_dependent: bool  = True,
        **kwargs,
    ):
        self.rank           = kwargs.get("n_gabor", rank)
        self.n_inputs       = n_inputs
        self.n_fourier      = n_fourier
        self.periodic_bc    = periodic_bc
        self.time_dependent = time_dependent
        self.x_mins, self.x_maxs = _domain_bounds(problem, n_inputs)

        act_dict = {
            "tanh": jnp.tanh,
            "silu": jax.nn.silu,
            "gelu": jax.nn.gelu,
            "sin": jnp.sin,
        }
        act_fn = act_dict.get(activation.lower(), jnp.tanh)

        branch_keys = jax.random.split(key, n_inputs + 1)
        k_branches, k_out = branch_keys[:-1], branch_keys[-1]

        branches = []
        n_spatial = n_inputs - 1 if time_dependent else n_inputs

        for i in range(n_inputs):
            is_spatial = i < n_spatial

            if n_fourier > 0 and is_spatial:
                embed_dim = 1 + 2 * n_fourier
            elif periodic_bc and is_spatial:
                embed_dim = 2
            else:
                embed_dim = 1

            branches.append(
                SPINNBranch(
                    key=k_branches[i],
                    in_dim=embed_dim,
                    hidden_dims=hidden_dims,
                    out_dim=self.rank,
                    activation=act_fn,
                )
            )

        self.branches = tuple(branches)

        self.W_out = _lecun_uniform(k_out, (n_outputs, self.rank))
        self.b_out = jnp.zeros(n_outputs)

        n_params = _count_params(self)
        print(
            f"[SPINN] rank={self.rank} | "
            f"branches={n_inputs}x{list(hidden_dims)} | "
            f"Fourier modes={n_fourier} | "
            f"{n_params:,} params"
        )

    def _embed_1d(self, x: Array, dim_idx: int) -> Array:
        """Embed one scalar coordinate, by dimension index."""
        x_min = jnp.array(self.x_mins[dim_idx])
        x_max = jnp.array(self.x_maxs[dim_idx])
        n_spatial = self.n_inputs - 1 if self.time_dependent else self.n_inputs
        is_spatial = dim_idx < n_spatial

        if self.n_fourier > 0 and is_spatial:
            x_norm  = (x - x_min) / (x_max - x_min)
            if self.time_dependent:
                modes = jnp.arange(1, self.n_fourier + 1, dtype=x.dtype)
            else:
                modes = 2 ** jnp.arange(0, self.n_fourier, dtype=x.dtype)
            theta   = 2.0 * jnp.pi * x_norm * modes
            fourier = jnp.stack([jnp.cos(theta), jnp.sin(theta)]).reshape(-1)
            return jnp.concatenate([jnp.array([x]), fourier])

        if self.periodic_bc and is_spatial:
            x_norm = (x - x_min) / (x_max - x_min)
            theta  = 2.0 * jnp.pi * x_norm
            return jnp.array([jnp.cos(theta), jnp.sin(theta)])

        return jnp.array([x])

    def __call__(self, coords: Array, *, key: PRNGKey | None = None) -> Array:
        """Pointwise forward pass: coords : (n_inputs,) → (n_outputs,)"""
        features = []
        for i, branch in enumerate(self.branches):
            e_i = self._embed_1d(coords[i], i)
            features.append(branch(e_i))

        H = jnp.prod(jnp.stack(features), axis=0)

        return self.W_out @ H + self.b_out
