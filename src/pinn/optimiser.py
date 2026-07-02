"""Sketched trust-region/Levenberg–Marquardt optimiser which
selects a step with a target reduction ratio.

Usage
-----
    solver    = Optimiser(**cfg.solver_config)
    opt_state = solver.init_state(key, initial_radius=1.0)

    loss, params, opt_state, metrics = solver.step(
        params, static, opt_state, conditions, target_rho, weights,
    )

``conditions`` is a sequence of ``(residual_fn, points)`` pairs, one per
loss term (PDE residual, boundary / initial conditions, ...).

All hyper-parameters are frozen at construction time so the step can be
JIT-compiled once and reused.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, NamedTuple, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.random import PRNGKey

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

Pytree      = Any
ResidualFn  = Callable[[Any, Array], Array]   # (model, coords) → residual
AuxCondition = tuple[ResidualFn, Array]       # one extra (fn, points) pair


# ---------------------------------------------------------------------------
# Solver state
# ---------------------------------------------------------------------------

class SolverState(NamedTuple):
    """Mutable state carried between optimiser steps."""
    radius: Array   # current trust-region radius
    rho:    Array   # quality ratio from the last accepted step
    lam:    Array   # Levenberg–Marquardt regulariser
    key:    Array


# ---------------------------------------------------------------------------
# SRCT (right-side sketch)
# ---------------------------------------------------------------------------

def _make_srct(key: PRNGKey, n_params: int, parameter_sketch: int,
               dtype=jnp.float32) -> tuple[Array, Array, Array]:
    """Generate SRCT parameters once; reuse across scan steps."""
    k_signs, k_perm, k_idx = jax.random.split(key, 3)
    signs   = jax.random.choice(k_signs, jnp.array([-1., 1.], dtype=dtype),
                                shape=(n_params,))
    perm    = jax.random.permutation(k_perm, n_params)
    indices = jax.random.choice(k_idx, n_params, shape=(parameter_sketch,),
                                replace=False)
    return signs, perm, indices


def _apply_srct(J: Array,
                srct: tuple[Array, Array, Array]) -> Array:
    """Right sketch: signs → permute → DCT → subsample.

    J   : (n_rows, n_params)
    out : (n_rows, parameter_sketch)
    """
    signs, perm, indices = srct
    J = J * signs[None, :]
    J = J[:, perm]
    J = jax.scipy.fft.dct(J, type=2, norm="ortho", axis=1)
    return J[:, indices]


def _lift_update(y: Array,
                 srct: tuple[Array, Array, Array],
                 n_params: int) -> Array:
    """Adjoint SRCT: map from sketch space back to full parameter space."""
    signs, perm, indices = srct
    v = jnp.zeros((n_params,), dtype=y.dtype).at[indices].set(y)
    v = jax.scipy.fft.idct(v, type=2, norm="ortho")
    v_full = jnp.zeros_like(v).at[perm].set(v)
    return v_full * signs


# ---------------------------------------------------------------------------
# Count sketch (left-side sketch)
# ---------------------------------------------------------------------------

def _count_sketch(J: Array, r: Array,
                  residual_sketch: int, n_hashes: int,
                  key: PRNGKey) -> tuple[Array, Array]:
    """Map (n_rows, n_cols) → (residual_sketch, n_cols) via hashed projections."""
    scale     = 1.0 / jnp.sqrt(n_hashes)
    hash_keys = jax.random.split(key, n_hashes)

    def _one_hash(carry, h_key):
        B_acc, r_acc = carry
        k_s, k_b = jax.random.split(h_key)
        signs   = jax.random.rademacher(k_s, shape=(J.shape[0],),
                                        dtype=J.dtype)
        buckets = jax.random.randint(k_b, shape=(J.shape[0],),
                                     minval=0, maxval=residual_sketch)
        signed  = signs * scale
        B_acc  += jax.ops.segment_sum(J * signed[:, None], buckets, residual_sketch)
        r_acc  += jax.ops.segment_sum(r * signed,          buckets, residual_sketch)
        return (B_acc, r_acc), None

    (B, r_sk), _ = jax.lax.scan(
        _one_hash,
        (jnp.zeros((residual_sketch, J.shape[1]), J.dtype),
         jnp.zeros((residual_sketch,),           J.dtype)),
        hash_keys,
    )
    return B, r_sk


# ---------------------------------------------------------------------------
# Jacobian block computation
# ---------------------------------------------------------------------------

def _jacobian_block(params_flat: Array, unflatten: Callable,
                    static: Pytree, residual_fn: ResidualFn,
                    x_batch: Array,
                    sub_batch_size: int) -> tuple[Array, Array]:
    """Compute (J, r) for one macro-batch via blocked jacrev + vmap."""

    def _sub_jac(p, x_sub):
        m = eqx.combine(unflatten(p), static)
        r = jax.vmap(partial(residual_fn, m))(x_sub)
        return r, r

    jac_fn = jax.jacrev(_sub_jac, argnums=0, has_aux=True)
    x_subs = x_batch.reshape(-1, sub_batch_size, x_batch.shape[-1])
    J_blocks, r_blocks = jax.vmap(jac_fn, in_axes=(None, 0))(params_flat, x_subs)

    return (
        J_blocks.reshape(-1, params_flat.shape[0]),
        r_blocks.reshape(-1),
    )


# ---------------------------------------------------------------------------
# Doubly-sketched Jacobian accumulation
# ---------------------------------------------------------------------------

def _kernel_sketch(
    params_flat:        Array,
    unflatten:          Callable,
    static:             Pytree,
    residual_fn:        ResidualFn,
    collocation_points: Array,
    key:                PRNGKey,
    srct:               tuple[Array, Array, Array],
    *,
    residual_sketch:    int,
    parameter_sketch:    int,
    n_hashes:       int,
    sub_batch_size: int,
    batch_size:     int,
) -> tuple[Array, Array, Array]:
    """Accumulate doubly-sketched (B, r_sketch) and the full residual vector.

    Returns
    -------
    B        : (residual_sketch, parameter_sketch)
    r_sketch : (residual_sketch,)
    r_full   : (n_points * output_dim,)
    """
    n_pts = collocation_points.shape[0]
    assert n_pts   % batch_size     == 0, (
        f"n_points ({n_pts}) must be divisible by batch_size ({batch_size})")
    assert batch_size % sub_batch_size == 0, (
        f"batch_size ({batch_size}) must be divisible by sub_batch_size "
        f"({sub_batch_size})")

    batched = collocation_points.reshape(n_pts // batch_size, batch_size, -1)
    key, sk_key = jax.random.split(key)

    init = (
        jnp.zeros((residual_sketch, parameter_sketch), params_flat.dtype),
        jnp.zeros((residual_sketch,),             params_flat.dtype),
        sk_key,
    )

    def _body(carry, x_macro):
        B_acc, r_acc, step_key = carry
        step_key, sk = jax.random.split(step_key)
        J, r          = _jacobian_block(params_flat, unflatten, static,
                                        residual_fn, x_macro, sub_batch_size)
        J_right       = _apply_srct(J, srct)
        dB, dr        = _count_sketch(J_right, r, residual_sketch, n_hashes, sk)
        return (B_acc + dB, r_acc + dr, step_key), r

    (B, r_sk, _), r_batched = jax.lax.scan(_body, init, batched)
    return B, r_sk, r_batched.reshape(-1)


# ---------------------------------------------------------------------------
# Batched secular equation solver  (trust-region sub-problem)
# ---------------------------------------------------------------------------

def _solve_subproblems(S: Array, w: Array,
                       target_radii: Array,
                       n_iters: int = 40) -> tuple[Array, Array]:
    """Newton solver: find λ_i s.t. ‖p(λ_i)‖ = target_radii_i.

    Returns
    -------
    coeffs : (n_probes, k)   positive coefficients  w / (S² + λ)
    lambdas: (n_probes,)
    """
    finfo = jnp.finfo(S.dtype)
    S2  = jnp.square(S)
    lam = jnp.zeros_like(target_radii)

    for _ in range(n_iters):
        denom  = S2[None, :] + lam[:, None]
        p      = w[None, :] / denom
        p_norm = jnp.linalg.norm(p, axis=1)
        phi    = p_norm - target_radii
        d_phi  = (-jnp.sum(jnp.square(p) / (denom + finfo.tiny), axis=1)
                  / (p_norm + finfo.tiny))
        lam    = jnp.maximum(lam - phi / (d_phi + finfo.tiny), 0.0)

    return lam


# ---------------------------------------------------------------------------
# Probe evaluation
# ---------------------------------------------------------------------------

def _eval_probes(ps: Array,
                 params_flat: Array,
                 unflatten: Callable,
                 static: Pytree,
                 residual_fn: ResidualFn,
                 points: Array,
                 batch_size: int) -> Array:
    """Evaluate all probe steps for a single condition.

    Returns residuals : (n_probes, n_points * output_dim)
    """
    n_batches = points.shape[0] // batch_size
    batched   = points[:n_batches * batch_size].reshape(n_batches, batch_size, -1)

    def _eval_batch(batch):
        def _one(step):
            m = eqx.combine(unflatten(params_flat + step), static)
            return jax.vmap(partial(residual_fn, m))(batch).ravel()
        return jax.vmap(_one)(ps)                      # (n_probes, batch * out)

    all_r = jax.lax.map(_eval_batch, batched)          # (n_batches, n_probes, …)
    return all_r.transpose(1, 0, 2).reshape(ps.shape[0], -1)


def _eval_all_conditions(ps, params_flat, unflatten, static,
                         residual_fns, points_list, weights_arr,
                         batch_size):
    """Weighted loss across all conditions for each probe step."""
    total = jnp.zeros(ps.shape[0])
    for i, (fn, pts) in enumerate(zip(residual_fns, points_list)):
        r = _eval_probes(ps, params_flat, unflatten, static, fn, pts, batch_size)
        w_i = weights_arr[i]
        total += w_i * jnp.mean(jnp.square(r), axis=1)
    return total


# ---------------------------------------------------------------------------
# PCHIP interpolation (JAX-native)
# ---------------------------------------------------------------------------

def _pchip(x: Array, y: Array, xq: Array) -> Array:
    """Monotonicity-preserving cubic Hermite spline (Fritsch–Carlson)."""
    finfo = jnp.finfo(x.dtype)
    hk = x[1:] - x[:-1]
    dk = (y[1:] - y[:-1]) / (hk + finfo.tiny)

    def _slope(d1, d2, h1, h2):
        w1, w2  = 2.0 * h2 + h1, h2 + 2.0 * h1
        mono    = (jnp.sign(d1) * jnp.sign(d2)) > 0
        val     = (w1 + w2) / (w1 / (d1 + finfo.tiny) + w2 / (d2 + finfo.tiny))
        return jnp.where(mono, val, 0.0)

    def _boundary(d_e, d_n, h_e, h_n):
        val = ((2.0 * h_e + h_n) * d_e - h_e * d_n) / (h_e + h_n)
        return jnp.where(jnp.sign(val) == jnp.sign(d_e), val, 0.0)

    d_mid = jax.vmap(_slope)(dk[:-1], dk[1:], hk[:-1], hk[1:])
    m_s   = _boundary(dk[0], dk[1], hk[0], hk[1])
    m_e   = _boundary(dk[-1], dk[-2], hk[-1], hk[-2])
    derivs = jnp.concatenate([jnp.atleast_1d(m_s), d_mid, jnp.atleast_1d(m_e)])

    idx = jnp.clip(jnp.searchsorted(x, xq) - 1, 0, x.shape[0] - 2)
    h   = x[idx + 1] - x[idx]
    t   = (xq - x[idx]) / (h + finfo.tiny)
    t2, t3 = t * t, t * t * t
    y0, y1 = y[idx], y[idx + 1]
    m0, m1 = derivs[idx], derivs[idx + 1]

    return (
        (2*t3 - 3*t2 + 1) * y0 + (t3 - 2*t2 + t) * h * m0
      + (-2*t3 + 3*t2) * y1 + (t3 - t2) * h * m1
    )


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

class Optimiser:
    """Sketched Trust-Region Levenberg–Marquardt optimiser.

    Parameters
    ----------
    residual_sketch      : rows in the count-sketch matrix
    parameter_sketch      : columns retained by the SRCT (right sketch)
    n_hashes         : number of hash functions in the count sketch
    sub_batch_size   : micro-batch size for blocked jacrev
    batch_size       : macro-batch size for the scan loop
    probe_batch_size : batch size used during probe evaluation
    n_probes         : number of candidate radii to evaluate per step
    window_scale     : probe radii span  [r/scale, r*scale]
    min_radius       : hard lower bound on the trust-region radius
    max_radius       : hard upper bound on the trust-region radius
    pchip_grid       : fine-grid size for PCHIP crossing search
    newton_iters     : Newton iterations in the secular equation solver
    """

    def __init__(
        self,
        *,
        residual_sketch:      int   = 4096,
        parameter_sketch:      int   = 2048,
        n_hashes:         int   = 4,
        sub_batch_size:   int   = 4,
        batch_size:       int   = 1024,
        probe_batch_size: int   = 64,
        n_probes:         int   = 32,
        window_scale:     float = 100.0,
        min_radius:       float = 1e-10,
        max_radius:       float = 1e5,
        pchip_grid:       int   = 1024,
        newton_iters:     int   = 60,
    ) -> None:
        self.residual_sketch      = residual_sketch
        self.parameter_sketch      = parameter_sketch
        self.n_hashes         = n_hashes
        self.sub_batch_size   = sub_batch_size
        self.batch_size       = batch_size
        self.probe_batch_size = probe_batch_size
        self.n_probes         = n_probes
        self.window_scale     = window_scale
        self.min_radius       = min_radius
        self.max_radius       = max_radius
        self.pchip_grid       = pchip_grid
        self.newton_iters     = newton_iters

        self._step_jit = jax.jit(
            _step_impl,
            static_argnums=(3,),          # residual_fns — tuple of callables
            static_argnames=(
                "residual_sketch",
                "parameter_sketch",
                "n_hashes",
                "sub_batch_size",
                "batch_size",
                "probe_batch_size",
                "n_probes",
                "window_scale",
                "min_radius",
                "max_radius",
                "pchip_grid",
                "newton_iters",
            ),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def init_state(key: PRNGKey, initial_radius: float = 1.0) -> SolverState:
        return SolverState(
            radius=jnp.array(initial_radius),
            rho=jnp.array(1.0),
            lam=jnp.array(1.0),
            key=key,
        )

    def step(
        self,
        params:     Pytree,
        static:     Pytree,
        state:      SolverState,
        conditions: Sequence[AuxCondition],
        target_rho: float,
        weights:    Sequence[float] | None = None,
    ) -> tuple[Array, Pytree, SolverState, dict]:
        residual_fns = tuple(fn  for fn, _  in conditions)
        points_list  = tuple(pts for _,  pts in conditions)
        if weights is None:
            weights = tuple(1.0 for _ in conditions)
        weights_arr = jnp.array(weights)

        return self._step_jit(
            params, static, state,
            residual_fns, points_list, weights_arr, target_rho,
            residual_sketch=self.residual_sketch,
            parameter_sketch=self.parameter_sketch,
            n_hashes=self.n_hashes,
            sub_batch_size=self.sub_batch_size,
            batch_size=self.batch_size,
            probe_batch_size=self.probe_batch_size,
            n_probes=self.n_probes,
            window_scale=self.window_scale,
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            pchip_grid=self.pchip_grid,
            newton_iters=self.newton_iters,
        )


def _step_impl(
    params:       Pytree,
    static:       Pytree,
    state:        SolverState,
    residual_fns: tuple[ResidualFn, ...],
    points_list:  tuple[Array, ...],
    weights_arr:  Array,
    target_rho:   float,
    *,
    residual_sketch:      int,
    parameter_sketch:      int,
    n_hashes:         int,
    sub_batch_size:   int,
    batch_size:       int,
    probe_batch_size: int,
    n_probes:         int,
    window_scale:     float,
    min_radius:       float,
    max_radius:       float,
    pchip_grid:       int,
    newton_iters:     int,
) -> tuple[Array, Pytree, SolverState, dict]:

    params_flat, unflatten = ravel_pytree(params)
    n_params = params_flat.shape[0]
    finfo    = jnp.finfo(params_flat.dtype)
    lift_vmap = jax.vmap(_lift_update, in_axes=(0, None, None))

    # ── 1. SRCT parameters (shared across all conditions) ──────────────
    key, subkey = jax.random.split(state.key)
    state = state._replace(key=key)

    subkey, srct_key = jax.random.split(subkey)
    srct = _make_srct(srct_key, n_params, parameter_sketch, params_flat.dtype)

    # ── 2. Sketch each condition; accumulate via linearity ─────────────
    B        = jnp.zeros((residual_sketch, parameter_sketch), params_flat.dtype)
    r_sketch = jnp.zeros((residual_sketch,),             params_flat.dtype)
    all_r    = []

    for i, (fn, pts) in enumerate(zip(residual_fns, points_list)):
        subkey, ck = jax.random.split(subkey)
        Bi, rsi, rfi = _kernel_sketch(
            params_flat, unflatten, static, fn, pts, ck, srct,
            residual_sketch=residual_sketch, parameter_sketch=parameter_sketch,
            n_hashes=n_hashes, sub_batch_size=sub_batch_size,
            batch_size=batch_size,
        )
        n_i  = rfi.shape[0]
        w_i  = weights_arr[i]
        # Scale sketch so accumulated system = weighted least-squares:
        #   loss = sum_i  w_i * mean(r_i^2)
        # Each condition's J rows and r are scaled by sqrt(w_i / n_i).
        scale = jnp.sqrt(w_i / n_i)
        B        += scale * Bi
        r_sketch += scale * rsi
        all_r.append(rfi)

    # Weighted loss:  sum_i w_i * mean(r_i^2)
    current_loss = jnp.array(0.0)
    for i, rfi in enumerate(all_r):
        w_i = weights_arr[i]
        current_loss += w_i * jnp.mean(jnp.square(rfi))

    # ── 3. SVD of the sketched core matrix ─────────────────────────────
    U, S, Vt = jnp.linalg.svd(B, full_matrices=False)
    g        = S * (U.T @ r_sketch)

    def _step_and_pred(lam_val: Array) -> tuple[Array, Array]:
        denom   = S**2 + lam_val
        step_sk = -(Vt.T @ (g / denom))
        pred    = jnp.sum(g**2 * (S**2 + 2*lam_val) / denom**2)
        return step_sk, pred

    # ── 4. Probe radii window ──────────────────────────────────────────
    cur_r      = jnp.clip(state.radius, min_radius, max_radius)
    r_lo       = jnp.maximum(cur_r / window_scale, min_radius)
    r_hi       = jnp.minimum(cur_r * window_scale, max_radius)
    probe_radii = jnp.geomspace(r_lo, r_hi, n_probes)

    # ── 5. Secular equation for all probes ─────────────────────────────
    probe_lambdas = _solve_subproblems(
        S, g, probe_radii, newton_iters)

    # Steps in sketch space, then lifted to full parameter space
    probe_steps_sk, probe_preds = jax.vmap(_step_and_pred)(probe_lambdas)
    ps = lift_vmap(probe_steps_sk, srct, n_params)       # (n_probes, n_params)

    # ── 6. Evaluate all probe steps ────────────────────────────────────
    aligned = tuple(
        p[:(p.shape[0] // probe_batch_size) * probe_batch_size]
        for p in points_list
    )
    probe_losses = _eval_all_conditions(
        ps, params_flat, unflatten, static,
        residual_fns, aligned, weights_arr, probe_batch_size,
    )

    # Predicted & actual reductions, quality ratios
    act_red    = current_loss - probe_losses
    probe_rhos = act_red / (probe_preds + finfo.tiny)

    # ── 7. PCHIP crossing to choose optimal radius ─────────────────────
    safe_rhos  = jnp.minimum.accumulate(jnp.clip(probe_rhos, -1.0, 1.0))
    log_radii  = jnp.log(probe_radii)
    fine_log_r = jnp.linspace(jnp.log(r_lo), jnp.log(r_hi), pchip_grid)
    fine_rho   = _pchip(log_radii, safe_rhos, fine_log_r)

    is_above  = fine_rho >= target_rho
    crossings = is_above[:-1] & ~is_above[1:]
    has_cross = jnp.any(crossings)

    last_idx  = jnp.max(
        jnp.where(crossings, jnp.arange(pchip_grid - 1), -1))
    opt_log_r   = 0.5 * (fine_log_r[last_idx] + fine_log_r[last_idx + 1])
    fallback_r  = jnp.where(jnp.all(is_above), r_hi, r_lo)
    final_rad   = jnp.where(has_cross, jnp.exp(opt_log_r), fallback_r)

    # ── 8. Final step at the chosen radius ─────────────────────────────
    f_lams = _solve_subproblems(
        S, g, jnp.atleast_1d(final_rad), newton_iters)
    f_lam = f_lams[0]

    f_step_sk, f_pred = _step_and_pred(f_lam)
    f_step = _lift_update(f_step_sk, srct, n_params)
    trial  = eqx.combine(unflatten(params_flat + f_step), static)

    f_loss = jnp.array(0.0)
    cond_losses = []
    for i, (fn, pts) in enumerate(zip(residual_fns, points_list)):
        nb      = pts.shape[0] // batch_size
        batched = pts[:nb * batch_size].reshape(nb, batch_size, -1)
        r_cond  = jax.lax.map(
            lambda b: jax.vmap(partial(fn, trial))(b).ravel(),
            batched,
        ).ravel()
        mean_sq = jnp.mean(jnp.square(r_cond))
        cond_losses.append(mean_sq)
        f_loss += weights_arr[i] * mean_sq

    f_act_red = current_loss - f_loss
    f_rho     = f_act_red / (f_pred + finfo.tiny)

    # ── 9. Accept / reject ─────────────────────────────────────────────
    rho_hi_thresh = target_rho + 0.1
    rho_too_high  = ~jnp.isnan(f_rho) & (f_rho > rho_hi_thresh)
    rho_positive  = ~jnp.isnan(f_rho) & (f_rho > 0.0)
    accepted      = rho_positive
    
    new_loss   = jnp.where(accepted, f_loss, current_loss)
    delta      = jnp.where(accepted, f_step, jnp.zeros_like(f_step))
    new_params = unflatten(params_flat + delta)
    
    next_radius = jnp.where(
        accepted,
        final_rad,      # normal accept: keep the PCHIP-chosen radius
        jnp.where(
            rho_too_high,
            r_hi,       # rho too high → expand so next step is less damped
            r_lo,       # rho ≤ 0      → shrink
        ),
    )
    
    new_state = SolverState(
        radius=next_radius,
        rho=f_rho,
        lam=jnp.where(accepted, f_lam, state.lam),
        key=key,
    )
    
    # Per-condition mean squared residuals (accepted values if step taken)
    current_cond_losses = []
    for i, rfi in enumerate(all_r):
        current_cond_losses.append(jnp.mean(jnp.square(rfi)))
    cond_losses_out = [
        jnp.where(accepted, cl, ccl)
        for cl, ccl in zip(cond_losses, current_cond_losses)
    ]

    metrics: dict = {
        "loss":         new_loss,
        "radius":       final_rad,
        "rho":          f_rho,
        "lam":       jnp.where(accepted, f_lam, state.lam),
        "accepted":     accepted,
        "cond_losses":  cond_losses_out,
        "probe_radii":  probe_radii,
        "probe_rhos":   safe_rhos,
        "probe_losses": probe_losses,
        "pchip_x":      jnp.exp(fine_log_r),
        "pchip_y":      fine_rho,
        "S": S
    }

    return new_loss, new_params, new_state, metrics