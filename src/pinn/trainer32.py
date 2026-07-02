"""Single-precision training loop.

Identical to trainer64 except each time slab is trained in one float32
pass — no precision switch.  See trainer64 for the slab scheme.
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # silence XLA/PjRt C++ warnings

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # hide findfont msgs

import numpy as np
from typing import Any
import time
from datetime import datetime
import itertools

import equinox as eqx
import jax
import jax.numpy as jnp

from .optimiser import Optimiser
from .plotting import Plotter
from .reference import Reference


def has_passed_minimum(sequence, window_size=10, min_slope=1e-4, min_correlation=0.1):
    """Detect whether a loss history has bottomed out and begun rising."""
    if len(sequence) < window_size:
        return False

    window = np.array(sequence[-window_size:], dtype=float)
    window = np.maximum(window, np.finfo(float).tiny)
    log_window = np.log10(window)
    x = np.arange(window_size)

    slope, _ = np.polyfit(x, log_window, 1)

    r_matrix = np.corrcoef(x, log_window)
    r_value = r_matrix[0, 1]

    # nan if the window is perfectly flat
    if np.isnan(r_value):
        return False

    return slope > min_slope and r_value > min_correlation


# ──────────────────────────────────────────────────────────────────────────────
# Internal utilities
# ──────────────────────────────────────────────────────────────────────────────

def _partition(model):
    return eqx.partition(model, eqx.is_array)


def _slab_iterator(t_start, t_end, dt):
    t = t_start
    while t < t_end:
        t0 = t
        t1 = min(t + dt, t_end)
        yield t0, t1
        t = t1


# ──────────────────────────────────────────────────────────────────────────────
# IC target computation from a previous slab's model
# ──────────────────────────────────────────────────────────────────────────────

def _u_targets_from_model(
    prev_params,
    prev_static,
    spatial_coords: Array,   # (n, n_spatial) — x only, or [x, y]
    t0: float,
) -> Array:                  # (n, out_dim)
    model = eqx.combine(prev_params, prev_static)

    t_col  = jnp.full((spatial_coords.shape[0], 1), t0)
    coords = jnp.concatenate([spatial_coords, t_col], axis=1)  # (n, n_spatial+1)

    return jax.jit(jax.vmap(model))(coords)


def _u_targets_from_model_t(
    prev_params,
    prev_static,
    spatial_coords: Array,   # (n, n_spatial) — x only, or [x, y]
    t0: float,
) -> Array:                  # (n, out_dim)
    model = eqx.combine(prev_params, prev_static)

    def compute_u_t_single(x):
        # x is a single spatial coordinate (n_spatial,)
        def model_at_x(t):
            coords = jnp.concatenate([x, jnp.array([t])])
            return model(coords)[0]  # Extract scalar from output
        return jax.grad(model_at_x)(t0)

    return jax.jit(jax.vmap(compute_u_t_single))(spatial_coords)[:, None]  # Reshape to (n, 1)

# ──────────────────────────────────────────────────────────────────────────────
# Generalised condition builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_conditions(problem, key, t0, t1,
                      prev_params, prev_static,
                      condition_weights: dict[str, float]):
    fns      = problem.residual_fns()
    samplers = problem.samplers()
    n_spatial = problem.D if hasattr(problem, 'D') else 1

    conditions: list        = []
    weights:    list[float] = []

    for name in fns:
        if name not in samplers:
            raise KeyError(
                f"Problem declares residual '{name}' but has no matching sampler."
            )

        sampler = samplers[name]
        key, sk = jax.random.split(key)

        if name == "ic":
            pts_coords = sampler(sk, problem.n_ic, t0)   # (n, n_spatial+1) — coords only

            if prev_params is not None:
                spatial_coords = pts_coords[:, :n_spatial]
                model_targets  = _u_targets_from_model(
                    prev_params, prev_static, spatial_coords, t0
                )  # (n, out_dim)
            else:
                # First slab: use analytical IC
                spatial_coords = pts_coords[:, :n_spatial]
                model_targets  = problem.analytical_ic(spatial_coords.T).T  # (n, out_dim)

            pts = jnp.concatenate([pts_coords, model_targets], axis=1)  # (n, n_spatial+1+out_dim)
        elif name == "ic_t":
            pts_coords = sampler(sk, problem.n_ic, t0)   # (n, n_spatial+1) — coords only

            if prev_params is not None:
                spatial_coords = pts_coords[:, :n_spatial]
                model_targets  = _u_targets_from_model_t(
                    prev_params, prev_static, spatial_coords, t0
                )  # (n, out_dim)
            else:
                # First slab: use analytical IC
                spatial_coords = pts_coords[:, :n_spatial]
                model_targets  = problem.analytical_ic_t(spatial_coords.T).T  # (n, out_dim)

            pts = jnp.concatenate([pts_coords, model_targets], axis=1)  # (n, n_spatial+1+out_dim)
        else:
            n   = getattr(problem, f"n_{name}", problem.n_pde)
            pts = sampler(sk, n, t0, t1)

        conditions.append((fns[name], pts))
        weights.append(condition_weights.get(name, 1.0))

    return conditions, weights, key


def _condition_names(problem) -> list[str]:
    """Ordered list of condition names matching _build_conditions output order."""
    return list(problem.residual_fns().keys())


def _default_weights(problem, config) -> dict[str, float]:
    """Build the initial weight dict from config fields."""
    names = _condition_names(problem)
    w: dict[str, float] = {n: 1.0 for n in names}

    if hasattr(config, "pde_weight") and "pde" in w:
        w["pde"] = float(config.pde_weight)
    if hasattr(config, "ic_weight") and "ic" in w:
        w["ic"] = float(config.ic_weight)
    if hasattr(config, "bc_weight") and "bc" in w:
        w["bc"] = float(config.bc_weight)
    if hasattr(config, "bc2_weight") and "bc2" in w:
        w["bc2"] = float(config.bc2_weight)
    if hasattr(config, "bc3_weight") and "bc3" in w:
        w["bc3"] = float(config.bc3_weight)
    if hasattr(config, "bc4_weight") and "bc4" in w:
        w["bc4"] = float(config.bc4_weight)
    if hasattr(config, "ic_t_weight") and "ic_t" in w:
        w["ic_t"] = float(config.ic_t_weight)
    if hasattr(config, "condition_weights") and config.condition_weights:
        for k, v in config.condition_weights.items():
            if k in w:
                w[k] = float(v)

    return w


# ──────────────────────────────────────────────────────────────────────────────
# Model evaluation on the reference grids
# ──────────────────────────────────────────────────────────────────────────────

@jax.jit
def _eval_model_jit(model, coords):
    return jax.vmap(lambda c: model(c))(coords)  # (n, out_dim)


def _eval_plot_grid(
    ref: Reference,
    frozen_slabs, active_params, active_static,
    active_t0, active_t1,
) -> dict[str, np.ndarray]:
    """Model prediction on the reference plot grid, as {channel: (n0, n1)}.

    When the second plot axis is time the slab models are stitched along it
    (NaN beyond the active slab).  A pressure channel, if present, is
    median-shifted onto the reference to remove its arbitrary constant.
    """
    _float = np.float64 if jax.config.jax_enable_x64 else np.float32
    coords = ref.plot_coords(_float)
    n0, n1 = ref.plot_shape

    if ref.plot_over_time:
        t_flat  = coords[:, -1]
        sample  = _eval_model_jit(
            eqx.combine(active_params, active_static), jnp.array(coords[:1])
        )
        out = np.full((coords.shape[0], sample.shape[-1]), np.nan, dtype=_float)

        all_slabs = frozen_slabs + [(active_t0, active_t1, active_params, active_static)]
        for i, (t0, t1, p, s) in enumerate(all_slabs):
            is_last = (i == len(all_slabs) - 1)
            mask    = (t_flat >= t0) & (t_flat <= t1 if is_last else t_flat < t1)
            if not np.any(mask):
                continue
            u = _eval_model_jit(eqx.combine(p, s), jnp.array(coords[mask]))
            out[mask, :] = np.array(u).reshape(-1, out.shape[1])
    else:
        u   = _eval_model_jit(
            eqx.combine(active_params, active_static), jnp.array(coords)
        )
        out = np.array(u)

    out   = out.reshape(n0, n1, -1)
    grids = {ch: out[..., i] for i, ch in enumerate(ref.channels)}

    if "p" in grids:
        grids["p"] = grids["p"] - np.nanmedian(grids["p"] - ref.plot_grids["p"])

    return grids


def predict_to_current_slab(
    frozen_slabs, active_params, active_static,
    active_t0, active_t1, all_coords, t_col,
) -> np.ndarray:
    _float = np.float64 if jax.config.jax_enable_x64 else np.float32

    _model  = eqx.combine(active_params, active_static)
    _sample = _eval_model_jit(_model, jnp.array(all_coords[:1], dtype=_float))
    out_dim = _sample.shape[-1]

    out = np.full((all_coords.shape[0], out_dim), np.nan, dtype=_float)

    n_slabs   = len(frozen_slabs) + 1
    all_slabs = frozen_slabs + [(active_t0, active_t1, active_params, active_static)]

    for i, (t0, t1, p, s) in enumerate(all_slabs):
        is_last = (i == n_slabs - 1)
        mask    = (t_col >= t0) & (t_col <= t1 if is_last else t_col < t1)
        if not np.any(mask):
            continue
        model        = eqx.combine(p, s)
        u            = _eval_model_jit(model, jnp.array(all_coords[mask], dtype=_float))
        out[mask, :] = np.array(u).reshape(-1, out_dim)

    return out  # (N, out_dim)


def compute_l2_error(
    u_pred: np.ndarray,
    u_ref: np.ndarray,
    active_t1: float,
    t_col: np.ndarray,
    channels: list[str],
):
    valid_rows = (t_col <= active_t1 + 1e-8) & ~np.any(np.isnan(u_pred), axis=-1)

    pred = u_pred[valid_rows].copy()
    ref  = u_ref[valid_rows].copy()

    pure_l2 = {}
    rel_l2  = {}

    for i, ch in enumerate(channels):
        if ch == "p":
            r_shifted = ref[:, i] - np.mean(ref[:, i])
            p_shifted = pred[:, i] - np.mean(pred[:, i])
            mask = np.abs(r_shifted) < 0.4
            r = r_shifted[mask]
            p = p_shifted[mask]
        else:
            p = pred[:, i]
            r = ref[:, i]

        abs_err  = float(np.sqrt(np.mean((p - r) ** 2)))
        ref_norm = float(np.sqrt(np.mean(r ** 2)))
        rel_err  = abs_err / (ref_norm + 1e-18)

        pure_l2[ch] = abs_err
        rel_l2[ch]  = rel_err

    return pure_l2, rel_l2


# ──────────────────────────────────────────────────────────────────────────────
# Helper: extract probe data from a metrics dict
# ──────────────────────────────────────────────────────────────────────────────

def _probe_data(metrics: dict | None) -> dict | None:
    """Convert JAX arrays in metrics to numpy for the plotter."""
    if metrics is None:
        return None
    return {
        "probe_radii": np.array(metrics["probe_radii"]),
        "probe_rhos":  np.array(metrics["probe_rhos"]),
        "pchip_x":     np.array(metrics["pchip_x"]),
        "pchip_y":     np.array(metrics["pchip_y"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Compilation pre-warming
# ──────────────────────────────────────────────────────────────────────────────

def precompile(problem_class, config):
    """Warm the XLA persistent compilation cache for `config`.

    Runs two solver steps in float32 so that a subsequent `train()` call with
    the same shapes and settings skips compilation entirely.  Safe to call
    multiple times; a cache hit is silent and instant.

    The cache is written to `config.cache_dir` (default `.cache/jax`).
    """
    config.resolve_precision_defaults(double_precision=False)

    jax.config.update("jax_compilation_cache_dir", os.path.abspath(config.cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

    problem = problem_class(**config.problem_config)

    is_steady = not hasattr(problem, "t_max")
    if is_steady:
        t0, t1 = 0.0, 0.0
    elif config.slab_boundaries is not None:
        t0, t1 = config.slab_boundaries[0], config.slab_boundaries[1]
    elif config.dt is not None:
        t0, t1 = 0.0, float(config.dt)
    else:
        raise ValueError("Time-dependent problem needs dt or slab_boundaries.")

    key = jax.random.PRNGKey(config.seed)
    key, model_key = jax.random.split(key)
    model          = config.network(model_key)
    params, static = _partition(model)
    solver         = Optimiser(**config.solver_config)

    saved_x64 = jax.config.jax_enable_x64

    # Mirror train()'s single float32 pass so the cached graphs match.
    jax.config.update("jax_enable_x64", False)

    params = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x,
        params,
    )

    conditions, weights, key = _build_conditions(
        problem, key, t0, t1,
        prev_params=None, prev_static=None,
        condition_weights=_default_weights(problem, config),
    )

    key, opt_key = jax.random.split(key)
    opt_state = solver.init_state(opt_key, initial_radius=config.initial_radius)

    for _ in range(2):
        _, params, opt_state, _ = solver.step(
            params=params,
            static=static,
            state=opt_state,
            conditions=conditions,
            target_rho=config.target_rho,
            weights=weights,
        )

    jax.config.update("jax_enable_x64", saved_x64)


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(problem_class, config):
    """Train a PINN on `problem_class` with the settings in `config` (RunConfig).

    Single-precision counterpart of trainer64: each time slab is trained in
    one float32 pass, with no precision switch. Steady problems are treated
    as a single slab.
    """
    config.resolve_precision_defaults(double_precision=False)

    problem = problem_class(**config.problem_config)

    # ── Persistent compilation cache ────────────────────────────────────
    # Compiled XLA executables are written to cache_dir, so a re-run with
    # the same shapes and settings skips the long compile.
    jax.config.update("jax_compilation_cache_dir", os.path.abspath(config.cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

    # ── Detect steady-state vs time-dependent ──────────────────────────
    is_steady_state = not hasattr(problem, 't_max')

    if not is_steady_state:
        if config.slab_boundaries is not None:
            n_slabs = len(config.slab_boundaries) - 1
            print(f"Problem: {problem.problem_name}")
            print(f"Slabs: {n_slabs}  (explicit boundaries)")
        else:
            if config.dt is None:
                raise ValueError("Time-dependent problem needs dt or slab_boundaries.")
            config.dt = config.dt + 1e-7   # avoid an empty final slab at t_max
            n_slabs = int(jnp.ceil(problem.t_max / config.dt))
            print(f"Problem: {problem.problem_name}")
            print(f"Slabs: {n_slabs}  (dt={config.dt:.3f})")
    else:
        print(f"Problem: {problem.problem_name} (steady-state)")

    cond_names = _condition_names(problem)

    # ── RNG ────────────────────────────────────────────────────────────────
    key = jax.random.PRNGKey(config.seed)

    # ── Model ──────────────────────────────────────────────────────────────
    key, model_key = jax.random.split(key)
    model          = config.network(model_key)
    params, static = _partition(model)

    # ── Optimiser ──────────────────────────────────────────────────────────
    solver = Optimiser(**config.solver_config)

    # ── Reference solution ─────────────────────────────────────────────────
    ref = Reference.load(problem.ref_path)
    print(f"Reference: {problem.ref_path} — {ref.describe()}")
    all_coords = ref.coords
    u_ref_flat = ref.solution
    t_col      = np.zeros(all_coords.shape[0]) if is_steady_state else all_coords[:, -1]

    # ── Plotter ─────────────────────────────────────────────────────────────
    plotter = Plotter(ref, save_path=None)
    plotter.prime()

    # ── History accumulators ────────────────────────────────────────────────
    frozen_slabs  = []
    global_step   = 0
    t_total       = 0.0
    prev_params   = None
    prev_static   = None

    hists             = {k: [] for k in ["loss", "lambda", "rho", "radius"]}
    l2_rel_iter:  list = []
    l2_rel_wall:  list = []
    l2_pure_iter: list = []
    l2_pure_wall: list = []
    weight_hists: dict[str, list[float]] = {n: [] for n in cond_names}
    latest_metrics: dict | None = None

    # ── Transition event accumulators ──────────────────────────────────────
    # Each entry: {global_step, wall_time, t0, t1, rel_l2, pure_l2, ...}
    rho_switch_events: list = []  # target_rho → 0.5 transitions
    slab_events:       list = []  # one entry per completed slab

    # ── Results accumulator ────────────────────────────────────────────────
    results: dict = {
        "problem_name": problem.problem_name,
        "all_coords":    all_coords,
        "t_col":         t_col,
        "u_ref_flat":    u_ref_flat,
        "hists":         hists,
        "l2_rel_iter":   l2_rel_iter,
        "l2_rel_wall":   l2_rel_wall,
        "l2_pure_iter":  l2_pure_iter,
        "l2_pure_wall":  l2_pure_wall,
        "weight_hists":  weight_hists,
        "rho_switch_events": rho_switch_events,
        "slab_events":       slab_events,
        "slabs": [],
    }

    # ── Slab loop ──────────────────────────────────────────────────────────
    if is_steady_state:
        _slab_seq = itertools.chain(
            [(True,  0.0, 0.0)],
            [(False, 0.0, 0.0)],
        )
    else:
        def _slabs(t0, t1):
            if config.slab_boundaries is not None:
                bounds = config.slab_boundaries
                return list(zip(bounds[:-1], bounds[1:]))
            return list(_slab_iterator(t0, t1, config.dt))

        _slab_seq = itertools.chain(
            [(True,  t0, t1) for t0, t1 in _slabs(0.0, config.dt)[:1]],
            [(False, t0, t1) for t0, t1 in _slabs(0.0, problem.t_max)],
        )

    t_compile_start = time.perf_counter()

    for is_dry_run, t0, t1 in _slab_seq:

        slab_init_weights = _default_weights(problem, config)
        conditions, weights, key = _build_conditions(
            problem, key, t0, t1,
            prev_params, prev_static,
            condition_weights=slab_init_weights,
        )

        plotter.log(f"\n{'─'*60}")
        plotter.log(f"  Slab (t={t0:.4f}→{t1:.4f}) (float32)")
        plotter.log(f"{'─'*60}")

        key, opt_key   = jax.random.split(key)
        opt_state      = solver.init_state(opt_key, initial_radius=config.initial_radius)
        target_rho     = config.target_rho
        lambda_history: list[float] = []  # for has_passed_minimum detection
        slab_step      = 0

        while True:
            t_step_start = time.perf_counter()
            loss, params, opt_state, metrics = solver.step(
                params=params,
                static=static,
                state=opt_state,
                conditions=conditions,
                target_rho=target_rho,
                weights=weights,
            )
            step_time = time.perf_counter() - t_step_start
            t_total  += step_time

            if is_dry_run and slab_step >= 2:
                plotter.log(f"Compile complete: {time.perf_counter() - t_compile_start:.2f}s")
                t_total = 0.0
                break

            latest_metrics = metrics

            if not is_dry_run:
                for k, v in zip(hists, [loss, opt_state.lam, opt_state.rho, opt_state.radius]):
                    hists[k].append(float(v))

                for name, w in zip(cond_names, weights):
                    weight_hists[name].append(float(w))

                lambda_history.append(float(opt_state.lam))

            # ── Periodic plot + log ─────────────────────────────────────
            if slab_step % config.log_every == 0:
                u_pred = predict_to_current_slab(
                    frozen_slabs, params, static, t0, t1, all_coords, t_col
                )
                pure_l2, rel_l2 = compute_l2_error(u_pred, u_ref_flat, t1, t_col, plotter.channels)

                if not is_dry_run:
                    l2_rel_iter.append((global_step, rel_l2))
                    l2_rel_wall.append((t_total,     rel_l2))
                    l2_pure_iter.append((global_step, pure_l2))
                    l2_pure_wall.append((t_total,     pure_l2))

                u_pred_plot = _eval_plot_grid(
                    ref, frozen_slabs, params, static, t0, t1
                )

                plotter.update(
                    hists,
                    {"l2_rel_vs_iter": l2_rel_iter, "l2_rel_vs_wall": l2_rel_wall,
                     "l2_pure_vs_iter": l2_pure_iter, "l2_pure_vs_wall": l2_pure_wall},
                    weight_hists=weight_hists,
                    active_t1=t1,
                    probe_data=_probe_data(metrics),
                    u_pred_plot=u_pred_plot,
                )

            # ── Adaptive reweighting ────────────────────────────────────
            if config.auto_adjust:
                weighted = [float(cl) * w for cl, w in zip(metrics["cond_losses"], weights)]
                mean_wl  = sum(weighted) / len(weighted)
                weights = [
                    w * (mean_wl / (wl + 1e-8)) ** config.alpha
                    for w, wl in zip(weights, weighted)
                ]
                mean_w  = sum(weights) / len(weights)
                weights = [w / mean_w for w in weights]

            slab_step   += 1
            if not is_dry_run:
                global_step += 1   # dry-run steps don't count (mirrors t_total reset)

            # ── λ-minimum / rho transition ──────────────────────────────
            if len(lambda_history) > config.lambda_grace_period and target_rho != 0.5:
                if has_passed_minimum(lambda_history, window_size=config.lambda_history_size):
                    plotter.mark_lambda_min(global_step)

                    if not is_dry_run:
                        u_pred_rho = predict_to_current_slab(
                            frozen_slabs, params, static, t0, t1, all_coords, t_col
                        )
                        _pure_rho, _rel_rho = compute_l2_error(
                            u_pred_rho, u_ref_flat, t1, t_col, plotter.channels
                        )
                        rho_switch_events.append({
                            "global_step": global_step,
                            "wall_time":   t_total,
                            "t0":          t0,
                            "t1":          t1,
                            "rel_l2":      _rel_rho,
                            "pure_l2":     _pure_rho,
                        })

                    target_rho = 0.5

            # ── Slab exit ───────────────────────────────────────────────
            if opt_state.radius < 10 * config.min_radius:
                exit_reason = "radius below threshold"

                # Fresh evaluation at slab exit
                u_pred = predict_to_current_slab(
                    frozen_slabs, params, static, t0, t1, all_coords, t_col
                )
                pure_l2, rel_l2 = compute_l2_error(u_pred, u_ref_flat, t1, t_col, plotter.channels)

                if not is_dry_run:
                    l2_rel_iter.append((global_step, rel_l2))
                    l2_rel_wall.append((t_total,     rel_l2))
                    l2_pure_iter.append((global_step, pure_l2))
                    l2_pure_wall.append((t_total,     pure_l2))

                plotter.mark_slab_end(global_step)

                plotter.update(
                    hists,
                    {"l2_rel_vs_iter": l2_rel_iter, "l2_rel_vs_wall": l2_rel_wall,
                     "l2_pure_vs_iter": l2_pure_iter, "l2_pure_vs_wall": l2_pure_wall},
                    weight_hists=weight_hists,
                    active_t1=t1,
                    probe_data=_probe_data(latest_metrics),
                )

                l2_scalar = float(np.mean(list(rel_l2.values())))
                plotter.wait()
                plotter.log(f"  {exit_reason} — moving to next slab.")
                plotter.log(
                    f"  solved up to t={t1:.2f} | "
                    f"final rel error = {l2_scalar:.3e}, compute time = {t_total:.2f}s"
                )

                break

        # ── Post-slab: fresh consistent evaluation & bookkeeping ───────────
        if not is_dry_run:
            u_pred_slab = predict_to_current_slab(
                frozen_slabs, params, static, t0, t1, all_coords, t_col
            )
            pure_l2_slab, rel_l2_slab = compute_l2_error(
                u_pred_slab, u_ref_flat, t1, t_col, plotter.channels
            )

            if not is_steady_state:
                frozen_slabs.append((t0, t1, params, static))
                prev_params = params
                prev_static = static

                key, model_key, perturb_key = jax.random.split(key, 3)
                model = config.network(model_key)
                fresh_params, static = _partition(model)
                params = jax.tree_util.tree_map(
                    lambda p_old, p_new: p_old + 0.01 * p_new,
                    params, fresh_params
                )

                # ── Slab event (flat, for easy curve-overlay) ───────────────
                slab_events.append({
                    "t0":              t0,
                    "t1":              t1,
                    "global_step":     global_step,
                    "wall_time":       t_total,
                    "rel_l2":          rel_l2_slab,
                    "pure_l2":         pure_l2_slab,
                    "n_steps":         slab_step,
                })

                results["slabs"].append({
                    "t0":              t0,
                    "t1":              t1,
                    "global_step":     global_step,
                    "wall_time":       t_total,
                    "u_pred":          u_pred_slab,
                    "rel_l2":          rel_l2_slab,
                    "pure_l2":         pure_l2_slab,
                    "lambda_history":  lambda_history,
                    "final_weights":   list(weights),
                    "n_steps":         slab_step,
                })
            else:
                frozen_slabs.append((0.0, 0.0, params, static))
                results["slabs"].append({
                    "t0":             0.0,
                    "t1":             0.0,
                    "global_step":    global_step,
                    "wall_time":      t_total,
                    "u_pred":         u_pred_slab,
                    "rel_l2":         rel_l2_slab,
                    "pure_l2":        pure_l2_slab,
                    "lambda_history": lambda_history,
                    "final_weights":  list(weights),
                    "n_steps":        slab_step,
                })

    plotter.wait()
    plotter.finalize()

    # ── Final full-domain prediction ───────────────────────────────────────
    _float = np.float64 if jax.config.jax_enable_x64 else np.float32
    if is_steady_state:
        u_pred_final = _eval_model_jit(
            eqx.combine(params, static),
            jnp.array(all_coords, dtype=_float)
        )
        u_pred_final = np.array(u_pred_final)
    else:
        u_pred_final = predict_to_current_slab(
            frozen_slabs, params, static,
            frozen_slabs[0][0], frozen_slabs[-1][1],
            all_coords, t_col,
        )

    # ── Plot-grid prediction (saved for post-hoc figures) ──────────────────
    t0_l, t1_l, p_l, s_l = frozen_slabs[-1]
    results["u_pred_plot"] = _eval_plot_grid(ref, frozen_slabs[:-1], p_l, s_l, t0_l, t1_l)
    results["plot_x0"]     = ref.plot_x0
    results["plot_x1"]     = ref.plot_x1
    results["plot_axes"]   = ref.plot_axes

    results["u_pred_final"] = u_pred_final

    if config.return_network:
        model = eqx.combine(params, static)
        return (results, model)

    return results
