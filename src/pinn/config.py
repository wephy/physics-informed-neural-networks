from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import ClassVar


@dataclass
class RunConfig:
    """All tunable settings for a single training run."""

    # ─────────────────────────────
    # Core experiment
    # ─────────────────────────────
    network: callable
    seed: int = 0
    return_network: bool = False

    # ─────────────────────────────
    # Solver
    # ─────────────────────────────
    residual_sketch:  int   = 2000
    parameter_sketch: int   = 2000
    n_hashes:         int   = 4
    sub_batch_size:   int   = 4
    batch_size:       int   = 1024
    probe_batch_size: int   = 128
    n_probes:         int   = 24
    window_scale:     float = 3.0
    # `None` → resolved per precision by the trainer:
    #   single precision (trainer32) → 1e-4, double precision (trainer64) → 1e-8.
    # Set an explicit value to override.
    min_radius:       float | None = None
    max_radius:       float = 1e3
    pchip_grid:       int   = 512
    newton_iters:     int   = 80
    initial_radius:   float = 1.0

    # ─────────────────────────────
    # Problem
    # ─────────────────────────────
    dt:    float | None = None
    slab_boundaries: list[float] | None = None

    n_pde: int   = 2**14
    n_ic:  int   = 2**13
    n_bc:  int   = 2**13

    # ─────────────────────────────
    # Training control
    # ─────────────────────────────
    target_rho:       float = 0.075

    auto_adjust:      bool  = True
    pde_weight:       float = 1e-2
    bc_weight:        float = 1.0
    bc2_weight:       float = 1e-2
    bc3_weight:       float = 1e-4
    bc4_weight:       float = 1e-6
    ic_weight:        float = 1.0
    ic_t_weight:      float = 1.0

    lambda_history_size: int = 30
    lambda_grace_period: int = 80

    float32_steps: int = 10
    alpha: float = 0.05

    # ─────────────────────────────
    # Display / IO
    # ─────────────────────────────
    log_every:  int      = 5
    plot_every: int      = 5
    cache_dir:  str      = ".cache/jax"

    # ─────────────────────────────
    # Explicit field groups
    # ─────────────────────────────
    _SOLVER_FIELDS: set[str] = field(
        default_factory=lambda: {
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
        },
        repr=False,
    )

    _PROBLEM_FIELDS: set[str] = field(
        default_factory=lambda: {
            "n_pde",
            "n_ic",
            "n_bc",
        },
        repr=False,
    )

    _DISPLAY_FIELDS: set[str] = field(
        default_factory=lambda: {
            "log_every",
            "plot_every",
            "cache_dir",
            "target_rho",
        },
        repr=False,
    )

    # Precision-dependent default for the minimum trust-region radius.
    SINGLE_PRECISION_MIN_RADIUS: ClassVar[float] = 1e-4
    DOUBLE_PRECISION_MIN_RADIUS: ClassVar[float] = 1e-8

    def resolve_precision_defaults(self, double_precision: bool) -> None:
        """Fill in precision-dependent defaults left unset by the user.

        `min_radius` defaults to 1e-4 for single precision and 1e-8 for double
        precision; an explicit value in the config is left untouched.
        """
        if self.min_radius is None:
            self.min_radius = (
                self.DOUBLE_PRECISION_MIN_RADIUS
                if double_precision
                else self.SINGLE_PRECISION_MIN_RADIUS
            )

    @property
    def solver_config(self) -> dict:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name in self._SOLVER_FIELDS
        }

    @property
    def problem_config(self) -> dict:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name in self._PROBLEM_FIELDS
        }

    @property
    def display_config(self) -> dict:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name in self._DISPLAY_FIELDS
        }