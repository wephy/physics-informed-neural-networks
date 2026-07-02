from __future__ import annotations

from typing import Callable


class Problem:
    """Base interface a PDE problem must implement.

    Subclasses expose their equations and domain through two dictionaries keyed
    by condition name (``"pde"``, ``"bc"``, ``"ic"``, ``"ic_t"``, ...), plus the
    domain bounds and metadata the trainer reads (``x_min``, ``x_max``,
    ``problem_name``, ``ref_path``, and ``t_max`` for time-dependent problems).
    See ``examples/`` for complete definitions.
    """

    def residual_fns(self) -> dict[str, Callable]:
        """Named residual functions, each ``(model, coords) -> Array``."""
        raise NotImplementedError

    def samplers(self) -> dict[str, Callable]:
        """Named samplers that draw collocation points for each condition.

        PDE / BC samplers have signature ``(key, n, t0, t1)``; IC samplers
        ``(key, n, t0)``. For steady problems the trainer passes ``t0 = t1 = 0``.
        Initial conditions are matched against a target carried in the last
        coordinate — from ``analytical_ic`` on the first slab, or the previous
        slab's model afterwards — so the residual graph is identical every slab.
        """
        raise NotImplementedError
