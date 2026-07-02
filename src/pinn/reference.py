"""Reference solutions — one .npz per problem, one schema for all of them.

Every reference file contains:

    coords    (N, in_dim)   points where the relative L2 error is evaluated
    solution  (N, out_dim)  reference values at those points
    channels  (out_dim,)    output names, e.g. ["u"] or ["u", "v", "p"]
    plot_axes (2,)          names of the two plotted inputs, e.g. ["x", "t"]
    plot_x0   (n0,)         grid along the first plot axis
    plot_x1   (n1,)         grid along the second plot axis
    plot_<ch> (n0, n1)      reference solution on the plot grid, per channel
    slice_val ()            value the remaining inputs are held at on the
                            plot grid (only present when in_dim > 2)

Plot-axis convention: the first plot axis is always input column 0; the
second is the time column (always last) when it is named "t", otherwise
input column 1.  Any remaining inputs are fixed at slice_val.  That is
enough to rebuild the full plot-grid coordinates for model evaluation —
see Reference.plot_coords().
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

import numpy as np


def reference_path(name: str) -> str:
    """Absolute path to a packaged reference file, e.g. ``reference_path("burgers")``.

    Resolves ``pinn/references/<name>.npz`` regardless of the working directory,
    so notebooks can load references from any location.
    """
    return str(files("pinn").joinpath("references", f"{name}.npz"))


def load_reference(name: str) -> "Reference":
    """Load a packaged reference by name (see :func:`reference_path`)."""
    return Reference.load(reference_path(name))


@dataclass(frozen=True)
class Reference:
    coords:     np.ndarray            # (N, in_dim)
    solution:   np.ndarray            # (N, out_dim)
    channels:   list[str]
    plot_axes:  tuple[str, str]
    plot_x0:    np.ndarray            # (n0,)
    plot_x1:    np.ndarray            # (n1,)
    plot_grids: dict[str, np.ndarray] # channel -> (n0, n1)
    slice_val:  float | None

    @classmethod
    def load(cls, path: str) -> Reference:
        z = dict(np.load(path))
        channels = [str(c) for c in z["channels"]]
        sol = z["solution"]
        if sol.ndim == 1:
            sol = sol[:, None]
        return cls(
            coords     = z["coords"],
            solution   = sol,
            channels   = channels,
            plot_axes  = tuple(str(a) for a in z["plot_axes"]),
            plot_x0    = z["plot_x0"],
            plot_x1    = z["plot_x1"],
            plot_grids = {ch: z[f"plot_{ch}"] for ch in channels},
            slice_val  = float(z["slice_val"]) if "slice_val" in z else None,
        )

    @property
    def in_dim(self) -> int:
        return self.coords.shape[1]

    @property
    def out_dim(self) -> int:
        return self.solution.shape[1]

    @property
    def plot_over_time(self) -> bool:
        """True when the second plot axis is time (so slab masking applies)."""
        return self.plot_axes[1] == "t"

    @property
    def plot_shape(self) -> tuple[int, int]:
        return self.plot_x0.shape[0], self.plot_x1.shape[0]

    def plot_coords(self, dtype=np.float64) -> np.ndarray:
        """Full input coordinates of the plot grid, shape (n0*n1, in_dim).

        Row-major over (plot_x0, plot_x1), so model output reshapes back to
        the (n0, n1) grids in plot_grids.
        """
        X0, X1 = np.meshgrid(self.plot_x0, self.plot_x1, indexing="ij")
        n      = X0.size
        cols   = [np.empty(n, dtype) for _ in range(self.in_dim)]

        i1 = self.in_dim - 1 if self.plot_over_time else 1
        for i in range(self.in_dim):
            if i == 0:
                cols[i] = X0.ravel()
            elif i == i1:
                cols[i] = X1.ravel()
            else:
                cols[i] = np.full(n, self.slice_val)
        return np.stack(cols, axis=1).astype(dtype)

    def describe(self) -> str:
        sliced = f", others at {self.slice_val}" if self.slice_val is not None else ""
        return (
            f"{self.coords.shape[0]} eval points in {self.in_dim}D, "
            f"channels {self.channels}, "
            f"plot {self.plot_axes[0]}×{self.plot_axes[1]} "
            f"({self.plot_shape[0]}×{self.plot_shape[1]}{sliced})"
        )


def save_reference(
    path:       str,
    *,
    coords:     np.ndarray,
    solution:   np.ndarray,
    channels:   list[str],
    plot_axes:  tuple[str, str],
    plot_x0:    np.ndarray,
    plot_x1:    np.ndarray,
    plot_grids: dict[str, np.ndarray],
    slice_val:  float | None = None,
    dtype                    = np.float64,
) -> None:
    """Write a reference .npz in the schema documented above."""
    solution = np.asarray(solution)
    if solution.ndim == 1:
        solution = solution[:, None]

    n0, n1 = plot_x0.shape[0], plot_x1.shape[0]
    assert solution.shape == (coords.shape[0], len(channels)), \
        f"solution {solution.shape} != ({coords.shape[0]}, {len(channels)})"
    assert set(plot_grids) == set(channels), \
        f"plot grids {sorted(plot_grids)} != channels {sorted(channels)}"
    for ch, g in plot_grids.items():
        assert g.shape == (n0, n1), f"plot_{ch} is {g.shape}, expected ({n0}, {n1})"
    if coords.shape[1] > 2:
        assert slice_val is not None, "slice_val required when in_dim > 2"

    arrays = {
        "coords":    coords.astype(dtype),
        "solution":  solution.astype(dtype),
        "channels":  np.array(channels),
        "plot_axes": np.array(plot_axes),
        "plot_x0":   plot_x0.astype(dtype),
        "plot_x1":   plot_x1.astype(dtype),
        **{f"plot_{ch}": g.astype(dtype) for ch, g in plot_grids.items()},
    }
    if slice_val is not None:
        arrays["slice_val"] = np.array(slice_val, dtype=dtype)

    np.savez(path, **arrays)
