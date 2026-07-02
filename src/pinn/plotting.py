"""Live plotting for PINN training — runs on a background thread.

All matplotlib work happens in a daemon thread.  The rendered PNG is pushed
into an ``ipywidgets.Image`` widget, which stays anchored in its original
position.

The plotter is driven by a ``reference.Reference``: the reference plot
grids give the left column, the trainer supplies matching prediction grids
via ``update(u_pred_plot=...)``, and axis names come from the reference's
``plot_axes``.

Figure layout (n_ch output channels, time-dependent problems):
    Rows 0 … n_ch-1    │ Reference │ Predicted │ |Error|    (per channel)
    Row  n_ch          │ Loss/iter │ Rel-ℓ₂/iter │ Weightings/iter
    Row  n_ch+1        │ λ/iter    │ ρ/iter    │ radius/iter
    Row  n_ch+2        │ Rel-ℓ₂ vs PDE time (cols 0-1) │ Rel-ℓ₂ vs wall time
    Row  n_ch+3        │ Probe ρ vs radius (cols 0-1)  │ stats panel

Steady problems drop the PDE-time row and pack the last row as
wall-time error │ probe │ stats.

Iteration-axis plots carry three kinds of vertical marker:
    Dark grey  ── time-marching boundaries   (``mark_slab_end``)
    Cyan       ── λ-min / ρ-transition  (``mark_lambda_min``)
    Magenta    ── float32 → float64 switch (``mark_precision_switch``)
"""

from __future__ import annotations

import io
import threading

import numpy as np
import matplotlib
matplotlib.use("Agg")                   # non-interactive — safe on all backends
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
from matplotlib.colors import LogNorm

from .reference import Reference


# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.size":            10,
    "axes.titlesize":       11,
    "axes.titleweight":     "semibold",
    "axes.labelsize":       9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      8,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.color":           "#E0E0E0",
    "grid.linewidth":       0.6,
    "grid.linestyle":       "-",
    "lines.linewidth":      1.8,
    "lines.markersize":     4,
    "patch.linewidth":      0.5,
})


# ── colour constants ──────────────────────────────────────────────────────────
_OI = {
    "orange":   "#E69F00",
    "sky":      "#56B4E9",
    "green":    "#009E73",
    "yellow":   "#F0E442",
    "blue":     "#0072B2",
    "red":      "#D55E00",
    "pink":     "#CC79A7",
    "black":    "#000000",
    "purple":   "#9467BD",
}

_C_LOSS   = _OI["purple"]
_C_LAMBDA = _OI["green"]
_C_RHO    = _OI["orange"]
_C_RADIUS = _OI["pink"]

_CHANNEL_COLOURS = [
    _OI["blue"], _OI["red"], _OI["green"],
    _OI["orange"], _OI["pink"], _OI["sky"],
]

# Distinct palette for the weighting lines so they don't clash with the
# channel colours on the same figure.
_WEIGHT_COLOURS = [
    _OI["red"], _OI["green"], _OI["orange"],
    _OI["pink"], _OI["sky"], _OI["yellow"],
]

_SLAB_COLOUR      = "#1A1A1A"
_LAM_COLOUR       = "#00C8D7"
_PRECISION_COLOUR = "#FF1493"

_CMAP_FIELD = "RdBu_r"
_CMAP_ERROR = "inferno"


def _l2_vs_time(pred_grid: np.ndarray, ref_grid: np.ndarray,
                t_values: np.ndarray, active_t1: float):
    """Relative L2 error per time column of the plot grid.

    Returns (t, rel) restricted to columns inside the solved window with a
    full (non-NaN) prediction.
    """
    valid = t_values <= (active_t1 + 1e-9)
    valid &= ~np.any(np.isnan(pred_grid), axis=0)
    if not np.any(valid):
        return np.array([]), np.array([])

    p = pred_grid[:, valid]
    r = ref_grid[:, valid]
    num   = np.sqrt(np.mean((p - r) ** 2, axis=0))
    denom = np.sqrt(np.mean(r ** 2, axis=0)) + 1e-12
    return t_values[valid], num / denom


class Plotter:
    """Background-thread live plotter for PINN training in Jupyter."""

    def __init__(self, ref: Reference, save_path: str = "training_plot.png"):
        self._ref        = ref
        self._save_path  = save_path
        self._lock       = threading.Lock()
        self._pending:   dict | None              = None
        self._thread:    threading.Thread | None  = None
        self._last_error: str | None              = None

        self._slab_end_iters:         list[int] = []
        self._lambda_min_iters:       list[int] = []
        self._precision_switch_iters: list[int] = []

        self._img_widget = None
        self._last_u_pred_plot = None

    # ── public properties ──────────────────────────────────────────────

    @property
    def channels(self) -> list[str]:
        return self._ref.channels

    # ── public API (main thread) ───────────────────────────────────────

    def prime(self) -> None:
        """Create and display the plot widget."""
        from IPython.display import display
        import ipywidgets as widgets

        if self._img_widget is None:
            self._img_widget = widgets.Image(
                format="png",
                layout=widgets.Layout(width="100%", max_width="1600px"),
            )
            display(self._img_widget)

    def log(self, *args, **kwargs) -> None:
        print(*args, **kwargs)

    def mark_slab_end(self, iteration: int) -> None:
        """Record a slab-end boundary at *iteration*."""
        with self._lock:
            self._slab_end_iters.append(iteration)

    def mark_lambda_min(self, iteration: int) -> None:
        """Record a λ-minimum / ρ-target transition at *iteration*."""
        with self._lock:
            self._lambda_min_iters.append(iteration)

    def mark_precision_switch(self, iteration: int) -> None:
        """Record a float32 → float64 precision switch at *iteration*."""
        with self._lock:
            self._precision_switch_iters.append(iteration)

    def update(
        self,
        hists:         dict,
        l2_metrics:    dict,
        weight_hists:  dict | None = None,
        active_t1:     float       = np.inf,
        probe_data:    dict | None = None,
        u_pred_plot:   dict | None = None,
    ) -> None:
        """Enqueue a redraw.  Safe from any thread.

        Parameters
        ----------
        hists :
            Scalar history lists — keys ``"loss"``, ``"lambda"``,
            ``"rho"``, ``"radius"``.
        l2_metrics :
            Keys ``"l2_rel_vs_iter"``, ``"l2_rel_vs_wall"``,
            ``"l2_pure_vs_iter"``, ``"l2_pure_vs_wall"``; each a list of
            ``(x_value, {channel: float})`` pairs.
        weight_hists :
            Condition name → list of weight values, one per iteration.
        active_t1 :
            End time of the current slab; bounds the PDE-time error panel.
        probe_data :
            Keys ``"probe_radii"``, ``"probe_rhos"``, ``"pchip_x"``,
            ``"pchip_y"`` (all array-like).
        u_pred_plot :
            Channel → (n0, n1) prediction on the reference plot grid.
            If omitted, the previous grids are reused.
        """
        if u_pred_plot is None:
            u_pred_plot = self._last_u_pred_plot
        else:
            self._last_u_pred_plot = u_pred_plot

        # Per-time-column error profile for the PDE-time panel
        l2_pde_time = {}
        if self._ref.plot_over_time and u_pred_plot is not None:
            t_values = self._ref.plot_x1
            for ch in self.channels:
                l2_pde_time[ch] = _l2_vs_time(
                    u_pred_plot[ch], self._ref.plot_grids[ch],
                    t_values, active_t1,
                )

        with self._lock:
            slab_ends    = list(self._slab_end_iters)
            lambda_mins  = list(self._lambda_min_iters)
            prec_switchs = list(self._precision_switch_iters)

        payload = {
            **hists,
            **l2_metrics,
            "active_t1":        active_t1,
            "l2_pde_time":      l2_pde_time,
            "weight_hists":     weight_hists or {},
            "probe_data":       probe_data,
            "slab_end_iters":   slab_ends,
            "lambda_min_iters": lambda_mins,
            "precision_switch_iters": prec_switchs,
            "u_pred_plot":      u_pred_plot,
        }
        with self._lock:
            self._pending = payload
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._draw_loop, daemon=True)
            self._thread.start()

    def wait(self) -> None:
        """Block until the background drawing thread has finished."""
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=30)

    def check_error(self) -> None:
        if self._last_error:
            print(self._last_error)
        else:
            print("No errors recorded.")

    def finalize(self) -> None:
        """Embed the final plot as a persistent static image in the notebook."""
        self.wait()
        import os
        from IPython.display import display, Image as IPImage
        if self._save_path and os.path.exists(self._save_path):
            display(IPImage(filename=self._save_path))

    # ── background draw loop ───────────────────────────────────────────

    def _draw_loop(self) -> None:
        import traceback
        while True:
            with self._lock:
                data, self._pending = self._pending, None
            if data is None:
                return
            try:
                self._redraw(data)
            except Exception:
                import sys
                msg = traceback.format_exc()
                self._last_error = msg
                print(f"\n[Plotter error]\n{msg}", file=sys.stderr, flush=True)
                return

    # ── master redraw ──────────────────────────────────────────────────

    def _redraw(self, data: dict) -> None:
        n_ch      = len(self.channels)
        is_steady = not self._ref.plot_over_time

        # Steady: 4 rows (probe and wall-time share the last row).
        # Time-dependent: 5 rows including the PDE-time profile.
        if is_steady:
            row_h = [3.5] * n_ch + [3.0, 3.0, 3.5]
            n_extra_rows = 3
        else:
            row_h = [3.5] * n_ch + [3.0, 3.0, 3.0, 3.5]
            n_extra_rows = 4

        fig = plt.figure(figsize=(18, sum(row_h) + 1.5),
                         facecolor="white")
        gs  = mgs.GridSpec(
            n_ch + n_extra_rows, 3, figure=fig,
            height_ratios=row_h,
            hspace=0.65, wspace=0.38,
        )

        self._draw_solution_rows(fig, gs, data, n_ch)
        self._draw_metrics_row(fig, gs, data, n_ch)
        self._draw_optimizer_row(fig, gs, data, n_ch)
        if is_steady:
            self._draw_steady_last_row(fig, gs, data, n_ch)
        else:
            self._draw_pde_time_row(fig, gs, data, n_ch)
            self._draw_probe_row(fig, gs, data, n_ch)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=110,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        png_bytes = buf.getvalue()

        if self._save_path:
            try:
                with open(self._save_path, "wb") as f:
                    f.write(png_bytes)
            except OSError:
                pass

        if self._img_widget is not None:
            self._img_widget.value = png_bytes

    # ── helper: vertical event markers ────────────────────────────────

    @staticmethod
    def _event_lines(ax, slab_ends: list[int], lambda_mins: list[int],
                     precision_switches: list[int] = None) -> None:
        if precision_switches is None:
            precision_switches = []

        for x in slab_ends:
            ax.axvline(x, color=_SLAB_COLOUR, ls="-", lw=1.8,
                       alpha=1.0, zorder=4)
        for x in lambda_mins:
            ax.axvline(x, color=_LAM_COLOUR, ls="--", lw=2.8,
                       alpha=1.0, zorder=5)
        for x in precision_switches:
            ax.axvline(x, color=_PRECISION_COLOUR, ls=":", lw=2.2,
                       alpha=0.95, zorder=4.5)

    # ── helper: colour-bar styling ─────────────────────────────────────

    @staticmethod
    def _add_colorbar(fig, mappable, ax, *, label: str = "") -> None:
        cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7, length=3)
        if label:
            cb.set_label(label, size=8)
        cb.outline.set_linewidth(0.5)

    @staticmethod
    def _ch_colour(i: int) -> str:
        return _CHANNEL_COLOURS[i % len(_CHANNEL_COLOURS)]

    @staticmethod
    def _wt_colour(i: int) -> str:
        return _WEIGHT_COLOURS[i % len(_WEIGHT_COLOURS)]

    # ── rows 0 … n_ch-1  :  Reference | Predicted | |Error| ──────────

    def _draw_solution_rows(self, fig, gs, data: dict, n_ch: int) -> None:
        u_pred_plot = data.get("u_pred_plot") or {}
        x0, x1       = self._ref.plot_x0, self._ref.plot_x1
        lab0, lab1   = self._ref.plot_axes

        for ch_i, ch_name in enumerate(self.channels):
            ref_2d  = self._ref.plot_grids[ch_name]
            pred_2d = u_pred_plot.get(ch_name)

            ax_ref  = fig.add_subplot(gs[ch_i, 0])
            ax_pred = fig.add_subplot(gs[ch_i, 1])
            ax_err  = fig.add_subplot(gs[ch_i, 2])
            for _ax in (ax_ref, ax_pred, ax_err):
                _ax.grid(False)

            # Colour limits: symmetric about zero when the field crosses it;
            # pressure is centred and clipped at the 99th percentile.
            if ch_name == "p":
                ref_2d = ref_2d - np.mean(ref_2d)
                if pred_2d is not None:
                    pred_2d = pred_2d - np.nanmean(pred_2d)
                v_ext = float(np.percentile(np.abs(ref_2d), 99))
                vmin, vmax = -v_ext, v_ext
            else:
                v_min, v_max = float(np.nanmin(ref_2d)), float(np.nanmax(ref_2d))
                if v_min < 0 < v_max:
                    v_ext = max(abs(v_min), abs(v_max))
                    vmin, vmax = -v_ext, v_ext
                else:
                    vmin, vmax = v_min, v_max
            if vmin == vmax:
                vmin -= 1e-3; vmax += 1e-3

            kw = dict(cmap=_CMAP_FIELD, vmin=vmin, vmax=vmax,
                      shading="auto", rasterized=True)

            # Grids are (n0, n1); pcolormesh wants (n1, n0)
            pc = ax_ref.pcolormesh(x0, x1, ref_2d.T, **kw)
            ax_ref.set_title(f"Reference — {ch_name}")
            ax_ref.set_xlabel(lab0); ax_ref.set_ylabel(lab1)
            self._add_colorbar(fig, pc, ax_ref)

            if pred_2d is not None:
                pc2 = ax_pred.pcolormesh(x0, x1, pred_2d.T, **kw)
                ax_pred.set_title(f"Predicted — {ch_name}")
                ax_pred.set_xlabel(lab0); ax_pred.set_ylabel(lab1)
                self._add_colorbar(fig, pc2, ax_pred)

                abs_err = np.abs(ref_2d - pred_2d)
                finite  = abs_err[~np.isnan(abs_err)]
                ef = finite[finite > 0]
                lo = max(float(ef.min()) if ef.size else 1e-10, 1e-10)
                hi = max(float(finite.max()) if finite.size else lo * 10,
                         lo * 10)
                pc3 = ax_err.pcolormesh(
                    x0, x1, np.clip(abs_err, lo, hi).T,
                    cmap=_CMAP_ERROR, norm=LogNorm(vmin=lo, vmax=hi),
                    shading="auto", rasterized=True,
                )
                ax_err.set_title(f"|Error| — {ch_name}")
                ax_err.set_xlabel(lab0); ax_err.set_ylabel(lab1)
                self._add_colorbar(fig, pc3, ax_err)
            else:
                ax_pred.set_visible(False)
                ax_err.set_visible(False)

    # ── row n_ch  :  Loss/iter | Rel-ℓ₂/iter | Weightings/iter ────────

    def _draw_metrics_row(self, fig, gs, data: dict, n_ch: int) -> None:
        sl = data.get("slab_end_iters",   [])
        lm = data.get("lambda_min_iters", [])
        ps = data.get("precision_switch_iters", [])

        loss_h       = data.get("loss",           [])
        l2_rel_iter  = data.get("l2_rel_vs_iter", [])
        weight_hists = data.get("weight_hists",   {})

        ax_loss = fig.add_subplot(gs[n_ch, 0])
        ax_li   = fig.add_subplot(gs[n_ch, 1])
        ax_wt   = fig.add_subplot(gs[n_ch, 2])

        # ── Loss vs iteration ──────────────────────────────────────────
        if loss_h:
            iters  = np.arange(1, len(loss_h) + 1)
            loss_h = np.asarray(loss_h, dtype=float)
            valid  = np.isfinite(loss_h) & (loss_h > 0)

            if np.any(valid):
                ax_loss.semilogy(iters, loss_h, lw=2.0, color=_C_LOSS,
                                 solid_capstyle="round")
                ax_loss.fill_between(iters, loss_h,
                                     alpha=0.10, color=_C_LOSS, lw=0)
            self._event_lines(ax_loss, sl, lm, ps)
        ax_loss.set_title("Training Loss")
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("Loss")

        # ── Rel-ℓ₂ vs iteration ───────────────────────────────────────
        if l2_rel_iter:
            xs = [e[0] for e in l2_rel_iter]
            for i, ch in enumerate(self.channels):
                ys = np.asarray(
                    [e[1].get(ch, np.nan) for e in l2_rel_iter], dtype=float)
                valid = np.isfinite(ys) & (ys > 0)

                if np.any(valid):
                    ax_li.semilogy(
                        xs, ys,
                        color=self._ch_colour(i),
                        lw=2.0, marker="o", ms=3.5,
                        markevery=max(1, len(xs) // 30),
                        solid_capstyle="round",
                        label=ch,
                    )
                    ax_li.fill_between(xs, ys, alpha=0.08,
                                       color=self._ch_colour(i), lw=0)
            self._event_lines(ax_li, sl, lm, ps)
        ax_li.set_title("Total Relative ℓ₂ Error")
        ax_li.set_xlabel("Iteration")
        ax_li.set_ylabel("Rel. ℓ₂ Error")
        if ax_li.get_legend_handles_labels()[0]:
            ax_li.legend(framealpha=0.85, edgecolor="#CCCCCC")

        # ── Condition weightings vs iteration ─────────────────────────
        if weight_hists:
            cond_names = list(weight_hists.keys())
            max_len    = max(len(v) for v in weight_hists.values())
            iters_wt   = np.arange(1, max_len + 1)
            for j, cname in enumerate(cond_names):
                ws = np.array(weight_hists[cname], dtype=float)
                if len(ws) < max_len:
                    ws = np.concatenate([ws,
                                         np.full(max_len - len(ws), np.nan)])
                ax_wt.semilogy(
                    iters_wt, ws,
                    lw=2.0, color=self._wt_colour(j),
                    solid_capstyle="round",
                    label=cname,
                )
            self._event_lines(ax_wt, sl, lm, ps)
            ax_wt.legend(framealpha=0.85, edgecolor="#CCCCCC")
        ax_wt.set_title("Condition Weightings")
        ax_wt.set_xlabel("Iteration")
        ax_wt.set_ylabel("Weight")

    # ── row n_ch+1  :  λ/iter | ρ/iter | radius/iter ──────────────────

    def _draw_optimizer_row(self, fig, gs, data: dict, n_ch: int) -> None:
        sl = data.get("slab_end_iters",   [])
        lm = data.get("lambda_min_iters", [])
        ps = data.get("precision_switch_iters", [])

        lam_h = data.get("lambda", [])
        rho_h = data.get("rho",    [])
        rad_h = data.get("radius", [])

        ax_lam = fig.add_subplot(gs[n_ch + 1, 0])
        ax_rho = fig.add_subplot(gs[n_ch + 1, 1])
        ax_rad = fig.add_subplot(gs[n_ch + 1, 2])

        if lam_h:
            iters = np.arange(1, len(lam_h) + 1)
            lam_h = np.asarray(lam_h, dtype=float)
            valid = np.isfinite(lam_h) & (lam_h > 0)

            if np.any(valid):
                ax_lam.semilogy(iters, lam_h, lw=2.0, color=_C_LAMBDA,
                                solid_capstyle="round")
                ax_lam.fill_between(iters, lam_h,
                                    alpha=0.10, color=_C_LAMBDA, lw=0)
            self._event_lines(ax_lam, sl, lm, ps)
        ax_lam.set_title("λ  (Regularisation)")
        ax_lam.set_xlabel("Iteration")
        ax_lam.set_ylabel("λ")

        if rho_h:
            iters = np.arange(1, len(rho_h) + 1)
            ax_rho.fill_between(iters, rho_h,
                                alpha=0.10, color=_C_RHO, lw=0)
            ax_rho.plot(iters, rho_h, lw=2.0, color=_C_RHO,
                        solid_capstyle="round")
            ax_rho.axhline(0, color="#555555", ls="--", lw=1.0, zorder=0)
            ax_rho.set_ylim(-0.1, 0.6)
            self._event_lines(ax_rho, sl, lm, ps)
        ax_rho.set_title("ρ  (Gain Ratio)")
        ax_rho.set_xlabel("Iteration")
        ax_rho.set_ylabel("ρ")

        if rad_h:
            iters = np.arange(1, len(rad_h) + 1)
            ax_rad.semilogy(iters, rad_h, lw=2.0, color=_C_RADIUS,
                            solid_capstyle="round")
            ax_rad.fill_between(iters, rad_h,
                                alpha=0.10, color=_C_RADIUS, lw=0)
            self._event_lines(ax_rad, sl, lm, ps)
        ax_rad.set_title("Trust-Region Radius")
        ax_rad.set_xlabel("Iteration")
        ax_rad.set_ylabel("Radius")

    # ── shared panels ──────────────────────────────────────────────────

    def _draw_wall_time_panel(self, ax, data: dict) -> None:
        l2_wall = data.get("l2_rel_vs_wall", [])
        if l2_wall:
            xs = [e[0] for e in l2_wall]
            for i, ch in enumerate(self.channels):
                ys = [e[1].get(ch, np.nan) for e in l2_wall]
                ax.semilogy(
                    xs, ys,
                    color=self._ch_colour(i),
                    lw=2.0, marker="s", ms=3.5,
                    markevery=max(1, len(xs) // 30),
                    solid_capstyle="round",
                    label=ch,
                )
                ax.fill_between(xs, ys, alpha=0.08,
                                color=self._ch_colour(i), lw=0)
        ax.set_title("Relative ℓ₂ Error  vs  Compute Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Rel. ℓ₂ error")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(framealpha=0.85, edgecolor="#CCCCCC")

    def _draw_probe_panel(self, ax, data: dict) -> None:
        pd = data.get("probe_data")
        if pd is not None:
            radii   = np.asarray(pd.get("probe_radii", []), dtype=float)
            rhos    = np.asarray(pd.get("probe_rhos",  []), dtype=float)
            pchip_x = np.asarray(pd.get("pchip_x",    []), dtype=float)
            pchip_y = np.asarray(pd.get("pchip_y",    []), dtype=float)

            drew = False
            ok = np.isfinite(rhos) & (radii > 0) if radii.size == rhos.size else \
                 np.zeros(0, dtype=bool)
            if np.any(ok):
                ax.scatter(
                    radii[ok], rhos[ok],
                    s=40, zorder=3, alpha=0.80,
                    color=_OI["sky"],
                    edgecolors=_OI["blue"], linewidths=0.6,
                    label="probe samples",
                )
                drew = True
            ok = np.isfinite(pchip_y) & (pchip_x > 0) if pchip_x.size == pchip_y.size else \
                 np.zeros(0, dtype=bool)
            if np.any(ok):
                ax.plot(
                    pchip_x[ok], pchip_y[ok],
                    lw=2.2, color=_OI["red"],
                    zorder=4, solid_capstyle="round",
                    label="PCHIP fit",
                )
                drew = True

            ax.axhline(0, color="#555555", ls="--", lw=1.0, zorder=0)
            if drew:
                ax.set_xscale("log")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(framealpha=0.85, edgecolor="#CCCCCC")

        ax.set_title("Probe  ρ  vs  Radius  (latest step)")
        ax.set_xlabel("Probe radius")
        ax.set_ylabel("ρ")

    def _draw_stats_panel(self, ax, data: dict) -> None:
        ax.axis("off")
        ax.set_facecolor("#F8F8F8")

        l2_iter      = data.get("l2_rel_vs_iter",  [])
        l2_pure_iter = data.get("l2_pure_vs_iter", [])

        l2_lines = l2_pure_lines = "  —"
        if l2_iter:
            l2_lines = "\n".join(
                "  {}:  {:.4e}".format(ch, l2_iter[-1][1].get(ch, float("nan")))
                for ch in self.channels
            )
        if l2_pure_iter:
            l2_pure_lines = "\n".join(
                "  {}:  {:.4e}".format(ch, l2_pure_iter[-1][1].get(ch, float("nan")))
                for ch in self.channels
            )

        l2_wall = data.get("l2_rel_vs_wall", [])
        wall_t  = l2_wall[-1][0] if l2_wall else 0.0

        text_body = "\n".join([
            "Relative l2 error",
            l2_lines,
            "",
            "Pure l2 error",
            l2_pure_lines,
            "",
            "Compute time",
            "  {:.1f} s  ({:.2f} min)".format(wall_t, wall_t / 60),
        ])
        ax.text(
            0.08, 0.92, text_body,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=11, family="monospace",
            linespacing=1.8,
            color="#222222",
        )

    # ── steady last row  :  wall-time error | probe | stats ───────────

    def _draw_steady_last_row(self, fig, gs, data: dict, n_ch: int) -> None:
        self._draw_wall_time_panel(fig.add_subplot(gs[n_ch + 2, 0]), data)
        self._draw_probe_panel(fig.add_subplot(gs[n_ch + 2, 1]), data)
        self._draw_stats_panel(fig.add_subplot(gs[n_ch + 2, 2]), data)

    # ── row n_ch+2  :  Rel-ℓ₂ vs PDE time | Rel-ℓ₂ vs wall time ───────
    # (time-dependent problems only)

    def _draw_pde_time_row(self, fig, gs, data: dict, n_ch: int) -> None:
        ax_pde  = fig.add_subplot(gs[n_ch + 2, :2])   # wide — cols 0 & 1
        ax_wall = fig.add_subplot(gs[n_ch + 2,  2])   # narrow — col 2

        l2_pde = data.get("l2_pde_time", {})
        any_pde_data = False
        for i, ch in enumerate(self.channels):
            if ch not in l2_pde:
                continue
            t_use, rel = l2_pde[ch]
            if t_use.size == 0:
                continue
            any_pde_data = True
            ax_pde.semilogy(
                t_use, rel,
                lw=2.0, color=self._ch_colour(i),
                solid_capstyle="round",
                label=ch,
            )
            ax_pde.fill_between(t_use, rel,
                                alpha=0.08, color=self._ch_colour(i), lw=0)

        ax_pde.set_title("Relative ℓ₂  vs  PDE Time")
        ax_pde.set_xlabel("Simulation time  t")
        ax_pde.set_ylabel("Rel. ℓ₂ error")
        if any_pde_data and len(self.channels) > 1:
            ax_pde.legend(framealpha=0.85, edgecolor="#CCCCCC")

        self._draw_wall_time_panel(ax_wall, data)

    # ── row n_ch+3  :  Probe ρ vs radius (cols 0-1) + stats (col 2) ───
    # (time-dependent problems only)

    def _draw_probe_row(self, fig, gs, data: dict, n_ch: int) -> None:
        self._draw_probe_panel(fig.add_subplot(gs[n_ch + 3, :2]), data)
        self._draw_stats_panel(fig.add_subplot(gs[n_ch + 3, 2]), data)
