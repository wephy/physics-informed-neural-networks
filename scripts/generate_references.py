"""Regenerate the reference solutions shipped in ``pinn/references/``.

Each PDE example is scored against a reference solution stored as an ``.npz``
(schema in ``pinn/reference.py``). Analytic references are exact; Burgers, KdV
and KS use self-contained pseudo-spectral solvers. The lid-driven cavity
reference is produced separately with Dedalus and shipped as data — it is not
regenerated here.

Usage:
    python scripts/generate_references.py [name ...]

With no arguments every reference below is regenerated.
"""

from __future__ import annotations

import sys

import numpy as np

from pinn.reference import reference_path, save_reference


def _save(name, **kw):
    path = reference_path(name)
    save_reference(path, **kw)
    print(f"  saved {name} -> {path}")


def burgers():
    """u_t + u·u_x - nu·u_xx = 0 on (-1, 1), pseudo-spectral (DOP853)."""
    from scipy.fft import fft, ifft
    from scipy.integrate import solve_ivp

    nu, nx, nt = 0.01, 1024, 100
    x = np.linspace(-1.0, 1.0, nx, endpoint=False)
    t = np.linspace(0.0, 1.0, nt)
    k = np.fft.fftfreq(nx, d=2.0 / nx) * 2 * np.pi

    def rhs(_, u):
        u_hat = fft(u)
        u_xx = np.real(ifft(-(k**2) * u_hat))
        u_sq_x = np.real(ifft(1j * k * fft(u**2)))
        return nu * u_xx - 0.5 * u_sq_x

    sol = solve_ivp(rhs, (0.0, 1.0), -np.sin(np.pi * x), t_eval=t,
                    method="DOP853", atol=1e-12, rtol=1e-12)
    assert sol.success, sol.message

    U = sol.y  # (nx, nt)
    X, T = np.meshgrid(x, t, indexing="ij")
    _save("burgers", coords=np.stack([X.ravel(), T.ravel()], 1),
          solution=U.ravel()[:, None], channels=["u"],
          plot_axes=("x", "t"), plot_x0=x, plot_x1=t, plot_grids={"u": U})


def kdv():
    """u_t + eta·u·u_x + mu²·u_xxx = 0 on (-1, 1), Fourier + RK4."""
    from scipy.fft import fft, ifft, fftfreq

    eta, mu, nx, nt = 1.0, 0.022, 512, 201
    x = np.linspace(-1.0, 1.0, nx, endpoint=False)
    k = fftfreq(nx, d=2.0 / nx) * 2.0 * np.pi

    def rhs(u):
        u_hat = fft(u)
        u_x = np.real(ifft(1j * k * u_hat))
        u_xxx = np.real(ifft((1j * k) ** 3 * u_hat))
        return -eta * u * u_x - mu**2 * u_xxx

    dt = 5e-6
    n_steps = int(np.ceil(1.0 / dt))
    save_every = max(1, n_steps // (nt - 1))
    U = np.zeros((nx, nt))
    u = np.cos(np.pi * x)
    U[:, 0] = u
    saved = 0
    for step in range(1, n_steps + 1):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if step % save_every == 0 and saved + 1 < nt:
            saved += 1
            U[:, saved] = u
    t = np.linspace(0.0, 1.0, nt)
    X, T = np.meshgrid(x, t, indexing="ij")
    _save("kdv", coords=np.stack([X.ravel(), T.ravel()], 1),
          solution=U.ravel()[:, None], channels=["u"],
          plot_axes=("x", "t"), plot_x0=x, plot_x1=t, plot_grids={"u": U})


def ks():
    """Kuramoto-Sivashinsky on [0, 2pi], periodic, ETDRK4 (Cox-Matthews)."""
    nx, nt, dt = 512, 501, 1e-6
    alpha, beta, gamma = 100.0 / 16.0, 100.0 / 16.0**2, 100.0 / 16.0**4
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=2 * np.pi / nx)
    L = beta * k**2 - gamma * k**4

    E, E2 = np.exp(dt * L), np.exp(dt * L / 2)
    M = 64
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (LR - 2)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))

    m, pad = nx // 2, 3 * nx // 2

    def nonlinear(v):
        v_pad = np.zeros(pad, dtype=complex)
        v_pad[:m], v_pad[-m:] = v[:m], v[-m:]
        u_pad = np.fft.ifft(v_pad).real * 1.5
        v_sq_pad = np.fft.fft(u_pad**2)
        v_sq = np.zeros(nx, dtype=complex)
        v_sq[:m], v_sq[-m:] = v_sq_pad[:m] / 1.5, v_sq_pad[-m:] / 1.5
        return -0.5j * alpha * k * v_sq

    u = np.cos(x) * (1.0 + np.sin(x))
    v = np.fft.fft(u)
    steps = int(round(1.0 / dt))
    save_every = steps // (nt - 1)
    U = np.zeros((nx, nt))
    U[:, 0] = u
    saved = 1
    for n in range(1, steps + 1):
        Nv = nonlinear(v)
        a = E2 * v + Q * Nv
        Na = nonlinear(a)
        b = E2 * v + Q * Na
        Nb = nonlinear(b)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = nonlinear(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if n % save_every == 0 and saved < nt:
            U[:, saved] = np.fft.ifft(v).real
            saved += 1
    t = np.linspace(0, 1.0, nt)
    X, T = np.meshgrid(x, t, indexing="ij")
    _save("ks", coords=np.stack([X.ravel(), T.ravel()], 1),
          solution=U.ravel()[:, None], channels=["u"],
          plot_axes=("x", "t"), plot_x0=x, plot_x1=t, plot_grids={"u": U})


def wave():
    """u_tt - c²·u_xx = 0, standing wave with a closed-form solution."""
    c, b, nx, nt = 2.0, 4.0, 1024, 101
    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    t = np.linspace(0.0, 1.0, nt)
    X, T = np.meshgrid(x, t, indexing="ij")
    U = (np.sin(np.pi * X) * np.cos(c * np.pi * T)
         + 0.5 * np.sin(b * np.pi * X) * np.cos(c * b * np.pi * T))
    _save("wave", coords=np.stack([X.ravel(), T.ravel()], 1),
          solution=U.ravel()[:, None], channels=["u"],
          plot_axes=("x", "t"), plot_x0=x, plot_x1=t, plot_grids={"u": U})


def laplacian():
    """-Δu = f with u = (1/n)·Σ sin(w·π·x)·sin(w·π·y) over multiple scales."""
    ws, n = [4, 8, 12, 16], 512
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    U = np.mean([np.sin(w * np.pi * X) * np.sin(w * np.pi * Y) for w in ws], axis=0)
    _save("laplacian", coords=np.stack([X.ravel(), Y.ravel()], 1),
          solution=U.ravel()[:, None], channels=["u"],
          plot_axes=("x", "y"), plot_x0=x, plot_x1=y, plot_grids={"u": U})


def poisson5d():
    """-Δu = π²·Σ cos(π·xᵢ) on [0, 1]⁵, exact u = Σ cos(π·xᵢ)."""
    n_grid, n_plot, slice_val = 15, 128, 0.5

    def u_exact(x):
        return np.sum(np.cos(np.pi * x), axis=1)

    grids = np.meshgrid(*[np.linspace(0, 1, n_grid)] * 5, indexing="ij")
    coords = np.stack([g.ravel() for g in grids], axis=1)

    xp = np.linspace(0, 1, n_plot)
    X, Y = np.meshgrid(xp, xp, indexing="ij")
    slice_coords = np.stack(
        [X.ravel(), Y.ravel(), *[np.full(n_plot**2, slice_val)] * 3], axis=1)
    U = u_exact(slice_coords).reshape(n_plot, n_plot)
    _save("poisson5d", coords=coords, solution=u_exact(coords), channels=["u"],
          plot_axes=("x1", "x2"), plot_x0=xp, plot_x1=xp,
          plot_grids={"u": U}, slice_val=slice_val)


def poisson10d():
    """Harmonic -Δu = 0 on [0, 1]¹⁰, exact u = Σ x_{2i-1}·x_{2i}."""
    n_eval, n_plot, slice_val = 50_000, 128, 0.5

    def u_exact(x):
        return np.sum(x[:, 0::2] * x[:, 1::2], axis=1)

    rng = np.random.default_rng(0)
    coords = rng.uniform(0.0, 1.0, size=(n_eval, 10))

    xp = np.linspace(0, 1, n_plot)
    X, Y = np.meshgrid(xp, xp, indexing="ij")
    slice_coords = np.concatenate(
        [X.ravel()[:, None], Y.ravel()[:, None],
         np.full((n_plot**2, 8), slice_val)], axis=1)
    U = u_exact(slice_coords).reshape(n_plot, n_plot)
    _save("poisson10d", coords=coords, solution=u_exact(coords), channels=["u"],
          plot_axes=("x1", "x2"), plot_x0=xp, plot_x1=xp,
          plot_grids={"u": U}, slice_val=slice_val)


# Lid-driven cavity is solved with Dedalus (see the ldc example) and shipped as
# data; it is intentionally absent from GENERATORS.
GENERATORS = {
    "burgers": burgers, "kdv": kdv, "ks": ks, "wave": wave,
    "laplacian": laplacian, "poisson5d": poisson5d, "poisson10d": poisson10d,
}


def main(argv):
    names = argv or list(GENERATORS)
    for name in names:
        if name not in GENERATORS:
            raise SystemExit(f"unknown reference '{name}'; choose from {list(GENERATORS)}")
        print(f"generating {name} ...")
        GENERATORS[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
