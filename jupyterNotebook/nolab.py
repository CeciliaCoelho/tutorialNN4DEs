"""Helpers for the Neural Operators hands-on lab.

Everything in here is *infrastructure*: the finite-difference Darcy solver,
matplotlib boilerplate, the weather download. None of it is the thing the lab
teaches -- the spectral layer, the model and the training loop stay in the
notebook where you can read and change them.

Import it with:

    from nolab import pick_device, grf, solve_darcy, show
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# --------------------------------------------------------------------------
# where to run
# --------------------------------------------------------------------------
def pick_device():
    """NVIDIA GPU if there is one, else the Apple-Silicon GPU, else the CPU.

    MPS (Metal) is only accepted if it can actually run a spectral layer:
    Experiment 2 needs complex-valued parameters and 2-D FFTs, which older
    PyTorch builds do not implement on Metal.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v = torch.randn(1, 2, 8, 8, device="mps")
                vh = torch.fft.rfft2(v)
                R = torch.randn(2, 2, 4, 4, dtype=torch.cfloat, device="mps")
                torch.fft.irfft2(
                    torch.einsum("bixy,ioxy->boxy", vh[..., :4, :4], R), s=(8, 8))
            return "mps"
        except Exception as e:
            print(f"MPS present but unusable ({type(e).__name__}); using the CPU. "
                  "A newer PyTorch usually fixes this.")
    return "cpu"


# --------------------------------------------------------------------------
# the data generator:  a(x)  ->  u(x)
# --------------------------------------------------------------------------
def grf(n, rng, alpha=2.0, tau=3.0):
    """A smooth Gaussian random field on an n x n grid.

    White noise filtered in Fourier space by (pi^2 |k|^2 + tau^2)^(-alpha/2):
    large `tau` damps the low frequencies less, giving finer-grained fields.
    Returned with zero mean and unit standard deviation.
    """
    xi = rng.standard_normal((n, n))
    kx, ky = np.meshgrid(*2 * [np.fft.fftfreq(n, 1 / n)], indexing="ij")
    coef = (np.pi ** 2 * (kx ** 2 + ky ** 2) + tau ** 2) ** (-alpha / 2)
    coef[0, 0] = 0.0
    g = np.fft.ifft2(np.fft.fft2(xi) * coef).real
    return g / (g.std() + 1e-12)


def solve_darcy(a, f_val=1.0):
    """Solve -div(a grad u) = f on (0,1)^2 with u = 0 on the boundary.

    Standard 5-point finite differences with harmonic averaging of `a` at the
    cell faces, assembled as one sparse system and solved directly. `a` is an
    (m, m) array of permeabilities; the result is the (m, m) pressure field.

    This is the *ground truth* the neural operator is trained against.
    """
    m = a.shape[0]
    h = 1.0 / (m + 1)
    idx = np.arange(m * m).reshape(m, m)
    harm = lambda x, y: 2 * x * y / (x + y)
    cx = harm(a[:, :-1], a[:, 1:])
    cy = harm(a[:-1, :], a[1:, :])
    bx = np.concatenate([a[:, :1], cx, a[:, -1:]], axis=1)
    by = np.concatenate([a[:1, :], cy, a[-1:, :]], axis=0)
    diag = (bx[:, :-1] + bx[:, 1:] + by[:-1, :] + by[1:, :]).ravel()
    rows = [idx.ravel(), idx[:, :-1].ravel(), idx[:, 1:].ravel(),
            idx[:-1, :].ravel(), idx[1:, :].ravel()]
    cols = [idx.ravel(), idx[:, 1:].ravel(), idx[:, :-1].ravel(),
            idx[1:, :].ravel(), idx[:-1, :].ravel()]
    vals = [diag, -cx.ravel(), -cx.ravel(), -cy.ravel(), -cy.ravel()]
    A = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(m * m, m * m))
    f = np.full(m * m, f_val * h ** 2)
    return spla.spsolve(A, f).reshape(m, m).astype(np.float32)


def two_phase(n, rng, tau=3.0, lo=3.0, hi=12.0):
    """A two-phase medium: threshold a random field into `lo` and `hi`."""
    return np.where(grf(n, rng, tau=tau) >= 0, hi, lo).astype(np.float32)


def ascii_medium(art, n=64, lo=3.0, hi=12.0):
    """Turn a block of text into a permeability field ('#' = high, else low)."""
    rows = [r for r in art.strip("\n").split("\n") if r.strip()]
    H, W = len(rows), max(len(r) for r in rows)
    coarse = np.full((H, W), lo, np.float32)
    for i, r in enumerate(rows):
        for j, ch in enumerate(r):
            if ch == "#":
                coarse[i, j] = hi
    ii = np.linspace(0, H - 1, n).astype(int)
    jj = np.linspace(0, W - 1, n).astype(int)
    return coarse[ii][:, jj]


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def show(panels, cmaps="inferno", suptitle=None, size=3.2):
    """Show a dict {title: 2-D array} as a row of images with colour bars.

    `cmaps` is one colormap name for all panels, or a list, one per panel.
    """
    titles = list(panels)
    if isinstance(cmaps, str):
        cmaps = [cmaps] * len(titles)
    fig, axs = plt.subplots(1, len(titles), figsize=(size * len(titles), size))
    axs = np.atleast_1d(axs)
    for ax, t, cm in zip(axs, titles, cmaps):
        im = ax.imshow(panels[t], cmap=cm)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.75)
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def show_pairs(A, U, n=6, suptitle="one operator, many input-output pairs"):
    """Media on the top row, their pressure fields underneath."""
    fig, axs = plt.subplots(2, n, figsize=(1.9 * n, 3.9))
    for j in range(n):
        axs[0, j].imshow(A[j], cmap="viridis")
        axs[1, j].imshow(U[j], cmap="inferno")
        axs[0, j].axis("off")
        axs[1, j].axis("off")
    axs[0, 0].set_title("$a$", fontsize=10)
    axs[1, 0].set_title("$u$", fontsize=10)
    fig.suptitle(suptitle)
    plt.show()


def bar_compare(series, labels_x, ylabel, title):
    """Grouped bar chart. `series` is a dict {name: [v1, v2, ...]}."""
    names = list(series)
    xs = np.arange(len(labels_x))
    w = 0.8 / len(names)
    colors = ["#1F7FA8", "#C3423F", "#7A9E7E", "#C3A972"]
    plt.figure(figsize=(5.8, 3.4))
    for i, name in enumerate(names):
        plt.bar(xs + (i - (len(names) - 1) / 2) * w, series[name], w,
                label=name, color=colors[i % len(colors)])
    plt.xticks(xs, labels_x)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


# --------------------------------------------------------------------------
# Experiment 4: the weather data
# --------------------------------------------------------------------------
def synthetic_weather(n=2922, ny=32, nx=64, seed=1):
    """Offline stand-in for ERA5: two drifting, slowly refreshed smooth fields."""
    r = np.random.default_rng(seed)
    cur = np.stack([grf(nx, r)[:ny, :] for _ in range(2)])
    X = np.empty((n, 2, ny, nx), np.float32)
    for t in range(n):
        cur = np.roll(cur, 1, axis=-1)
        if t % 40 == 0:
            cur = 0.9 * cur + 0.45 * np.stack(
                [grf(nx, r)[:ny, :] for _ in range(2)])
        X[t] = cur
    return X, False


def load_weather(years=("2015", "2016"), timeout_s=300):
    """Stream a coarse ERA5 slice from WeatherBench 2, or fall back offline.

    Returns (X, is_real) with X of shape (time, 2, 32, 64) -- geopotential at
    500 hPa and temperature at 850 hPa, six-hourly, latitude before longitude.

    Note the store itself is laid out (time, longitude, latitude), so we
    transpose: without that the maps come out sideways and `n_modes` ends up
    applied to the wrong axes.

    The download typically takes one to three minutes. Past `timeout_s` we
    stop waiting and hand back the synthetic stand-in, so a workshop never
    stalls on bad wifi; `load_weather(timeout_s=0)` skips it entirely.
    (With no network at all this fails fast, so the long budget costs nothing.)
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout

    def _fetch():
        import xarray as xr
        paths = [
            "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
            "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr",
        ]
        order = ("time", "latitude", "longitude")
        for path in paths:
            try:
                ds = xr.open_zarr(path, storage_options={"token": "anon"})
                z = ds["geopotential"].sel(level=500, time=slice(*years))
                t = ds["temperature"].sel(level=850, time=slice(*years))
                z, t = z.transpose(*order), t.transpose(*order)
                return np.stack([z.values, t.values], axis=1).astype(np.float32)
            except Exception:
                continue
        return None

    if not timeout_s:
        return synthetic_weather()
    print(f"streaming ~2 years of ERA5 from WeatherBench 2 -- usually 1-3 min "
          f"(giving up after {timeout_s}s; load_weather(timeout_s=0) skips it)...")
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            X = ex.submit(_fetch).result(timeout=timeout_s)
        if X is not None:
            print("loaded real ERA5:", X.shape, "(time, channel, lat, lon)")
            return X, True
        print("WeatherBench unreachable -- using the SYNTHETIC stand-in.")
    except FTimeout:
        print(f"download did not finish in {timeout_s}s -- using the SYNTHETIC "
              "stand-in. (Raise timeout_s if you want to insist on real data.)")
    except Exception as e:
        print(f"download failed ({type(e).__name__}) -- using the SYNTHETIC stand-in.")
    return synthetic_weather()
