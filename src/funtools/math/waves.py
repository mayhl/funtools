from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import scipy.signal as sig
from numpy.typing import NDArray
from scipy import interpolate


def newtonIterative(
    f: Callable,
    df: Callable,
    x0: float,
    tolerance: float = 10**-8,
    max_iterations: int = 100,
) -> float:

    x = x0
    for i in range(max_iterations):

        dx = -f(x) / df(x)
        x += dx
        if np.all(np.abs(dx) < tolerance):
            break

    return x


g = 9.81


def dispersion(k: float, h: float):

    w = np.sqrt(g * k * np.tanh(k * h))

    return Wave(w=w, k=k, h=h)


def _f(k: float, w: float, h: float) -> float:
    return g * k * np.tanh(k * h) - w**2


def _dfdk(k: float, w: float, h: float) -> float:
    return g * (np.tanh(k * h) + k * np.cosh(k * h) ** -2)


@dataclass
class Wave:
    k: float | NDArray
    h: float | NDArray
    w: float | NDArray

    @property
    def f(self):
        return self.w / (2 * np.pi)

    @property
    def tp(self):
        return 2 * np.pi / self.w

    @property
    def lp(self):
        return 2 * np.pi / self.k

    @property
    def kh(self):
        return self.h * self.k

    def filter(self, filt: NDArray):

        k = self.k[filt]
        h = self.h[filt]
        w = self.w[filt]

        return Wave(k=k, h=h, w=w)


def computeCritFreq(h: float, kh: float = np.pi) -> float:

    k = kh / h
    w = np.sqrt(g * k * np.tanh(kh))

    return w / (2 * np.pi)


def computeOptimalDepth(period, crit_padding=1.5, point_per_wavelength=60):

    _crit_depth_ratio_ = 1 / 15
    w = 2 * np.pi / period
    dx_ratio = _crit_depth_ratio_ * crit_padding
    r = dx_ratio * point_per_wavelength

    b = r * w**2
    a = 2 * np.pi * g * np.tanh(2 * np.pi / r)

    h = a / b
    return h, dx_ratio * h


def solveDispersion(
    w: float | None = None,
    h: float | None = None,
    k: float | None = None,
    tolerance: float = 10**-8,
    max_iterations: int = 100,
) -> Wave:

    is_w, is_h, is_k = chks = tuple((x is not None for x in [w, h, k]))

    n = int(np.sum(chks))
    if not n == 2:
        raise Exception("Must specify only two of w, h, and k")

    if not is_k:

        # Linear approximation
        k0 = np.sqrt(w**2 / (g * h))
        f = partial(_f, w=w, h=h)
        df = partial(_dfdk, w=w, h=h)

        k = newtonIterative(
            f, df, k0, tolerance=tolerance, max_iterations=max_iterations
        )
        return Wave(k=k, h=h, w=w)
    elif not is_w:
        raise NotImplementedError()
    elif not is_h:
        raise NotImplementedError()

    # Sanity check
    raise Exception("Unexpected State")


# sub_ffts Number of sub ffts to performs, i.e., nfft scales with size of data
def compute_spectra(
    eta, dt, t=None, sub_ffts=1, scaling="density", tlim=None, **welch_kwargs
):

    fs = 1.0 / dt

    if not t is None:
        if not tlim is None:
            i0, i1 = [int(np.round(x / dt)) for x in tlim]
            eta = eta[i0 : i1 + 1]
    else:

        # Get indices of limits
        t0, t1 = t.min(), t.max() if tlim is None else tlim

        # Interpolating onto equispaced grid
        n = int(np.round((t1 - t0) / dt))
        ti = np.arange(0, n + 1) * dt + t0
        eta = np.interp(ti, t, eta)

    # Setting default kwargs
    kwargs = dict(nfft=len(eta) // sub_ffts, scaling=scaling)
    # Overiding defaults
    kwargs.update(welch_kwargs)

    f, spec_den = sig.welch(eta, fs=fs, **kwargs)

    return f, spec_denc

    # energy = np.trapezoid(spec_den, f)

    # hmo = energy2hmo(energy)

    # return f, spec_den, hmo, energy


def energy2hmo(energy):
    hrms = np.sqrt(energy * 8)
    hmo = np.sqrt(2.0) * hrms
    return hmo
