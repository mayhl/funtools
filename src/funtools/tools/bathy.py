from __future__ import annotations

import holoviews as hv
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from funtools.ui import plot


class Bathy:

    def __init__(self, dx, dy, z):

        if np.any(np.isnan(z)):
            raise Exception("Bathymetry contains NaNs")

        self._z = z
        self._ny, self._nx = ny, nx = z.shape
        self._dx = dx
        self._dy = dy

        self._x = np.arange(nx) * dx
        self._y = np.arange(ny) * dy

        self.bounds = (0, 0, nx * dx, ny * dy)

        self._log_history = []

    def _log(self, name, **kwargs):
        self._log_history.append((name, kwargs))

    @property
    def shape(self):
        return self.z.shape

    @property
    def x(self):
        return self._x

    @property
    def dx(self):
        return self._dx

    @property
    def nx(self):
        return self._nx

    @property
    def y(self):
        return self._y

    @property
    def dy(self):
        return self._dy

    @property
    def ny(self):
        return self._ny

    @property
    def z(self):
        return self._z

    def plot(self, stride=1, xy_scale=1, **opts):

        bounds = [s / xy_scale for s in self.bounds]
        return plot.image_raw(np.flipud(self.z[::stride, ::stride]), bounds, **opts)

    def _copy(self, z, dx=None, dy=None):

        if dx is None:
            dx = self.dx

        if dy is None:
            dy = self.dy

        new_bathy = Bathy(dx, dy, z)
        new_bathy._log_history = self._log_history
        return new_bathy

    def blend_y_boundary(self, extend_length, sigma):

        # TODO: Fix doubling
        n_pad = (round(extend_length / self.dy) // 2) * 2
        extend_length = 2 * n_pad * self.dy

        self._log("blend north south", extend_length=extend_length, sigma=sigma)

        south_boundary = self._z[0, :]
        north_boundary = self._z[-1, :]
        extend = np.zeros((2 * n_pad, self.nx))
        ramp = (np.arange(2 * n_pad) + 1) / (2 * n_pad + 1)
        for i in range(2 * n_pad):
            extend[i, :] = south_boundary * ramp[i] + north_boundary * (1 - ramp[i])

        extend_smooth = gaussian_filter(extend, sigma)
        extend_final = extend_smooth
        ramp = 2 * np.abs(0.5 - ramp)
        power = 12
        for i in range(2 * n_pad):
            extend_final[i, :] = extend[i, :] * ramp[i] ** power + extend_smooth[
                i, :
            ] * (1 - ramp[i] ** power)
        z_new = np.concatenate(
            [extend_final[n_pad:, :], self.z, extend_final[:n_pad, :]], axis=0
        )

        return self._copy(z_new)

    def _get_idx_bound(self, x0, y0, x1, y1, relative, args={}):

        is_x0 = not x0 is None
        is_x1 = not x1 is None
        is_y0 = not y0 is None
        is_y1 = not y1 is None

        if is_x0:
            args["x0"] = x0
            i0 = round(x0 / self.dx)
        else:
            i0 = None

        if is_x1:
            args["x1"] = x1

            if relative:
                x1 = self.nx * self.dx - x1
            i1 = round(x1 / self.dx) + 1
        else:
            i1 = None

        if is_y0:
            args["y0"] = y0
            j0 = round(y0 / self.dy)
        else:
            j0 = None

        if is_y1:
            args["y1"] = y1
            if relative:
                y1 = self.ny * self.dy - y1

            j1 = round(y1 / self.dy) + 1
        else:
            j1 = None

        return slice(i0, i1), slice(j0, j1), args

    def flatten(self, max_depth, x0=None, y0=None, x1=None, y1=None, relative=False):

        args = {"max_depth": max_depth}
        sx, sy, args = self._get_idx_bound(x0, y0, x1, y1, relative, args)

        print(sx, sy)
        self._log("flatten", **args)

        z_new = self.z.copy()

        sub_z = z_new[sy, sx].copy()

        filt = z_new < -max_depth
        sub_z[filt] = -max_depth

        z_new[sy, sx] = sub_z
        return self._copy(z_new)

    def crop(self, x0=None, y0=None, x1=None, y1=None, relative=False):

        sx, sy, args = self._get_idx_bound(x0, y0, x1, y1, relative)
        self._log("crop", **args)
        return self._copy(self.z[sy, sx])

    def contours(self, levels):

        img = self.plot()

        levels = hv.operation.contours(img, levels=levels)

        return levels

    def largestContourBounds(self, level):
        data = self.contours([level]).data[0]

        idxs = np.argwhere(np.isnan(data["x"])).flatten()
        idxs = np.hstack([[0], idxs, [len(data["x"])]])

        i = np.argmax(np.diff(idxs))
        i0, i1 = idxs[i : i + 2]
        sl = slice(i0 + 1, i1)
        x = data["x"][sl]
        y = data["y"][sl]

        return x.min(), y.min(), x.max(), y.max()

    def blend_x0_flat(self, z_flat, x_length):

        self._log("blend_x0_flat", z_flat=z_flat, x_length=x_length)

        nx = round(x_length // self.dx)
        # z_ext = np.zeros([self.ny, nx])
        s = 1 - np.linspace(0, 1, nx + 1)[:-1]

        a = np.full(self.ny, z_flat)
        b = self.z[:, 0]
        z1 = np.outer(a, s) + np.outer(b, 1 - s)
        z2 = z_flat * s + b.mean() * (1 - s)

        power = 8
        s = s**power
        z_ext = z1 * (1 - s)[np.newaxis, :] + (z2 * s)[np.newaxis, :]

        z_new = np.concatenate([z_ext, self.z], axis=1)
        return self._copy(z_new)

    def extend_x(self, x0=None, x1=None):

        args = {}
        zs = [self.z]

        if not x0 is None:
            args["x0"] = x0

            nx = round(x0 // self.dx)
            z_ext = np.tile(self.z[:, 0], (nx, 1)).T
            zs.insert(0, z_ext)

        if not x1 is None:
            args["x1"] = x1

            nx = round(x1 // self.dx)
            z_ext = np.tile(self.z[:, -1], (nx, 1)).T
            zs.append(z_ext)

        assert len(args) > 0, "No arguments entered."
        self._log("extend_x", **args)

        z_new = np.concatenate(zs, axis=1)

        return self._copy(z_new)

    def smooth(self, sigma: float) -> Bathy:
        self._log("smooth", sigma=sigma)
        z = gaussian_filter(self.z, sigma)
        return self._copy(z)

    def slope(self):

        dx = (self._z[:, 2:] - self._z[:, :-2]) / (2 * self._dx)

        dx = np.concatenate([dx[:, :1], dx, dx[:, -1:]], axis=1)
        dy = (self._z[2:, :] - self._z[:-2, :]) / (2 * self._dy)

        dy = np.concatenate([dy[:1, :], dy, dy[-1:, :]], axis=0)
        z = np.hypot(dx, dy)
        return self._copy(z)

    def interpolate(self, dx, dy):

        self._log("interpolate", dx=dx, dy=dy)
        *_, xl, yl = self.bounds

        args = (self.y, self.x), self.z
        f = RegularGridInterpolator(*args)

        x = np.arange(xl // dx) * dx
        y = np.arange(yl // dy) * dy

        x = x[(self.x[0] < x) & (x < self.x[-1])]
        y = y[(self.y[0] < y) & (y < self.y[-1])]

        x, y = np.meshgrid(x, y)

        z_new = f((y, x))
        return self._copy(z_new, dx=dx, dy=dy)
