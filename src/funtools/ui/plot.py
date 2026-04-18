from __future__ import annotations

import holoviews as hv
import numpy as np

from ..math.geometry import Polygon
from .colorbar import ColorBar

hv.extension("bokeh")


def image(x, y, z, **opts):
    bounds = (x[0], y[0], x[-1], y[-1])
    return image_raw(z, bounds)


def _parse_custom_opts(**opts):

    opts = opts.copy()
    opts.update(ColorBar.parse_custom_opts(**opts))
    return opts


class Tiles:

    def __init__(self, name: str | None) -> None:

        self._is_tiles = not name is None

        if self._is_tiles:
            tile_srcs = hv.element.tiles.tile_sources
            if not name in tile_srcs:
                raise ValueError(f"Invalid tiles '{name}': {tile_srcs.keys()}")

            self._tiles = tile_srcs[name]()

        else:
            self._tiles = None

    def merge(self, plt):

        if self._is_tiles:
            return self._tiles * plt
        else:
            return plt


def parse_tiles_opts(projection_key: str, **opts):

    __DEFAULT__ = "EsriImagery"
    opts = opts.copy()

    if "tiles" in opts:
        name = opts.pop("tiles")
    else:
        name = None

    is_tile_mode = projection_key == "mrc"

    if is_tile_mode and name is None:
        name = __DEFAULT__

    return opts, Tiles(name)


def getFontSize(
    title: int,
    labels: int,
    ticks: int,
    disable_x: bool = False,
    disable_y: bool = False,
) -> dict:

    opts = dict(
        title=title,
        xlabel=labels,
        ylabel=labels,
        clabel=labels,
        xticks=ticks,
        yticks=ticks,
        cticks=ticks,
        legend=ticks,
    )

    if disable_x:
        opts["xticks"] = 0
        opts["xlabel"] = 0

    if disable_y:
        opts["yticks"] = 0
        opts["ylabel"] = 0

    return opts


def image_raw(data, bounds, **opts):

    opts = _parse_custom_opts(**opts)
    return hv.Image(data, bounds=bounds).opts(**opts)


def scatter(data, label: str | None = None, **opts):

    opts = _parse_custom_opts(**opts)
    kwargs = {"vdims": ["y", "z"]}
    if not label is None:
        kwargs["label"] = label

    return hv.Scatter(data, **kwargs).opts(**opts)


def curve(data, label: str | None = None, **opts):

    kwargs = {}
    if not label is None:
        kwargs["label"] = label

    return hv.Curve(data, **kwargs).opts(**opts)


def arrow(
    origin: tuple[float, float],
    angle: float,
    scale: float,
    opts: dict = {},
    head_angle: float = 35,
    head_length: float = 0.15,
):

    head_angle = np.deg2rad(head_angle)
    angle = -np.deg2rad(angle)

    dx = head_length * np.cos(head_angle)
    dy = head_length * np.sin(head_angle)

    x = np.array([-0.5, 0.5, 0.5 - dx, 0.5, 0.5 - dx]) * scale
    y = np.array([0, 0, dy, 0, -dy]) * scale

    x0, y0 = origin

    u = x0 + x * np.cos(angle) + y * np.sin(angle)
    v = y0 - x * np.sin(angle) + y * np.cos(angle)

    return hv.Curve((u, v)).opts(**opts)


def polygons(data: list[dict] | Polygon, **opts):

    if isinstance(data, Polygon):
        data = data.to_hv_dict()

    return hv.Polygons(data).opts(**opts)
