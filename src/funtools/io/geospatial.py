from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import rasterio
from pyproj import CRS

from ..math.geometry import Polygon
from ..math.kdtree import KDTree
from ..math.projection import (BoxProjection, PolygonProjection,
                               RotationProjection, ScatterProjection)
from ..ui import plot
from ..ui.colorbar import ColorBar

__METERS2FEET__ = 3.28084
__FEET2METERS__ = 0.3048


class Datum:

    def __init__(
        self,
        data,
        key: str,
        source_crs: Any,
        target_crs: Any,
        invert_z: bool = False,
        is_feet: bool = False,
        z_offset: float = 0.0,
        poly_args: dict = {},
    ) -> None:

        self._crs_pair = source_crs, target_crs
        self._xy = ScatterProjection(source_crs, target_crs, data[:, :2].T)

        self._crs_pair = self._xy._src_crs, self._xy._trg_crs

        factor = -1 if invert_z else 1
        if is_feet:
            factor *= __FEET2METERS__

        z = data[:, 2:].T.copy()
        z[0, :] = factor * (z[0, :] + z_offset)
        self._z = z

        self._key = key
        # self._invert_z = invert_z
        # self._is_feet = is_feet
        # self._z_offset = z_offset

        self._poly_args = poly_args
        self._plot_idxs = None
        self._poly = None

        self._is_filter = False

    def get_scatter(self, key: str) -> np.ndarray:
        return np.concatenate([self._xy[key], self._z], axis=0).T

    @property
    def key(self) -> str:
        return self._key

    def set_max_plot_count(self, maximum_points: int, workers: int = 1) -> None:
        self._plot_idxs, *_ = KDTree(self._xy["src"].T)._filter_cell_centered(
            maximum_points, workers=workers
        )

    def plot(self, key="src", label: None | str = None, **opts):
        """Wrapper  method filtering points for plotting"""

        data = self.get_scatter(key)

        if not self._plot_idxs is None:
            data = data[self._plot_idxs]

        need_vmin, need_vmax = ColorBar.check_ranges_required(**opts)

        if need_vmin:
            vmin = data[:, -1].min()
        else:
            vmin = None

        if need_vmax:
            vmax = data[:, -1].max()
        else:
            vmax = None

        opts = ColorBar.get_updated_range_opts(new_vmin=vmin, new_vmax=vmax, **opts)

        opts, tiles = plot.parse_tiles_opts(projection_key=key, **opts)

        plt = plot.scatter(data, label=label, **opts)
        return plt  # tiles.merge(plt)

    @property
    def poly(self) -> PolygonProjection:

        if self._poly is None:
            raise RuntimeError("Boundary polygon not created/initialized.")

        return self._poly

    def create_boundary_poly(
        self, key: str, method: str, debug: bool = False, **opts
    ) -> None:

        if debug and not self._plot_idxs is None:
            data = self._xy[key][:, self._plot_idxs].T
        else:
            data = self._xy[key].T

        match method:
            case "nearest":
                poly = Polygon.from_scatter(data, method, **opts)
            case "equipartition":

                args = self._poly_args
                poly = Polygon.from_equipartition(args["x"], args["y"], args["data"])
                self._poly_args = {}
            case _:
                raise ValueError(f"Invalid method '{method}'.")

        self._poly = PolygonProjection(*self._crs_pair, poly)

    def plot_poly(self, key: str, **opts):

        if self._poly is None:
            raise RuntimeError("Boundary polygon not created/initialized.")

        return plot.polygons(self._poly[key].to_hv_dict(), **opts)

    @classmethod
    def from_file(cls, dpath: Path | str, fname: str, target_crs: Any) -> Datum:

        fpath = Path(dpath) / fname
        with open(fpath, "rb") as fh:
            info = pickle.load((fh))

        crs = info["src_crs"], target_crs
        self = Datum(info["xyz"], fpath.stem, *crs)

        self._poly = PolygonProjection(*crs, Polygon(info["poly"]))

        if "rot_proj" in info:

            rot_proj = RotationProjection.from_dict(info["rot_proj"])

            self._xy._rot_proj = rot_proj
            self._poly._rot_proj = rot_proj

        return self

    def to_file(self, dpath: Path | str) -> str:

        fname = f"{self._key}.pkl"
        fpath = Path(dpath) / fname

        src_crs, _ = self._crs_pair
        data = dict(
            xyz=self.get_scatter("src"),
            poly=self.poly["src"]._poly,
            src_crs=src_crs,
        )

        rot_proj = self._xy._rot_proj
        if not rot_proj is None:
            data["rot_proj"] = rot_proj.to_dict()

        with open(fpath, "wb") as fh:
            pickle.dump(data, fh)

        return fname

    def filter_data_in_poly(self, target_n: int, n_filter: int) -> None:

        d = self.get_scatter("src")
        indices = self._poly["src"].equi_subdivide(d, target_n, n_filter)

        self._xy.apply_filter(indices)

        if len(indices) > 0:
            self._z = self._z[:, indices]
        else:
            d, _ = self._z.shape
            self._z = np.zeros((d, 0))

    def apply_rotated_window(self, key: str, box: BoxProjection) -> None:

        p = self._poly[key]
        p._poly = box.poly_interp.intersection(p._poly)

        self._xy._rot_proj = box.proj
        self._poly._rot_proj = box.proj


class Datums:

    def __init__(self, target_crs: Any) -> None:
        self._items = {}
        self._target_crs = target_crs

        # for d in items:
        #    if d.key in self._items:
        #        raise ValueError(f"Duplicate key, {d.key:s}, in Datum list.")

    #
    #           self._items[d.key] = d

    def keys(self) -> list[str]:
        return list(self._items.keys())

    @property
    def target_projection(self):
        key = self.keys()[0]
        return self[key]._xy._trg_proj

    @property
    def rotation_projection(self) -> RotationProjection:
        key = self.keys()[0]
        proj = self[key]._xy._rot_proj

        assert not proj is None
        return proj

    @classmethod
    def from_file(cls, dpath: Path | str, dname: str) -> Datums:
        """Return instance of class from files"""

        dpath = Path(dpath) / dname
        fpath = dpath / "metadata.pkl"
        with open(fpath, "rb") as fh:
            info = pickle.load(fh)

        self = Datums(info["trg_crs"])

        datums = [Datum.from_file(dpath, f, info["trg_crs"]) for f in info["files"]]

        self._items = {d.key: d for d in datums}
        return self

    def to_file(self, dpath: Path | str, dname: str) -> None:
        """Save class data to files"""

        dpath = Path(dpath) / dname
        dpath.mkdir(exist_ok=True, parents=True)

        fnames = [d.to_file(dpath) for d in self._items.values()]

        info = dict(
            # src_crs=self._source_crs,
            trg_crs=self._target_crs,
            files=fnames,
        )

        fpath = dpath / "metadata.pkl"
        with open(fpath, "wb") as fh:
            pickle.dump(info, fh)

    def _filter_duplicates(self, data: np.ndarray, submsg: str) -> np.ndarray:
        """Returns data filtering duplicate grid points replacing duplicate
        grid point values with their mean"""
        # Removing duplicate points with same grid values
        n1, _ = data.shape
        data = np.unique(data, axis=0)
        n2, _ = data.shape

        dn = n1 - n2
        if dn > 0:
            logging.warning(
                f"{dn:d} ({100*dn/n1:.2f}%%) duplicate points removed in {submsg}."
            )

        # Wrapper method for extracting indices for an individual duplicate point
        def _filter(dups: np.ndarray) -> np.ndarray:
            idxs = np.argwhere(np.all(dups == unique, axis=1))
            return np.argwhere(idxs[0] == inverse)[:, 0]

        # Getting indices all unique duplicate point
        kwargs = dict(
            axis=0, return_index=True, return_inverse=True, return_counts=True
        )
        unique, indices, inverse, counts = np.unique(data[:, :2], **kwargs)
        duplicates = unique[counts > 1]
        dup_indices = [_filter(d) for d in duplicates]
        nd = len(dup_indices)

        if nd == 0:
            return data

        # Computing statitics
        z_vals = [data[i, 2:] for i in dup_indices]
        z_means = np.array([z.mean(axis=0) for z in z_vals]).flatten()
        z_std = np.array([z.std(mean=z0) for z, z0 in zip(z_vals, z_means)])
        z_rtd = 100 * z_std / np.abs(z_means)
        stats = dict(
            {k: float(getattr(np, k)(z_rtd)) for k in ["min", "median", "max", "std"]}
        )

        # Update values with mean and removing duplicates data
        data = data.copy()
        for i in range(len(z_means)):
            j = dup_indices[i]
            data[j, 2] = z_means[i]

        data = data[indices, :]

        logging.warning(
            f"{dn:d} ({100*dn/n1:.2f}%%) duplicate points with varying values in {submsg}."
        )

        for k, d in stats.items():

            logging.warning(f"{k:6s} - {d:.4f}%")

        return data
        n3, _ = tmp_data.shape

        dn = n2 - n3
        if dn > 0:
            raise ValueError(
                f"{dn:d} ({100*dn/n2:.2f}%%) duplicate grid points width different grid values found."
            )

        return data

    def __getitem__(self, key: str) -> Datum:
        """Return Datum by key"""
        return self._items[key]

    def merge(self, key: str) -> Datums:
        """Return new Datums merging Datum with shared CRS"""
        keys = list(self._items.keys())

        d: ScatterProjection = self._items[keys[0]]._xy
        for k in keys[1:]:

            if not d.is_same_src_crs(self._items[k]._xy):
                raise ValueError("Can not combine datums of different source CRS")

        data = np.concatenate([d.get_scatter("src") for d in self._items.values()])

        msg = f"merge of {key}."
        data = self._filter_duplicates(data, msg)

        datum = Datum(data, key, source_crs=d._src_crs, target_crs=self._target_crs)

        datums = Datums(target_crs=self._target_crs)
        datums._append(datum)
        return datums

    @classmethod
    def _get_files(cls, dpath: Path | str, mask: str) -> list[Path]:
        """Wrapper method for get file list in directory"""
        if isinstance(dpath, str):
            dpath = Path(dpath)

        return list(dpath.glob(mask))

    def _get_updated_range_opts(self, z_idx: int = 0, **opts) -> dict:
        """Returns Holoviews clim opts fitting data range"""
        need_vmin, need_vmax = ColorBar.check_ranges_required(**opts)

        if need_vmin:
            vmin = min([d._z[z_idx, :].min() for d in self._items.values()])
        else:
            vmin = None

        if need_vmax:
            vmax = max([d._z[z_idx, :].max() for d in self._items.values()])
        else:
            vmax = None

        return ColorBar.get_updated_range_opts(new_vmin=vmin, new_vmax=vmax, **opts)

    def plot(self, key: str = "src", **opts):
        """Return scatter plot"""
        opts = self._get_updated_range_opts(**opts)

        opts, tiles = plot.parse_tiles_opts(key, **opts)

        if "color" in opts:
            plts = [d.plot(key, **opts) for d in self._items.values()]
        else:
            plts = [d.plot(key, label=d.key, **opts) for d in self._items.values()]

        plt = plts[0]
        for p in plts[1:]:
            plt = plt * p

        return tiles.merge(plt)

    def plot_poly(self, key: str = "src", **opts):
        """Return boundary polygon plot"""
        opts = self._get_updated_range_opts(**opts)

        opts, tiles = plot.parse_tiles_opts(key, **opts)

        if "color" in opts:
            plts = [d.plot_poly(key, **opts) for d in self._items.values()]
        else:
            plts = [d.plot_poly(key, **opts) for d in self._items.values()]

        plt = plts[0]
        for p in plts[1:]:
            plt = plt * p

        return tiles.merge(plt)

    def project(self, name: str, source: str, target: str) -> None:
        """Projects data member by key name"""
        if name == "poly":
            method_name = "_poly"

        elif name == "scatter":
            method_name = "_xy"
        else:
            raise ValueError(f"Invalid data type name '{name}'")

        for d in self._items.values():
            getattr(d, method_name).project(source, target)

    def create_boundary_poly(
        self, key: str, method: str, debug: bool = False, **opts
    ) -> None:
        """Construct boundary polygon fromm grid data"""
        for d in self._items.values():
            d.create_boundary_poly(key, method, debug=debug, **opts)

    def _parse_fpath_key(self, fpath: str | Path, key: None | str) -> tuple[Path, str]:
        """Wrapper method for validating key from file name"""
        if isinstance(fpath, str):
            fpath = Path(fpath)

        key = fpath.stem if key is None else key

        if key in self._items:
            raise ValueError(f"Duplicate key, {key:s}, in Datum list.")

        return fpath, key

    def read_csv(
        self,
        fpath: Path | str,
        source_crs: Any,
        indices: None | list[int] = None,
        key: None | str = None,
        invert_z: bool = False,
        is_feet: bool = False,
        z_offset: float = 0.0,
        **kwargs,
    ) -> None:
        """Read CSV file"""
        fpath, key = self._parse_fpath_key(fpath, key)

        data = np.loadtxt(fpath, **kwargs)

        _, d = data.shape

        if d < 3:
            raise Exception(f"Less than three columns of data in file: {fpath:s}")

        if not indices is None:
            data = data[:, indices]
        else:
            data = data[:, :3]

        msg = f"file {key}"
        data = self._filter_duplicates(data, msg)
        args = (data, key)

        kwargs = {
            "source_crs": source_crs,
            "target_crs": self._target_crs,
            "invert_z": invert_z,
            "is_feet": is_feet,
            "z_offset": z_offset,
        }

        self._append(Datum(*args, **kwargs))

    def read_csvs(
        self,
        dpath: Path,
        mask: str,
        source_crs: Any,
        indices: None | list[int] = None,
        key: None | str = None,
        invert_z: bool = False,
        is_feet: bool = False,
        z_offset: float = 0.0,
    ) -> None:
        """Read CSV files in directory"""
        kwargs = dict(
            source_crs=source_crs,
            indices=indices,
            invert_z=invert_z,
            is_feet=is_feet,
            z_offset=z_offset,
        )

        fpaths = self._get_files(dpath, mask)
        for i, f in enumerate(fpaths, start=1):
            self.read_csv(f, key=f"{key} {i:d}", **kwargs)

    def read_geotiff(
        self,
        fpath: Path | str,
        key: str | None = None,
        invert_z: bool = False,
        is_feet: bool = False,
        z_offset: float = 0.0,
        band_id: int = 1,
    ) -> None:
        """Read data from geotiff file."""
        # Reading data
        img = rasterio.open(fpath)
        m, n = img.height, img.width
        z = img.read(band_id)

        # Transforming pixel points to coordinates
        n, m = z.shape
        cols, rows = np.meshgrid(np.arange(m), np.arange(n))
        x, y = rasterio.transform.xy(img.transform, rows, cols)

        #  Orienting matrix for increase x, y points
        data = [np.array(s) for s in [x, y, z]]
        x, y = data[0][0, :], data[1][:, 0]
        if y[1] - y[0] < 0:
            data = [np.flipud(s) for s in data]
        if x[1] - x[0] < 0:
            data = [np.fliplr(s) for s in data]

        idxs = data[2] != img.nodata
        x, y = data[0][0, :], data[1][:, 0]
        mask = dict(x=x, y=y, data=idxs)

        # Flattening 2D grid to scatter grid and remove null points
        data = np.vstack([s.flatten() for s in data]).T[idxs.flatten(), :]

        # Converting rasterio CRS class to pyproj CRS class
        source_crs = pyproj.CRS(img.read_crs())  # .to_epsg()

        args = (data, key)
        kwargs = dict(
            target_crs=self._target_crs,
            source_crs=source_crs,
            invert_z=invert_z,
            is_feet=is_feet,
            z_offset=z_offset,
            poly_args=mask,
        )

        return self._append(Datum(*args, **kwargs))

    def read_geotiffs(
        self,
        dpath: Path,
        mask: str = "*.tif",
        key: str | None = None,
        invert_z: bool = False,
        is_feet: bool = False,
        z_offset: float = 0.0,
        band_id: int = 1,
    ) -> None:
        """Read data from geotiffs files in directory"""
        kwargs = dict(
            invert_z=invert_z, is_feet=is_feet, z_offset=z_offset, band_id=band_id
        )

        fpaths = self._get_files(dpath, mask)
        for i, f in enumerate(fpaths, start=1):
            self.read_geotiff(f, key=f"{key} {i:d}", **kwargs)

    def _append(self, datum: Datum):
        """Wrapper method for add a Datum"""
        assert not datum._key in self._items

        src_crs, trg_crs = datum._crs_pair

        if len(self._items) == 0:
            # self._source_crs = src_crs
            self._target_crs = trg_crs
        else:
            pass
            # if not self._source_crs.equals(src_crs):
            #    raise ValueError("Adding datum with different source  CRS.")

        self._items[datum._key] = datum

    def update(self, other: Datums):
        """Add collection of Datum for other Datums to current Datums"""
        self._items.update(other._items)

    def remove_boundary_overlaps(
        self, src: str = "trg", priority_list: list[str] | None = None
    ) -> None:
        """Remove overlaps between datums based on priority. If no priority list is given,
        natural ordering will be used"""

        if priority_list is None:
            keys = list(self._items.keys())
        else:
            keys = priority_list

        rkeys = list(reversed(keys))
        for i, k0 in enumerate(rkeys[:-1]):

            p0 = self[k0].poly[src]._poly

            for k1 in rkeys[i + 1 :]:
                p1 = self[k1].poly[src]._poly
                p0 = p0.difference(p1)

            self[k0].poly[src]._poly = p0
            self[k0]._is_filter = True

    def filter_data_in_poly(self, target_n: int, n_filter: int) -> None:
        for k, d in self._items.items():

            # if not d._is_filter:
            #    continue

            print(k)
            d.filter_data_in_poly(target_n, n_filter)

    def apply_rotated_window(self, key: str, box: BoxProjection) -> None:

        for d in self._items.values():
            d.apply_rotated_window(key, box)

        del_keys = []

        for k, d in self._items.items():
            if not d.poly[key]._poly.area > 0:
                del_keys.append(k)

        for k in del_keys:
            del self._items[k]
