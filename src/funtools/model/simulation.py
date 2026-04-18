import json
import pickle
import typing
from pathlib import Path

import holoviews as hv
import numpy as np
from cmocean import cm
from pandas._config.config import is_instance_factory
from param import Callable
from scipy.interpolate import RegularGridInterpolator

from funtools.io.field import Parser, ProjectionParser
from funtools.io.input.main import InputFile
from funtools.math.geometry import Polygon, equipartitioned_mask2shape
from funtools.math.projection import LinkedProjections
from funtools.parallel.multi import simple
from funtools.ui.colorbar import ColorBar

from ..math import grid
from ..parallel.mydask import ClassJob, Scheduler

# from holoviews import opts


try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm

hv.extension("bokeh")


class Simulation:
    def __init__(
        self,
        dpath: Path | str,
        view: dict = {},
        force_2d: bool = False,
        dry_run: bool = False,
    ):
        """Parse FUNWAVE simulations by folder containing input.txt file"""

        if isinstance(dpath, str):
            dpath = Path(dpath)

        # FUTURE: add input valid check to class
        self._input = InputFile.from_file(dpath)
        self._dpath = dpath
        self._data = Parser(dpath, self._input)

        self._post_dpath = dpath / "postprocessing"

        self._plot_kwargs = {}
        self._init = {
            "args": (str(dpath),),
            "kwargs": {"view": view, "force_2d": force_2d},
        }

        self._dry_run = dry_run
        self._job = ClassJob(self.__class__, **self._init)

    @property
    def data(self) -> Parser:
        """ "Returns class for reading FUNWAVE 2D field output data by name and timestep"""
        return self._data

    @property
    def input(self) -> InputFile:
        """Returns class containing FUNWAVE input/driver file variables"""
        return self._input

    @property
    def dry_run(self) -> bool:
        """Returns if simulations is in dry mode skipping plots"""
        return self._dry_run

    def set_view(self, **kwargs) -> None:
        """Set view of 2D field data. See io.field.Parser for more info"""

        self._job.create("set_view", kwargs=kwargs, name="Set New View")
        self._data.set_view(**kwargs)

    def _parse_colorbar(self, kwargs: dict):
        """Parses custom colobar options into holoviews options"""
        if not "colorbar" in kwargs:
            return {}

        if isinstance(kwargs["colorbar"], bool):
            return {}

        colorbar = "clabel" in kwargs
        return {"colorbar": colorbar, **ColorBar(**kwargs["colorbar"]).to_holoviews()}

    def _plot_step(
        self,
        data: np.ndarray,
        bounds: tuple[float, float, float, float],
        kwargs: dict = {},
    ) -> hv.element.raster.Image:
        """Returns holoviews plots of main variable"""
        opts = {}
        opts.update(kwargs)

        opts.update(self._parse_colorbar(opts))

        img = hv.Image(np.flipud(data), bounds=bounds)
        img.opts(**opts)

        return img

    def _get_scalebar_opts(self, kwargs: dict = {}) -> dict:
        """ "Parse scalebar kwargs to holoviews options"""
        if len(kwargs) == 0:
            return {}

        new_kwargs = {
            "unit": "m",
            "range": "x",
            "location": "bottom_left",
        }
        new_kwargs.update(**kwargs)

        mapped_kwargs = {f"scalebar_{k}": d for k, d in new_kwargs.items()}
        mapped_kwargs["scalebar"] = True

        return mapped_kwargs

    def _plot_contours(
        self,
        data: np.ndarray | hv.element.raster.Image,
        kwargs: dict = {},
        gbl_kwargs: dict = {},
    ):
        """Returns contour lines at specfied label. Note if label kwargs is specfied
        Contour is converted to collection of Path"""

        opts = {"show_legend": False}

        opts.update(kwargs)

        # filt = np.isnan(data)
        # data = np.ma.masked_array(data, mask=filt)

        if isinstance(data, np.ndarray):
            data = hv.Image(data, bounds=self.data.view_bounds)

        levels = opts.pop("levels")

        is_label = "label" in opts

        if is_label:
            labels = opts.pop("label")

        def parse(val) -> float:
            if isinstance(val, float):
                return val
            if isinstance(val, int):
                return val

            try:
                water_level = self._input.get_flt("WaterLevel")
            except KeyError:
                water_level = 0.0

            var_map = {"WaterLevel": -water_level}
            return var_map[val]

        levels = [parse(it) for it in levels]
        contours = hv.operation.contours(data, levels=levels)
        contours.opts(**opts)

        if not is_label:
            return contours.opts(gbl_kwargs)

        # Seperating contour into lines to apply labels
        contour_data = [x for x in contours.data]

        opts["show_legend"] = True
        colors = opts["cmap"]
        args = list(zip(labels, colors, contour_data))

        label, color, contour = args[0]
        plt = hv.Path(contour, label=label).opts(color=color, **opts, **gbl_kwargs)
        for label, color, contour in args[1:]:
            plt = plt * hv.Path(contour, label=label).opts(
                color=color, **opts, **gbl_kwargs
            )

        return plt

    def plot(
        self,
        name: str,
        index: int,
        kwargs: dict = {},
        data=None,
        no_mask=False,
        bypass_data=None,
    ):
        """ "Returns holoviews/bokeh plot of FUNWAVE 2D field variable
        at specfied timestep with optional configuration"""

        self._plot_args = (name, index)
        self._plot_kwargs = kwargs
        self._job.create("plot", (name, index), kwargs, f"Plot {name}_{index:05d}")
        # Saving method args/kwargs for exporting
        if self.dry_run:
            return

        # Quick method for return empty dict if key does not exsist
        get_kwargs = lambda kw: kwargs[kw] if kw in kwargs else {}

        plt_kwargs = get_kwargs("plot")
        # Optional scalebar
        plt_kwargs.update(self._get_scalebar_opts(get_kwargs("scalebar")))

        if data is None:
            data = self.data.read_step(name, index, bypass_data=bypass_data)
        mask = self.data.read_mask_step(index)

        if no_mask:
            data_masked = data
        else:
            data_masked = np.ma.masked_array(data, mask=mask)
            data_masked[mask] = np.nan
        bounds = self.data.view_bounds
        plt = self._plot_step(data_masked, bounds, plt_kwargs)

        # Optional bathy contour lines
        opts = get_kwargs("bathy_contour")
        if len(opts) > 0:
            bathy = np.flipud(self.data.read_bathy())
            plt = plt * self._plot_contours(bathy, opts, kwargs["gbl"])

        # Setting default global options
        x0, y0, x1, y1 = self.data.view_bounds
        gbl_kwargs = {
            "aspect": "equal",
            "xlim": (x0, x1),
            "ylim": (y0, y1),
        }

        if "gbl" in kwargs:
            gbl_kwargs.update(kwargs["gbl"])

        return plt.opts(**gbl_kwargs)

    def plot_flood(self, name: str, index: int, kwargs: dict = {}):
        """ "Returns holoviews/bokeh plot of FUNWAVE 2D field variable
        at specfied timestep with optional configuration"""

        self._plot_args = (name, index)
        self._plot_kwargs = kwargs
        self._job.create("plot", (name, index), kwargs, f"Plot {name}_{index:05d}")

        # Quick method for return empty dict if key does not exsist
        get_kwargs = lambda kw: kwargs[kw] if kw in kwargs else {}

        # Saving method args/kwargs for exporting
        if self.dry_run:
            return

        water_level = self._input.get_flt("WaterLevel")

        plt_kwargs = get_kwargs("plot")
        # Optional scalebar
        plt_kwargs.update(self._get_scalebar_opts(get_kwargs("scalebar")))

        data = np.flipud(self.data.read_step(name, index))
        mask = np.flipud(self.data.read_mask_step(index))
        bathy = np.flipud(self.data.read_bathy())
        data_masked = np.ma.masked_array(data, mask=bathy >= -water_level)

        bounds = self.data.view_bounds

        if "bathy" in kwargs:
            plt_kwargs = get_kwargs("bathy")
            plt = self._plot_step(bathy, bounds, plt_kwargs)
        else:
            plt = self._plot_step(data_masked, bounds, plt_kwargs)

        n, m = mask.shape

        mask2 = np.zeros((n, m)).astype(bool)

        for j in range(n):
            for i in range(m):
                mask2[j, i] = bathy[j, i] < -water_level or (mask[j, i] == True)
        # mask2 = (bathy <= 0) & mask

        data_land = np.ma.masked_array(data, mask=mask2)

        bathy2 = bathy.copy()
        bathy2[bathy2 > 0] = 0

        data_land -= bathy2

        data_land += water_level

        plt_kwargs = get_kwargs("flood")

        plt = self._plot_step(data_land, bounds, plt_kwargs)
        # plt = plt * self._plot_step(data_land, bounds, plt_kwargs)
        # plt = self._plot_step(data_land, bounds, {})
        # Optional bathy contour lines
        opts = get_kwargs("bathy_contour")
        if len(opts) > 0:
            bathy = np.flipud(self.data.read_bathy())
            plt = plt * self._plot_contours(bathy, opts)

        # Setting default global options
        x0, y0, x1, y1 = self.data.view_bounds
        gbl_kwargs = {
            "aspect": "equal",
            "xlim": (x0, x1),
            "ylim": (y0, y1),
        }

        if "global" in kwargs:
            gbl_kwargs.update(kwargs["global"])

        return plt.opts(**gbl_kwargs), data_land

    def save_plot(
        self,
        name: str,
        index: int,
        kwargs: dict = {},
        output_dpath: str | None = None,
    ):
        """Saves one or more plots as png files in postprocessing subdirectory of simulation folder.
        Optional directory alternate file placement"""

        if output_dpath is None:
            output_dpath = self._dpath / "postprocessing"
        elif isinstance(output_dpath, str):
            output_dpath = Path(output_dpath)

        output_dpath.mkdir(parents=True, exist_ok=True)

        assert isinstance(output_dpath, Path)
        fpath = output_dpath / f"{name}_{index:05d}.png"
        plt = self.plot(name, index, kwargs)
        hv.save(plt, fpath)
        del plt

    def save_plots(
        self,
        name: str,
        index: int | list[int],
        kwargs: dict = {},
        output_dpath: str | None = None,
    ):
        """Saves one or more plots as png files in postprocessing subdirectory of simulation folder.
        Optional directory alternate file placement"""

        if isinstance(index, int):
            index = [index]

        if output_dpath is None:
            output_dpath = self._dpath / "postprocessing"
        elif isinstance(output_dpath, str):
            output_dpath = Path(output_dpath)

        output_dpath: Path = output_dpath / f"{name}_timeseries"
        output_dpath.mkdir(parents=True, exist_ok=True)

        assert isinstance(output_dpath, Path)
        for i in tqdm(index, desc="Plotting"):
            fpath = output_dpath / f"plot_{i:05d}.png"
            plt = self.plot(name, i, kwargs)

            # Calling plot function to save args/kwargs in dry run
            if self.dry_run:
                break

            hv.save(plt, fpath)
            del plt

    def export_batch_json(
        self, fpath: Path, subbatch_size=1, output_dpath: Path | str | None = None
    ) -> None:
        """Converts class history into Scheular manifest and applies to all timesteps for parallel execution"""

        scheduler = Scheduler()
        tasks = self._job._tasks

        last_view_task = None
        for n, t in tasks:
            if t._method == "set_view":
                last_view_task = t
                continue

            if t._method == "plot":
                name, _ = t._args

                idxs = self.data.get_time_steps(name)

                n = len(idxs)
                n_batches = round(n / subbatch_size)
                n_batches = max(n_batches, 1)
                idxs = [idxs[s] for s in grid.even_divide_slices(n, n_batches)]

                for i, subidxs in enumerate(idxs, start=1):
                    job = ClassJob(self.__class__, self._job._args, self._job._kwargs)

                    if not last_view_task is None:
                        job.add(last_view_task)

                    kwargs = {"kwargs": t._kwargs}

                    if not output_dpath is None:
                        kwargs["output_dpath"] = str(output_dpath)

                    job.create(
                        "save_plots",
                        (name, subidxs),
                        kwargs,
                    )

                    job_name = f"Batch Plot {name} {i:d}"
                    scheduler.add(job, name=job_name)

            scheduler.export_manifest_json(fpath)

    def get_save_plot_manifest(self, name: str | None = None) -> dict:
        """Return dict manifest of last plot call for future execution. Option input dict to override plot kwargs"""
        job = ClassJob(self.__class__, **self._init)
        job.create("plot", self._plot_args, self._plot_kwargs, name=name)
        return job.get_manifest()

    def compute_flooding(self, n_procs: int = 1, subatch_size: int = 1):
        """Return the total flooding as a function of time and the number of times grid point was dry"""
        bathy = self.data.read_bathy()
        land = bathy > 0

        try:
            water_level = self.input.get_flt("WaterLevel")

        except KeyError:
            water_level = 0

        no_water_level = bathy > -water_level

        idxs = self.data.get_time_steps("eta")

        dx, dy = [self.input.get_flt(s) for s in ["DX", "DY"]]
        cell_area = dx * dy

        if n_procs == 1:
            subidxs = [idxs]
        else:
            n = len(idxs)
            n_batches = n // subatch_size
            subidxs = [idxs[s] for s in grid.even_divide_slices(n, n_batches)]

        args = [(self, i, land) for i in subidxs]
        # args = [(self._dpath, i) for i in subidxs]
        flood, dry_counts = zip(
            *[simple(_compute_flood_steps, n_procs, args, p_desc="Computing")][0]
        )

        flood = np.concatenate(flood)

        # Reducing to find total dry count
        dry_count = dry_counts[0]
        for m in dry_counts[1:]:
            dry_count = dry_count + m

        flood = flood * cell_area

        flood_dpath = self._post_dpath / "flood"
        flood_dpath.mkdir(parents=True, exist_ok=True)

        fpath = flood_dpath / "dry_time.npy"
        with open(fpath, "wb") as fh:
            np.save(fh, dry_count)

        flood_area = (dry_count < n) & land
        fpath = flood_dpath / "mask.npy"
        with open(fpath, "wb") as fh:
            np.save(fh, flood_area)

        x = self.data.x
        y = self.data.y
        poly = equipartitioned_mask2shape(x, y, flood_area, n_procs=n_procs)
        fpath = flood_dpath / "poly.pkl"
        with open(fpath, "wb") as fh:
            pickle.dump(poly, fh, pickle.HIGHEST_PROTOCOL)

        t = np.array(idxs) * self.input.get_flt("PLOT_INTV")
        water_level = no_water_level & ~land
        land = np.sum(land) * cell_area

        poly = equipartitioned_mask2shape(x, y, water_level, n_procs=n_procs)
        fpath = flood_dpath / "water_level_poly.pkl"
        with open(fpath, "wb") as fh:
            pickle.dump(poly, fh, pickle.HIGHEST_PROTOCOL)

        no_water_level = np.sum(no_water_level) * cell_area - land

        fpath = flood_dpath / "data.json"
        data = {
            "t": t.tolist(),
            "total": flood.tolist(),
            "land": float(land),
            "water_level": float(no_water_level),
        }

        with open(fpath, "w") as fh:
            json.dump(data, fh)

        plt = hv.Curve((t / 3600, flood / (1000**2)))

        opts = {
            "width": 600,
            "xlabel": "Time (hr)",
            "ylabel": "Flooding (km²)",
            "fontsize": {
                "title": 25,
                "labels": 18,
                "xticks": 16,
                "yticks": 14,
            },
        }

        plt.opts(**opts)

        fpath = flood_dpath / "plot.png"
        hv.save(plt, fpath)

        return plt

    def plot_shapes(self, fpaths: list[str], kwargs: dict = {}):
        labels = kwargs["labels"]
        colors = kwargs["colors"]

        plts = []
        for fpath, label, color in zip(fpaths, labels, colors):
            p = self.data.read_shape(fpath)
            plt = hv.Polygons(p.to_hv_dict(), label=label)
            # NOTE: show_legend needed here
            plt.opts(
                fill_color=color,
                **kwargs["common_opts"],
                show_legend=True,
                **kwargs["global"],
            )
            plts.append(plt)

        plt = plts[0]
        for p in plts[1:]:
            plt = plt * p

        if "bathy" in kwargs:
            get_kwargs = lambda kw: kwargs[kw] if kw in kwargs else {}

            bounds = self.data.view_bounds
            bathy = self.data.read_bathy()
            plt_kwargs = get_kwargs("bathy")

            plt = self._plot_step(bathy, bounds, plt_kwargs) * plt

        x0, y0, x1, y1 = self.data.view_bounds
        gbl_kwargs = {
            "aspect": "equal",
            "xlim": (x0, x1),
            "ylim": (y0, y1),
            "show_legend": True,
        }

        kwargs = {}
        if "global" in kwargs:
            gbl_kwargs.update(kwargs["global"])

        return plt.opts(**gbl_kwargs)


from bokeh.models import ColumnDataSource
from shapely.geometry import MultiPolygon, Polygon


def shapely_2_bokeh_datasource(poly):
    xs_dict = []
    ys_dict = []

    polys = [poly] if isinstance(poly, Polygon) else poly.geoms

    polys = [p for p in polys if isinstance(p, Polygon)]

    xs = []
    ys = []
    holes = []
    for p in polys:
        holes = list(zip(*[s.xy for s in p.interiors]))
        xi, yi = holes if len(holes) == 2 else [], []
        xe, ye = p.exterior.xy

        xs.append(xs)
        ys.append(ys)

        xs_dict.append([{"exterior": list(xe), "holes": list(xi)}])
        ys_dict.append([{"exterior": list(ye), "holes": list(yi)}])

    xs = [[[p["exterior"], *p["holes"]] for p in mp] for mp in xs_dict]
    ys = [[[p["exterior"], *p["holes"]] for p in mp] for mp in ys_dict]

    return ColumnDataSource(dict(xs=xs, ys=ys))


def _compute_flood_steps(self, steps, land):
    # def _compute_flood_steps(dpath, steps):
    # self = Simulation(dpath)
    # bathy = self.data.read_bathy()
    # land = bathy > 0

    dry_count = np.zeros(land.shape)
    flood = []
    for i in steps:
        mask = self.data.read_mask_step(i)

        flood.append(np.sum(~mask & land))
        dry_count = dry_count + mask

    return flood, dry_count


class ProjectedSimulation(Simulation):
    """Derived class for projected FUNWAVE data into Geospatial coordinates"""

    def __init__(
        self, dpath: Path, espg_code: int, transform_fpath: Path, view: dict = {}
    ):
        super().__init__(dpath)

        if isinstance(dpath, str):
            dpath = Path(dpath)

        # Swapping parser so plots method uses projected data instead
        self._raw_data = self._data
        self._data = ProjectionParser(dpath, espg_code, transform_fpath)

        self._init = {
            "args": (str(dpath), espg_code, str(transform_fpath)),
            "kwargs": {"view": view},
        }

        self._job = ClassJob(self.__class__, **self._init)

    @property
    def raw_data(self) -> Parser:
        return self._raw_data

    def plot_shapes(self, fpaths: list[str], kwargs: dict = {}):
        plt = super().plot_shapes(fpaths, kwargs)

        if "tile_map" in kwargs:
            tiles = hv.element.tiles.tile_sources[kwargs["tile_map"]]()
            plt = tiles * plt

        x0, y0, x1, y1 = self.data.view_bounds
        gbl_kwargs = {
            "aspect": "equal",
            "xlim": (x0, x1),
            "ylim": (y0, y1),
        }

        kwargs = {}
        if "global" in kwargs:
            gbl_kwargs.update(kwargs["global"])

        return plt.opts(**gbl_kwargs)

    def plot(
        self,
        name: str,
        index: int,
        kwargs: dict = {},
        data=None,
        no_mask=False,
        bypass_data=None,
    ):
        # Calls same paraent plot routines with projected data
        plt = super().plot(
            name, index, kwargs, data=data, no_mask=no_mask, bypass_data=bypass_data
        )

        # Call parent plot first to saving arg/kwargs in dry run mode
        if self.dry_run:
            return

        if "tile_map" in kwargs:
            tiles = hv.element.tiles.tile_sources[kwargs["tile_map"]]()
            plt = tiles * plt

        x0, y0, x1, y1 = self.data.view_bounds
        gbl_kwargs = {
            "aspect": "equal",
            "xlim": (x0, x1),
            "ylim": (y0, y1),
        }

        if "global" in kwargs:
            gbl_kwargs.update(kwargs["global"])

        return plt.opts(**gbl_kwargs)

    def plot_flood(self, name: str, index: int, kwargs: dict = {}):
        # Calls same paraent plot routines with projected data
        plt, data = super().plot_flood(name, index, kwargs)

        # Call parent plot first to saving arg/kwargs in dry run mode
        if self.dry_run:
            return

        if "tile_map" in kwargs:
            tiles = hv.element.tiles.tile_sources[kwargs["tile_map"]]()
            plt = tiles * plt

        x0, y0, x1, y1 = self.data.view_bounds
        gbl_kwargs = {
            "aspect": "equal",
            "xlim": (x0, x1),
            "ylim": (y0, y1),
        }

        if "global" in kwargs:
            gbl_kwargs.update(kwargs["global"])

        return plt.opts(**gbl_kwargs), data
