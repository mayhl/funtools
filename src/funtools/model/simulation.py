from pathlib import Path
import typing
from cmocean import cm
import holoviews as hv

# from holoviews import opts


from pandas._config.config import is_instance_factory
from param import Callable
from scipy.interpolate import RegularGridInterpolator

import numpy as np
from pathlib import Path

from funtools.io.input import InputFile
from funtools.io.field import Parser, ProjectionParser
from funtools.math.projection import LinkedProjections
from funtools.math import grid
from funtools.ui.colorbar import ColorBar

from ..parallel.mydask import ClassJob, Scheduler

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

        return {"colorbar": True, **ColorBar(**kwargs["colorbar"]).to_holoviews()}

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

        img = hv.Image(data, bounds=bounds)
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
        self, data: np.ndarray | hv.element.raster.Image, kwargs: dict = {}
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

            var_map = {"WaterLevel": -self._input.get_flt("WaterLevel")}
            return var_map[val]

        levels = [parse(it) for it in levels]
        contours = hv.operation.contours(data, levels=levels)
        contours.opts(**opts)

        if not is_label:
            return contours

        # Seperating contour into lines to apply labels
        contour_data = [x for x in contours.data]

        opts["show_legend"] = True
        colors = opts["cmap"]
        args = list(zip(labels, colors, contour_data))

        label, color, contour = args[0]
        plt = hv.Path(contour, label=label).opts(color=color, **opts)
        for label, color, contour in args[1:]:
            plt = plt * hv.Path(contour, label=label).opts(color=color, **opts)

        return plt

    def plot(self, name: str, index: int, kwargs: dict = {}):
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

        data = np.flipud(self.data.read_step(name, index))
        mask = np.flipud(self.data.read_mask_step(index))
        data_masked = np.ma.masked_array(data, mask=mask)

        bounds = self.data.view_bounds
        plt = self._plot_step(data_masked, bounds, plt_kwargs)

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

        return plt.opts(**gbl_kwargs)

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
        for i in index:
            fpath = output_dpath / f"eta_{i:05d}.png"
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

    def plot(self, name: str, index: int, kwargs: dict = {}):
        # Calls same paraent plot routines with projected data
        plt = super().plot(name, index, kwargs)

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
