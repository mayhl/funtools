import funtools.io as funio
import funtools.grid as grid
import funtools.projection

import numpy as np
from pathlib import Path
from cmocean import cm

import holoviews as hv
from holoviews import opts

from typing import Tuple

from scipy.interpolate import RegularGridInterpolator


class Simulation():
    """Class for parseing a FUNWAVE simulation directory output."""
    def __init__(self, sim_dpath: str, stride: int = 1,  mask_dry:bool = False, mask_sponges: bool = False, mask_wavemaker: bool = False):

        # Reading parameters from input.txt file
        self._dpath = dpath = Path(sim_dpath)
        input_fpath = dpath / "input.txt"
        self._params = funio.read_input_file(input_fpath)

        # Constructing grid
        self._n, self._m = tuple((self._params[s] for s in ['Nglob', 'Mglob']))
        self._dx, self._dy = (self._params[s] for s in ['DX', 'DY']) 
        x0, y0, x1, y1 = 0, 0 , self._m*self._dx, self._n*self._dy
        x, y = grid.linspace2d(x0, x1, self._m, y0, y1, self._n, mode='left')
        # Bounds for image plotting
        self._bounds = (x0, y0, x1, y1)

        # Adjusting index range based on masks
        i0, i1 = 0, self._m
        j0, j1 = 0, self._n

        print(mask_sponges, mask_wavemaker)
        if mask_sponges:
            i0 = round(self.params['Sponge_west_width']/self._dx)
            i1 -= round(self.params['Sponge_east_width']/self._dx)
            
            j0 = round(self.params['Sponge_south_width']/self._dy)
            j1 -= round(self.params['Sponge_north_width']/self._dy)

        if mask_wavemaker:

            key = 'Wc_WK'
            if key not in self.params:
                raise Exception("Set wavemaker mask, but '%s' not found in input.txt." % key)
            
            di = round(self.params[key]/self._dx)

            if mask_sponges and di < i0:
                raise Exception("Wavemaker detected in sponge layer.")
            i0 = di

        # Initializing I/O parameters
        self._out_dpath = dpath / self._params['RESULT_FOLDER']
        self._sx = slice(i0, i1, stride)
        self._sy = slice(j0, j1, stride)

        self._x = x[self._sx]
        self._y = y[self._sy]


        is_stations = np.all([k in self.params for k in ['STATIONS_FILE', 'NumberStations']])

        if is_stations:
            is_stations = self.params['NumberStations'] > 0

        if is_stations: 
            fpath = dpath / self.params["STATIONS_FILE"]

            sta_idx = np.loadtxt(fpath).astype(int) - 1

            self._sta_x = (sta_idx[:,1]*self._dx - 0 )
            self._sta_y = (sta_idx[:,0]*self._dy - 00 )


            i = 3
            self._sta_x[3] += 300
            self._sta_y[6] -= 100

   
        print(is_stations)
        self._dx *= stride
        self._dy *= stride

        self._is_stations = is_stations
        self._mask_dry = mask_dry

    @property
    def params(self) -> dict:
        """FUNWAVE input file parameters"""
        return self._params

    @property
    def x(self) -> np.ndarray:
        return self._x
    @property
    def y(self) -> np.ndarray:
        return self._y

    def load_data_step(self, name: str, step: int) -> np.ndarray:
        return funio.load_data_step(self._out_dpath, name, step, self._n, self._m)[self._sy, self._sx]

    def load_data(self, name: str) -> np.ndarray:
        return funio.load_data(self._out_dpath, name, self._n, self._m)[self._sy, self._sx]

    def load_bathy(self) -> np.ndarray:
        return -self.load_data('dep.out')
            
    def load_mask_step(self, step: int) -> np.ndarray:
        return self.load_data_step("mask", step).astype(bool)

    def plot_data(self, data: np.ndarray, **opts):
        # NOTE: Image follows the standard of image storage, i.e.,
        #       positive y points downwards, hence np.flipud
        return hv.Image(np.flipud(data), bounds=self._bounds).opts(**opts)



class ProjectedSimulation(Simulation):

    def __init__(self, sim_dpath: str, 
                 espg_code: int, 
                 transform_fpath: str, 
                 stride: int = 1, 
                 mask_dry: bool = False, 
                 mask_sponges: bool = False,
                 mask_wavemaker: bool = False):
        super().__init__(sim_dpath, stride=stride,  mask_dry=mask_dry, mask_sponges=mask_sponges, mask_wavemaker=mask_wavemaker)


        # Setting up projection from FUNWAVE grid to Bokeh/Holowviews plot 
        proj_geo = funtools.projection.GeoProjection(espg_code)
        proj_fun = funtools.projection.RotationProjection.from_funwave_info(transform_fpath)
        proj_bok = funtools.projection.BokehProjection()
        self._proj = funtools.projection.LinkedProjections([proj_bok, proj_geo, proj_fun])

        # Creating rectilinear grid in projected coordinates
        u0, v0, u1, v1 = self._get_proj_bounds(self._x ,self._y)
        ds = np.mean([self._dx, self._dy])
        self._u, self._v = grid.nearest_linspace2d(u0, u1, ds, v0, v1, ds)
        (uu, vv), self._uv_shp = grid.flat_meshgrid(self._u, self._v)
        xx, yy = self._proj.to_target(uu, vv)
        self._pts_interp = yy, xx


        if self._is_stations:
            self._sta_u, self._sta_v = self.proj.to_source(self._sta_x, self._sta_y)
    
    @property
    def proj(self) -> funtools.projection.LinkedProjections:
        return self._proj

    def _get_proj_bounds(self, x: np.ndarray, y: np.ndarray, is_inverse: bool = False) -> Tuple[int, int, int, int]:

        func = self._proj.to_target if is_inverse else self._proj.to_source
        m, n = len(x), len(y)
        xb = np.concatenate([[x.min()]*n, [x.max()]*n, x          , x         ])
        yb = np.concatenate([ y         , y          , [y.min()]*m,[y.max()]*m])
        args = ((s.min(), s.max()) for s in func(xb, yb))
        (u_lower, u_upper), (v_lower, v_upper) = args
        return u_lower, v_lower, u_upper, v_upper

    def _interpolate_data(self, data: np.ndarray) -> np.ndarray:
        args = (self._y, self._x), data
        kwargs = dict(bounds_error=False)
        return RegularGridInterpolator(*args, **kwargs)(self._pts_interp).reshape(self._uv_shp) 


    def load_data(self, name: str) -> np.ndarray:
        return self._interpolate_data(super().load_data(name))

    def load_data_step(self, name: str, step: int) -> np.ndarray:
        return self._interpolate_data(super().load_data_step(name, step))

    #def load_mask_step(self, step: int) -> np.ndarray:
     #   return self._interpolate_data(super().load_mask_step(step))

    def plot_utm(self, name: str, step: int,
                 gbl_opts: dict = {}, 
                 plot_opts: dict = {},
                 bathy_opts:  dict = {},
                 station_opts: dict = {},
                 tile_source: str| None = None):

        # Wrapper for adding optional plots
        plots = []

        if tile_source is not None:
            tiles =  hv.element.tiles.tile_sources[tile_source]() 
            plots.append(tiles)


        # Hacky solution for using same code for bathy 
        is_bathy_main =  name == 'bathy'
        if is_bathy_main:
            data = self.load_bathy()
        else:
            data = self.load_data_step(name, step)

        is_bathy = len(bathy_opts) > 0
        if self._mask_dry and not is_bathy_main:

            mask = self.load_mask_step(step)
            data[~mask] = np.nan
            
            if is_bathy:
                bathy  = self.load_bathy()
                bathy[mask] = np.nan

        elif is_bathy:
            raise Exception("Dry mask set to False, but bathy_opts not specified.")

        u, v = self._u, self._v
        bounds = (u.min(), v.min(), u.max(), v.max())

        img = hv.Image(np.flipud(data), bounds = bounds).opts(**plot_opts)


        plots.append(img)

        if is_bathy:
            bimg = hv.Image(np.flipud(bathy), bounds = bounds).opts(**bathy_opts)
            plots.append(bimg)


        self._is_stations = False
        if self._is_stations and len(station_opts)>0:

            key = "label_opts"
            is_labels = key in station_opts

            if is_labels:
                label_opts = station_opts.pop(key)

            pts = list(zip(self._sta_u, self._sta_v))
            scat = hv.Scatter(pts).opts(**station_opts)
            plots.append(scat)

            if is_labels:

                labels = ["%d" % (i+1) for i in range(len(pts))]
                labels = hv.Labels({('x', 'y'): pts, 'text': labels}, ['x', 'y'], 'text').opts(**label_opts)
                plots.append(labels)

                station_opts[key] = label_opts

        return hv.Overlay(plots).opts(**gbl_opts)

        








    
