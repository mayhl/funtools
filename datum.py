import os
import pickle
import numpy as np
from scipy.spatial import cKDTree as KDTree

from shapely import Polygon, MultiPolygon

import funtools.bokeh as qbokeh

from funtools.error import *
from funtools.create_scatter_poly import generate as generate_scatter_poly
from funtools.create_equipartition_poly import generate as generate_equipartition_poly
from funtools.buffer_fill_poly import create as buffer_fill_poly



class BufferData:
    def __init__(self, data=None):

        self._data = data
        self._fpath = None
        

    def match_existing_fpath(self, dpath, base_name):
        
        fpath = os.path.join(dpath, base_name)
        npy_fpath = fpath + ".npy"
        pkl_fpath = fpath + ".pkl"
        is_npy  = os.path.exists(npy_fpath)
        is_pkl  = os.path.exists(pkl_fpath)

   
        if is_npy == is_pkl:
            if is_npy: raise Exception ("Numpy and pickle file detected.")
            # No files exist 
            return 

        self._fpath = npy_fpath if is_npy else pkl_fpath
     

    def get(self):
        if self._data is None and not self._fpath is None:
            self.load()
        return self._data
        
    def save(self, dpath, base_name):
        fpath = os.path.join(dpath, base_name)
        if isinstance(self._data, np.ndarray):
            fpath += ".npy"
            np.save(fpath, self._data)
        else:
            fpath += ".pkl"
            with open(fpath, "wb") as f:
                pickle.dump(self._data, f)

        self._fpath = fpath

    def load(self):
        if self._fpath is None:
            raise Exception(
                "Tried load data that has not been previously saved or set."
            )

        ext = self._fpath.split(".")[-1]

        if ext == "npy":
            self._data = np.load(self._fpath)
        else:
            with open(self._fpath, "rb") as f:
                self._data = pickle.load(f)

    def flush(self, name):
        if self._data is None:
            return
        if self._fpath is None:
            warnings.warn("Flushing data that does not seem to be have saved.")
        self._data = None

class Datum:
    def __init__(self, xyz=None, poly=None, is_satellite=False):
        self.xyz = xyz
        self.poly = poly
        self._is_satellite = is_satellite
        self._fig = None
        self._container = None
        self._renderer_poly = None
        self._renderer_xyz = None

        self.poly_plot_kwargs = None
        self.xyz_plot_kwargs = None

    @property
    def _smaps(self):
        return [
            # (self._xyz, "x"),
            (self._poly, "_poly"),
            (self._kdtree, "_kdtree"),
        ]

    def match_existing_fpaths(self, dpath, base_name):

        for var, suffix in self._smaps:
            fname = base_name + suffix
            if var is None: 
                var = BufferData()
                setattr(self, suffix, var)
            var.match_existing_fpath(dpath, fname)  
            if var.get() is None: var = None
    
    def save(self, dpath, base_name):
        for var, suffix in self._smaps:
            if var is None: continue
            if not var.get() is None:
                fname = base_name + suffix
                var.save(dpath, fname)
                var.flush(fname)

    def construct_boundary_poly(self, method, **kwargs):
        assert self.poly is None, "Polygon already exists."

        mmap = {
            "simple": generate_scatter_poly,
            "equipartition": generate_equipartition_poly,
        }

        assert method in mmap.keys(), "Unknown boundary polygon method '%s'." % method
        #self.poly = mmap[method](self.xyz[:, 0:2], **kwargs)
        self.poly = mmap[method](self.kdtree, **kwargs)

    def modify_boundary_poly(self, method, **kwargs):
        assert (
            not self.poly is None
        ), "Boundary polygon must first be created before modifying."

        mmap = {"buff_fill": buffer_fill_poly}

        assert method in mmap.keys(), "Unknown modify polygon method '%s'." % method
        self.poly = mmap[method](self.poly, **kwargs)

    @property
    def xyz(self):
        kdtree = self.kdtree
        return None if kdtree is None else kdtree.data

    @xyz.setter
    def xyz(self, value):
        
        if value is not None:
            self.kdtree = KDTree(value)
        else:
            self._kdtree = None
     
    @property
    def kdtree(self):
        return None if self._kdtree is None else self._kdtree.get()

    @kdtree.setter
    def kdtree(self, value):
        self._kdtree = BufferData(value)

    @property
    def poly(self):
        return self._poly.get()

    @poly.setter
    def poly(self, value):
        self._poly = BufferData(value)

    @property
    def has_data(self):
        return not (self.xyz is None and self.poly is None)

    @property
    def fig(self):
        return self._fig

    @property
    def renderer_poly(self):
        return self._renderer_poly

    @property
    def renderer_xyz(self):
        return self._renderer_xyz

    def init_figure(self, **kwargs):
        # Changing default value
        key = "match_aspect"
        if not key in kwargs.keys():
            kwargs[key] = True

        kwargs["is_satellite"] = self._is_satellite
  
        self._fig, self._container = qbokeh.qcreate_figure(**kwargs)

        # # Extract figure sizes to use in 'container' for better sizing_mode
        # cont_kwargs = pop_dict_keys(['width', 'height'], kwargs)
        # # Don't set sizing_mode keyword if already defined,
        # # or if both width and height are specified
        # key =  'sizing_mode'
        # if not key in kwargs.keys() and len(cont_kwargs) < 2 :
        #     # Defaults to 'scale_height' mode if width keyword is not specified
        #     kwargs[key]  = 'stretch_width' if  'height' in  cont_kwargs  else 'stretch_height'

        # #is_width = 'width' in kwargs.keys()
        # #is_height = 'height' in kwargs.keys()
        # #if not (is_width and is_height):
        #     # Don't set keyword argument if both keyword args are specified
        #     key =  'sizing_mode'
        #     # Defaults to 'scale_height' mode if width keyword is not specified
        #     key_val = 'stretch_width' if  is_height  else 'stretch_width'
        #     if not key in kwargs.keys(): kwargs[key] = key_val

        # self._fig = qbokeh.create_figure(self._is_satellite, **kwargs)
        # self._container = qbokeh.create_container(self._fig, **cont_kwargs)

    def _init_figure(self, fig):
        is_fig = not fig is None
        is_sfig = not self._fig is None

        # Error if both are None or both are not None
        if is_fig == is_sfig:
            if is_fig:
                msg = "Figure has already been initialized."
            else:
                msg = """Argument 'fig' must be specified or class function, or
                                class method 'init_figure' must be called first."""
            raise qerror(TypeError, msg)

        # Setting class member _fig if fig argument is not None
        if is_fig:
            self._fig, self._container = qbokeh.qcreate_container(fig)

        # GHOST CASE: is_fig = False, is_sfig = True
        # In this case, class member _fig has already been set

    def create_poly_plot(
        self, fig=None, color="blue", label=None, is_skip_checks=False, **fig_kwargs
    ):
        if not is_skip_checks:
            assert not self.xyz is None, "No polygon data."
            self._init_figure(fig, **fig_kwargs)

        
        args = (self.fig, self.poly)
        
        plt_kwargs = dict(color=color, line_width=2)
        if label is None: 
            self._renderer_poly = qbokeh.plot_poly(*args, **plt_kwargs)
        else:

            # NOTE: Messy
            if type(self.poly) in [MultiPolygon, Polygon]:    
                plt_kwargs['legend_label'] = label
                self._renderer_poly, self._renderer_poly_mark = qbokeh.plot_poly(*args, **plt_kwargs)
            else:
                self._renderer_poly= qbokeh.plot_poly(*args, **plt_kwargs) 

    def create_xyz_plot(self, fig=None, color="blue", is_skip_checks=False, **fig_kwargs
    ):
        if not is_skip_checks:
            assert not self.xyz is None, "No XYZ scatter data."
            self._init_figure(fig, **fig_kwargs)

        args = (self.fig, self.xyz)

        if self.xyz_plot_kwargs is None: 
            plt_kwargs = dict(color=color)
        else:
            plt_kwargs = self.xyz_plot_kwargs 
            
        self._renderer_xyz = qbokeh.plot_scatter(*args, **plt_kwargs)

    def create_xyz_cmap_plot(self, fig=None, **fig_kwargs):

        assert not self.xyz is None, "No XYZ scatter data."
        self._init_figure(fig, **fig_kwargs)

        args = (self.fig, self.xyz)
        plt_kwargs = dict(size = 4)
        self._renderer_xyz_cmap = qbokeh.plot_scatter_cmap(*args, **plt_kwargs)

    def create_plot(self, fig=None, label=None, color="blue", **fig_kwargs):
        assert self.has_data, "No data to plot."

        self._init_figure(fig, **fig_kwargs)

        if not self.xyz is None:
            self.create_xyz_plot(fig, color=color, is_skip_checks=True, **fig_kwargs)

        if not self.poly is None:
            print(fig_kwargs)
            self.create_poly_plot(fig, color=color, label=label, is_skip_checks=True, **fig_kwargs)

    def show(self, **kwargs):
        assert not self._container is None, "No figure created."
        qbokeh.show(self._container, **kwargs)

    def _plot(self, plot_func, fig_kwargs={}, plt_kwargs={}, **kwargs):
        
        show_kwargs = qbokeh.pop_show_kwargs(kwargs)

        if len(kwargs) > 0:
            raise qerror(TypeError, "Unknown keyword arguments: %s." % kwargs.keys())

        self.init_figure(**fig_kwargs)
        plot_func(**plt_kwargs)
        self.show(**show_kwargs)

    def plot(self, fig_kwargs={}, plt_kwargs={}, **kwargs):
        self._plot(self.create_plot, fig_kwargs, plt_kwargs, **kwargs)

    def plot_xyz(self, fig_kwargs={}, plt_kwargs={}, **kwargs):
        self._plot(self.create_xyz_plot, fig_kwargs, plt_kwargs, **kwargs)
        
    def plot_xyz_cmap(self, fig_kwargs={}, plt_kwargs={}, **kwargs):
        self._plot(self.create_xyz_cmap_plot, fig_kwargs, plt_kwargs, **kwargs)

    def plot_poly(self, fig_kwargs={}, plt_kwargs={}, **kwargs):
        self._polt(self.create_poly, fig_kwargs, plt_kwargs, **kwargs)
