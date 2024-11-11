from funtools.parse_raw_data import * 
#from src.create_scatter_poly import generate as generate_scatter_poly
#from src.create_equipartition_poly import generate as generate_equipartition_poly
#from src.buffer_fill_poly import fill as buffer_fill_poly
from funtools.datum import Datum
import funtools.bokeh as qbokeh

from funtools.misc import pop_dict_keys
from funtools.error import *
from funtools.transform import Transformer

from shapely import GeometryCollection, MultiPolygon, Polygon, LineString, MultiLineString
from bokeh.models import ColorBar

import numpy as np
from glob import glob
import os

import pickle
import re

import warnings


class SourceGrids:
    @classmethod
    def bulk_read_extension(
        cls, dpath, extension, exclude_masks=[], is_debug=False, **kwargs
    ):
        ext = extension
        if ext[0:2] == "*.":
            ext = ext[2:]
        if ext[0:1] == ".":
            ext = ext[1:]
        ext_mask = "*." + ext

        grids = cls._bulk_read_mask(dpath, ext_mask, exclude_masks, is_debug, **kwargs)
        # Removing extension of key
        return cls({k.replace("." + ext, ""): d for k, d in grids.items()})

    @classmethod
    def _bulk_read_mask(
        cls, dpath, file_mask, exclude_masks=[], is_debug=False, **kwargs
    ):
        def match(string, mask):
            # Hack to use simple wildcards
            mask = "^" + mask.replace("*", ".*") + "$"
            regex = re.compile(mask)
            matches = re.findall(regex, string)
            return len(matches) == 0

        def is_keep(fpath):
            name = os.path.basename(fpath)
            return np.all([match(name, mask) for mask in exclude_masks])

        fpath_mask = os.path.join(dpath, file_mask)
        fpaths = glob(fpath_mask)
        fpaths = [f for f in fpaths if is_keep(f)]
        if is_debug:
            pass
            #fpaths = fpaths[0:1]

        if len(fpaths) == 0:
            raise Exception("No files found matching file path mask '%s'." % fpath_mask)

        rkwargs = kwargs["reader"] if "reader" in kwargs else kwargs

        def qgrid(fpath):
            rkwargs["fpath"] = fpath
            return UnstructuredGrid(**kwargs)

        return {os.path.basename(f): qgrid(f) for f in fpaths}

    @classmethod
    def merge(cls, sources):
        sources = [s._data for s in sources]
        merged_srcs = sources[0].copy()
        for s in sources[1:]:
            conflicts = [k for k in s if k in merged_srcs]
            if len(conflicts):
                raise Exception(
                    "Can not merge data sources, conflicting keys: '%s'." % conflicts
                )

            merged_srcs.update(s)

        return cls(merged_srcs)
        
    @classmethod
    def simple_contours(cls, segs, src_crs, key):
 
        NewClass = lambda s: UnstructuredGrid.simple_contour(s, src_crs, key)
        grids={i: NewClass(s) for i, s in enumerate(segs)}
        return cls(grids)

        
    def __init__(self, data=[]):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def keys(self):
        return self._data.keys()

    def items(self): return self._data.items()

    def values(self): return self._data.values()

    def save(self, dpath, fname=None):
        
        for k, d in self._data.items(): d.save(dpath, k, is_nested=True)

        info = dict(keys=list(self.keys()))
        
        if fname is None: fname = "grids_info"
        fpath = os.path.join(dpath, fname)
        with open(fpath + ".pkl", "wb") as f: pickle.dump(info, f)

    @classmethod
    def load(cls, dpath, fname=None):
        if fname is None: fname = "grids_info"
        fpath = os.path.join(dpath, fname)
        
        with open(fpath + ".pkl", "rb") as f: info = pickle.load(f)

        keys = info['keys']

        data = {k: UnstructuredGrid() for k in keys}

        for k, d in data.items(): d.load(dpath, base_name=k)
            
        return cls(data)

    def construct_boundary_polys(self, method, **kwargs):
        for d in self._data.values():
            d.construct_boundary_poly(method, **kwargs)

    def transform(self, src, trg, is_xyz_only=False, is_poly_only=False):
        for k, d in self._data.items(): d.transform(src, trg, is_xyz_only, is_poly_only)

    def set_target_crs(self, crs):

        for d in self._data.values():
            d._transformer.set_target_crs(crs)

    def plot(self, key, fig_kwargs={}, plt_kwargs={}, cmapper=None, fig=None,  **kwargs):

        is_cbar = cmapper is not None
        colors = cmapper.palette if is_cbar else qbokeh.quick_palette(self._data)

            
        items = list(zip(self._data.items(),colors))

        (k, g), c = items[0]
        if not key in g.keys(): 
            raise Exception("Invalid datum type")

        is_fig = fig is not None
        
        d = g._data[key]

        if not is_fig:
            d.init_figure(**fig_kwargs)
            d.create_plot(color=c, label=k, **plt_kwargs)
        else:
            d.create_plot(fig=fig, color=c, label=k, **plt_kwargs)

        #for (k, g), c in items[1:]: g._data[key].create_plot(fig=fig, color=c, **plt_kwargs, label=k)
        for (k, g), c in items[1:]: g._data[key].create_plot(fig=d.fig, color=c, **plt_kwargs, label=k)

        if len(d.fig.legend) > 0:
            pass
            d.fig.legend.location = "center"
            d.fig.add_layout(d.fig.legend[0], 'below')

        if is_cbar:
            cb = ColorBar(color_mapper = cmapper, 
                         label_standoff = 14,
                         location = (0,0),
                         title = 'Topography/Bathymetry (m)')
            d.fig.add_layout(cb, 'below')
        
        show_kwargs = qbokeh.pop_show_kwargs(kwargs)
        if not is_fig: d.show(**show_kwargs)


    def filter_keys(self, keys):
        self._data = {k: d for k,d in self._data.items() if k in keys}
        
    def filter_poly_intersection(self, poly, key, key_poly=None):
        for d in self._data.values(): d.filter_poly_intersection(poly,  key, key_poly=key_poly)
        self._data = {k: d for k, d in self._data.items() if d.data[key].poly.area>0}

    def filter_xyz_in_poly(self, key, n_procs=1):
        for k, d in self._data.items(): 
            d.filter_xyz_in_poly(key, n_procs=n_procs)

    
    def remove_overlap_by_rank(self, key, ranks):

        for i, ri in enumerate(ranks):
            if not ri in self._data: continue
            pi = self._data[ri].data[key].poly
            
            for rj in ranks[i+1:]:
                if not rj in self._data: continue
                
                pj = self._data[rj].data[key].poly
                pj = pj.difference(pi)
       
                if isinstance(pj, GeometryCollection):
                    tpolys = [s for s in pj.geoms if type(s) in [Polygon, MultiPolygon]]
                    pj = MultiPolygon(tpolys)
    
                self._data[rj].data[key].poly =pj

        dkeys = [k for k in self._data.keys() if self._data[k].data[key].poly.area <= 1]
        for k in dkeys: del self._data[k]
    

class UnstructuredGrid:

    def keys(self):
        return self._data.keys()

    @classmethod
    def simple_poly(cls, x , y, src_crs, key):
        grid = cls()
        data = np.vstack([x, y]).T
        grid._data[key].xyz = data
        grid._data[key].poly = Polygon(data)
        grid._transformer = Transformer(src_crs)
        grid._is_data_read = True
        return grid
        
    @classmethod
    def simple_contour(cls, segs, src_crs, key):

        grid=cls()
        grid._data[key].poly = MultiLineString(segs) 
        grid._transformer = Transformer(src_crs)
        grid._is_data_read = True
        return grid
    
    def __init__(self, **kwargs):
        self._is_data_read = False

        keys = ["source", "target", "satellite"]
        is_satellite = [False, False, True]
        self._data = {k: Datum(is_satellite=i) for k, i in zip(keys, is_satellite)}

        if np.all([not x in kwargs for x in ["reader", "fpath"]]): return

        skwargs = kwargs["reader"] if "reader" in kwargs else kwargs
        self.read_data(**skwargs)
        
        key = "poly_initial"
        if not key in kwargs: return
        self.construct_boundary_poly(**kwargs[key])#, **self._temp_args)

        key = "poly_modifies"
        if not key in kwargs: return
        
        for a in kwargs[key]:
            self.modify_boundary_poly(**a)

    @property
    def data(self): return self._data

    def __getitem__(self, key):
        return self._data[key]
        
    def load(self, dpath, base_name=None):
        
        def get_key(k):
            return k if base_name is None else "%s_%s" % (base_name, k)
        
        info_fpath = os.path.join(dpath, get_key("info.pkl"))
        with open(info_fpath, "rb") as f: info = pickle.load(f)    

        self._transformer = info['transformer']
        self._is_data_read = info['is_data_read']

        for k, d in self._data.items():
            d.match_existing_fpaths(dpath, get_key(k))
        
    def save(self, dpath, base_name=None, is_nested=False):
        
        def get_key(k):
                return k if base_name is None else "%s_%s" % (base_name, k)
            
        info = dict(transformer=self._transformer,
                           is_data_read=self._is_data_read)
        
        info_fpath = os.path.join(dpath, get_key("info.pkl"))   
        with open(info_fpath, "wb") as f: pickle.dump(info, f)
        
        data = {k: d for k, d in self._data.items() if not d is None}

        for k, d in data.items():
            d.save(dpath, get_key(k))
        
        if not is_nested:
            raise NotImplementedError()

        return info_fpath

    @property
    def pts(self):
        assert not self._data is None, "Data not read."
        return self._data[:, 0:2]

    @property
    def x(self):
        assert not self._data is None, "Data not read."
        return self._data[:, 0]

    @property
    def y(self):
        assert not self._data is None, "Data not read."
        return self._data[:, 1]

    @property
    def z(self):
        assert not self._data is None, "Data not read."
        return self._data[:, 2]

    def read_data(self, fpath=None, method=None, **kwargs):
        is_fpath = not fpath is None
        is_method = not method is None

        if not is_fpath == is_method:
            raise Exception("fpath and method must both be specfied.")

        if self._is_data_read:
            raise Exception("Can load data; data has already been read.")

        pmap = {
            "csv": read_csv,
            "table": read_pandas_csv,
            "geotiff": read_geotiff,
            "ww3": read_ww3_mesh,
        }

        assert method in pmap.keys(), "Unknown read method '%s'." % method

        xyz, src_crs, extra_args = pmap[method](fpath, **kwargs)

        for i, var in enumerate(["x", "y", "z"]):
            idx = np.isnan(xyz[:, i])
            n = np.sum(idx)
            if n > 0:
                warnings.warn(
                    "Removing %d points due to NaN %s values, File Path: %s."
                    % (n, var, fpath)
                )
                xyz = xyz[~idx, :]

        
        self._transformer = Transformer(src_crs)
        #self._src_crs = src_crs
        #self._data["raw"] = Datum(xyz=xyz)
        self._data['source'].xyz = xyz
        self._is_data_read = True

        # Tempoary linkage args
        self._temp_args = extra_args

    def construct_boundary_poly(self, method, **kwargs):
        self._data['source'].construct_boundary_poly(method, **kwargs)

    def modify_boundary_poly(self, method, **kwargs):
        self._data['source'].modify_boundary_poly(method, **kwargs)

    def filter_poly_intersection(self, poly, key, key_poly=None):
        assert key in self._data, "Invalid key"
        p = self._data[key].poly 
        assert p is not None, "No poly in datum"

        if key_poly is not None:
            poly = self._transformer.proj_poly(poly, key_poly, key)

        self._data[key].poly = p.intersection(poly)

    def filter_xyz_in_poly(self, key, n_procs=1):

        from funtools.polyfilter import filter_scatter_poly as filter_xyz_in_poly
        assert key in self._data, "Invalid data key"
        xyz = self._data[key].xyz
        poly = self._data[key].poly

        xyz = filter_xyz_in_poly(xyz, poly, n_procs=n_procs)
        self._data[key].xyz = xyz
    
    def transform(self, src, trg, is_xyz_only=False, is_poly_only=False):

        assert src in self._data, "Source datum '%s' does not exist." % src
        assert trg in self._data, "Target datum '%s' does not exist." % trg

        poly = self._data[src].poly
        if not is_xyz_only and poly is not None:
            self._data[trg].poly = self._transformer.proj_poly(poly, src, trg)

        if not is_poly_only and self._data[src].xyz is not None:
            xyz = self._data[src].xyz
            x, y = xyz[:,0], xyz[:,1] 
            x, y = self._transformer.proj_xyz(x, y, src, trg)
            xyz[:,0], xyz[:,1] = x, y
            self._data[trg].xyz = xyz




def linspace(s0, s1, n, mode='centered'):

    mmap = {
        'centered': (False, lambda dx: dx/2),
        'left'    : (False, lambda dx: 0   ),
        'right'   : (False, lambda dx: dx  ),
        'border'  : (True , lambda dx: 0   )
    }

    e, o = mmap[mode]
    o = o((s1-s0)/n)
    return np.linspace(s0+o, s1+o, n, endpoint=e) 

def linspace2d(x0, x1, nx, y0, y1, ny, mode='centered'):
    x = linspace(x0, x1, nx, mode)
    y = linspace(y0, y1, ny, mode)
    return x , y



def rectilinear(n, ds, s0=0, mode='centered'):
    mmap = {
        'centered': (0, 1/2),
        'left'    : (0, 0  ),
        'right'   : (0, 1  ),
        'border'  : (1, 0  )
    }
    ep, o = mmap[mode]
    return (np.arange(n+ep) + o)*ds + s0

def rectilinear2d(nx, dx, ny, dy, x0=0, y0=0, mode='centered'):
    x = rectilinear(nx, dx, x0, mode)
    y = rectilinear(ny, dy, y0, mode)
    return x , y




# Generated linear space for some given spacing centered on some range
def nearest_linspace(s0, s1, ds, mode='centered'):
    
    l = s1-s0
    n = int(np.floor(l/ds))
    if n < 1: n = 1

    mmap = {
        'centered': (False, lambda dx: dx/2),
        'left'    : (False, lambda dx: 0   ),
        'right'   : (False, lambda dx: dx  ),
        'border'  : (True , lambda dx: 0   )
    }

    # Centering grid on domain

    e, o = mmap[mode]
    o = o(ds) + (l-n*ds)/2
    return np.linspace(s0+o, s1+o, n, endpoint=e) 


def nearest_linspace2d(x0, x1, dx, y0, y1, dy, mode='centered'):
    x = nearest_linspace(x0, x1, dx, mode)
    y = nearest_linspace(y0, y1, dy, mode)
    return x , y
    

def flat_meshgrid(*args, **kwargs):
    aargs = np.meshgrid(*args, **kwargs)
    shp = aargs[0].shape
    aargs = (ss.flatten() for ss in aargs)
    return aargs, shp



    

