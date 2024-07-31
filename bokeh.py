from funtools.error import *
from funtools.misc import pop_dict_keys, pop_dict_key

import numpy as np
from bokeh.palettes import all_palettes

from shapely.geometry import Polygon, MultiPolygon, Point, GeometryCollection
from shapely.geometry import LineString, MultiLineString

from bokeh.models import ColumnDataSource, MultiPolygons, Div, Button, CustomJS
from bokeh.models import LinearColorMapper, ColorBar
import xyzservices.providers as xyz
# from bokeh.models import PolyDrawTool
from bokeh.plotting import figure, output_notebook, reset_output
from bokeh.plotting import show as bshow
from bokeh.layouts import row, column
import warnings

import matplotlib.colors as mcolors

# Constant module parameters
LONG_KEY = "is_notebook_output"
SHORT_KEY = "is_nb_out"
IS_NB_OUT_DEFAULT = False
RETURN_KEY = SHORT_KEY

# Module parameter for switching between
# notebook output and default output
IS_NB_OUT = IS_NB_OUT_DEFAULT


def _set_output_mode(is_nb_out):
    global IS_NB_OUT
    if is_nb_out == IS_NB_OUT:
        return
    reset_output() if IS_NB_OUT else output_notebook()
    IS_NB_OUT = is_nb_out


def show(fig, is_nb_out=IS_NB_OUT_DEFAULT):
    _set_output_mode(is_nb_out)
    bshow(fig)


def pop_show_kwargs(kwargs):
    global LONG_KEY, SHORT_KEY, RETURN_KEY, IS_NB_OUT_DEFAULT

    is_s_key, s_key_val = pop_dict_key(SHORT_KEY, kwargs)
    is_l_key, l_key_val = pop_dict_key(LONG_KEY, kwargs)

    if is_s_key == is_l_key:
        # Neither keyword argument, returns the default value
        if not is_s_key:
            return {RETURN_KEY: IS_NB_OUT_DEFAULT}

        # Both keyword arguments are defined
        if not s_key_val == l_key_val:
            msg = """Both keyword arguments '%s' and '%s are specfied and have .
                             different values.""" % (LONG_KEY, SHORT_KEY)
            raise qerror(TypeError, msg)

        msg = """Both keyword arguments '%s' and '%s are specfied. Please only 
                         use one in the future.""" % (LONG_KEY, SHORT_KEY)

        qwarn(msg)

    # Either both keyword arguments are defined and have the same value,
    # or only one of the keyword arguments is defined
    return {RETURN_KEY: s_key_val if is_s_key else l_key_val}


def shapely_2_bokeh_datasource(poly):
    xs_dict = []
    ys_dict = []

    polys = [poly] if isinstance(poly, Polygon) else poly.geoms

    polys = [p for p in polys if isinstance(p, Polygon)]

    for p in polys:
        holes = list(zip(*[s.xy for s in p.interiors]))
        xi, yi = holes if len(holes) == 2 else [], []
        xe, ye = p.exterior.xy
        xs_dict.append([{"exterior": list(xe), "holes": list(yi)}])
        ys_dict.append([{"exterior": list(ye), "holes": list(yi)}])

    xs = [[[p["exterior"], *p["holes"]] for p in mp] for mp in xs_dict]
    ys = [[[p["exterior"], *p["holes"]] for p in mp] for mp in ys_dict]

    return ColumnDataSource(dict(xs=xs, ys=ys))


def quick_palette(var):
    n = var if isinstance(var, int) else len(var)
    if n <= 2:
        n = 3

    if n > 20:
        raise Exception("Only a maximum of 20 palettes allowed.")

    if n <= 10:
        return all_palettes["Category10"][n]
    else:
        return all_palettes["Category20"][n]



def _plot_poly(fig, poly, legend_label=None, **plt_kwargs):
    key = "alpha"
    if key not in plt_kwargs:
        plt_kwargs[key] = 0.5

    src = shapely_2_bokeh_datasource(poly)
    r = fig.multi_polygons(
        xs="xs", ys="ys", source=src, **plt_kwargs
    )  # , color=color, alpha=alpha)

    if legend_label is None:
        return r

    x0, y0, x1, y1 = poly.bounds
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    rm = fig.scatter([xc], [yc], marker="square", size=0.1, legend_label=legend_label, **plt_kwargs) 
    
    rm.visible = False
    return r, rm

def _plot_line(fig, line, **plt_kwargs):
    lines = [line] if isinstance(line, LineString) else line.geoms
    return [fig.line(*l.xy, **plt_kwargs) for l in lines]


def plot_poly(fig, poly, legend_label=None, **plt_kwargs):

    if type(poly) in [MultiPolygon, Polygon]:
        return _plot_poly(fig, poly, legend_label=legend_label, **plt_kwargs)

    return _plot_line(fig, poly, **plt_kwargs)


def _scale_divergence_color_map(cm, vmin, vmax, vmid, N):
    # Creating an higher resolution equispace linear space for mapping
    u = np.linspace(vmin,vmax,2*N)
    # Dual linear map from [vmin, vmid] -> [0,0.5] and [vmid, vmax] -> [0.5,1] 
    norm =  mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vmid, vmax=vmax)
    # Creating new scaled list of colors 
    new_colors = [mcolors.rgb2hex(color) for color in cm(norm(u))]
    # Returning color map
    return LinearColorMapper(palette = new_colors, low = vmin, high = vmax, nan_color=(0,0,0,0))  


def get_color_mapper(vmin, vmax, cmap, cmap_mode='auto', n=256):

    # FUTURE: Add vmin, vmax, vcenter checks
    nmap={'auto'  : (mcolors.Normalize, {}),
          'diverg': (mcolors.TwoSlopeNorm, {'vcenter': 0})}
    
    assert cmap_mode in nmap, "Unknown colormap mode."
    
    func, kwargs = nmap[cmap_mode]
    norm = func(**kwargs)

    u = np.linspace(vmin,vmax,2*n)
    new_colors = [mcolors.rgb2hex(color) for color in cmap(norm(u))]

    kwargs = {'low'      : vmin,
              'high'     : vmax,
              'nan_color': (0,0,0,0)}
                                  
    return LinearColorMapper(palette = new_colors, **kwargs) 

def plot_pcolor(fig, x, y, z, cmap, cmap_mode='auto'):

    linear_info = lambda s: (s.min(), s.max(), s.max()-s.min())
    x0, x1, xl = linear_info(x)
    y0, y1, yl = linear_info(y)
    z0, z1, zl = linear_info(z[~np.isnan(z)])
    
    print(z0,z1,zl)

    data = dict(z=[z],
                dw=[xl], dh=[yl],
                x=[x0]  ,y=[y0])
    
    data = ColumnDataSource(data)

    cmapper = get_color_mapper(z0, z1, cmap, cmap_mode=cmap_mode, n=256)

    fig.image(source=data, image='z', x='x', y='y', dw='dw', dh='dh', color_mapper=cmapper)

# Color Bar
    cb = ColorBar(color_mapper = cmapper, 
                         label_standoff = 14,
                         location = (0,0),
                         title = 'Topography/Bathymetry (m)')
    fig.add_layout(cb, 'below') #below, left, right or center



def plot_scatter(fig, data, is_visible=True, **plt_kwargs):
    r = fig.scatter(data[:, 0], data[:, 1], **plt_kwargs)
    r.visible = is_visible
    return r
    # r = p.scatter([xc], [yc], marker='square', size=0.1, color=color, alpha=alpha, legend_label=key)


def plot_scatter_cmap(fig, data, **plt_kwargs):

    src = {x: data[:,i] for i, x in enumerate(['x', 'y', 'z'])}
    src = ColumnDataSource(src)

    cmap  = LinearColorMapper(palette="Viridis256", 
                             low = min(data[:,2]), 
                             high = max(data[:,2]))

    fill_color = fill_color={"field": "z", "transform": cmap}
    plt_kwargs = dict(size=3, line_color=None, fill_color=fill_color)
    fig.scatter("x", "y", source=src, **plt_kwargs)
    
    from bokeh.models import ColorBar
    bar = ColorBar(color_mapper=cmap, location=(0,0))
    fig.add_layout(bar, "left")


def create_figure(is_satellite=False, **plt_kwargs):
    if is_satellite:
        plt_kwargs.update(dict(x_axis_type="mercator", y_axis_type="mercator"))

    fig = figure(**plt_kwargs)

    if is_satellite:
        fig.add_tile(xyz.Esri.WorldImagery)

    return fig


def create_container(fig, **kwargs):
    return row(fig, sizing_mode="stretch_both", **kwargs)
    # return row(fig, max_width=300, **kwargs)


def qcreate_container(fig, **kwargs):
    return fig, create_container(fig, **kwargs)


def qcreate_figure(**kwargs):
    # Extract figure sizes to use in 'container' for better sizing_mode
    cont_kwargs = pop_dict_keys(["width", "height"], kwargs)

    # # Try automatic resizing with match_aspect = True
    # key = 'match_aspect'
    # if key in kwargs and kwargs[key]:
    #     # Don't set sizing_mode keyword if already defined, or if both width and height are specified
    #     key =  'sizing_mode'
    #     if not key in kwargs.keys() and len(cont_kwargs) < 2 :
    #         # Defaults to 'scale_height' mode if width keyword is not specified
    #         kwargs[key]  = 'scale_height' if  'height' in  cont_kwargs  else 'stretch_height'

    # kwargs['max_width'] = cont_kwargs['width']
    fig = create_figure(**kwargs)
    return fig, create_container(fig, **cont_kwargs)
