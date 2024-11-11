from funtools.error import *

import pyproj
from functools import partial 
from shapely.geometry import MultiPolygon, Polygon, MultiLineString, LineString
import numpy as np
import shapely 


class Transformer():

    def __init__(self, src_crs):

        self._proj_src = pyproj.Proj(src_crs)
        self._proj_trg = None

    def set_target_crs(self, crs):
        self._proj_trg = pyproj.Proj(crs)

    def _get_proj(self, src, trg):

        key = src, trg
        mapper = {
            ('source', 'satellite'): partial(proj2bokeh, proj=self._proj_src),
            ('target', 'satellite'): partial(proj2bokeh, proj=self._proj_trg),
            ('satellite', 'source'): partial(bokeh2proj, proj=self._proj_src),
            ('satellite', 'target'): partial(bokeh2proj, proj=self._proj_trg),
            ('source', 'target'): partial(proj2proj, src_proj=self._proj_src,  trg_proj=self._proj_trg),
            ('target', 'source'): partial(proj2proj, src_proj=self._proj_trg,  trg_proj=self._proj_src),
        }
        
        if key not in  mapper:
            msg = """Transform %s to %s not implemented.""" % key
            raise qerror(TypeError, msg)

        return  mapper[key] 
    
    def proj_xyz(self, x, y, src, trg):
        return  self._get_proj(src, trg)(x, y)

    def proj_poly(self, poly, src, trg):
        return transform_poly(poly, self._get_proj(src, trg))


def _transform_poly(poly, transform):
    
    holes = [list(zip(*transform(*s.xy))) for s in poly.interiors]
    pts = list(zip(*transform(*poly.exterior.xy)))

    return Polygon(pts, holes=holes)


def _transform_line(line, transform):
    lines = [line] if isinstance(line, LineString) else line.geoms
    NewLine = lambda l: LineString(list(zip(*transform(*l.xy))))
    return shapely.union_all([NewLine(l) for l in lines])

def transform_poly(poly, transform):

    ptype = type(poly)
    # Hacking working around for lines 
    if ptype in [LineString, MultiLineString]: 
        return _transform_line(poly, transform)
    
    polys = [poly] if ptype is Polygon else poly.geoms
    polys = [_transform_poly(p, transform) for p in polys]

    polys = [p for p in polys if p.is_valid]
    
    #print([t.is_valid for t in polys])
    
    return shapely.union_all(polys)


def proj2proj(x, y, src_proj, trg_proj):
    x, y = src_proj(np.array(x), np.array(y), inverse=True)
    x, y = trg_proj(np.array(x), np.array(y))
    return list(x), list(y)   

def proj2bokeh(x, y, proj):
    x, y = proj(np.array(x), np.array(y), inverse=True)
    x, y = degrees2meters(x, y)
    return list(x), list(y)   

def bokeh2proj(x, y, proj):
    x, y = meters2degrees(x, y)
    x, y = proj(x, y)
    return list(x), list(y)   


def degrees2meters (lon,lat):
        x = lon * 20037508.34 / 180;
        y = np.log(np.tan((90 + lat) * np.pi / 360)) / (np.pi / 180);
        y = y * 20037508.34 / 180;
        return x, y
    
def meters2degrees (x ,y):
    if not type(x) is np.ndarray: x = np.array(x)
    if not type(y) is np.ndarray: y = np.array(y)
        
    lon = x*(180.0/20037508.34)
    y = y/(20037508.34 / 180.0)
    lat = (np.arctan(np.exp(y*(np.pi / 180)))*360/np.pi - 90)

    return lon, lat     
    

