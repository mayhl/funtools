from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List
import pyproj
import json

class _Projection(ABC):

    """Interface for projecting between coordinates"""

    @abstractmethod
    def to_target(self, x: np.ndarray, y:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def to_source(self, u: np.ndarray, v:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

class GeoProjection(_Projection):

    """Project Geographical Lon/Lat coordinates to some ESPG projections"""

    def __init__(self, target_espg: int) -> None:
        crs = pyproj.CRS.from_epsg(target_espg)
        self._proj = pyproj.Proj(crs)

    def to_source(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert projection coordinates to Geographical coordinates"""
        return self._proj(u, v, inverse=True)

    def to_target(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert Geographical coordinates to projection coordinates"""
        return self._proj(x, y)

class BokehProjection(_Projection):

    """Projects Geographical Lon/Lat coordinates to Bokeh Tile coordinates for plotting"""

    def to_source(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Converts Geographical coodinates to Bokeh Tile coordinates"""
        x = u * 20037508.34 / 180;
        y = np.log(np.tan((90 + v) * np.pi / 360)) / (np.pi / 180);
        y = y * 20037508.34 / 180;
        return x, y
    
    def to_target(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Converts Bokeh Tile coodinates to Geographical coordinates"""
        u = x*(180.0/20037508.34)
        y = y/(20037508.34 / 180.0)
        v = (np.arctan(np.exp(y*(np.pi / 180)))*360/np.pi - 90)
        return u, v

class RotationProjection(_Projection):
    """Project from one coordinate system to another by rotation around some point of rotation and shifting coordinates"""
    def __init__(self, rotation_x0: float, rotation_y0:  float, angle: float,  offset_x: float, offset_y: float) -> None:

        self._rot_x0 = rotation_x0
        self._rot_y0 = rotation_y0
        self._angle = angle
        self._off_x = offset_x
        self._off_y = offset_y

    def to_source(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Converts rotated coordinates to original coordinates"""
        u = u + self._off_x
        v = v + self._off_y

        angle = self._angle
        x =  u*np.cos(angle) + v*np.sin(angle)
        y = -u*np.sin(angle) + v*np.cos(angle)

        x = x + self._rot_x0
        y = y + self._rot_y0

        return x, y

    def to_target(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Converts original coordinates to rotated coordinates"""
        x = x - self._rot_x0
        y = y - self._rot_y0

        #NOTE: Negative sign
        angle = -self._angle
        u =  x*np.cos(angle) + y*np.sin(angle)
        v = -x*np.sin(angle) + y*np.cos(angle)
       
        u = u - self._off_x
        v = v - self._off_y

        return u, v

    @classmethod
    def from_funwave_info(cls, fpath: str) -> RotationProjection:
        """Wrapper method for creating class from FUNWAVE JSON file"""        
        with open(fpath, 'r') as fh:
            trans_info = json.load(fh)

        off_x = trans_info['x_off'] - trans_info['x0_extend']
        off_y = trans_info['y_off'] - float(trans_info['yl_blend'])/2

        rot_x0 = trans_info['x0_rot']
        rot_y0 = trans_info['y0_rot']

        angle = np.deg2rad(trans_info['angle']) 

        return RotationProjection(rot_x0, rot_y0, angle, off_x, off_y)


class LinkedProjections(_Projection):
    """Class for chaining projections"""
    def __init__(self, projections: List) -> None:
        self._projections = projections

    def to_source(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        "Convert from final target coordinates to first source coordinates in projection list"""
        x, y = u, v
        for proj in reversed(self._projections):
            x, y = proj.to_source(x, y)
        return x, y

    def to_target(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        "Convert from firstl src coordinates to final target coordinates in projection list"""
        u, v = x, y
        for proj in self._projections:
            u, v = proj.to_target(u, v)
        return u, v
