# NOTE: _ Avoids subsequent import overides
from .core import PlotInterface as _Interface
from .core import SimplePlotInterface as _SimpleInterface
from .objects import *

import param


class ShapePlot(_SimpleInterface):
    plot = param.ClassSelector(default=ShapeObject(), class_=ShapeObject)
