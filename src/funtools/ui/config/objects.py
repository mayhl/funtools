# NOTE: _ Avoids subsequent import overides
from .core import ObjectInterface as _Interface
from .elements import *

import param


class LineObject(_Interface):

    _LABELS_ = {"line": ""}
    line = param.ClassSelector(default=LineElement(), class_=LineElement)


class ShapeObject(_Interface):

    _LABELS_ = {"line": "Line", "fill": "Fill"}

    line = param.ClassSelector(default=LineElement(), class_=LineElement)
    fill = param.ClassSelector(default=FillElement(), class_=FillElement)
