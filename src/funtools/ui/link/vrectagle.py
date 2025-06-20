
from holoviews.element.sankey import _Y_N_DECIMAL_DIGITS
from pandas._libs.tslibs.offsets import YearOffset



from .core import DataModelLink, LinkedDataModel

from bokeh.model import DataModel
from bokeh.core.properties import Float
from bokeh.models import CustomJS

import holoviews as hv
from holoviews.plotting.bokeh import LinkCallback


class WavemakerModel(LinkedDataModel):

        xc = Float(default = 0)
        yc = Float(default = 0)

        direction = Float(default=0)

        xs = Float(default = 0)
        xe = Float(default = 0)
        ys = Float(default = 0)
        ye = Float(default = 0)

        width = Float(default=99999)


        def __init__(self, **kwargs):
            super().__init__(self, **kwargs)
        


            y0 = self.yc - self.width/2
            y1 = self.yc + self.width/2

            if y1 > ye: y1 = ye
            if y0 < ys: y0 = ys 

            wavemaker = [(xc, y0), (xc, y1)]

            wavemaker = Line()
            
    @classmethod
    def DataModelLink(cls):
        return WavemakerLink


class WavemakerCallback(LinkCallback):


    source_model = 'plot'
    source_handles = ['cds']

    target_model = 'cds'

    source_code = """

    """

class WavemakerLink(DataModelLink):
    _MODEL_CLASS_ = WavemakerModel
    _CALLBACK_CLASS = WavemakerCallback
    _requires_target = True


WavemakerModel.register_callback()
