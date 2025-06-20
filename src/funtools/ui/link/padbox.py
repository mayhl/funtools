from .core import DataModelLink, LinkedDataModel

from ..config.hvbokeh.plots import ShapePlot


from bokeh.core.properties import Float, Int, Bool, String

import holoviews as hv
from holoviews.plotting.links import SelectionLink
from holoviews.plotting.bokeh import LinkCallback

import panel as pn

pn.extension(
    js_files={"padbox": "assets\\js\\padbox.min.js"}
    # js_files={"padbox": "assets\\js\\padbox.min.js"}
)


class PaddedBoxModel(LinkedDataModel):
    is_move = Bool(default=False)
    xs = Float(default=0)
    xe = Float(default=0)
    x0 = Float(default=0)
    x1 = Float(default=0)
    ys = Float(default=0)
    ye = Float(default=0)
    y0 = Float(default=0)
    y1 = Float(default=0)
    dx = Float(default=0)
    dy = Float(default=0)
    i0 = Int(default=0)
    i1 = Int(default=0)
    j0 = Int(default=0)
    j1 = Int(default=0)

    # Members to attach callback too
    _ON_CHANGE_ = ["x0", "x1", "y0", "y1"]

    # Wrapper method for generating object from 2D equispaced grid
    @classmethod
    def from_grid(cls, x, y, north_width=0, south_width=0, east_width=0, west_width=0):
        kwargs = dict(
            x0=x[0] + west_width,
            x1=x[-1] - east_width,
            y0=y[0] + south_width,
            y1=y[-1] - north_width,
            xs=x[0],
            xe=x[-1],
            ys=y[0],
            ye=y[-1],
            dx=(x[-1] - x[0]) / len(x),
            dy=(y[-1] - y[0]) / len(y),
        )

        return cls(**kwargs)

    def __init__(self, widgets=None, **kwargs):
        ###########################
        # Setting up Glyphs/Plots #
        ###########################

        super().__init__(**kwargs)

        x0, x1 = self.x0, self.x1
        y0, y1 = self.y0, self.y1

        xs, xe = self.xs, self.xe
        ys, ye = self.ys, self.ye

        # Setting up bounding box for un-sponged area
        lines = [
            [(x0, y0), (x1, y0)],
            [(x1, y0), (x1, y1)],
            [(x1, y1), (x0, y1)],
            [(x0, y1), (x0, y0)],
        ]

        self._lines = lines = hv.Path(data=lines, label="TEST")

        lines.opts(tools=["lasso_select", "box_select", "tap"])

        # Glyphs for marking sponged areas
        rects = [(x0, ys, x1, y0), (x1, ys, xe, ye), (x0, y1, x1, ye), (xs, ys, x0, ye)]

        self._rects = rects = hv.Rectangles(data=rects, label="Test")

        rects.opts(
            line_width=0,
        )

        self._hv_plot = lines * rects

        kwargs = dict(
            line=lines,
            fill=rects,
        )

        self._hv_control_panel = _get_hv_controls().jslinked_panel(kwargs)

        # Attaching Javascript code
        self._link = self.attach_callback(lines, rects)
        SelectionLink(lines, rects)

        ##################################
        # Widget => Plots Sync Callbacks #
        ##################################

        # FUTURE: Add widget input
        self._widgets = dict(
            north_width=pn.widgets.FloatInput(
                name="Sponge_west_width",
                step=self.dy,
                description="Width (m) of sponge layer at north boundary",
            ),
            south_width=pn.widgets.FloatInput(
                name="Sponge_south_width",
                description="Width (m) of sponge layer at south boundry",
                step=self.dy,
            ),
            east_width=pn.widgets.FloatInput(
                name="Sponge_east_width",
                description="Width (m) of sponge layer at east boundry",
                step=self.dx,
            ),
            west_width=pn.widgets.FloatInput(
                name="Sponge_north_width",
                description="Width (m) of sponge layer at west boundry",
                step=self.dx,
            ),
        )

        to_render = lambda x: hv.render(x).renderers[-1]
        args = dict(rects=rects, lines=lines, model=self)

        #################
        #     NORTH     #
        #################
        # NOTE: Figure out render tagging
        code = """
            const lines_cds = rects.renderers[0].data_source
            const rects_cds = rects.renderers[1].data_source
        
            sync_north(source, lines_cds, rects_cds, model)
        """
        self._widgets["north_width"].jscallback(value=code, args=args)

        #################
        #     SOUTH     #
        #################

        code = """
            const lines_cds = rects.renderers[0].data_source
            const rects_cds = rects.renderers[1].data_source
              
            sync_south(source, lines_cds, rects_cds, model)
        """

        self._widgets["south_width"].jscallback(value=code, args=args)

        ################
        #     EAST     #
        ################

        code = """
            const lines_cds = rects.renderers[0].data_source
            const rects_cds = rects.renderers[1].data_source

            sync_east(source, lines_cds, rects_cds, model)
        """

        self._widgets["east_width"].jscallback(value=code, args=args)

        ################
        #     WEST     #
        ################

        code = """
            const lines_cds = rects.renderers[0].data_source
            const rects_cds = rects.renderers[1].data_source

            sync_west(source, lines_cds, rects_cds, model)
        """

        self._widgets["west_width"].jscallback(value=code, args=args)

    def datamodel_widgets(self):
        return self._widgets

    # Server-side data syncing callback, triggered by client-side JS update
    def on_change_hook(self, attr, old, new):
        self._widgets["south_width"].value = self.y0 - self.ys
        self._widgets["north_width"].value = self.ye - self.y1

        self._widgets["west_width"].value = self.x0 - self.xs
        self._widgets["east_width"].value = self.xe - self.x1

    def hv_control_panel(self):
        return self._hv_control_panel

    def hv_plot(self):
        return self._hv_plot

    # Delayed/runtime linking with PaddedBoxLink class
    @classmethod
    def DataModelLink(cls):
        return PaddedBoxLink


# Class for defining client-side Holoview/Bokeh Javascript callback
class PaddedBoxCallback(LinkCallback):
    source_model = "plot"
    source_handles = ["cds"]
    # on_source_changes = ['indices']#'indices']
    on_source_events = ["tap", "mousemove"]
    target_model = "cds"

    source_code = """
        var event = arguments[4]
        console.log(source_plot)
        update_plot(event, source_cds, target_cds, model) 
    """

    def validate(self):
        assert True


class PaddedBoxLink(DataModelLink):
    _MODEL_CLASS_ = PaddedBoxModel
    _CALLBACK_CLASS_ = PaddedBoxCallback
    _requires_target = True


# NOTE: Seems to work with import
PaddedBoxModel.register_callback()


def _get_hv_controls():
    params = dict(
        figure=dict(
            width=500,
            height=500,
        ),
        line=dict(
            color="red",
            width=2,
        ),
        fill=dict(color="green", alpha=0.5),
    )

    return ShapePlot(**params)
