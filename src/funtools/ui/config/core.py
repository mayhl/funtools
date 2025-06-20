import param
import panel as pn

from ...link.hvbokeh import Native as _Native


class NestedParameterized(param.Parameterized):

    def __init__(self, **params):

        is_cls = lambda k: isinstance(self.param[k], param.ClassSelector)
        keys = [k for k in params if is_cls(k)]

        key_cls = {k: self.param[k].class_ for k in keys}
        is_nested = lambda c: issubclass(c, NestedParameterized)
        key_cls = {k: c for k, c in key_cls.items() if is_nested(c)}

        for k, cls in key_cls.items():
            params[k] = cls(**params.pop(k))

        super().__init__(**params)
        self._nested_keys = [*key_cls]


# Base clase for interfacingw with a Holowviews/Bokeh element, e.g., line, fill, etc
# PURPOSE:
#    1) Derived classes for each element defines the mapping between corresponding
#           i)  Panel widget and
#           ii) Holoviews/Bokeh Javascript member
#    2) BaseInterface automates the generation of widgets and jslink with passed Holoviews/Bokeh renderer
# NOTES:
#    a) Widgets are only generated for param initialized in the contstructor
#    b) [WIP] Optional Group field for collapsing interface
class ElementInterface(NestedParameterized):

    _WIDGET_CONFIGS_ = {}
    _LABEL_ = ""

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self._tagged_keys = [*kwargs]  # [k for k in kwargs]

    def jslinked_widgets(self, hv_plot):

        configs = {k: self._WIDGET_CONFIGS_[k] for k in self._tagged_keys}
        tuples = {k: self._config2widget(k, hv_plot, **c) for k, c in configs.items()}

        groups = _unique_grouping([g for _, (g, _) in tuples.items()])
        sname = {k: d["name"] for k, d in self._WIDGET_CONFIGS_.items()}
        get_args = lambda p, n, w: {"prefix": p, "name": n, "widget": w}

        if len(groups) == 1:
            assert groups[0] == "Standard", "Must link at least standard key."
            args = {k: get_args("", sname[k], w) for k, (_, w) in tuples.items()}
        else:
            args = {k: get_args("%s " % g, sname[k], w) for k, (g, w) in tuples.items()}

        return args

    def jslinked_panel(self, hv_plot):
        widgets = [d["widget"] for d in self.jslinked_widgets(hv_plot).values()]
        return pn.WidgetBox(*widgets)

    # Wrapper method for creating widget from param and apply jslink to Holoviews plot
    def _config2widget(self, k, hv_plot, group="Standard", **config):
        # Extracting widget and linking configs

        linker, widget_cls = [config.pop(k) for k in ["linker", "widget"]]
        widget = widget_cls.from_param(getattr(self.param, k), **config)

        linker.jslink(widget, hv_plot)
        return group, widget


def _unique_grouping(items):

    if len(items) == 0:
        return items
    u_item = items[0]
    u_items = [u_item]
    for item in items[1:]:
        if not item == u_item:
            assert not item in u_items, "Unique items are not items in list"
            u_items.append(item)
            u_item = item

    return u_items


# An object is defined as a  group of glyphs
# Base inteface for glyphs (Holoviews/Bokeh renderers) with widgets


# Base class for collecting different elements/glyphs into a single object and UI panel
# NOTES
#    1) Use param.ClassSelector to add element interfaces
#    2) Holowviews renderers/glyphs are linked to element interface via dict keys
#       corosponding to a defined param.ClassSelector
class ObjectInterface(NestedParameterized):

    _LABELS_ = {}

    # elements a dict of keys
    def jslinked_panel(self, elements):

        # Allows for easier method calls if only a single elementis to be passed
        if not isinstance(elements, dict):
            elements = {k: elements for k in self._keys}

        # FUTURE: add check that all elements are set

        get_dict = lambda k, p: getattr(self, k).jslinked_widgets(p)  #
        args = {k: get_dict(k, p) for k, p in elements.items()}

        is_single = len(args) == 1
        suffix = lambda k: (
            "" if is_single else "%s " % getattr(self, k).__class__._LABEL_
        )

        hv_widgets = []
        for k, d in args.items():
            for sk, sd in d.items():
                hv_widgets.append(("%s.%s" % (k, sk), sd, suffix(k)))

        # Flattening dict of dict by merging keys seperated by '.'
        # widget_info = [("%d.%d" % (k, sk), sd) for sk, sd in d.items() for k, d in args.items()]

        def patch(d, val, key="prefix"):
            d[key] = val + d[key]
            return d

        args = {k: patch(d, p) for k, d, p in hv_widgets}

        # Get unique values in order of appearance
        prefixes = _unique_grouping([d["prefix"] for d in args.values()])
        widget_boxes = {k: [] for k in prefixes}

        # Combining widgets to a single panel
        for k, d in args.items():
            widget_boxes[d["prefix"]].append(d["widget"])

        widget_boxes = {k: pn.WidgetBox(*wb) for k, wb in widget_boxes.items()}

        return pn.Column(*[d for _, d in widget_boxes.items()])

    def __init__(self, **params):
        super().__init__(**params)

        nl = len(self._LABELS_)
        if nl < 1:
            raise NotImplementedError(
                "No _LABELS_ not set in derived class %s." % self.__class__
            )

        self._keys = [*params]
        nk = len(self._keys)
        if not nl == nk:
            pass
            # raise Exception("Number of _LABELS_ does not match number of BaseInterfaces, got %d and %d, respectively. Found BaseInterfaces: %s." % (nl, nk, self._keys))


# Special case of ElementInterface for figure attributes global to all objects
class FigureElement(ElementInterface):

    width = param.Number(
        default=500,
        bounds=(0, None),
        inclusive_bounds=(False, None),
        softbounds=(100, 4000),
    )

    height = param.Number(
        default=500,
        bounds=(0, None),
        inclusive_bounds=(False, None),
        softbounds=(100, 4000),
    )

    _WIDGET_CONFIGS_ = dict(
        width=dict(
            # Linking kwargs
            widget=pn.widgets.FloatSlider,
            linker=_Native("value", "width"),
            # target_attr = 'width',
            # source_attr = 'value',
            # Widget kwargs
            name="Width",
            # description = "Width of figure in pixels.",
        ),
        height=dict(
            # Linking kwargs
            widget=pn.widgets.IntSlider,
            linker=_Native("value", "height"),
            # target_attr = 'height',
            # source_attr = 'value',
            # Widget kwargs
            name="Height",
            # description = "Height of figure in pixels.",
        ),
    )


# Final plot, attaches figure
class PlotInterface(NestedParameterized):

    _LABELS_ = {}

    figure = param.ClassSelector(
        default=FigureElement(),
        class_=FigureElement,
    )

    def jslinked_panel(self, objects):

        print(objects)
        # if not isinstance(objects, dict):

        # Grabing random object to attach figure properties to
        # NOTE: jslink do not seems work when targeting overlayed plots
        obj = list(objects.values())[0]
        figure = list(obj.values())[0]

        # FUTURE: add check that all elements are set
        # Getting config list and enforcing figure to be first

        objects = {"figure": figure, **objects}
        panels = {k: getattr(self, k).jslinked_panel(g) for k, g in objects.items()}

        labels = self._LABELS_
        key = "figure"
        if not key in labels:
            labels[key] = "Figure"

        return pn.Accordion(*[(labels[k], p) for k, p in panels.items()])


# Wrapper class for removing unnecessary for a single object
# Assume plot = param.ClassSelector(class_=OBJECT) is implemented in derived class
class SimplePlotInterface(PlotInterface):

    _LABELS_ = {"plot": "Plot"}

    def jslinked_panel(self, objects):
        return super().jslinked_panel({"plot": objects})

    def __init__(self, **params):
        params = {"plot": params}
        params["figure"] = params["plot"].pop("figure")
        super().__init__(**params)
