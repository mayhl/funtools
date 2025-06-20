import param
import panel as pn

from bokeh.model import DataModel
from bokeh.core.properties import Bool, Int, String

from holoviews.plotting.links import Link
from holoviews.plotting.bokeh import LinkCallback

pn.extension(notifications=True)

# Shared data structure for syncing data between client-side Javascript code and server-side Python code.
# Motivation:
#    AFAIK, Holoviews/Bokeh javascript can not target Panel widgets, therefore, bokeh's DataModel is used as a
#    target and callback may be attached to the DataModel's members and responds to client-side JavaScript changes
# Purposes:
#    1) Data stucture that can be targeted by Holoviews/Bokeh plots javascript callbacks, i.e., Holoviews' LinkCallback
#    2) Watch tagged members, _ON_CHANGE_, and link with callback method
#    3) Optional user notification method


class LinkedDataModel(DataModel):
    alert_id = Int(default=0)
    alert_msg = String(default="")
    # NOTE: Using seperate variable to explicitly trigger callback and avoid
    #       assignment ordering requirement for alert_id  and alert_msg
    trigger_alert = Bool(default=False)
    # Members to watch for Python server-side callback
    _ON_CHANGE_ = []

    def on_change_hook(self, attr, old, new):
        raise NotImplementedError(
            "Method on_change_hook is not defined in derived class %s." % self.__class__
        )

    # Wrapper method for error/message handling
    def _alert(self):
        if not self.trigger_alert:
            return

        if self.alert_id == 0:
            notify_method = pn.state.notifications.info
        elif self.alert_id == 1:
            notify_method = pn.state.notifications.success
        elif self.alert_id == 2:
            notify_method = pn.state.notifications.warning
        elif self.alert_id == 3:
            notify_method = pn.state.notifications.error
        else:
            raise Exception("Invalid alter_id, got %d." % self.alert_id)

        notify_method(self.alert_msg, duration=4000)
        self.alert_id = 0
        self.alert_msg = ""
        self.trigger_alert = False

    def _on_change_hook(self, attr, old, new):
        self._alert()
        self.on_change_hook(attr, old, new)

    # Delayed/Runtime linking of derived DataModeLink (back) to derived LinkedDataModel (and LinkCallback)
    @classmethod
    def DataModelLink(cls):
        raise NotImplementedError(
            "Class method DataModelLink is not defined in derived class %s." % cls
        )

    # Wrapper methods for cconvenient access to linked data model members

    @classmethod
    def register_callback(cls):
        cls.DataModelLink()._register_callback()

    def attach_callback(self, source, target=None):
        Link = self.__class__.DataModelLink()
        if target is None:
            return Link(source, model=self)
        else:
            return Link(source, target, model=self)

    # Standard reference to Holoview/Bokeh plot
    def hv_plot(self):
        raise NotImplementedError(
            "Method hv_plot is not defined in derived class %s." % self.__class__
        )

    # Panel with widgets to control visual elements of Holoviews/Bokeh plot
    def hv_control_panel(self):
        raise NotImplementedError(
            "Method hv_control_panel is not defined in derived class %s."
            % self.__class__
        )

    # Dictionary of widgets corosponding to relevant derived LikedDataModel members
    # NOTE: Deferring integration of widgets into a panel to allow integration with other components
    def datamodel_widgets(self):
        raise NotImplementedError(
            "Method datamodel_widgets is not defined in derived class %s."
            % self.__class__
        )

    # Wrapper method for standalone demo panel
    def demo_panel(self, title="Quick Demo"):

        plot = pn.pane.HoloViews(self.hv_plot())
        crt_panel = self.hv_control_panel()
        widgets = pn.WidgetBox(*[d for _, d in self.datamodel_widgets().items()])

        template = pn.template.MaterialTemplate(
            title=title,
            sidebar=[pn.Accordion(("Inputs", widgets), ("Plot Options", crt_panel))],
        )

        template.main.append(plot)
        return template


# Base abstract class for linking derived LinkedDataModel to derived LinkCallback
class DataModelLink(Link):

    # Configs to automate linking of derived classes
    _MODEL_CLASS_ = LinkedDataModel
    _CALLBACK_CLASS_ = LinkCallback

    model = param.ClassSelector(class_=_MODEL_CLASS_)

    # Initalizing shared data structure and linking callback hooks
    def __init__(self, *args, model=None):
        super().__init__(*args)

        if model is None:
            model = self._MODEL_CLASS_()
        self.model = model

        if len(model._ON_CHANGE_) > 0:

            model.on_change("trigger_alert", model._on_change_hook)
            for name in model._ON_CHANGE_:
                model.on_change(name, model._on_change_hook)

    # Wrapper automatically hooking register_callback of derived DataModelLink
    # class to the derived LinkedData model class
    @classmethod
    def _register_callback(cls):
        cls.register_callback("bokeh", cls._CALLBACK_CLASS_)
