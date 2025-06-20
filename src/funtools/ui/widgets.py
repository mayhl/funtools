import param

import panel as pn

pn.extension("floatpanel")  # type: ignore


class AnyFile(param.Parameterized):
    def panel(self):
        self.file_input = file_input = pn.widgets.FileInput()

        button = pn.widgets.Button(name="Click me", button_type="primary")

        pn.bind(self.callback, button, watch=True)

        dummy = pn.layout.FloatPanel(status="closed")
        self._layout = pn.Row(button, dummy)

        return self._layout

    def callback(self, event):
        is_new = self._layout[1].status == "closed"  # type: ignore
        if is_new:
            tmp = self._layout[1]
            self._layout[1] = pn.layout.FloatPanel(name="Basic FloatPanel", margin=20)
            del tmp


test = AnyFile()


test.panel().servable()
