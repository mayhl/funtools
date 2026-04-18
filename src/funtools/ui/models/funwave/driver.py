# NOTE: For quick testing, can't use relative import
# from ....models.funwave.driver2 import Driver as _Driver
from chlhydrotoolbox.models.funwave.driver2 import (
    Grid as _Grid,
    Bathy as _Bathy,
    Driver as _Driver,
)

import param


class Panel(param.Parameterized):
    def foo(self):
        print("bar")


class Grid(_Grid, Panel):
    _is_2d = param.Boolean(default=True, constant=True)
    _is_equal = param.Boolean(default=True, constant=True)

    _xl = param.Number()
    _yl = param.Number()

    def __init__(self, **params):
        super().__init__(**params)

        qwatch = self.param.watch
        qwatch(self._check_x, ["dx", "px"])

        if self._is_2d:
            qwatch(self._check_y, ["dy", "px"])
        else:
            self.py = 1
            self.dy = self.dx
            self.ny = 3

        if self._is_equal:
            qwatch(self._sync_y, ["dx"], precedence=1)

    def _check_x(self, event):
        is_valid, msg = self.check_x()

    def _check_y(self, event):
        is_valid, msg = self.check_y()

    def _sync_y(self, event):
        self.param.update(dy=event.new)


class Bathy(_Bathy):
    def __init__(self, **params):
        super().__init__(**params)


class Driver(_Driver):
    grid = param.ClassSelector(default=Grid(), class_=Grid)
    bathy = param.ClassSelector(default=Bathy(), class_=Bathy)

    def panel(self):
        pass

    def control_box(self):
        pass


# c = Grid()

# print(c)

# c.foo()


class A(param.Parameterized):
    a = param.Integer(default=0)

    @param.depends("a", watch=True)
    def _test(self):
        print("sub")


class B(param.Parameterized):
    a = param.ClassSelector(default=A(a=3), class_=A)

    @param.depends("a.a", watch=True)
    def _test(self):
        print("HERE")


a = Grid()


a.recommened_mpi()
