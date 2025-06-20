# from .core import LinkedDataModel, DataModelLink, LinkCallback
from typing import Dict
import param
from bokeh.core.properties import Int


from enum import Enum as _Enum


# NOTE 1: param.Dict is used to store data between JavaScript calls
# NOTE 2: param.Dict is passed by reference (hence JavaScript changes are caputred), whereas
#         intrinsic types are pass by values
# NOTE 3: Since callbacks are not need, additional machinary of LinkedDataModel is not needed
class Base(param.Parameterized):

    src_attr = param.String()
    trg_attr = param.String()
    data = param.Dict(default={})

    @classmethod
    def init_data(self):
        return {}

    def __init__(self, source_attr, target_attr):
        params = dict(
            src_attr=source_attr,
            trg_attr=target_attr,
            data=self.__class__.init_data(),
        )
        params.update(**params)

        super().__init__(**params)
        self._link = None

    @classmethod
    def code(cls):
        raise NotImplementedError()

    def jslink(self, source, target):

        args = dict(
            src_attr=self.src_attr,
            trg_attr=self.trg_attr,
            data=self.data,
        )

        self._link = source.js_on_change(target, args=args, code=self.__class__.code())


class Native(Base):

    # Overiding base class for special case
    def jslink(self, source, target):
        self._link = source.jslink(target, **{self.src_attr: self.trg_attr})


class Simple(Base):

    # Overiding base class for other special case
    def jslink(self, source, target):
        code = "target.%s = source.%s; console.log(target); console.log(source)" % (
            self.trg_attr,
            self.src_attr,
        )
        # code = "console.log(target)"
        print({self.src_attr: code})

        print(source)
        print(source)

        self._link = source.jslink(target, code={"value": code})


class Legend(Base):

    _CODE_ = """ 
        let i = 0;
        console.log(target);
         for (let i in target.panels) {
            if (target.panels[i].type == "Legend") {

                let id = target.renderers[%d].id;
                let labels = target.panels[i].items;

                for (let j in labels) {

                    let id_label = labels[j].renderers[0].id;

                    if (id == id_label) {
                        labels[j].%%s.value = source.%%s;
                        break;
                    }
                }

                break;
            }
         }
         target.change.emit();"""


class Colorbar(Base):

    _CODE_ = """let rend = target.renderers[%d];
     console.log(source)

     for (let i in source.items) {

        let name = source.items[i][0]

        if (name == source.value) {
            let vals = source.items[i][1]

            console.log(vals)
            rend.glyph.color_mapper.%%s = vals

            break;
            let dummmy = source.%%s;
        }

     }
     rend.data_source.change.emit();
     console.log(target)"""  # % idxs
