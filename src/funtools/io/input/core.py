from __future__ import annotations

import dataclasses
import importlib.resources
import json
from enum import Enum
from operator import ne
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Union, get_origin

from pydantic import BaseModel

_IN_META_ = importlib.resources.files("metadata").joinpath("input.json")
_OUT_META_ = importlib.resources.files("metadata").joinpath("output.json")

with _IN_META_.open("r", encoding="utf-8") as f:
    _IN_META_ = json.load(f)

with _OUT_META_.open("r", encoding="utf-8") as f:
    _OUT_META_ = json.load(f)


@dataclasses.dataclass
class Variable:
    short_name: str
    long_name: Optional[str] = None
    standard_name: Optional[str] = None
    units: Optional[str] = None
    comment: Optional[str] = None
    latex: Optional[str] = None

    @property
    def name(self) -> str:
        return self.short_name


@dataclasses.dataclass
class Parameter:
    name: str
    funwave_name: str
    short_name: Optional[str] = None
    long_name: Optional[str] = None
    units: Optional[str] = None


class LinkedMetadata(BaseModel):

    def __getSplitAttrs(self) -> tuple[list[str], list[str]]:
        """Return list of names of local and nested variables"""
        items = [
            (n, issubclass(v.__class__, LinkedMetadata))
            for n, v in self
            if not v is None and not n == "meta"
        ]

        nested = [n for n, f in items if f]
        local = [n for n, f in items if not f]
        return local, nested

    def toInputDict(self) -> dict:
        """Returns nested values as FUNWAVE input key/value pair dict"""

        local, nested = self.__getSplitAttrs()
        nested = [getattr(self, n).toInputDict() for n in nested]

        assert hasattr(self, "meta")
        local = {getattr(self.meta, n).funwave_name: getattr(self, n) for n in local}

        for k, d in local.items():
            if issubclass(d.__class__, Enum):
                local[k] = d.value

            if issubclass(d.__class__, PathLike):
                local[k] = str(d)

            if issubclass(d.__class__, bool):
                local[k] = "T" if d else "F"
        for n in nested:
            local.update(n)

        return local

    def initCompleted(self, **kwargs):
        pass

    @classmethod
    def _getNested(cls) -> tuple[list[tuple[str, type[LinkedMetadata]]], type[Any]]:
        """Returns nested fields as key/class type pairs, and meta field class type."""
        items = cls.model_fields

        assert "meta" in cls.model_fields
        MetaCls = cls.model_fields["meta"].annotation

        assert not MetaCls is None

        def parse(d) -> None | type[LinkedMetadata]:
            obj = d.annotation
            if get_origin(obj) is Union:
                obj = obj.__class__

            if issubclass(obj, LinkedMetadata):
                return obj
            else:
                return None

        nested = [(k, parse(d)) for k, d in items.items()]

        return [(k, d) for k, d in nested if not d is None], MetaCls

    @classmethod
    def getInputMap(cls) -> dict:
        """Returns dict mapping nested keys to all FUNWAVE inputs"""

        nested, MetaCls = cls._getNested()

        local = MetaCls.getInputMap()
        nested = [((k,), d.getInputMap()) for k, d in nested]
        nested = {k1 + k2: d for k1, sub in nested for k2, d in sub.items()}

        return {**local, **nested}

    @classmethod
    def getFileMap(cls) -> dict:
        """Returns dict mapping nested keys to FUNWAVE input file"""
        nested, MetaCls = cls._getNested()
        items = cls.model_fields.items()
        local = [k for k, d in items if d.annotation is Path]

        if len(local) > 0:
            meta = MetaCls()
            local = {(k,): getattr(meta, k).funwave_name for k in local}
        else:
            local = {}

        nested = [((k,), d.getFileMap()) for k, d in nested]
        nested = {k1 + k2: d for k1, sub in nested for k2, d in sub.items()}

        return {**local, **nested}


class Metadata(BaseModel):

    @classmethod
    def getInputMap(cls) -> dict:
        """Returns dict mapping local variables to FUNWAVE input."""
        meta = cls()
        return {(f,): getattr(meta, f).funwave_name for f in cls.model_fields}


class _Template(LinkedMetadata):

    class _TemplateMetadata(Metadata):
        #: Parameter = Parameter(name="", funwave_name="")
        pass

    meta: _TemplateMetadata = _TemplateMetadata()
