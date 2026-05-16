from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..core.resources import getPath

_ENV_ = Environment(
    loader=FileSystemLoader(getPath("templates")), undefined=StrictUndefined
)


class TemplateWriter:

    def __init__(self, path: str | Path) -> None:
        if isinstance(path, str):
            path = Path(path)

        self._path = path
        self.__is_first = True

    def append(self, source: Path | str, **kwargs) -> None:

        template = _ENV_.get_template(source)

        mode = "w" if self.__is_first else "a"

        if self.__is_first:
            self.__is_first = False

        content = template.render(**kwargs)
        with open(self._path, mode=mode, encoding="utf-8") as fh:
            fh.write(content)
            fh.write("\n\n")
