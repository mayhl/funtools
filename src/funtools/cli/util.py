import logging
from pathlib import Path
from typing import Annotated

import typer

from ..core.log import getLogger
from ..io.input.main import Input
from .core import VerboseType, _typerExitWrapper, getTyperOpts
from .job import SubmitScript

app = typer.Typer(
    help="""
        Tools for miscellaneous FUNWAVE tasks\n
    """,
    **getTyperOpts(),
)


@app.command(
    rich_help_panel="Input File",
)
def prettify(
    src: Annotated[str, typer.Argument(help="Path to source input file")] = "input.txt",
    dest: Annotated[
        str, typer.Argument(help="Path to output input file")
    ] = "formatted_input.txt",
    in_place: Annotated[
        bool,
        typer.Option(
            "-i", "--in-place", help="Perform prettify in place, ignores DEST."
        ),
    ] = False,
    verbose: VerboseType = False,
) -> None:
    """[TESTING] Prettify input file"""

    if in_place and dest != "formatted_input.txt":
        getLogger(verbose).error(
            "Can not set in-place flag '-i/--in-place' and 'dest' (destination) file."
        )
        raise typer.Exit(1)

    def _process(path: str) -> Path:
        new_path = Path(path)
        if new_path.suffix == "":
            new_path.with_suffix(".txt")

        return new_path

    psrc = _process(src)
    pdest = _process(dest)

    print(psrc)
    test = Input.fromFile("input.txt")

    test.write("input.out")


@app.command()
def my_func(name: str):
    print(f"Hi {name}")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: VerboseType = False,
) -> None:
    """sdass"""
    _typerExitWrapper(False, ctx)
