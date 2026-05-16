from typing import Annotated

import typer

from ..core.log import getLogger
from .core import VerboseType, getHPCParameters, getTyperOpts
from .job import SubmitScript

app = typer.Typer(
    help="""
        Tools for submiting FUNWAVE simulations to the HPC\n
        Run create COMMAND first to build a CMake build directory. Compiling COMMANDS 
        are wrappers around make and ccmake commands.
    """,
    **getTyperOpts()
)


@app.command(
    rich_help_panel="Single",
)
def simple(
    path: Annotated[str, typer.Argument(help="Path to build directory")] = ".",
    verbose: VerboseType = False,
) -> None:
    """[TESTING] Submit simulation mirroring $HOME to $WORK"""

    SubmitScript("pbspro")


@app.command(
    rich_help_panel="Single",
)
def id(
    path: Annotated[str, typer.Argument(help="Path to build directory")] = ".",
    verbose: VerboseType = False,
) -> None:
    """[NOT IMPLEMENTED] Submit simulation mirroring $HOME to $WORK with job ID suffix"""


@app.command(
    rich_help_panel="Batch",
)
def batch(
    path: Annotated[str, typer.Argument(help="Path to build directory")] = ".",
    verbose: VerboseType = False,
) -> None:
    """[NOT IMPLEMENTED] Submit an array job of simulations"""


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: VerboseType = False,
) -> None:
    """sdass"""
    getHPCParameters(ctx)
