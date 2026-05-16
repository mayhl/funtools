from typing import Annotated, Optional

import shellingham
import typer

from ..core.log import getLogger
from .build import app as build_app
from .core import VerboseType, checkEnvironment
from .submit import app as submit_app
from .util import app as util_app

app = typer.Typer(
    help="FUNWAVE-TVD HPC CLI tool.",
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=True,
    no_args_is_help=False,
)

app.add_typer(
    build_app,
    name="build",
)

app.add_typer(
    submit_app,
    name="submit",
)

app.add_typer(
    util_app,
    name="util",
)


def _install_aliases(value: bool):
    """Install alias shorthands"""
    if value:
        typer.echo("Installing custom completion...")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: VerboseType = 0,  # pyright: ignore
    install_aliases: bool = typer.Option(
        False,
        "--install-alias",
        callback=_install_aliases,
        is_eager=True,
        help="Install shorthand aliases for the current shell.",
    ),
):
    """Setup custom option flags"""
    checkEnvironment(ctx)


@app.command(rich_help_panel="Executing")
def submit() -> None:
    """Submit a FUNWAVE simulation to the HPC queue"""
