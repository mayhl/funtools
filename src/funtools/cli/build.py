import subprocess
from functools import partial
from logging import log
from sys import stderr, stdout
from typing import Annotated

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.log import getLogger
from .core import (
    ShellProcess,
    VerboseType,
    getRepoPath,
    getTyperOpts,
    parsePath,
    runSimpleProgress,
)

app = typer.Typer(
    help="""
        Tools for building & compiling FUNWAVE code\n
        Run create COMMAND first to build a CMake build directory. Compiling COMMANDS 
        are wrappers around make and ccmake commands.
    """,
    **getTyperOpts(),
)


@app.command(
    rich_help_panel="Building",
)
def create(
    path: Annotated[str, typer.Argument(help="Path to build directory")] = ".",
    verbose: VerboseType = False,
) -> None:
    """Create CMake build directory for code"""

    dest_path = parsePath("path", path, mkdir=True)
    capture_output = verbose < 1
    cmd = ["cmake", "-S", getRepoPath(), path]
    msg = f"Cmake build files written to: '{dest_path.absolute()}'"

    ShellProcess("Building", cmd, capture_output, verbose, msg).runSpinner()


@app.command(rich_help_panel="Building")
def copy(
    path: Annotated[
        str, typer.Argument(help="Destination path for source code.")
    ] = "funwave",
    verbose: VerboseType = 0,
) -> None:
    """Copy source code to specified directory"""

    dest_path = parsePath("path", path, mkdir=True)
    capture_output = verbose < 1
    cmd = ["cp", "-r", getRepoPath(), path]
    msg = f"FUNWAVE source code copied to: '{dest_path.absolute()}'"
    ShellProcess("Compiling", cmd, capture_output, verbose, msg, err_map).runSpinner()


@app.command(
    rich_help_panel="Compiling",
)
def make(
    ctx: typer.Context,
    verbose: VerboseType = False,
) -> None:
    """Compile code using make\n
    Wrapper around 'make'"""
    capture_output = verbose < 1
    cmd = ["make"]

    msg = f"Compiled FUNWAVE exectuable"
    err_map = {
        "Not in a valid build directory, can not make!": "No targets specified and no makefile found",
    }
    ShellProcess("Compiling", cmd, capture_output, verbose, msg, err_map).runSpinner()


# @app.callback(
#   invoke_without_command=True,
@app.command(
    rich_help_panel="Compiling",
)
def config(a, verbose: VerboseType = 0) -> None:
    """Launch configure UI for FUNWAVE modules"""
    result = subprocess.run(
        ["ccmake", "."], capture_output=False, text=True, check=True
    )
    typer.Exit(result.returncode)


@app.command(
    rich_help_panel="Compiling",
)
def install(verbose: VerboseType = 0) -> None:
    """Install executable to configured path, see config"""

    capture_output = verbose < 1
    cmd = ["make", "install"]
    msg = f"Install FUNWAVE exectuable"
    err_map = {
        "Not in a valid build directory, can not make!": "No targets specified and no makefile found",
    }

    ShellProcess("Installing", cmd, capture_output, verbose, msg, err_map).runSpinner()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: VerboseType = False,
) -> None:
    """sdass"""
    getRepoPath(ctx)
