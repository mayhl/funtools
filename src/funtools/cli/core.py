import os
import subprocess
from pathlib import Path
from re import A
from sys import stderr, stdout
from typing import Annotated, Callable

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.log import getLogger

# Shorthand for verbose flag
VerboseType = Annotated[
    int, typer.Option("--verbose", "-v", count=True, help="Enable verbose mode")
]
_common_opts_ = dict(
    no_args_is_help=False,
)


def _getVar(name: str) -> tuple[bool, str]:
    var = os.environ.get(name)

    is_read = not (var is None or var.strip() == "")
    if not is_read:
        getLogger(verbose_lvl=0).critical(f"Environment variable '{name}' not defined.")
        var = ""

    return is_read, var


def _getIntVar(name: str) -> tuple[bool, int]:
    is_read, var = _getVar(name)

    if not is_read:
        return False, -9999

    try:
        var = int(var)

    except Exception:

        getLogger(verbose_lvl=0).critical(
            f"Value of environment variable '{name}' is not an integer: '{var}'."
        )
        return False, -9999

    return True, var


def parsePath(
    name: str,
    raw_path: str,
    ctx: typer.Context | None = None,
    is_file: bool | None = None,
    is_dir: bool | None = None,
    mkdir: bool | None = None,
    is_critical: bool = False,
) -> Path:

    logger = getLogger(verbose_lvl=0)
    if is_critical:
        log_msg = logger.critical
    else:
        log_msg = logger.error

    try:
        path = Path(raw_path)
    except Exception as e:
        msg = f"{name} is not a valid path: '{raw_path}'."
        log_msg(msg)

    assert isinstance(path, Path)

    all_flags = [
        (is_file, "is_file"),
        (is_dir, "is_dir"),
        (mkdir, "mkdir"),
    ]

    flags = [n for f, n in all_flags if not f is None]

    if len(flags) > 1:
        raise Exception(f"Can not specify more than one flag: {flags}")

    if is_file and not path.is_file():
        msg = f"{name} is not a valid path: '{raw_path}'."
        log_msg(msg)
        raise typer.Exit(code=1)
    if mkdir:

        apath = path.absolute()
        if apath.exists():
            msg = f"{name} already exists: '{apath}'."
            log_msg(msg)
            raise typer.Exit(code=1)

        ppath = apath.parent
        if not ppath.exists():
            create = typer.confirm(
                f"Parent folder '{ppath}' does not exists, do you want to created it?"
            )

            if not create:
                logger.warning("Nothing was done")
                raise typer.Exit()

            ppath.mkdir(parents=True)

    return path


_REPO_PATH_ = "FUNWAVE_REPO_PATH"


def getRepoPath(
    ctx: typer.Context | None = None,
) -> str:
    """Check FUNWAVE repository path variable"""
    is_read, var = _getVar(_REPO_PATH_)
    _typerExitWrapper(not is_read, ctx)
    return var


def _getHPCParameters() -> tuple[bool, tuple]:
    """Returns True if HPC environment variables are defined"""

    is_read_all = True
    is_read, nproc = _getIntVar("FUNWAVE_HPC_NPROC")
    is_read_all = is_read_all and is_read

    is_read, nproc = _getVar("FUNWAVE_HPC_ABREV")
    is_read_all = is_read_all and is_read

    return is_read_all, (nproc,)


def getHPCParameters(
    ctx: typer.Context | None = None,
) -> tuple:
    is_read, var = _getHPCParameters()
    _typerExitWrapper(not is_read, ctx)
    return var


def checkEnvironment(
    ctx: typer.Context | None = None,
) -> None:
    """Check all environment variables"""

    is_read_all = True
    is_read, _ = _getVar(_REPO_PATH_)
    is_read_all = is_read_all and is_read

    is_read, _ = _getHPCParameters()
    is_read_all = is_read_all and is_read

    _typerExitWrapper(not is_read_all, ctx)


def _typerExitWrapper(
    is_error: bool,
    ctx: typer.Context | None = None,
) -> None:
    """Wrapper for printing help on no errors when ctx is passed and exiting on error."""
    if ctx is None:
        if is_error:
            raise typer.Exit(is_error)
    else:
        if ctx.invoked_subcommand is None:
            if not is_error:
                print(ctx.get_help())
            raise typer.Exit(is_error)


class LineFilter:

    def __init__(self, filter_map: dict[str, list[str] | str] | None = None) -> None:

        map = {} if filter_map is None else filter_map
        map = {k: [d] if isinstance(d, str) else d for k, d in map.items()}
        map = [(f, k) for k, filters in map.items() for f in filters]
        self._map = map
        self._errors = {}

    def match(self, lines: list[str]) -> dict[str, dict]:
        """Filter lines matching string masks with keys"""

        def update(key: str, line_number: int):
            if key in errors:
                d = errors[key]
            else:
                errors[key] = d = {"count": 0, "lines": []}

            d["count"] += 1
            d["lines"].append(line_number)

        errors = {}

        print(lines)
        for i, line in enumerate(lines, start=1):
            for filt, key in self._map:
                if filt in line:
                    print(filt, line)
                    update(key, i)

        self._errors = errors

        print(errors)
        self._n = sum([d["count"] for d in errors.values()])
        return errors

    def getCount(self) -> int:
        """Get number of matches"""
        return self._n

    def isSingle(self) -> bool:
        """Returns true of only on match is found"""
        return self._n == 1

    def isEmpty(self) -> bool:
        """Returns True if not matches are found"""
        print(self._n)
        return self._n == 0

    def getMatches(self) -> dict[str, dict]:
        """Returns a dict of matched filter keys with match counts and line numbers"""
        return self._errors

    def get(self) -> str:
        """Returns first/only key matched by filter"""
        assert not self.isEmpty(), "No filter match to return"
        return list(self._errors.keys())[0]


class Process:

    def __init__(self, label: str, verbose: int) -> None:
        self._label = label
        self.__is_done = False
        self.__is_running = False
        self.__is_success = False
        self._logger = getLogger(verbose)

    def _setDone(self) -> None:
        assert self.__is_running, f"Process {self.__class__.__name__} not running"
        self.__is_running = False
        self.__is_done = True

    def _setRunning(self) -> None:
        assert not self.__is_done, f"Process {self.__class__.__name__} is already done"
        self.__is_running = True

    def _setSuccess(self) -> None:
        self.__is_success = False

    def isSuccess(self) -> bool:
        return self.__is_success

    def isStarted(self) -> bool:
        if self.__is_done:
            return True
        return self.__is_running

    def isRunning(self) -> bool:
        return self.__is_running

    def isDone(self) -> bool:
        """"""
        return self.__is_done

    def getLabel(self) -> str:
        return self._label

    def run(self, *args, **kwargs):
        """"""

        """"""

    def runSpinner(self) -> None:
        with getSpinnerFormat() as progress:
            progress.add_task(self._label, total=None)
            results = self.run()

        return results


class ShellProcess(Process):

    def __init__(
        self,
        label: str,
        cmd: list[str],
        capture_output: bool,
        verbose: int,
        msg: str | None = None,
        err_map: dict[str, str | list[str]] | None = None,
    ) -> None:
        super().__init__(label, verbose)

        self._cmd = cmd
        self._opts = dict(
            capture_output=capture_output,
            text=False,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
            check=True,
        )
        self._capture = capture_output
        self._proc = None

        self._suc_msg = msg
        self._err_filter = LineFilter(err_map)
        self.__exit_hook = typer.Exit

    def run(self):

        self._setRunning()

        try:
            proc: subprocess.CompletedProcess = subprocess.run(self._cmd, **self._opts)
            stdout = proc.stdout
            stderr = proc.stderr
            return_code = proc.returncode

        except subprocess.CalledProcessError as e:
            stderr = e.stderr
            stdout = e.stdout
            return_code = e.returncode

        self._setDone()

        def _process(string: str) -> list[str]:
            string = string + "\n"
            lines = string.splitlines()

            if self._capture:
                lines = [l for l in lines]

            return lines

        logger = self._logger
        if return_code != 0:

            if stderr is None:
                logger.critical(
                    "Unknown shell exit state, non-zero exit code with no stderr"
                )
                raise self.__exit_hook(return_code)

            stderr = _process(stderr)

            filt = self._err_filter
            print(stderr)
            filt.match(stderr)

            if filt.isEmpty():
                logger.error(self._suc_msg)
                logger.critical("Unhandled Exception")
                raise RuntimeError("".join(stderr))

            else:
                logger.error(filt.get())
                raise self.__exit_hook(return_code)

        if return_code == 0:
            msg = self._suc_msg
            if not msg is None:
                logger.success(msg)

    def runSpinner(self) -> None:
        if self._capture:
            return super().runSpinner()
        else:
            return self.run()


def getSpinnerFormat() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}..."),
        transient=True,
    )


def runSimpleProgress(
    tasks: list[tuple[str, Callable]] | tuple[str, Callable],
):

    if isinstance(tasks, tuple):
        tasks = [tasks]
        is_single = True
    else:
        is_single = False

    results = []
    with getSpinnerFormat() as progress:
        for name, cmd in tasks:
            task = progress.add_task(description=name, total=None)
            results.append(cmd())
            progress.remove_task(task)

    if is_single:
        return results[0]
    else:
        return results


def getTyperOpts() -> dict:
    return _common_opts_
