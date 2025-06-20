from __future__ import annotations
import json
from pathlib import Path
from dask.distributed import Client
from collections.abc import Callable
import importlib
import typing

# TODO: Remove
from .cluster import DASK_SCHEDULER_ADDRESS


def simple(
    self,
    funcs: list[Callable] | Callable,
    args: list[tuple] | tuple | None,
    kwargs: list[dict] | None = None,
) -> list:
    """Executes embarrassingly simple parallel tasks using dask"""
    client = Client(DASK_SCHEDULER_ADDRESS)  # , asynchronous=True)
    futures = []

    nc = len(funcs) if isinstance(funcs, list) else 1

    if args is None:
        na = 1
        args: tuple = ()
    else:
        na = len(args) if isinstance(args, list) else 1

    if kwargs is None:
        nk = 1
        kwargs: dict = {}
    else:
        nk = len(kwargs) if issubclass(kwargs, dict) else 1

    are_same = (nc == nk) and (nc == na)

    if are_same and na == 1:
        raise TypeError("At least one of funcs, args, or kwargs must be a list")

    n = max(nc, nk, na)

    if not nc == n or nc == 1:
        raise TypeError(f"Length of funcs, {nc:d}, does not match total, {n:d}")

    if not na == n or na == 1:
        raise TypeError(f"Length of args, {na:d}, does not match total, {n:d}")

    if not nk == n or nk == 1:
        raise TypeError(f"Length of kwargs, {nk:d}, does not match total, {n:d}")

    if nc == 1:
        funcs = [funcs] * n
    if nk == 1:
        kwargs = [kwargs] * n
    if na == 1:
        args = [args] * n

    for (
        f,
        a,
        k,
    ) in zip(funcs, args, kwargs):
        future = client.submit(f, *a, **k)
        futures.append(future)

    return client.gather(futures)


class ClassTask:
    """Wrapper class for linking class methods for delayed execution as tasks"""

    def __init__(self, method_name: str, args: tuple = (), kwargs: dict = {}) -> None:
        self._method = method_name
        self._args = args
        self._kwargs = kwargs

    def link(self, cls: typing.Any) -> None:
        """Link class to call methods from"""
        self._cls = cls

    def execute(self) -> typing.Any:
        """Executes tasks and returns results"""
        return getattr(self._cls, self._method)(*self._args, **self._kwargs)

    def init_to_dict(self) -> dict:
        return {"method": self._method, "args": self._args, "kwargs": self._kwargs}

    @classmethod
    def from_dict(self, **kwargs) -> ClassTask:
        return ClassTask(kwargs["method"], *kwargs["args"], **kwargs["kwargs"])


class ClassJob:
    """Wrapper class for queuing collection of ClassTask for delayed exeuction as a job"""

    def __init__(
        self, class_name: str | type, args: tuple = (), kwargs: dict = {}
    ) -> None:

        if not isinstance(class_name, str):
            class_name = str(class_name)[8:-2]

        self._tasks = []

        self._class = class_name
        self._args = args
        self._kwargs = kwargs

        self._linked_cls = None

    def init_class(self) -> None:

        if not self._linked_cls is None:
            raise RuntimeError(
                "Can not initialize linked class as it has already been linked or initialized"
            )

        *module, cls_str = self._class.split(".")
        module = ".".join(module)

        module = importlib.import_module(module)
        my_class = getattr(module, cls_str)
        self._linked_cls = my_class(*self._args, **self._kwargs)

    def add(self, task: ClassTask, name: str | None = None) -> None:
        """Add task to queue with optional name (default: Run #). If link is set,
        new task is automatically linked"""
        if name is None:
            i = len(self._tasks) + 1
            name = f"Task {i:d}"
        self._tasks.append((name, task))

        if not self._linked_cls is None:
            task.link(self._linked_cls)

    def create(
        self, method: str, args: tuple = (), kwargs: dict = {}, name: str | None = None
    ) -> None:
        """Creates and adds ClassTask to queue using init args and/or kwargs"""
        self.add(ClassTask(method, args, kwargs), name=name)

    def link(self, cls: typing.Any) -> None:
        """Links class to tasks"""
        for t in self._tasks:
            t.link(cls)

    def execute(self) -> dict:
        """Execute all tasks and returns results as dict using task name as key"""
        return {n: t.execute() for n, t in self._tasks}

    @property
    def n(self) -> int:
        """Returns the number of tasks"""
        return len(self._tasks)

    def get_manifest(self) -> dict:
        """Returns dict manifest for initalizing job with tasks"""
        return {
            "class_name": self._class,
            "args": self._args,
            "kwargs": self._kwargs,
            "tasks": {n: t.init_to_dict() for n, t in self._tasks},
        }

    @classmethod
    def execute_manifest(cls, **kwargs) -> dict:
        """Executes class tasks/methods from mainifest and returns results"""

        self = ClassJob(kwargs["class_name"], kwargs["args"], kwargs["kwargs"])
        self.init_class()

        for k, d in kwargs["tasks"].items():
            self.create(**d, name=k)

        return self.execute()

    @classmethod
    def from_manifest(cls, **kwargs) -> ClassJob:
        self = ClassJob(kwargs["class_name"], kwargs["args"], kwargs["kwargs"])
        self.init_class()

        for k, d in kwargs["tasks"].items():
            self.create(**d, name=k)

        return self


class Scheduler:
    """Class for queuing collection of ClassJob for execution in parallel"""

    def __init__(self) -> None:
        self._jobs = []

    def add(self, job: ClassJob, name: str | None = None) -> None:

        if name is None:
            i = len(self._jobs) + 1
            name = f"Job {i:d}"

        self._jobs.append((name, job))

    def execute(self) -> dict:
        """Executes jobs in parallel using dask and returns results as dict with job name as key"""

        client = Client(DASK_SCHEDULER_ADDRESS)

        futures = []
        names = []
        for name, job in self._jobs:
            # Messy solution, want class creation on job node, not host scheduler
            manifest = job.get_manifest()

            # ClassJob.execute_manifest(**manifest)
            future = client.submit(ClassJob.execute_manifest, **manifest)
            futures.append(future)
            names.append(name)

        results = client.gather(futures)

        return {n: r for n, r in zip(names, results)}

    def get_manifest(self) -> dict:
        """Returns dict manifest for initializing a collection of jobs with tasks"""
        return {n: j.get_manifest() for n, j in self._jobs}

    def export_manifest_json(self, fpath: Path) -> None:
        with open(fpath, "w") as f:
            json.dump(self.get_manifest(), f, indent=2)

    @classmethod
    def execute_manifest_json(cls, fpath: Path) -> dict:
        """Executes jobs in parallel from manifest json file"""
        with open(fpath, "r") as file:
            return cls.execute_manifest(json.load(file))

    @classmethod
    def execute_manifest(cls, kwargs: dict) -> dict:
        """Executes jobs in dask parallel from manifest dict"""
        self = cls()
        for n, j in kwargs.items():
            self.add(ClassJob.from_manifest(**j), name=n)

        return self.execute()
