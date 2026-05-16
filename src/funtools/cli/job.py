from enum import Enum

from ..core.resources import getJSON
from ..io.template import TemplateWriter


class SchedulerVendorEnum(str, Enum):
    PBSPRO = ("pbspro", "pbs")
    SLURM = ("slurm", "slurm")

    def __new__(cls, value, ext):
        obj = str.__new__(cls)
        obj._value_ = value
        obj._ext = ext
        return obj

    @property
    def ext(self) -> str:
        return self._ext


# Reading template files
def __load_files(name: str, extension: str, keys: list[str]) -> dict:
    return {k: f"{name}/{k}.{extension}" for k in keys}


_files = ["header"]
_HPC_FILES_ = {
    k.value: __load_files(k.value, k.ext, _files) for k in SchedulerVendorEnum
}
_files = ["init_funwave", "run_funwave"]
_SCRIPTS_ = __load_files("bash", "bash", _files)


def __wrap(data: dict) -> dict:
    return {k: f"${{{d}}}" for k, d in data.items()}


_HPC_VARS_ = {
    SchedulerVendorEnum(k).value: __wrap(d) for k, d in getJSON("hpc_env").items()
}

del _files, __load_files, __wrap


class SubmitScript:

    def __init__(self, system_type: SchedulerVendorEnum | str):

        if isinstance(system_type, str):
            system_type = SchedulerVendorEnum(system_type)

        self._type = system_type
        self._hpc_templates = _HPC_FILES_[system_type.value]
        self._hpc_vars = _HPC_VARS_[system_type.value]
        kwargs = dict(
            account="ERDCV00898FUN",
            queue="standard",
            name="test",
            nodes=1,
            cpus=192,
            mpi=192,
            walltime="0:30:00",
            email="test",
            array_range="1-5",
            nproc=192,
        )

        writer = TemplateWriter(f"test.{self._type.ext}")

        writer.append(self._hpc_templates["header"], **kwargs)

        kwargs = dict(hpc=self._hpc_vars)

        writer.append(_SCRIPTS_["init_funwave"], **kwargs)

        kwargs = dict(
            hpc=self._hpc_vars,
            files=["path1", "path2", "path3"],
            nproc=128,
            folder_type=2,
        )
        writer.append(
            _SCRIPTS_["run_funwave"],
            **kwargs,
        )

    def generateCustom(self, bash_script_path: str) -> None:
        pass
