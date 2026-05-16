import importlib.resources
import json


def getPath(path: str) -> str:

    # WORKAROUND: Weird bug MultiplexedPath not casting to string correctly
    return str(importlib.resources.files(path)._paths[0])


def getJSON(path: str) -> dict:
    fpath = importlib.resources.files("metadata").joinpath(path)
    with open(str(fpath) + ".json", "r", encoding="utf-8") as f:
        return json.load(f)
