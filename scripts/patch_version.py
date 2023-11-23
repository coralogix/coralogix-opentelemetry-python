import sys

import tomlkit
from time import time


def run() -> None:
    if len(sys.argv) != 2:
        raise Exception("usage: patch_version.py <branch>")
    _, branch = sys.argv
    if branch == "master":
        print("not patching master branch")
        return

    with open("pyproject.toml", "r") as py_project:
        content = tomlkit.load(py_project)
        content["tool"]["poetry"][  # type: ignore
            "version"
        ] = f'{content["tool"]["poetry"]["version"]}-{int(time())}'  # type: ignore
    if content is None:
        raise Exception('Could not find "tool.poetry.version" in pyproject.toml')
    with open("pyproject.toml", "w") as py_project:
        tomlkit.dump(content, py_project)
