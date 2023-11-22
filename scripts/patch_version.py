import sys

import tomlkit


def run() -> None:
    if len(sys.argv) != 3:
        raise Exception("usage: patch_version.py <patch> <branch>")
    _, patch, branch = sys.argv
    if branch == "master":
        print("not patching master breach")
        return

    with open("pyproject.toml", "r") as py_project:
        content = tomlkit.load(py_project)
        content["tool"]["poetry"][  # type: ignore #
            "version"
        ] = f'{content["tool"]["poetry"]["version"]}-{patch}'  # type: ignore
    if content is None:
        raise Exception('Could not find "tool.poetry.version" in pyproject.toml')
    with open("pyproject.toml", "w") as py_project:
        tomlkit.dump(content, py_project)
