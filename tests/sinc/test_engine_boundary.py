"""The sinc engine is file-free and does not reach into the rest of yippy."""

import ast
from pathlib import Path

import yippy.sinc

ENGINE = Path(yippy.sinc.__file__).parent


def _imported_modules(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = ["yippy", "sinc", *path.relative_to(ENGINE).parts[:-1]]
                base = package[: len(package) - (node.level - 1)]
                yield ".".join([*base, node.module] if node.module else base)
            else:
                yield node.module


def test_engine_imports_only_itself_from_yippy():
    offenders = []
    for path in sorted(ENGINE.rglob("*.py")):
        for module in _imported_modules(path):
            top = module.split(".")
            if top[0] == "yippy" and top[:2] != ["yippy", "sinc"]:
                offenders.append((path.name, module))
    assert not offenders


def test_engine_does_not_import_astropy():
    offenders = [
        (path.name, module)
        for path in sorted(ENGINE.rglob("*.py"))
        for module in _imported_modules(path)
        if module.split(".")[0] == "astropy"
    ]
    assert not offenders
