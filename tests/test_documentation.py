"""Enforce the public docstring contract and smoke-test pdoc generation."""

from __future__ import annotations

import ast
from pathlib import Path

from scripts.generate_docs import DOCUMENTED_MODULES, build_documentation

PROJECT_ROOT = Path(__file__).parents[1]
PACKAGE_ROOTS = (
    "game",
    "agents",
    "solvers",
    "ml_logger",
    "integrations",
    "scripts",
    "ui",
)
CONTROL_FLOW_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Match,
    ast.Raise,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
)


def test_modules_classes_and_public_callables_have_docstrings():
    """Report every public documentation gap in one actionable assertion."""
    violations = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative_path = path.relative_to(PROJECT_ROOT)
        if path.name != "__init__.py" and not ast.get_docstring(tree):
            violations.append(f"{relative_path}:1 module")
        parents = _parent_map(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and not ast.get_docstring(node):
                violations.append(
                    f"{relative_path}:{node.lineno} class {node.name}"
                )
            if _is_public_module_or_class_callable(node, parents):
                if not ast.get_docstring(node):
                    violations.append(
                        f"{relative_path}:{node.lineno} callable {node.name}"
                    )
    assert violations == []


def test_pdoc_builds_every_documented_package(tmp_path):
    """Render the full reference and verify each requested module has output."""
    output_directory = build_documentation(tmp_path / "api")

    assert (output_directory / "index.html").exists()
    for module in DOCUMENTED_MODULES:
        expected_path = output_directory / module.replace(".", "/")
        assert expected_path.with_suffix(".html").exists() or (
            expected_path / "index.html"
        ).exists()


def test_nontrivial_private_callables_have_docstrings():
    """Protect algorithmic and lifecycle helpers not visible in pdoc."""
    violations = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = _parent_map(tree)
        for node in ast.walk(tree):
            if _is_nontrivial_private_callable(node, parents):
                if not ast.get_docstring(node):
                    relative_path = path.relative_to(PROJECT_ROOT)
                    violations.append(
                        f"{relative_path}:{node.lineno} callable {node.name}"
                    )
    assert violations == []


def _python_sources():
    """Yield application sources covered by the documentation policy."""
    for package in PACKAGE_ROOTS:
        yield from (PROJECT_ROOT / package).rglob("*.py")
    yield PROJECT_ROOT / "benchmark.py"


def _parent_map(tree):
    """Map AST nodes to their immediate parents."""
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _is_public_module_or_class_callable(node, parents):
    """Return whether a node is a public top-level function or class method."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    if node.name.startswith("_"):
        return False
    return isinstance(parents.get(node), (ast.Module, ast.ClassDef))


def _is_nontrivial_private_callable(node, parents):
    """Identify complex private module functions and methods conservatively."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    if not isinstance(parents.get(node), (ast.Module, ast.ClassDef)):
        return False
    if not node.name.startswith("_") or (
        node.name.startswith("__") and node.name.endswith("__")
    ):
        return False
    line_count = node.end_lineno - node.lineno + 1
    control_flow_count = sum(
        isinstance(child, CONTROL_FLOW_NODES) for child in ast.walk(node)
    )
    return line_count > 12 or control_flow_count >= 2
