#!/usr/bin/env python3
"""Pre-commit guard for public docstring/signature contracts.

Usage: python3 examples/notebooks/check_docstring_params.py [file...]

The guard intentionally stays syntactic.  It cannot prove scientific claims, but
it catches the common contract drift introduced by broad docstring passes:
invented parameters, missing parameters inside a ``Parameters`` section, missing
``Returns`` sections for documented non-None returns, and known misleading
phrases such as "Named tuple" for dataclass result objects.
"""

from __future__ import annotations

import ast
import glob
import os
import re
import sys


SECTION_HEADERS = {
    "Parameters",
    "Returns",
    "Yields",
    "Raises",
    "Notes",
    "Examples",
    "Attributes",
    "Warnings",
    "See Also",
}

FORBIDDEN_DOCSTRING_PATTERNS = (
    (re.compile(r"\bNamed tuple\b", re.IGNORECASE), "use the concrete dataclass/result type"),
    (re.compile(r"with\s+``x``", re.IGNORECASE), "result fields use parameter_vector, not x"),
    (re.compile(r"26-parameter model", re.IGNORECASE), "do not conflate this API with the 26p paper model"),
    (re.compile(r"docs/DOCSTRING_TODO\.md"), "the docstring TODO file has been completed and deleted"),
)


def _section(doc: str, header: str) -> str:
    lines = doc.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i + 1
            break
    if start is None:
        return ""
    if start < len(lines) and set(lines[start].strip()) == {"-"}:
        start += 1
    end = len(lines)
    for i in range(start, len(lines)):
        if lines[i].strip() in SECTION_HEADERS:
            end = i
            break
    return "\n".join(lines[start:end])


def _documented_parameters(doc: str) -> set[str]:
    params_section = _section(doc, "Parameters")
    if not params_section:
        return set()
    doc_params: set[str] = set()
    for line in params_section.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-"):
            continue
        if ":" in stripped:
            name_part = stripped.split(":", 1)[0].strip()
        else:
            name = stripped.strip().strip("`").lstrip("*")
            if name.isidentifier() and " " not in stripped and "," not in stripped:
                doc_params.add(name)
            continue
        for raw_name in name_part.split(","):
            name = raw_name.strip().strip("`").lstrip("*")
            if name.isidentifier():
                doc_params.add(name)
    return doc_params


def _real_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    real_args = {a.arg for a in node.args.posonlyargs}
    real_args |= {a.arg for a in node.args.args}
    real_args |= {a.arg for a in node.args.kwonlyargs}
    if node.args.vararg is not None:
        real_args.add(node.args.vararg.arg)
    if node.args.kwarg is not None:
        real_args.add(node.args.kwarg.arg)
    return real_args - {"self", "cls"}


def _has_non_none_return(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if node.returns is None:
        return False
    return not (isinstance(node.returns, ast.Constant) and node.returns.value is None)


def check_file(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=path)
    errors: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node)
        if doc is None:
            continue

        for pattern, fix_hint in FORBIDDEN_DOCSTRING_PATTERNS:
            if pattern.search(doc):
                errors.append(f"{path}:{node.lineno} {node.name}() has stale phrase: {fix_hint}")

        real_args = _real_parameters(node)
        if real_args and "Parameters" in doc:
            doc_params = _documented_parameters(doc)
            invented = doc_params - real_args
            missing = real_args - doc_params
            if invented:
                errors.append(
                    f"{path}:{node.lineno} {node.name}() invents params: {sorted(invented)} "
                    f"(real: {sorted(real_args)})"
                )
            if missing:
                errors.append(
                    f"{path}:{node.lineno} {node.name}() omits params: {sorted(missing)}"
                )

        if _has_non_none_return(node) and "Parameters" in doc and not _section(doc, "Returns"):
            errors.append(f"{path}:{node.lineno} {node.name}() documents parameters but no Returns")

    return errors


if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = glob.glob("src/stereocomplex/**/*.py", recursive=True)

    all_errors: list[str] = []
    checked = 0
    for f in files:
        if os.path.isfile(f):
            checked += 1
            all_errors.extend(check_file(f))

    if all_errors:
        print(f"DOCSTRING CONTRACT ERRORS: {len(all_errors)} issue(s)\n")
        for e in all_errors:
            print(f"  {e}")
        print("\nFix: align docstrings with the current signature and concrete result types.")
        sys.exit(1)

    print(f"OK: docstring contracts match signatures ({checked} files)")
    sys.exit(0)
