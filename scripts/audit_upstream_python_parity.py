#!/usr/bin/env python3
"""Check that local Python modules retain every public upstream symbol.

This is a structural audit for ZIP-based upstream synchronization. It does not
require Isaac Sim and intentionally permits compatible function-body changes.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


def parse_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.add(f"{node.name}.{child.name}")
    return symbols


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root; defaults to the parent of scripts/.",
    )
    args = parser.parse_args()
    upstream_root = args.upstream_root.resolve()
    local_root = args.local_root.resolve()
    if not upstream_root.is_dir():
        parser.error(f"upstream root does not exist: {upstream_root}")
    if not local_root.is_dir():
        parser.error(f"local root does not exist: {local_root}")

    missing_files: list[str] = []
    missing_symbols: dict[str, list[str]] = {}
    parse_errors: list[str] = []
    checked = 0
    for upstream_file in sorted(upstream_root.rglob("*.py")):
        relative = upstream_file.relative_to(upstream_root)
        local_file = local_root / relative
        if not local_file.is_file():
            missing_files.append(str(relative))
            continue
        checked += 1
        try:
            missing = sorted(parse_symbols(upstream_file) - parse_symbols(local_file))
        except (OSError, SyntaxError) as exc:
            parse_errors.append(f"{relative}: {exc}")
            continue
        if missing:
            missing_symbols[str(relative)] = missing

    result = {
        "checked_python_files": checked,
        "missing_files": missing_files,
        "missing_symbol_files": missing_symbols,
        "parse_errors": parse_errors,
        "status": "PASS" if not (missing_files or missing_symbols or parse_errors) else "FAIL",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
