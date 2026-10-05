"""Collect what a deployed run actually used: numeric constants in its scripts and numeric
scalar settings in its output JSON. Read-only; limited to the deployment snapshot.
"""
from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

from common import SKIP_DIR_NAMES, in_deployment_snapshot, rel

MAX_JSON_BYTES = 2_000_000


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def tokens(name: str) -> set[str]:
    parts = re.split(r"[^a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower())
    stop = {"n", "num", "the", "of", "per", "default", "min", "max", ""}
    return {p for p in parts if p not in stop}


def run_files(run_dir: Path, suffix: str):
    out = []
    for dp, dns, fns in os.walk(run_dir):
        dns[:] = sorted(d for d in dns if d not in SKIP_DIR_NAMES and not d.startswith("._"))
        for f in sorted(fns):
            if f.startswith("._") or not f.endswith(suffix):
                continue
            p = Path(dp) / f
            if in_deployment_snapshot(p):
                out.append(p)
    return out


def _env_default(node):
    """int(os.environ.get("X", "50")) / float(os.getenv("X", "0.5")) -> 50.0"""
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("int", "float") and node.args:
        inner = node.args[0]
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr in ("get", "getenv"):
            if len(inner.args) >= 2 and isinstance(inner.args[1], ast.Constant):
                try:
                    return float(inner.args[1].value)
                except (TypeError, ValueError):
                    return None
    return None


def _numlist(node):
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
        vals = [_num(e) for e in node.elts]
        if all(v is not None for v in vals):
            return vals
    return None


def _num(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _num(node.operand)
        return -v if v is not None else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Pow)):
        a, b = _num(node.left), _num(node.right)
        if a is not None and b is not None:
            return a * b if isinstance(node.op, ast.Mult) else a ** b
    return None


def script_constants(py: Path):
    """(name, value, line, kind, function) for numeric assignments, keyword args, defaults, argparse."""
    src = py.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [], "syntax-error"
    out = []
    func_stack = []
    if_stack = []

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            func_stack.append(node.name)
            args = node.args
            defaults = args.defaults
            pos = args.args[len(args.args) - len(defaults):] if defaults else []
            for a, d in zip(pos, defaults):
                v = _num(d)
                if v is not None:
                    out.append((a.arg, v, node.lineno, "func_default", node.name))
            for a, d in zip(args.kwonlyargs, args.kw_defaults):
                v = _num(d) if d is not None else None
                if v is not None:
                    out.append((a.arg, v, node.lineno, "func_default", node.name))
            self.generic_visit(node)
            func_stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_If(self, node):
            if_stack.append(node)
            for n in node.body + node.orelse:
                self.visit(n)
            if_stack.pop()
            self.visit(node.test)

        def visit_Assign(self, node):
            v = _num(node.value)
            kind = "assign"
            if v is None:
                v = _env_default(node.value)
                kind = "env_default"
            if v is None:
                v = _numlist(node.value)
                kind = "assign_list"
            if v is not None:
                if if_stack:
                    kind += "(conditional)"
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        out.append((t.id, v, node.lineno, kind, func_stack[-1] if func_stack else ""))
                    elif isinstance(t, ast.Attribute):
                        out.append((t.attr, v, node.lineno, kind, func_stack[-1] if func_stack else ""))
            self.generic_visit(node)

        def visit_AnnAssign(self, node):
            v = _num(node.value) if node.value is not None else None
            if v is not None and isinstance(node.target, ast.Name):
                out.append((node.target.id, v, node.lineno, "assign", func_stack[-1] if func_stack else ""))
            self.generic_visit(node)

        def visit_Call(self, node):
            fname = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if fname == "add_argument":
                dest = None
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--"):
                        dest = a.value.lstrip("-").replace("-", "_")
                for kw in node.keywords:
                    if kw.arg == "default" and dest:
                        v = _num(kw.value)
                        if v is not None:
                            out.append((dest, v, node.lineno, "argparse", func_stack[-1] if func_stack else ""))
            for kw in node.keywords:
                if kw.arg and kw.arg != "default":
                    v = _num(kw.value)
                    if v is not None:
                        out.append((kw.arg, v, node.lineno, f"kwarg:{fname}", func_stack[-1] if func_stack else ""))
            self.generic_visit(node)

    V().visit(tree)
    return out, "ok"


def json_leaves(p: Path, max_list=3):
    """Numeric scalar leaves as (dotted_key_path, value). Lists: only first `max_list` items."""
    try:
        if p.stat().st_size > MAX_JSON_BYTES:
            return None
        obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None
    out = []

    def walk(o, path, depth):
        if depth > 6:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + [str(k)], depth + 1)
        elif isinstance(o, list):
            for i, v in enumerate(o[:max_list]):
                if isinstance(v, (dict, list)):
                    walk(v, path + [f"[{i}]"], depth + 1)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            out.append((".".join(path), float(o)))
    walk(obj, [], 0)
    return out


def collect(run_dir: Path):
    scripts = run_files(run_dir, ".py")
    consts, parse_status = [], {}
    for py in scripts:
        rows, st = script_constants(py)
        parse_status[rel(py)] = st
        for name, v, line, kind, fn in rows:
            consts.append(dict(source="script", file=rel(py), line=line, name=name, value=v, kind=kind, function=fn))
    jleaves = []
    jfiles = run_files(run_dir, ".json")
    n_skipped = 0
    for jp in jfiles:
        leaves = json_leaves(jp)
        if leaves is None:
            n_skipped += 1
            continue
        for kp, v in leaves:
            jleaves.append(dict(source="json", file=rel(jp), line=None, name=kp, value=v, kind="json", function=""))
    return dict(scripts=[rel(s) for s in scripts], script_parse=parse_status, constants=consts,
                json_files=len(jfiles), json_skipped=n_skipped, json_leaves=jleaves)
