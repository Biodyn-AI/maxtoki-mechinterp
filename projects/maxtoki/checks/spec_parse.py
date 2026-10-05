"""Parsing helpers for pipeline markdown specs: sections, the parameter table, numbers.

These helpers are deliberately conservative. When a table cell cannot be read as a
number, a set, a range or a one-sided bound, the parser returns None and the caller
reports "not machine-readable". It never guesses a number out of free text.
"""
from __future__ import annotations

import math
import re

CANON = [
    ("Overview", r"^overview\b"),
    ("Source", r"^sources?\b"),
    ("Inputs", r"^inputs?\b"),
    ("Outputs", r"^outputs?\b"),
    ("Dependencies", r"^dependenc"),
    ("Methodology", r"^methodology\b"),
    ("Code references", r"^code\s+ref"),
    ("Parameters", r"^parameters?\b"),
    ("Validation", r"^validation\b"),
    ("Known pitfalls", r"^(known\s+)?pitfalls?\b"),
]

H2 = re.compile(r"^##\s+(?!#)(.*)$")
NUM_PREFIX = re.compile(r"^\s*(\d+)\.\s*(.*)$")
TABLE_ROW = re.compile(r"^\|.*\|\s*$")
TABLE_SEP = re.compile(r"^\|\s*:?-{2,}")


def split_sections(text: str):
    """Return (lines, [(title, start_idx, end_idx)]) for level-2 headings outside code fences."""
    lines = text.splitlines()
    starts = []
    in_code = False
    for i, ln in enumerate(lines):
        if ln.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = H2.match(ln)
        if m:
            starts.append((m.group(1).strip(), i))
    out = []
    for k, (title, i) in enumerate(starts):
        j = starts[k + 1][1] if k + 1 < len(starts) else len(lines)
        out.append((title, i, j))
    return lines, out


def canon_index(title: str):
    """Map a heading to its canonical template index (0..9) and its written number."""
    t = title
    num = None
    m = NUM_PREFIX.match(t)
    if m:
        num = int(m.group(1))
        t = m.group(2)
    t = re.sub(r"[*_`]", "", t).strip().lower()
    for idx, (_, pat) in enumerate(CANON):
        if re.search(pat, t):
            return idx, num
    return None, num


def section_map(text: str):
    """{canonical_index: dict(title, line (1-based), body_lines)} for the first match of each."""
    lines, secs = split_sections(text)
    found = {}
    for title, i, j in secs:
        idx, num = canon_index(title)
        if idx is not None and idx not in found:
            found[idx] = dict(title=title, line=i + 1, end=j, num=num, body=lines[i + 1:j])
    return lines, secs, found


# ---------------------------------------------------------------- numbers in spec cells
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
NUM_TOKEN = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"


def _clean(cell: str) -> str:
    s = cell.replace("**", "").replace("`", "").replace("\\|", "|")
    s = s.translate(SUP)
    s = s.replace("−", "-").replace("—", " — ").replace("≈", "~")
    s = re.sub(r"(\d)\s*[×x]\s*10\s*\^?\s*\{?\s*([-+]?\d+)\}?", r"\1e\2", s)   # 3×10-4 -> 3e-4
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)                                # 2,000 -> 2000
    s = re.sub(r"(?<=\d)\s*[×x](?![\w])", "", s)                                  # 4× -> 4
    return s.strip()


def _head(s: str) -> str:
    """Leading part of a cell before notes: cut at ';', '(' and ' — '."""
    s = re.split(r";|\(| — |\bsee\b|\bper\b|\bfor\b|\bwith\b|\bin paper\b", s)[0]
    return s.strip()


def parse_numeric(cell: str | None):
    """Parse a spec cell into a machine-readable constraint, or None.

    Returns dict(kind=..., ...):
      value  -> {'kind':'value','v':x}
      set    -> {'kind':'set','vals':[...]}
      range  -> {'kind':'range','lo':a,'hi':b}
      min    -> {'kind':'min','lo':a,'strict':bool}
      max    -> {'kind':'max','hi':b,'strict':bool}
    """
    if cell is None:
        return None
    s = _clean(cell)
    if not s or s in {"—", "-", "–"}:
        return None
    bold = re.findall(r"\*\*(.+?)\*\*", cell.replace("`", ""))
    cands = [_clean(b) for b in bold] if bold else [_head(s)]
    parsed = [_parse_expr(c) for c in cands]
    parsed = [p for p in parsed if p is not None]
    if not parsed:
        return None
    if len(parsed) == 1:
        return parsed[0]
    # several bold defaults, e.g. "**2000** (K562), **3309** (RPE1)": treat as alternatives
    vals = []
    for p in parsed:
        if p["kind"] == "value":
            vals.append(p["v"])
        elif p["kind"] == "set":
            vals.extend(p["vals"])
        else:
            return parsed[0]
    return {"kind": "set", "vals": vals, "note": "several bold values (alternatives by context)"}


def _parse_expr(s: str):
    s = s.strip().rstrip(".,")
    # strip a leading symbol name before a comparator / membership, e.g. "|d| > 0.5", "α ∈ {2, 5}"
    s = re.sub(r"^(?:\|?[A-Za-zα-ωΑ-Ω_][\w|]*\|?\s*)(?=(?:[<>≤≥=∈]))", "", s)
    s = s.replace("∈", "").replace("~", "").strip()
    s = s.replace("%", "")
    m = re.fullmatch(r"(>=|≥|>|<=|≤|<|at least|at most|more than|less than)\s*(" + NUM_TOKEN + r")\+?", s, re.I)
    if m:
        op, v = m.group(1).lower(), float(m.group(2))
        if op in (">=", "≥", "at least"):
            return {"kind": "min", "lo": v, "strict": False}
        if op in (">", "more than"):
            return {"kind": "min", "lo": v, "strict": True}
        if op in ("<=", "≤", "at most"):
            return {"kind": "max", "hi": v, "strict": False}
        return {"kind": "max", "hi": v, "strict": True}
    m = re.fullmatch(r"(" + NUM_TOKEN + r")\s*\+", s)
    if m:
        return {"kind": "min", "lo": float(m.group(1)), "strict": False}
    m = re.fullmatch(r"(" + NUM_TOKEN + r")\s*(?:–|-|to)\s*(" + NUM_TOKEN + r")", s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if a < b:
            return {"kind": "range", "lo": a, "hi": b}
    m = re.fullmatch(r"[\[{]?\s*(" + NUM_TOKEN + r"(?:\s*,\s*" + NUM_TOKEN + r")+)\s*[\]}]?", s)
    if m:
        return {"kind": "set", "vals": [float(x) for x in re.split(r"\s*,\s*", m.group(1))]}
    m = re.fullmatch(r"[\[{]?\s*(" + NUM_TOKEN + r")\s*[\]}]?", s)
    if m:
        return {"kind": "value", "v": float(m.group(1))}
    return None


def satisfies(x: float, c: dict, rel_tol: float = 1e-9) -> bool:
    """Does value x satisfy constraint c (as produced by parse_numeric)?"""
    def eq(a, b):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-12)
    k = c["kind"]
    if k == "value":
        return eq(x, c["v"])
    if k == "set":
        return any(eq(x, v) for v in c["vals"])
    if k == "range":
        return c["lo"] - 1e-12 <= x <= c["hi"] + 1e-12
    if k == "min":
        return x > c["lo"] if c.get("strict") else x >= c["lo"] - 1e-12
    if k == "max":
        return x < c["hi"] if c.get("strict") else x <= c["hi"] + 1e-12
    return False


def describe(c: dict | None) -> str:
    if c is None:
        return "not machine-readable"
    k = c["kind"]
    f = lambda v: ("%g" % v)
    if k == "value":
        return f(c["v"])
    if k == "set":
        return "{" + ", ".join(f(v) for v in c["vals"]) + "}"
    if k == "range":
        return f"{f(c['lo'])}–{f(c['hi'])}"
    if k == "min":
        return (">" if c.get("strict") else "≥") + " " + f(c["lo"])
    if k == "max":
        return ("<" if c.get("strict") else "≤") + " " + f(c["hi"])
    return str(c)


# ---------------------------------------------------------------- parameter table
def _cells(row: str):
    row = row.strip()
    row = row.replace("\\|", "\u0000")
    parts = [p.replace("\u0000", "|").strip() for p in row.strip("|").split("|")]
    return parts


def parse_param_table(body_lines):
    """Parse every markdown table in the Parameters section.

    Returns (rows, header_info). Each row: dict(name, qualifier, default_raw, range_raw,
    used_in_raw, default, range, nonnegotiable, line_offset).
    """
    rows = []
    headers = []
    i = 0
    n = len(body_lines)
    while i < n:
        s = body_lines[i].strip()
        if TABLE_ROW.match(s) and i + 1 < n and TABLE_SEP.match(body_lines[i + 1].strip()):
            hdr = [h.lower() for h in _cells(s)]
            headers.append(hdr)
            def col(pat, default=None):
                for k, h in enumerate(hdr):
                    if re.search(pat, h):
                        return k
                return default
            c_name = col(r"^param|^name", 0)
            c_def = col(r"default", 1)
            c_rng = col(r"range|alternate|sensitiv|valid|allowed")
            c_use = col(r"used in|stage|phase")
            i += 2
            while i < n and TABLE_ROW.match(body_lines[i].strip()):
                cells = _cells(body_lines[i])
                name_cell = cells[c_name] if c_name < len(cells) else ""
                m = re.search(r"`([^`]+)`", name_cell)
                name = m.group(1).strip() if m else re.sub(r"[*_]", "", name_cell).strip()
                qual = re.sub(r"`[^`]+`", "", name_cell).strip()
                other = [c for k, c in enumerate(cells) if k != c_name]
                if not m and all(not c.strip() for c in other):
                    i += 1           # group header row such as "**Experiment 1**"
                    continue
                d_raw = cells[c_def] if c_def is not None and c_def < len(cells) else ""
                r_raw = cells[c_rng] if c_rng is not None and c_rng < len(cells) else ""
                u_raw = cells[c_use] if c_use is not None and c_use < len(cells) else ""
                nonneg = bool(re.search(r"non-?negotiable", r_raw + " " + d_raw, re.I))
                rows.append(dict(
                    name=name, qualifier=qual, default_raw=d_raw, range_raw=r_raw,
                    used_in_raw=u_raw, default=parse_numeric(d_raw),
                    range=None if nonneg else parse_numeric(r_raw),
                    nonnegotiable=nonneg, line_offset=i,
                ))
                i += 1
            continue
        i += 1
    return rows, headers
