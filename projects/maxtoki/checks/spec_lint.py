#!/usr/bin/env python3
"""spec_lint.py -- deterministic lint of pipeline markdown specs (read-only).

Checks per spec (ids are used in README.md and CHECKER_REPORT.md):
  S1  the ten template sections are present as level-2 headings, in template order,
      with heading numbers equal to the template position.
  S2  each present section has a body of >= 200 characters and no TBD/TODO marker.
  S3  the Source section names a 40-hex pinned commit that (a) equals the text in
      repos/<name>/PINNED_COMMIT.txt and (b) where the clone still has .git, is the
      clone's HEAD. An explicit "no repository / N/A" statement is recorded as a
      waiver that needs a human decision.
  S4  code references: counts the required `repos/<name>/path:LINE` form, other forms
      (repos/ path without a line, bare paths, name-only), and whether each file
      exists, whether LINE is within the file, and whether a named function/class is
      defined in the file (or found within +-3 lines of LINE).
  S5  the Parameters section has a table with a name, default and range column;
      counts rows whose default / range can be read as numbers.
  S6  Validation section: splits into items and counts items with a numeric
      threshold (number + comparator/range/tolerance), with a number only, or with no
      number; also counts items anchored to the source model and "if X then broken" rules;
      checks for a machine-readable validation block (fenced yaml with `validation:`).
  S7  named slots: does the spec have a labelled slot (heading or bold label) for a
      null model, a trivial baseline, a positive control and a scope statement?
      Also counts plain keyword mentions.

What it does NOT establish: that any section is correct, sufficient or was followed;
that the working tree of a clone without .git is the pinned commit; that a keyword
mention is a real control. See README.md.

Outputs (checks/results/): spec_lint.json, spec_lint_summary.csv, spec_lint_coderefs.csv,
spec_lint_params.csv, spec_lint_validation_items.csv, spec_lint_run_config.json.
Exit code: 0 (report mode). With --strict: 1 if any spec fails S1 or S3 or cites a code path that
does not resolve (the blocking rule proposed for a pre-run gate).
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEPLOYED, OTHER_SPECS, RESULTS, ROOT, rel, run_config, write_json  # noqa: E402
from spec_parse import CANON, TABLE_ROW, canon_index, describe, parse_param_table, split_sections  # noqa: E402

MIN_BODY = 200
HEX40 = re.compile(r"\b[0-9a-f]{40}\b")
CODE_EXT = r"(?:py|sh|ipynb|R|r|jl|js|ts)"

# --- S4 patterns ---------------------------------------------------------------
# repos/<name>/<path>[:LINE[-LINE2]] (optionally preceded by ../ segments)
RE_REPOS = re.compile(r"(?:\.\./)*repos/([A-Za-z0-9_.\-]+)/([^\s`'\"():,\]\[;]+)(?::(\d+)(?:\s*[-–]\s*(\d+))?)?")
# backticked bare code path containing a slash, e.g. `src/03_x/06_y.py` or `src/a.py:120`
RE_BARE = re.compile(r"`((?!repos/|\.\./|https?:)[A-Za-z0-9_.\-{}*,/]+/[A-Za-z0-9_.\-{}*,]+\." + CODE_EXT + r")(?::(\d+))?`")
# backticked bare file name without a slash, e.g. `c1_hyperparam_ablation.py`
RE_NAME = re.compile(r"`([A-Za-z0-9_.\-]+\." + CODE_EXT + r")(?::(\d+))?`")
# line-only pointer attached to a symbol, e.g. `ScGPTRuntime` (`:449`)
RE_SYM_LINE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`\s*\(`:(\d+)`\)")
RE_REL_BASE = re.compile(r"(?:paths?|All paths)\s+relative to\s+`?((?:\.\./)*repos/[A-Za-z0-9_.\-]+/?)`?", re.I)

# --- S6 patterns (validation items) --------------------------------------------
LABEL_NUM = re.compile(
    r"\b(?:phase|step|stage|layer|iter(?:ation)?|fig(?:ure)?|table|supp(?:lementary)?|note|section|"
    r"experiment|family|test family|condition|iter_)\s*[-_]?\d+|§\s*\d+", re.I)
LABEL_ID = re.compile(r"\b(?:[HLPECGSV]|SV|iter_|K)\d+\b|\bK562\b|\bRPE1\b|\bV2\b")
NUMBER = re.compile(r"(?<![A-Za-z_])[-+−]?\d+(?:[.,]\d+)?(?:\s*[×x]\s*10[⁻-]?\d+)?%?")
COMPARATOR = re.compile(
    r"(≥|≤|>=|<=|(?<![-=])>\s*[-+−]?\d|<\s*[-+−]?\d|±|\d\s*[–—-]\s*\d|"
    r"\bbetween\s+[-+]?\d|\bwithin\s+[-+±]?\d|\btolerance\b|\bat least\s+\d|\bat most\s+\d|"
    r"\bmore than\s+\d|\bless than\s+\d|\babove\s+\d|\bbelow\s+\d|\bexceed)", re.I)
ITEM_START = re.compile(r"^(\d+\.|[-*])\s+")
TABLE_SEP = re.compile(r"^\|\s*:?-{2,}")
SOURCE_ANCHOR = re.compile(r"(geneformer|scgpt|source paper|the paper|paper'?s|table s\d|supp(lementary)? note|"
                           r"fig\. ?s\d|in the source)", re.I)
BUG_RULE = re.compile(r"\bif\b[^.]{0,200}?(broken|bug|wrong|not actually|mis-?|is not|are not|check|indicat|"
                      r"suspicious|too weak|too strict|flipped|leak|fail)", re.I)

# --- S7 slot patterns -----------------------------------------------------------
SLOTS = {
    "null_model": r"null[ -]model|\bnulls?\b",
    "trivial_baseline": r"trivial[ -]baseline|gene[- ]level baseline|simple baseline",
    "positive_control": r"positive[ -]control",
    "scope_statement": r"\bscope\b",
}


def git(args, cwd):
    r = subprocess.run(["git", "-C", str(cwd)] + args, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def pinned_text(name: str) -> str:
    p = ROOT / "repos" / name / "PINNED_COMMIT.txt"
    return p.read_text(errors="ignore") if p.exists() else ""


# ------------------------------------------------------------------ S1 / S2
def check_sections(lines, secs):
    found, order, num_mismatch, extras = {}, [], [], []
    for title, i, j in secs:
        idx, num = canon_index(title)
        body = "\n".join(lines[i + 1:j]).strip()
        if idx is None:
            extras.append({"title": title, "line": i + 1, "after_section_10": bool(found.get(9))})
            continue
        order.append(idx)
        if idx not in found:
            found[idx] = dict(title=title, line=i + 1, chars=len(body), body=lines[i + 1:j], start=i, end=j)
        if num is not None and num != idx + 1:
            num_mismatch.append({"title": title, "written_number": num, "template_position": idx + 1})
    missing = [CANON[k][0] for k in range(10) if k not in found]
    in_order = all(a < b for a, b in zip(order, order[1:]))
    extras_before_10 = [e["title"] for e in extras if not e["after_section_10"]]
    short = [CANON[k][0] for k, v in found.items() if v["chars"] < MIN_BODY]
    tbd = [CANON[k][0] for k, v in found.items()
           if re.search(r"\b(TBD|TODO|to be (written|added|filled))\b", "\n".join(v["body"]), re.I)]
    s1 = "PASS" if (not missing and in_order and not num_mismatch and not extras_before_10) else "FAIL"
    s2 = "PASS" if (not short and not tbd) else "FAIL"
    return found, dict(
        S1=s1, present=[CANON[k][0] for k in sorted(found)], missing=missing, in_order=in_order,
        heading_number_mismatches=num_mismatch, extra_sections=[e["title"] for e in extras],
        extra_sections_before_section_10=extras_before_10,
        S2=s2, short_sections=short, tbd_sections=tbd,
        section_chars={CANON[k][0]: v["chars"] for k, v in sorted(found.items())},
    )


# ------------------------------------------------------------------ S3
def check_pin(found):
    if 1 not in found:
        return dict(S3="FAIL", status="NO_SOURCE_SECTION", hashes=[])
    body = "\n".join(found[1]["body"])
    clones = sorted(set(re.findall(r"repos/([A-Za-z0-9_.\-]+)/", body)))
    hashes = sorted(set(HEX40.findall(body)))
    rows = []
    for h in hashes:
        owners = [c.parent.name for c in sorted((ROOT / "repos").glob("*/PINNED_COMMIT.txt")) if h in c.read_text(errors="ignore")]
        row = dict(hash=h, pinned_commit_txt_in=owners)
        for name in owners:
            clone = ROOT / "repos" / name
            if (clone / ".git").exists():
                rc, typ = git(["cat-file", "-t", h], clone)
                _, head = git(["rev-parse", "HEAD"], clone)
                _, dirty = git(["status", "--porcelain", "--untracked-files=no"], clone)
                row.update(git_present=True, commit_in_clone=(rc == 0 and typ == "commit"),
                           head=head, head_equals_pin=(head == h),
                           modified_tracked_files=len([x for x in dirty.splitlines() if x.strip()]))
            else:
                row.update(git_present=False, commit_in_clone=None, head=None, head_equals_pin=None,
                           modified_tracked_files=None)
        rows.append(row)
    waiver = bool(re.search(r"(N/A|no public repository|not have a reference implementation|no repo)", body, re.I))
    if not hashes:
        status = "NONE_WITH_REASON" if waiver else "MISSING"
        s3 = "WAIVER" if waiver else "FAIL"
    else:
        ok_text = all(r["pinned_commit_txt_in"] for r in rows)
        verified = all(r.get("head_equals_pin") for r in rows)
        clone_mismatch = [c for c in clones if (ROOT / "repos" / c / "PINNED_COMMIT.txt").exists()
                          and not any(h in pinned_text(c) for h in hashes)]
        if not ok_text:
            status, s3 = "HASH_NOT_IN_ANY_PINNED_COMMIT_TXT", "FAIL"
        elif verified:
            status, s3 = "VERIFIED_GIT_HEAD", "PASS"
        else:
            status, s3 = "TEXT_MATCH_ONLY_NO_GIT", "WARN"
        if clone_mismatch:
            status += f"; other clones named in Source without their commit in this spec: {clone_mismatch}"
    return dict(S3=s3, status=status, hashes=rows, clones_named_in_source=clones)


# ------------------------------------------------------------------ S4
def file_lines(p: Path):
    try:
        with open(p, "rb") as fh:
            return fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None


def symbol_defined(text_lines, sym, near=None, window=3):
    pat = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+" + re.escape(sym) + r"\b|^\s*" + re.escape(sym) + r"\s*[:=]")
    hits = [k + 1 for k, ln in enumerate(text_lines) if pat.search(ln)]
    if near is None:
        return bool(hits), hits[:3]
    return any(abs(h - near) <= window for h in hits), hits[:3]


def section_of(line_idx, found):
    for k, v in found.items():
        if v["start"] <= line_idx < v["end"]:
            return CANON[k][0]
    return "(outside template sections)"


_FILE_CACHE: dict = {}


def _all_files(base: Path):
    if base not in _FILE_CACHE:
        files = []
        if base.exists():
            for dp, dns, fns in __import__("os").walk(base):
                dns[:] = [d for d in dns if d not in {".git", "node_modules", "__pycache__"}]
                files += [Path(dp) / f for f in fns if not f.startswith("._")]
        _FILE_CACHE[base] = files
    return _FILE_CACHE[base]


def resolve_bare(path, bases):
    """Resolve a bare path: as written against each base, then against ROOT, then as a
    unique path suffix inside a base (reported separately as 'ok_by_suffix')."""
    if any(ch in path for ch in "{}*"):
        return None, "pattern"
    for b in bases:
        p = (b / path)
        if p.exists():
            return p, "ok"
    p = ROOT / path
    if p.exists():
        return p, "ok"
    hits = []
    for b in bases:
        hits += [f for f in _all_files(b) if f.as_posix().endswith("/" + path)]
    if len(hits) == 1:
        return hits[0], "ok_by_suffix"
    if len(hits) > 1:
        return None, "ambiguous_suffix"
    return None, "missing"


SYM_TOKEN = re.compile(r"\s*(?:`([A-Za-z_][A-Za-z0-9_.]*)(?:\(\))?`|([A-Za-z_][A-Za-z0-9_.]*)\(\))")
SYM_SEP = re.compile(r"\s*(?:,|and|\(|\)|/)\s*")
INTRO = re.compile(r"^`?\s*(?:→|->|:)\s*")


def symbols_after(ln: str, end: int):
    """Backticked identifiers (or name()) right after an arrow/colon that follows a path."""
    m = INTRO.match(ln[end:])
    if not m:
        return []
    pos = end + m.end()
    syms = []
    while True:
        t = SYM_TOKEN.match(ln, pos)
        if not t:
            break
        name = (t.group(1) or t.group(2)).split(".")[-1]
        syms.append(name)
        pos = t.end()
        sep = SYM_SEP.match(ln, pos)
        if sep and sep.end() > pos:
            pos = sep.end()
        elif not SYM_TOKEN.match(ln, pos):
            break
    return syms


def check_coderefs(text, lines, found, spec_path: Path, pin_clones):
    bases = [ROOT / "repos" / c for c in pin_clones]
    for m in RE_REL_BASE.finditer(text):
        b = (spec_path.parent / m.group(1)).resolve() if m.group(1).startswith("..") else ROOT / m.group(1)
        if b not in bases:
            bases.append(b)
    refs = []
    in_code = False
    pending_names = []
    for li, ln in enumerate(lines):
        if ln.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        spans = []
        line_refs = []
        for m in RE_REPOS.finditer(ln):
            name, path, line_no = m.group(1), m.group(2).rstrip(".,"), m.group(3)
            spans.append(m.span())
            f = ROOT / "repos" / name / path
            form = "repos_path_line" if line_no else "repos_path_no_line"
            r = dict(raw=m.group(0), form=form, repo=name, path=path, file=f if f.exists() else None,
                     line=int(line_no) if line_no else None, spec_line=li + 1, section=section_of(li, found),
                     relative_prefix=m.group(0).startswith("../"), end=m.end(),
                     resolve_status="ok" if f.exists() else "missing")
            refs.append(r); line_refs.append(r)
        for m in RE_BARE.finditer(ln):
            if any(a <= m.start() < b for a, b in spans):
                continue
            path, line_no = m.group(1), m.group(2)
            f, st = resolve_bare(path, bases)
            r = dict(raw=m.group(0), form="bare_path_line" if line_no else "bare_path", repo=None, path=path,
                     file=f, line=int(line_no) if line_no else None, spec_line=li + 1,
                     section=section_of(li, found), resolve_status=st, relative_prefix=False, end=m.end())
            refs.append(r); line_refs.append(r)
            spans.append(m.span())
        for m in RE_NAME.finditer(ln):
            if any(a <= m.start() < b for a, b in spans):
                continue
            r = dict(raw=m.group(0), form="name_only", repo=None, path=m.group(1), file=None,
                     line=int(m.group(2)) if m.group(2) else None, spec_line=li + 1,
                     section=section_of(li, found), resolve_status="pending", relative_prefix=False, end=m.end())
            refs.append(r); line_refs.append(r); pending_names.append(r)
        for r in line_refs:
            r["symbols"] = symbols_after(ln, r["end"])
        for m in RE_SYM_LINE.finditer(ln):
            prev = [r for r in line_refs if r["end"] <= m.start()] or line_refs
            if prev:
                prev[-1].setdefault("symbol_lines", []).append((m.group(1), int(m.group(2))))
    # name-only references: search the pinned clones and the folders of resolved paths in this spec
    name_bases = list(bases) + sorted({Path(r["file"]).parent for r in refs
                                       if r["file"] is not None and r["form"] != "name_only"})
    for r in pending_names:
        hits = []
        for b in name_bases:
            if b.is_dir() and b.parent != ROOT / "repos":
                cand = b / r["path"]
                if cand.exists() and cand not in hits:
                    hits.append(cand)
        if not hits:
            for b in bases:
                hits += [f for f in _all_files(b) if f.name == r["path"] and f not in hits]
        r["n_matches"] = len(hits)
        r["file"] = hits[0] if len(hits) == 1 else None
        r["resolve_status"] = "ok_by_name" if len(hits) == 1 else ("ambiguous" if hits else "missing")
    # evaluate
    out = []
    for r in refs:
        f = r["file"]
        exists = bool(f and Path(f).exists())
        tl = file_lines(f) if exists and Path(f).is_file() else None
        n = len(tl) if tl is not None else None
        line_ok = None
        if r["line"] is not None:
            line_ok = bool(n is not None and 1 <= r["line"] <= n)
        sym_res = []
        for k, s in enumerate(r.get("symbols", [])):
            ok_file, where = symbol_defined(tl, s) if tl is not None else (False, [])
            near = None
            if k == 0 and r["line"] is not None and tl is not None:
                near, _ = symbol_defined(tl, s, near=r["line"])
            sym_res.append({"symbol": s, "cited_line": r["line"] if k == 0 else None, "found": ok_file,
                            "at_cited_line": near, "defined_at": where})
        for s, ln_ in r.get("symbol_lines", []):
            ok_file, where = symbol_defined(tl, s) if tl is not None else (False, [])
            near, _ = symbol_defined(tl, s, near=ln_) if tl is not None else (False, [])
            sym_res.append({"symbol": s, "cited_line": ln_, "found": ok_file, "at_cited_line": near,
                            "defined_at": where})
        git_backed = None
        if f:
            try:
                parts = Path(f).resolve().relative_to(ROOT).parts
                git_backed = (ROOT / parts[0] / parts[1] / ".git").exists() if parts[0] == "repos" else None
            except ValueError:
                git_backed = None
        elif r["repo"]:
            git_backed = (ROOT / "repos" / r["repo"] / ".git").exists()
        out.append(dict(
            spec_line=r["spec_line"], section=r["section"], form=r["form"], raw=r["raw"],
            required_form=(r["form"] == "repos_path_line"), relative_prefix=r["relative_prefix"],
            resolved_file=rel(f) if f else None, exists=exists, resolve_status=r["resolve_status"],
            file_lines=n, cited_line=r["line"], line_in_range=line_ok,
            clone_has_git=git_backed, symbols=sym_res,
        ))
    return out, [rel(b) for b in bases]


def summarize_refs(refs):
    by_form, by_status = {}, {}
    for r in refs:
        by_form[r["form"]] = by_form.get(r["form"], 0) + 1
        by_status[r["resolve_status"]] = by_status.get(r["resolve_status"], 0) + 1
    n = len(refs)
    strict = [r for r in refs if r["required_form"]]
    sym = [s for r in refs for s in r["symbols"]]
    sym_line = [s for s in sym if s["cited_line"] is not None]
    res = dict(
        n_refs=n, by_form=by_form, by_resolve_status=by_status, n_required_form=len(strict),
        n_exists=sum(r["exists"] for r in refs),
        n_exists_as_written=sum(r["exists"] and r["resolve_status"] == "ok" for r in refs),
        missing=[r["raw"] for r in refs if not r["exists"] and r["resolve_status"] != "pattern"],
        n_pattern_unchecked=sum(r["resolve_status"] == "pattern" for r in refs),
        n_with_line=sum(r["cited_line"] is not None for r in refs),
        n_line_in_range=sum(bool(r["line_in_range"]) for r in refs),
        lines_out_of_range=[r["raw"] for r in refs if r["line_in_range"] is False],
        n_symbols_checked=len(sym), n_symbols_found_in_file=sum(s["found"] for s in sym),
        symbols_not_found=[f"{s['symbol']}" for s in sym if not s["found"]],
        n_symbol_line_checks=len(sym_line), n_symbol_at_cited_line=sum(bool(s["at_cited_line"]) for s in sym_line),
        symbol_line_mismatches=[f"{s['symbol']}@{s['cited_line']} (defined at {s['defined_at']})"
                                for s in sym_line if not s["at_cited_line"]],
        n_in_code_reference_section=sum(r["section"] == "Code references" for r in refs),
        n_resolved_in_clone_without_git=sum(bool(r["exists"] and r["clone_has_git"] is False) for r in refs),
    )
    ok = (n > 0 and len(strict) == n and not res["missing"] and not res["lines_out_of_range"]
          and res["n_exists_as_written"] == n and not res["symbol_line_mismatches"])
    res["S4"] = "PASS" if ok else "FAIL"
    return res


# ------------------------------------------------------------------ S5
def check_params(found):
    if 7 not in found:
        return dict(S5="FAIL", reason="no Parameters section", rows=[]), []
    rows, headers = parse_param_table(found[7]["body"])
    names = [r["name"] for r in rows]
    dup = sorted({n for n in names if names.count(n) > 1})
    snake = [n for n in names if not re.fullmatch(r"[a-z][a-z0-9_]*", n)]
    has_range_col = any(any(re.search(r"range|alternate|sensitiv|valid", h) for h in hdr) for hdr in headers)
    n_def = sum(r["default"] is not None for r in rows)
    n_rng = sum(r["range"] is not None for r in rows)
    s5 = "PASS" if rows and has_range_col and not dup else "FAIL"
    return dict(S5=s5, n_tables=len(headers), headers=headers, n_rows=len(rows),
                has_range_column=has_range_col, duplicate_names=dup, non_snake_case_names=snake,
                n_default_machine_readable=n_def, n_range_machine_readable=n_rng,
                n_nonnegotiable=sum(r["nonnegotiable"] for r in rows)), rows


# ------------------------------------------------------------------ S6
def split_items(body_lines):
    items, cur, in_code = [], None, False
    for ln in body_lines:
        if ln.strip().startswith("```"):
            in_code = not in_code
            if cur is not None:
                cur.append(ln)
            continue
        if in_code:
            if cur is not None:
                cur.append(ln)
            continue
        if ITEM_START.match(ln):
            if cur:
                items.append("\n".join(cur))
            cur = [ln]
        elif TABLE_ROW.match(ln.strip()) and not TABLE_SEP.match(ln.strip()):
            if cur:
                items.append("\n".join(cur))
                cur = None
            items.append(ln.strip())
        elif ln.startswith("#"):
            if cur:
                items.append("\n".join(cur))
                cur = None
        elif cur is not None:
            cur.append(ln)
    if cur:
        items.append("\n".join(cur))
    return [it for it in items if it.strip()]


def classify(item: str):
    item = re.sub(r"^\s*(\d+\.|[-*])\s+", "", item)
    s = LABEL_NUM.sub(" ", item)
    s = LABEL_ID.sub(" ", s)
    s = re.sub(r"\(Phase[^)]*\)", " ", s, flags=re.I)
    has_num = bool(NUMBER.search(s))
    if has_num and COMPARATOR.search(s):
        return "numeric-threshold"
    if has_num:
        return "numeric-mention"
    return "qualitative"


def check_validation(found):
    if 8 not in found:
        return dict(S6="FAIL", reason="no Validation section"), []
    body = found[8]["body"]
    items = []
    for it in split_items(body):
        first = it.strip().splitlines()[0]
        if TABLE_ROW.match(first) and not re.search(r"\d", first):
            continue                      # table header row
        items.append(dict(cls=classify(it), first_line=first[:200],
                          source_anchored=bool(SOURCE_ANCHOR.search(it)), if_broken_rule=bool(BUG_RULE.search(it))))
    text = "\n".join(body)
    mr_block = bool(re.search(r"```ya?ml[\s\S]*?^\s*validation\s*:", text, re.M))
    counts = {c: sum(i["cls"] == c for i in items) for c in ("numeric-threshold", "numeric-mention", "qualitative")}
    intro = " ".join(body[:15])
    return dict(
        S6="PASS" if mr_block else "FAIL",
        n_items=len(items), **counts,
        n_source_anchored=sum(i["source_anchored"] for i in items),
        n_if_broken_rule=sum(i["if_broken_rule"] for i in items),
        n_threshold_not_source_anchored=sum(i["cls"] == "numeric-threshold" and not i["source_anchored"] for i in items),
        machine_readable_block=mr_block,
        framed_as_source_replication=bool(re.search(r"must reproduce[^.]{0,250}", intro, re.I)),
    ), items


# ------------------------------------------------------------------ S7
BOLD_LABEL = re.compile(r"^\s*(?:[-*]\s+|\d+\.\s+)?\*\*([^*]{1,60})\*\*")


def check_slots(text, lines):
    """A 'labelled slot' = a heading line, or a line that starts with a short bold label,
    whose text contains the keyword. Plain mentions are counted separately."""
    res = {}
    for slot, pat in SLOTS.items():
        rx = re.compile(pat, re.I)
        labelled = []
        for k, ln in enumerate(lines):
            if ln.lstrip().startswith("#") and rx.search(ln):
                labelled.append(k + 1)
                continue
            m = BOLD_LABEL.match(ln)
            if m and rx.search(m.group(1)):
                labelled.append(k + 1)
        res[slot] = dict(mentions=len(rx.findall(text)), labelled_slot_lines=labelled[:10],
                         has_labelled_slot=bool(labelled))
    n_slots = sum(v["has_labelled_slot"] for v in res.values())
    res["S7"] = "PASS" if n_slots == len(SLOTS) else "FAIL"
    return res


# ------------------------------------------------------------------ main
def lint(relpath: str, deployed: bool, run: str | None):
    p = ROOT / relpath
    text = p.read_text(encoding="utf-8")
    lines, secs = split_sections(text)
    found, s12 = check_sections(lines, secs)
    pin = check_pin(found)
    pin_clones = []
    for h in pin.get("hashes", []):
        pin_clones += h["pinned_commit_txt_in"]
    pin_clones += [c for c in pin.get("clones_named_in_source", []) if c not in pin_clones]
    refs, bases = check_coderefs(text, lines, found, p, pin_clones)
    ref_sum = summarize_refs(refs)
    par, par_rows = check_params(found)
    val, val_items = check_validation(found)
    slots = check_slots(text, lines)
    verdicts = {"S1": s12["S1"], "S2": s12["S2"], "S3": pin["S3"], "S4": ref_sum["S4"], "S5": par["S5"],
                "S6": val["S6"], "S7": slots["S7"]}
    return dict(
        spec=relpath, run=run, deployed_on_maxtoki=deployed, chars=len(text),
        verdicts=verdicts,
        template_verdict=("ACCEPT" if s12["S1"] == "PASS" else "REJECT"),
        sections=s12, pin=pin, code_refs=ref_sum, code_ref_bases=bases, parameters=par, validation=val,
        slots=slots,
    ), refs, par_rows, val_items


def main():
    strict = "--strict" in sys.argv
    RESULTS.mkdir(parents=True, exist_ok=True)
    targets = [(s, True, r) for r, s in DEPLOYED] + [(s, False, None) for s in OTHER_SPECS]
    allres, ref_rows, par_rows, val_rows = [], [], [], []
    for spec, dep, run in targets:
        res, refs, prow, vitems = lint(spec, dep, run)
        allres.append(res)
        for r in refs:
            ref_rows.append([spec, r["spec_line"], r["section"], r["form"], r["required_form"], r["raw"],
                             r["resolved_file"], r["exists"], r["resolve_status"], r["cited_line"], r["file_lines"],
                             r["line_in_range"], r["clone_has_git"],
                             "; ".join(f"{s['symbol']}:{'found' if s['found'] else 'NOT FOUND'}"
                                       + ("" if s["cited_line"] is None else (" at line" if s["at_cited_line"] else " NOT at cited line"))
                                       for s in r["symbols"])])
        for r in prow:
            par_rows.append([spec, r["name"], r["qualifier"], r["default_raw"], describe(r["default"]),
                             r["range_raw"], describe(r["range"]), r["nonnegotiable"], r["used_in_raw"]])
        for it in vitems:
            val_rows.append([spec, it["cls"], it["source_anchored"], it["if_broken_rule"], it["first_line"]])
    write_json(RESULTS / "spec_lint.json", allres)
    with open(RESULTS / "spec_lint_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["spec", "deployed", "template_verdict", "S1_sections", "S2_nonempty", "S3_pin", "S4_coderefs",
                    "S5_params", "S6_validation_block", "S7_named_slots", "n_sections_present", "missing_sections",
                    "in_order", "heading_number_mismatches", "pin_status", "n_code_refs", "n_required_form",
                    "n_code_refs_exist", "n_exist_as_written", "n_code_refs_missing", "n_with_line", "n_line_in_range",
                    "n_symbols_checked", "n_symbols_found_in_file", "n_symbol_line_checks", "n_symbol_at_cited_line", "param_rows", "param_default_numeric", "param_range_numeric",
                    "val_items", "val_numeric_threshold", "val_numeric_mention", "val_qualitative",
                    "val_source_anchored", "val_if_broken_rule", "slots_labelled"])
        for r in allres:
            s, c, pa, v = r["sections"], r["code_refs"], r["parameters"], r["validation"]
            w.writerow([r["spec"], r["deployed_on_maxtoki"], r["template_verdict"], *[r["verdicts"][k] for k in
                        ("S1", "S2", "S3", "S4", "S5", "S6", "S7")], len(s["present"]), "; ".join(s["missing"]),
                        s["in_order"], len(s["heading_number_mismatches"]), r["pin"]["status"], c["n_refs"],
                        c["n_required_form"], c["n_exists"], c["n_exists_as_written"], len(c["missing"]), c["n_with_line"],
                        c["n_line_in_range"], c["n_symbols_checked"], c["n_symbols_found_in_file"],
                        c["n_symbol_line_checks"], c["n_symbol_at_cited_line"], pa.get("n_rows", 0),
                        pa.get("n_default_machine_readable", 0), pa.get("n_range_machine_readable", 0),
                        v.get("n_items", 0), v.get("numeric-threshold", 0), v.get("numeric-mention", 0),
                        v.get("qualitative", 0), v.get("n_source_anchored", 0), v.get("n_if_broken_rule", 0),
                        "; ".join(k for k in ("null_model", "trivial_baseline", "positive_control", "scope_statement")
                                  if r["slots"][k]["has_labelled_slot"])])
    with open(RESULTS / "spec_lint_coderefs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["spec", "spec_line", "section", "form", "required_form", "raw", "resolved_file", "exists",
                    "resolve_status", "cited_line", "file_lines", "line_in_range", "clone_has_git", "symbols"])
        w.writerows(ref_rows)
    with open(RESULTS / "spec_lint_params.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["spec", "name", "qualifier", "default_raw", "default_parsed", "range_raw", "range_parsed",
                    "nonnegotiable", "used_in"])
        w.writerows(par_rows)
    with open(RESULTS / "spec_lint_validation_items.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["spec", "class", "source_anchored", "if_broken_rule", "item_first_line"])
        w.writerows(val_rows)
    inputs = [ROOT / s for s, _, _ in targets] + sorted((ROOT / "repos").glob("*/PINNED_COMMIT.txt"))
    write_json(RESULTS / "spec_lint_run_config.json", run_config("spec_lint.py", inputs))
    for r in allres:
        print(f"{r['template_verdict']:6s} {r['spec']}")
        print("       " + " ".join(f"{k}={v}" for k, v in r["verdicts"].items()) + f"  pin={r['pin']['status']}")
        c = r["code_refs"]
        print(f"       refs={c['n_refs']} required_form={c['n_required_form']} exist={c['n_exists']} "
              f"as_written={c['n_exists_as_written']} missing={len(c['missing'])} with_line={c['n_with_line']} "
              f"line_ok={c['n_line_in_range']} symbols_in_file={c['n_symbols_found_in_file']}/{c['n_symbols_checked']} "
              f"symbol_at_line={c['n_symbol_at_cited_line']}/{c['n_symbol_line_checks']}")
    dep = [r for r in allres if r["deployed_on_maxtoki"]]
    print(f"\nDeployed specs rejected by the literal template rule (S1): "
          f"{sum(r['template_verdict'] == 'REJECT' for r in dep)}/{len(dep)}")
    if strict:
        # blocking rule when used as a pre-run gate: S1 (sections) or S3 (pin) FAIL, or a missing code path
        bad = [r["spec"] for r in allres if r["verdicts"]["S1"] == "FAIL" or r["verdicts"]["S3"] == "FAIL"
               or r["code_refs"]["missing"]]
        print(f"--strict: {len(bad)} spec(s) would be blocked: {bad}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
