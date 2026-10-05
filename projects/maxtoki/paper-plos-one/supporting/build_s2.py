#!/usr/bin/env python3
"""Build S2 Appendix (protocol of the controlled studies) as Markdown and PDF.

Reads the evaluation workspace (EV) without changing it, assembles
S2_Appendix.md next to this script, and runs pandoc + xelatex to make
S2_Appendix.pdf.

Run:  python3 build_s2.py

Rules the script follows:
- Text documents are copied word for word. Only heading levels change
  (headings inside copied documents are demoted and left unnumbered).
- Prompt templates are cut out of the workflow scripts by a small JS
  scanner, so they match the scripts exactly. Every `${...}` placeholder
  must have a note in PLACEHOLDER_NOTES, or the build stops.
- Key files are laid out field by field without changing their text.
- Checks that back statements in the appendix text (freeze-log hashes,
  brief interpreter paths, Study B item list, Study C key) are run here;
  the build stops if one fails.
"""

import csv
import difflib
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
EV = Path("<EVAL_ROOT>")
AT = Path("<CODE_ROOT>/analysis-tasks")  # deployed task folders (optional check)
PANDOC = "<CONDA_ROOT>/bin/pandoc"
OUT_MD = HERE / "S2_Appendix.md"
OUT_PDF = HERE / "S2_Appendix.pdf"
HAIKU_SECTION = HERE / "s2_haiku_section.md"

PROTO = EV / "protocol"
TOOLS = EV / "tools"
TASKS = ["T1", "T2", "T3", "T4"]

TITLE = "S2 Appendix. Protocol of the controlled studies"
RELEASE_FOLDER = "controlled-evaluation/"

EV_PY = "<EVAL_ROOT>/bin/python"
AT_PY = "<CODE_ROOT>/analysis-tasks/bin/python"

MAINFONT = "DejaVu Sans"       # has every glyph used (incl. arrows, subset sign, box drawing)
MONOFONT = "DejaVu Sans Mono"


# ---------------------------------------------------------------- small helpers
def read(p):
    return Path(p).read_text(encoding="utf-8")


def fail(msg):
    sys.exit("BUILD STOPPED: " + msg)


MD_SPECIAL = set("\\`*_{}[]<>#|~^$@!")


def esc(s):
    """Escape text so pandoc markdown shows it literally."""
    s = "".join("\\" + c if c in MD_SPECIAL else c for c in str(s))
    # things that would start a block at the beginning of a line
    s = re.sub(r"(?m)^(\s*)([-+])(\s)", r"\1\\\2\3", s)
    s = re.sub(r"(?m)^(\s*\d+)([.)])(\s)", r"\1\\\2\3", s)
    return s


def code(s):
    """Inline code span that is safe for any content."""
    s = str(s)
    n = max([len(r) for r in re.findall(r"`+", s)] + [0]) + 1
    pad = " " if s.startswith("`") or s.endswith("`") else ""
    return "`" * n + pad + s + pad + "`" * n


def block(text, cls="code"):
    """Fenced code block; the Lua filter turns it into a wrapping Verbatim."""
    if "\\end{Verbatim}" in text:
        fail("code block contains \\end{Verbatim}")
    n = max([len(r) for r in re.findall(r"`{3,}", text)] + [2]) + 1
    fence = "`" * n
    return f"{fence}{{.{cls}}}\n{text.rstrip(chr(10))}\n{fence}\n"


def raw_latex(tex):
    return "```{=latex}\n" + tex + "\n```\n"


LIST_MARK = re.compile(r"^(\s*)([-*+]|\d+\.)\s")
# A line inside a paragraph that pandoc's list parser would take as a list marker,
# but GitHub-flavoured Markdown would not, for example "(b) the generic checklist".
FALSE_MARK = re.compile(r"^\s*(\([A-Za-z0-9]{1,4}\)|[A-Za-z]{1,4}[.)]|\d+\))\s")

# Nested emphasis with the same delimiter: pandoc's reader closes the outer span early,
# GitHub-flavoured Markdown nests it. The inner span is rewritten with underscores
# (same text, same nesting). Each pattern must occur exactly once.
EMPH_FIXES = {
    "CHECKLIST_GENERIC.md": [("report properties of *those* — that's", "report properties of _those_ — that's")],
    "CHECKLIST_DEPLOYED.md": [("report properties of *those* — that's", "report properties of _those_ — that's")],
}


def fix_md(md):
    """Make pandoc markdown read a GitHub-style document the way GitHub does.

    - a list that follows a paragraph line gets a blank line before it;
    - a paragraph line that starts like "(b) " is joined to the line before
      (a line break inside a paragraph is shown as a space, so the text is unchanged).
    """
    out, fence, blk = [], None, []
    for line in md.split("\n"):
        if fence:
            if re.match(r"^" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$", line):
                fence = None
            out.append(line)
            continue
        m = re.match(r"^(`{3,}|~{3,})", line)
        if m:
            fence, blk = m.group(1), []
            out.append(line)
            continue
        if not line.strip():
            blk = []
            out.append(line)
            continue
        if blk and FALSE_MARK.match(line) and not out[-1].lstrip().startswith(("#", "|")):
            out[-1] = out[-1].rstrip() + " " + line.strip()
            blk.append(line)
            continue
        if LIST_MARK.match(line) and blk and not any(LIST_MARK.match(b) for b in blk):
            out.append("")
            blk = []
        out.append(line)
        blk.append(line)
    return "\n".join(out)


def _ast(md, fmt):
    r = subprocess.run([PANDOC, "-f", fmt, "-t", "json"], input=md, capture_output=True, text=True)
    if r.returncode:
        fail("pandoc could not parse a document: " + r.stderr)
    return json.loads(r.stdout)["blocks"]


def _norm(x):
    """Drop differences that do not change the text: soft breaks, tight/loose lists,
    heading ids, list number styles, table column widths."""
    if isinstance(x, dict):
        t, c = x.get("t"), x.get("c")
        if t == "SoftBreak":
            return {"t": "Space"}
        if t == "Plain":
            t = "Para"
        if t == "Header":
            return {"t": "Header", "c": [c[0], _norm(c[2])]}
        if t == "OrderedList":
            return {"t": "OL", "c": [c[0][0], _norm(c[1])]}
        if t == "Table":
            return {"t": "Table", "c": _norm(c[3:])}
        if t == "RawInline" and c[0] == "html":
            # the gfm reader keeps "<package>" as raw HTML even without raw_html; ours shows it as text
            return {"t": "Str", "c": c[1]}
        return {"t": t, "c": _norm(c)} if c is not None else {"t": t}
    if isinstance(x, list):
        out = []
        for y in map(_norm, x):
            if out and isinstance(y, dict) and y.get("t") == "Str" and isinstance(out[-1], dict) and out[-1].get("t") == "Str":
                out[-1] = {"t": "Str", "c": out[-1]["c"] + y["c"]}
            else:
                out.append(y)
        return out
    return x


def copied(text, name):
    """Prepare a copied document and check it parses as GitHub-flavoured Markdown does."""
    fixed = text
    for old, new in EMPH_FIXES.get(name, []):
        if fixed.count(old) != 1:
            fail(f"{name}: emphasis fix pattern not found exactly once")
        fixed = fixed.replace(old, new)
    fixed = fix_md(fixed)
    if _norm(_ast(fixed, READER)) != _norm(_ast(text, "gfm-raw_html")):
        fail(f"{name}: pandoc reads it differently from GitHub-flavoured Markdown")
    print(f"[check] {name}: same parse as GitHub-flavoured Markdown")
    return fixed


def demote(md, by, unnumbered=True):
    """Shift ATX headings down by `by` levels (outside code fences)."""
    out, fence = [], None
    for line in md.split("\n"):
        if fence:
            if re.match(r"^" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$", line):
                fence = None
            out.append(line)
            continue
        m = re.match(r"^(`{3,}|~{3,})", line)
        if m:
            fence = m.group(1)
            out.append(line)
            continue
        h = re.match(r"^(#{1,6})\s+(.*?)\s*$", line)
        if h:
            level = len(h.group(1)) + by
            if level > 6:
                fail("heading too deep after demotion: " + line)
            out.append("#" * level + " " + h.group(2) + (" {-}" if unnumbered else ""))
        else:
            out.append(line)
    return "\n".join(out)


def md_table(header, rows, widths):
    """Pipe table. A long separator line makes pandoc use these relative widths."""
    total = 120  # > pandoc --columns, so widths come from the dash counts
    dashes = [max(3, round(total * w / sum(widths))) for w in widths]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("-" * d for d in dashes) + "|"]
    for r in rows:
        cells = [str(c).replace("\n", " ") for c in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def small(md, size="\\footnotesize"):
    return raw_latex("\\begingroup" + size) + "\n" + md + "\n" + raw_latex("\\endgroup")


# ---------------------------------------------------------------- JSON with raw numbers
class RawNum(str):
    """A number kept exactly as written in the JSON file."""


def load_json_raw(p):
    return json.loads(read(p), parse_float=RawNum, parse_int=RawNum)


def jtext(v, top=True):
    if isinstance(v, RawNum):
        return str(v)
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return v if top else json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(jtext(x, False) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(json.dumps(k, ensure_ascii=False) + ": " + jtext(x, False) for k, x in v.items()) + "}"
    return str(v)


# ---------------------------------------------------------------- JS scanning
def scan_string(src, i):
    q, j = src[i], i + 1
    while src[j] != q:
        j += 2 if src[j] == "\\" else 1
    return j + 1


def scan_expr(src, j):
    """j is just after an opening '{'; return index just after the matching '}'."""
    depth = 1
    while depth:
        c = src[j]
        if c in "'\"":
            j = scan_string(src, j)
            continue
        if c == "`":
            j, _ = scan_template(src, j)
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        j += 1
    return j


def scan_template(src, i):
    """src[i] is a backtick; return (index after closing backtick, raw content)."""
    assert src[i] == "`"
    j = i + 1
    while True:
        c = src[j]
        if c == "\\":
            j += 2
            continue
        if c == "`":
            return j + 1, src[i + 1:j]
        if c == "$" and src[j + 1] == "{":
            j = scan_expr(src, j + 2)
            continue
        j += 1


def placeholders(raw):
    """All ${...} expressions in a raw template, in order."""
    out, j = [], 0
    while True:
        k = raw.find("${", j)
        if k < 0:
            return out
        end = scan_expr(raw, k + 2)
        out.append(raw[k:end])
        j = end


def show_template(raw):
    """Template as the agent saw it, placeholders kept: only \\n becomes a line break."""
    if re.search(r"\\(?!n)", raw):
        fail("unexpected escape in template: " + raw[:80])
    return raw.replace("\\n", "\n")


def fn_template(src, name):
    m = re.search(r"function " + re.escape(name) + r"\([^)]*\)\s*\{", src)
    if not m:
        fail("function not found: " + name)
    k = src.index("return `", m.end())
    return scan_template(src, k + len("return "))[1]


def const_value_start(src, name):
    m = re.search(r"^const " + re.escape(name) + r"\s*=\s*", src, re.M)
    if not m:
        fail("const not found: " + name)
    return m.start(), m.end()


def const_template(src, name):
    _, v = const_value_start(src, name)
    if src[v] == "`":
        return scan_template(src, v)[1]
    if src[v] == "'":
        return src[v + 1:scan_string(src, v) - 1]
    fail("const is not a string: " + name)


def const_source(src, name):
    """Exact source text of `const NAME = ...` (object, template or one-line value)."""
    s, v = const_value_start(src, name)
    if src[v] == "{":
        e = scan_expr(src, v + 1)
    elif src[v] == "`":
        e = scan_template(src, v)[0]
    else:
        e = src.index("\n", v)
    return src[s:e].rstrip()


def obj_templates(src, name):
    """Property name -> raw string/template for an object const of strings."""
    _, v = const_value_start(src, name)
    end = scan_expr(src, v + 1)
    body = src[v + 1:end - 1]
    out, j = {}, 0
    for m in re.finditer(r"(?m)^\s*(\w+):\s*", body):
        if m.start() < j:
            continue
        p = m.end()
        if body[p] == "`":
            j, raw = scan_template(body, p)
        elif body[p] == "'":
            j = scan_string(body, p)
            raw = body[p + 1:j - 1]
        else:
            continue
        out[m.group(1)] = raw
    return out


def schema(src, name, env=None):
    text = const_source(src, name).split("=", 1)[1].strip()
    for k, v in (env or {}).items():
        text = re.sub(r"\b" + k + r"\b", v, text)
    text = re.sub(r"'([^'\\]*)'", lambda m: json.dumps(m.group(1)), text)
    text = re.sub(r"([{,]\s*)([A-Za-z_]\w*)\s*:", r'\1"\2":', text)
    return json.loads(text)


def fieldlist(s):
    parts = []
    for name, sub in s["properties"].items():
        t = sub.get("type")
        if t == "object" and "properties" in sub:
            parts.append(f"{name} {{{fieldlist(sub)}}}")
        elif t == "array" and isinstance(sub.get("items"), dict) and "properties" in sub["items"]:
            parts.append(f"{name}[] {{{fieldlist(sub['items'])}}}")
        elif t == "array":
            parts.append(f"{name}[]")
        else:
            parts.append(name)
    return ", ".join(parts)


def js_lines(src, start_pat, end_pat=None):
    """Exact source lines from the first line matching start_pat (to end_pat, inclusive)."""
    lines = src.split("\n")
    for i, l in enumerate(lines):
        if re.match(start_pat, l):
            if not end_pat:
                return l
            for k in range(i, len(lines)):
                if re.match(end_pat, lines[k]):
                    return "\n".join(lines[i:k + 1])
    fail("lines not found: " + start_pat)


# ---------------------------------------------------------------- placeholder notes (plain words)
NOTE_PKG_A = "the task folder for this run (an opaque folder name under analysis-tasks/; Amendment 1, item 3)."
COMMON = {
    "${PY}": "the Python interpreter path (constant PY above).",
    "${BUDGET}": "the budget sentence (constant BUDGET above).",
    "${ARM[arm]}": "the arm sentence: ARM.generic or ARM.checklist (shown above).",
    "${arm === 'checklist' ? ' and the checklist file' : ''}": "adds the words \" and the checklist file\" in the checklist arm only.",
    "${g}": "the grader number: 1 or 2, or 3 for the extra grader.",
    "${CHECKLIST}": "the path of the generic checklist (constant CHECKLIST above).",
}
PLACEHOLDER_NOTES = {
    "A.exec": {"${pkg}": NOTE_PKG_A},
    "A.review": {"${pkg}": NOTE_PKG_A,
                 "${d.report}": "the executor's Markdown report.",
                 "${JSON.stringify(d.results)}": "the executor's results object (verdict, estimates, analysis script), as JSON."},
    "A.repair": {"${pkg}": NOTE_PKG_A,
                 "${d.report}": "the executor's Markdown report.",
                 "${JSON.stringify(d.results)}": "the executor's results object, as JSON.",
                 "${JSON.stringify(rv.points)}": "the points returned by the review, as JSON."},
    "A.grade": {"${pkg}": NOTE_PKG_A,
                "${key}": "the path of the frozen answer key (studyA/keys/T<n>/key.json in the evaluation workspace).",
                "${red.report}": "the report of the deliverable being graded, after redact(). The deliverable is the executor's output (review arm none) or the repaired output (generic or checklist arm).",
                "${JSON.stringify(red.results)}": "the results object of the same deliverable, after redact(), as JSON."},
    "A.rgrade": {"${pkg}": NOTE_PKG_A,
                 "${key}": "the path of the frozen answer key.",
                 "${d.report}": "the executor's Markdown report (the one that was reviewed; not redacted).",
                 "${JSON.stringify(d.results)}": "the executor's results object, as JSON (not redacted).",
                 "${JSON.stringify(rv.points)}": "the review's points, as JSON."},
    "A.run": {"${x.rep}": "the replicate number (1 to 5)."},
    "B.review": {"${N}": "the folder that holds the item packages (constant N above).",
                 "${id}": "the item's opaque id, for example B1144."},
    "B.grade": {"${N}": "the folder that holds the item packages (constant N above).",
                "${id}": "the item's opaque id.",
                "${WS}": "the evaluation workspace folder (constant WS above).",
                "${JSON.stringify(rv.findings)}": "the review's findings, as JSON."},
    "B.verify": {"${N}": "the folder that holds the item packages (constant N above).",
                 "${x.id}": "the item's opaque id.",
                 "${WS}": "the evaluation workspace folder (constant WS above).",
                 "${JSON.stringify(disputed)}": "the findings that at least one grader marked 'disputed', as JSON."},
    "B.run": {"${x.rep}": "the replicate number (1 to 3)."},
    "C.context": {"${SNAP}": "the snapshot folder of the project (constant SNAP above).",
                  "${PY}": "the Python interpreter path (constant PY above)."},
    "C.arm": {"${WS}": "the evaluation workspace folder (constant WS above).",
              "${SNAP}": "the snapshot folder of the project (constant SNAP above)."},
    "C.output": {},
    "C.budget": {},
    "C.run": {"${x.r}": "the replicate number (1 to 3)."},
    "C.grade": {"${KEY}": "the path of the frozen key, STUDYC_KEY.csv (constant KEY above).",
                "${'68'}": "the literal text 68, the number of rows in the key.",
                "${SNAP}": "the snapshot folder of the project.",
                "${JSON.stringify(review.findings)}": "the audit's findings, as JSON."},
    "C.adjudicate": {"${KEY}": "the path of the frozen key, STUDYC_KEY.csv.",
                     "${SNAP}": "the snapshot folder of the project.",
                     "${JSON.stringify(review.findings)}": "the audit's findings, as JSON.",
                     "${JSON.stringify(a)}": "the full grading returned by grader 1, as JSON.",
                     "${JSON.stringify(b)}": "the full grading returned by grader 2, as JSON."},
    "C.verify": {"${SNAP}": "the snapshot folder of the project.",
                 "${JSON.stringify(unmatched)}": "the audit findings that match no key row in the adjudicated grading, as JSON."},
}


def template_section(title, raw, notes_key, intro=None):
    """Heading, optional intro line, template block, and one note per placeholder."""
    notes = dict(COMMON)
    notes.update(PLACEHOLDER_NOTES[notes_key])
    ph = []
    for p in placeholders(raw):
        if p not in ph:
            ph.append(p)
    missing = [p for p in ph if p not in notes]
    if missing:
        fail(f"no note for placeholders {missing} in {notes_key}")
    out = [f"### {title}\n"]
    if intro:
        out.append(intro + "\n")
    out.append(block(show_template(raw), "prompt"))
    if ph:
        out.append("Placeholders:\n")
        for p in ph:
            out.append(f"- {code(p)}: {esc(notes[p])}")
        out.append("")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- section 1
def sec_contents():
    return f"""# Contents of this appendix

This appendix gives the protocol of the three controlled studies of LLM agents (Studies A, B and C). Section 2 is the pre-registration, and Section 3 gives its three amendments. Section 4 is the freeze log: the time and the sha256 hash of each file that was frozen before use. Section 5 is the generic ten-pattern checklist, and Section 6 is the checklist as it was deployed. Section 7 gives the prompt templates sent to the agents and the fields of their structured output. Section 8 gives the Study A task briefs, and Section 9 the Study A answer keys. Section 10 lists the Study B items and their keys. Section 11 lists the 68 known errors that form the Study C reference set. Section 12 covers the exploratory arm with Claude Haiku 4.5. Sections 2 to 6 and the briefs in Section 8 are copied word for word; only their heading levels were changed. Sections 9 to 11 lay out the key files without changing their text. All files are also in the code and data release, in the folder {code(RELEASE_FOLDER)}. Absolute local paths are shown with the placeholders used in the release: {code("<EVAL_ROOT>")} is the evaluation workspace and {code("<CODE_ROOT>")} the folder that held it ({code("PATH_MAP.md")} in the release lists all placeholders). The freeze-log hashes are those of the original files.
"""


# ---------------------------------------------------------------- sections 2-6
def sec_prereg():
    return ("# Pre-registration\n\n"
            "The pre-registration is copied word for word below. Its sha256 hash is in the freeze log (Section 4).\n\n"
            + demote(copied(read(PROTO / "PREREGISTRATION.md"), "PREREGISTRATION.md"), 1) + "\n")


def sec_amendments():
    parts = ["# Amendments\n",
             "The pre-registration has three amendments. They are given in order, word for word. "
             "Amendments 2 and 3 concern the exploratory arm with Claude Haiku 4.5 (Section 12).\n"]
    for i in (1, 2, 3):
        parts.append(demote(copied(read(PROTO / f"AMENDMENT_{i}.md"), f"AMENDMENT_{i}.md"), 1) + "\n")
    note = PROTO / "AMENDMENT_3_NOTE.md"
    if note.exists():
        parts.append("A dated note, written after the re-run and also hashed in the freeze log, corrects one count "
                     "in Amendment 3. It is copied word for word below.\n")
        parts.append(demote(copied(read(note), "AMENDMENT_3_NOTE.md"), 1) + "\n")
    return "\n".join(parts)


def check_freeze_log():
    """Recompute every hash in the freeze log. Returns a sentence for the appendix."""
    entries = []
    for line in read(PROTO / "FREEZE_LOG.md").split("\n"):
        m = re.match(r"- (\S+) `([0-9a-f]{64})` (\S+)(.*)", line.strip())
        if m:
            entries.append(m.groups())
    by_file = {}
    for t, h, p, rest in entries:
        by_file.setdefault(p, []).append((t, h, rest.strip()))
    later_only = []
    for p, lst in by_file.items():
        cur = hashlib.sha256((PROTO / p).read_bytes()).hexdigest()
        if cur != lst[-1][1]:
            fail(f"freeze log: current {p} does not match its last logged hash")
        if len(lst) > 1:
            if len(lst) != 2 or "batch 2" not in lst[-1][2]:
                fail(f"freeze log: unexpected repeated entries for {p}")
            later_only.append(Path(p).name)
    n = len(by_file)
    s = (f"When this appendix was built, the build script recomputed the sha256 hash of each of the {n} files in the log. "
         f"All {n} match their last entry in the log.")
    if later_only:
        s += (" " + ", ".join(later_only) + (" has" if len(later_only) == 1 else " have")
              + " two entries; the file on disk matches the second (batch 2) entry.")
    print(f"[check] freeze log: {n} files, all match their last entry; two entries for: {later_only}")
    return s


def sec_freeze():
    note = check_freeze_log()
    return ("# Freeze log\n\n"
            "The freeze log lists the UTC time and the sha256 hash of each frozen file. It is copied word for word below.\n\n"
            + demote(copied(read(PROTO / "FREEZE_LOG.md"), "FREEZE_LOG.md"), 1) + "\n\n" + note + "\n")


def sec_generic_checklist():
    return ("# The generic checklist\n\n"
            "This is the generic ten-pattern checklist. It was the checklist used by the checklist review arm in Studies A and B "
            "and by the checklist arm of Study C. It is copied word for word below.\n\n"
            + demote(copied(read(PROTO / "CHECKLIST_GENERIC.md"), "CHECKLIST_GENERIC.md"), 1) + "\n")


def sec_deployed_checklist():
    return ("# The checklist as deployed\n\n"
            "This is the audit specification as it was used in the MaxToki deployment. In Study C it was given to the "
            "deployed-checklist arm (named `deployed` in the workflow script). It is copied word for word below, "
            "including its first section, which says where its ten patterns came from.\n\n"
            + demote(copied(read(PROTO / "CHECKLIST_DEPLOYED.md"), "CHECKLIST_DEPLOYED.md"), 1) + "\n")


# ---------------------------------------------------------------- section 7
def schema_block(rows):
    lines = [f"{name} ({use}):\n    {fieldlist(s)}" for name, use, s in rows]
    return block("\n".join(lines), "prompt")


def changed_phrases(a, b, gap=16, ctx=6):
    """Every place where two texts differ, as (old phrase, new phrase).

    Word-level comparison; changes less than about eight words apart are joined into
    one phrase. A pure insertion or deletion gets about three words of context on
    each side so the reader can find it. Together the phrases cover every difference.
    """
    ta = re.findall(r"\S+|\s+", a)
    tb = re.findall(r"\S+|\s+", b)
    regs = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if regs and i1 - regs[-1][1] <= gap and j1 - regs[-1][3] <= gap:
            regs[-1][1], regs[-1][3] = i2, j2
        else:
            regs.append([i1, i2, j1, j2])
    out = []
    for i1, i2, j1, j2 in regs:
        if i1 == i2 or j1 == j2:
            k1, k2 = min(ctx, i1, j1), min(ctx, len(ta) - i2, len(tb) - j2)
            i1, j1, i2, j2 = i1 - k1, j1 - k1, i2 + k2, j2 + k2
        out.append(("".join(ta[i1:i2]).strip(), "".join(tb[j1:j2]).strip()))
    return out


def sec_prompts():
    A = read(TOOLS / "studyA_workflow.js")
    B = read(TOOLS / "studyB_workflow.js")
    C = read(TOOLS / "studyC_workflow.js")
    H = read(TOOLS / "studyA_haiku_isolated_workflow.js")

    # every agent call in A, B and C is read-only Opus
    for name, src in (("A", A), ("B", B), ("C", C)):
        calls = re.findall(r"agent\(", src)
        explore = re.findall(r"agentType: 'Explore', model: 'opus'", src)
        if len(calls) != len(explore):
            fail(f"study {name}: not every agent call is Explore + opus")

    out = ["# Agent prompts\n",
           "All prompts are template strings in three workflow scripts in the folder `tools/`: "
           "`studyA_workflow.js`, `studyB_workflow.js` and `studyC_workflow.js`. "
           "Each template is shown as it is written in the script. Placeholders of the form `${...}` are filled in by the "
           "script at run time; a note under each template says what each one holds. "
           "Line breaks written as `\\n` in the scripts are shown as real line breaks. "
           "In all three scripts, every agent call uses the read-only agent type (`agentType: 'Explore'`) "
           "with `model: 'opus'` (Claude Opus 5.5). "
           "The JSON output schemas are given by their field names only: `[]` marks a list, and braces give the fields of each object.\n"]

    # ---- Study A
    out.append("## Study A (`tools/studyA_workflow.js`)\n")
    out.append("### Constants\n")
    consts = "\n".join([js_lines(A, r"^const WS ="), js_lines(A, r"^const PY ="), js_lines(A, r"^const CHECKLIST ="),
                        "", js_lines(A, r"^const BUDGET ="), js_lines(A, r"^const ARM = \{", r"^\}")])
    out.append("These lines are copied from the script. The review prompt uses `BUDGET` and one of the two `ARM` sentences.\n")
    out.append(block(consts, "prompt"))
    out.append(template_section("Executor prompt", fn_template(A, "execPrompt"), "A.exec",
                                "The script adds a last line, `(Run ${x.rep}.)`, where `${x.rep}` is the replicate number (1 to 5)."))
    out.append(template_section("Review prompt (generic and checklist arms)", fn_template(A, "reviewPrompt"), "A.review",
                                "The two review arms differ only in `${ARM[arm]}` and in the words added in the checklist arm."))
    out.append(template_section("Repair prompt", fn_template(A, "repairPrompt"), "A.repair",
                                "The same repair prompt follows both review arms."))
    out.append("### Redaction before grading\n")
    out.append("Before grading, `redact()` replaces the words below with `[REDACTED]` in the deliverable's report and results. "
               "The first pattern ignores case. `SPEC.md` and the labels P1 to P10 are matched with their case as written. "
               "The function is copied from the script.\n")
    out.append(block(js_lines(A, r"^function redact\(s\) \{", r"^\}"), "prompt"))
    out.append(template_section("Grading prompt", fn_template(A, "gradePrompt"), "A.grade",
                                "Two graders score each final deliverable. A third grader is run when the two disagree on whether the "
                                "verdict is correct, or when one of them returns nothing."))
    out.append(template_section("Review-point grading prompt", fn_template(A, "rgradePrompt"), "A.rgrade",
                                "Each review (generic and checklist) is judged point by point by one grader."))
    est = const_source(A, "EST").split("=", 1)[1].strip()
    out.append("### Output schemas\n")
    out.append(schema_block([
        ("EXEC_SCHEMA", "executor and repair", schema(A, "EXEC_SCHEMA", {"EST": est})),
        ("REVIEW_SCHEMA", "review", schema(A, "REVIEW_SCHEMA")),
        ("GRADE_SCHEMA", "grading", schema(A, "GRADE_SCHEMA")),
        ("RGRADE_SCHEMA", "review-point grading", schema(A, "RGRADE_SCHEMA")),
    ]))

    # ---- Study B
    out.append("## Study B (`tools/studyB_workflow.js`)\n")
    out.append("### Constants\n")
    consts = "\n".join([js_lines(B, r"^const WS ="), js_lines(B, r"^const N ="), js_lines(B, r"^const PY ="),
                        js_lines(B, r"^const CHECKLIST ="), js_lines(B, r"^const BUDGET ="), js_lines(B, r"^const ARM = \{", r"^\}")])
    out.append("These lines are copied from the script.\n")
    out.append(block(consts, "prompt"))
    out.append(template_section("Review prompt (generic and checklist arms)", fn_template(B, "reviewPrompt"), "B.review",
                                "The script adds a last line, `(Run ${x.rep}.)`, where `${x.rep}` is the replicate number (1 to 3)."))
    out.append(template_section("Grading prompt", fn_template(B, "gradePrompt"), "B.grade",
                                "Two graders score each review. A third grader is run when the two differ on `error_detected` or on the "
                                "number of load-bearing false alarms, or when one of them returns nothing."))
    k = B.index("agent(`Check the findings below")
    ver_raw = scan_template(B, k + len("agent("))[1]
    out.append(template_section("Verifier prompt", ver_raw, "B.verify",
                                "This prompt is written inline in the script. It is run once for a review when any grader marks one "
                                "or more of its findings 'disputed'."))
    out.append("### Output schemas\n")
    out.append(schema_block([
        ("REVIEW_SCHEMA", "review", schema(B, "REVIEW_SCHEMA")),
        ("GRADE_SCHEMA", "grading", schema(B, "GRADE_SCHEMA")),
        ("VER_SCHEMA", "verifier", schema(B, "VER_SCHEMA")),
    ]))

    # ---- Study C
    out.append("## Study C (`tools/studyC_workflow.js`)\n")
    out.append("### Constants\n")
    consts = "\n".join([js_lines(C, r"^const WS ="), js_lines(C, r"^const SNAP ="), js_lines(C, r"^const PY ="),
                        js_lines(C, r"^const KEY =")])
    out.append("These lines are copied from the script.\n")
    out.append(block(consts, "prompt"))
    assembly = re.search(r"agent\((CONTEXT \+ .*?\(Run \$\{x\.r\}\.\)`)", C).group(1)
    out.append("### Audit prompt\n")
    out.append("The audit prompt is put together from four parts, in this order (copied from the script):\n")
    out.append(block(assembly, "prompt"))
    out.append("`'\\n\\n'` is a blank line. `ARMS[x.arm]` is one of the three arm texts below. "
               "`(Run ${x.r}.)` is a last line with the replicate number (1 to 3).\n")
    out.append(template_section("CONTEXT", const_template(C, "CONTEXT"), "C.context"))
    arms = obj_templates(C, "ARMS")
    if list(arms) != ["deployed", "checklist", "generic"]:
        fail("unexpected ARMS keys in Study C: " + str(list(arms)))
    labels = {"deployed": "ARMS.deployed (deployed-checklist arm)", "checklist": "ARMS.checklist (checklist arm)",
              "generic": "ARMS.generic (generic review arm)"}
    for a in ("deployed", "checklist", "generic"):
        out.append(template_section(labels[a], arms[a], "C.arm"))
    out.append(template_section("OUTPUT", const_template(C, "OUTPUT"), "C.output"))
    out.append(template_section("BUDGET", const_template(C, "BUDGET"), "C.budget"))
    out.append(template_section("Grading prompt", fn_template(C, "gradePrompt"), "C.grade",
                                "Two graders score each audit against the key."))
    out.append(template_section("Adjudication prompt", fn_template(C, "adjudicatePrompt"), "C.adjudicate",
                                "An adjudicator is always run after the two graders. Its grading is the final one."))
    out.append(template_section("Verifier prompt", fn_template(C, "verifyPrompt"), "C.verify",
                                "The verifier is run once per audit, on the findings that match no key row in the final grading."))
    out.append("### Output schemas\n")
    out.append(schema_block([
        ("REVIEW_SCHEMA", "audit", schema(C, "REVIEW_SCHEMA")),
        ("GRADE_SCHEMA", "grading and adjudication", schema(C, "GRADE_SCHEMA")),
        ("VERIFY_SCHEMA", "verifier", schema(C, "VERIFY_SCHEMA")),
    ]))

    # ---- Haiku re-run script
    out.append("## Exploratory arm (`tools/studyA_haiku_isolated_workflow.js`)\n")
    if "const SUBJ = args.subject_model || 'haiku'" not in H:
        fail("SUBJ line not found in Haiku script")
    subj_uses = re.findall(r"model: SUBJ, schema: (\w+), label: `([^`]*)`", H)
    opus_uses = re.findall(r"model: 'opus', schema: (\w+), label: `([^`]*)`", H)
    if sorted(s for s, _ in subj_uses) != ["EXEC_SCHEMA", "EXEC_SCHEMA", "REVIEW_SCHEMA"] or \
       sorted(s for s, _ in opus_uses) != ["GRADE_SCHEMA", "GRADE_SCHEMA", "RGRADE_SCHEMA", "RGRADE_SCHEMA"]:
        fail("unexpected model use in Haiku script")
    out.append("The exploratory re-run with Claude Haiku 4.5 (Amendment 3) used this script. It is a copy of "
               "`studyA_workflow.js` with two kinds of change: the model of the executor, review and repair agents, and the "
               "folder instructions. The executor, review and repair agents use `model: SUBJ`, which is Haiku unless the "
               "run passes another model. The graders and review-point graders still use `model: 'opus'`. Every agent gets "
               "its own folder, named by `F(label)`, and its agent label starts with `AI:` in place of `A:`. "
               "These lines are copied from the script:\n")
    hl = "\n".join([js_lines(H, r"^const SUBJ ="), js_lines(H, r"^const AT ="), js_lines(H, r"^const F ="),
                    js_lines(H, r"^const ISO =")])
    out.append(block(hl, "prompt"))
    out.append("The `ISO` sentence as written. `${f}` is the agent's own folder. `${extra || ''}` is empty, or the "
               "extra words given in the call (listed below).\n")
    iso_raw = re.search(r"^const ISO = \(f, extra\) => `(.*)`$", H, re.M).group(1)
    out.append(block(show_template(iso_raw), "prompt"))
    uses = []
    for fn, label in (("execPrompt", "executor prompt"), ("reviewPrompt", "review prompt"), ("repairPrompt", "repair prompt"),
                      ("gradePrompt", "grading prompt"), ("rgradePrompt", "review-point grading prompt")):
        raw = fn_template(H, fn)
        calls = [p for p in placeholders(raw) if p.startswith("${ISO(")]
        if len(calls) != 1:
            fail(f"expected one ISO call in Haiku {fn}")
        uses.append((fn, label, calls[0]))
    out.append("Five prompts use `ISO`. In each, `pkg` is the agent's own folder. The call in each prompt is:\n")
    for fn, label, call in uses:
        out.append(f"- {esc(label)} ({code(fn)}): {code(call)}")
    out.append("")
    out.append("Below is every place where the wording of these five prompts differs between the two scripts. "
               "A line starting with `-` is the wording in `studyA_workflow.js`. The line starting with `+` under it is "
               "the wording in the re-run script. All other prompt text is the same.\n")
    for fn, label, _ in uses:
        out.append(f"**{esc(label[0].upper() + label[1:])}** ({code(fn)}):\n")
        ph = changed_phrases(show_template(fn_template(A, fn)), show_template(fn_template(H, fn)))
        out.append(block("\n".join(f"- {a}\n+ {b}" for a, b in ph), "prompt"))
    # the constants, schemas and redact() must be identical
    for name in ("BUDGET", "ARM", "EST", "EXEC_SCHEMA", "REVIEW_SCHEMA", "GRADE_SCHEMA", "RGRADE_SCHEMA"):
        if const_source(A, name) != const_source(H, name):
            fail(f"Haiku script differs from Study A in {name}")
    if js_lines(A, r"^function redact\(s\) \{", r"^\}") != js_lines(H, r"^function redact\(s\) \{", r"^\}"):
        fail("Haiku script differs from Study A in redact()")
    out.append("The `BUDGET` and `ARM` constants, the output schemas and `redact()` are the same in both scripts.\n")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- section 8
def check_brief_paths():
    """Back the interpreter-path note with checks."""
    for t in TASKS:
        for arm in ("paper", "contract"):
            s = read(EV / "studyA" / "tasks" / f"{t}-{arm}" / "BRIEF.md")
            has_ev, has_at = EV_PY in s, AT_PY in s
            if t == "T2" and not (has_at and not has_ev):
                fail(f"{t}-{arm} brief: expected the analysis-tasks interpreter path")
            if t != "T2" and not (has_ev and not has_at):
                fail(f"{t}-{arm} brief: expected the evaluation-workspace interpreter path")
    pmap = json.loads(read(PROTO / "STUDYA_PACKAGE_MAP.json"))
    if not AT.exists():
        print("[check] deployed task folders not found; skipped the deployed-brief comparison")
        return
    for oid, info in pmap.items():
        dep = AT / oid / "BRIEF.md"
        if not dep.exists():
            print(f"[check] {dep} missing; skipped")
            continue
        src = read(Path(info["source"]) / "BRIEF.md")
        if src.replace(EV_PY, AT_PY) != read(dep):
            fail(f"deployed brief {oid} differs from its source in more than the interpreter path")
    print("[check] deployed briefs: each equals its source after the interpreter-path change")


def sec_briefs():
    check_brief_paths()
    out = ["# Study A task briefs\n",
           "Each Study A task had two briefs, one per executor arm. For each task, the paper-arm brief is given in full. "
           "For the contract arm, only the lines that differ are shown, as a unified diff: a line starting with `+` was added. "
           "The briefs are the frozen copies in `studyA/tasks/`. In the task folders the agents used, the interpreter path in the "
           f"T1, T3 and T4 briefs was {code(AT_PY)} in place of {code(EV_PY)}. The T2 briefs already had that path. "
           "This is the only other difference. The specifications (`contract/SPEC.md`), the source-method papers and the data are "
           "not reproduced here.\n"]
    for t in TASKS:
        p = EV / "studyA" / "tasks" / f"{t}-paper" / "BRIEF.md"
        c = EV / "studyA" / "tasks" / f"{t}-contract" / "BRIEF.md"
        out.append(f"## Task {t}\n")
        out.append(f"The paper-arm brief, copied from {code(f'studyA/tasks/{t}-paper/BRIEF.md')}:\n")
        out.append(demote(copied(read(p), f"{t}-paper/BRIEF.md"), 2) + "\n")
        out.append(f"### Contract-arm brief: lines that differ {{-}}\n")
        diff = "\n".join(difflib.unified_diff(read(p).split("\n"), read(c).split("\n"),
                                              fromfile=f"{t}-paper/BRIEF.md", tofile=f"{t}-contract/BRIEF.md",
                                              lineterm="", n=0))
        out.append(block(diff, "code"))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- section 9
def sec_keys():
    out = ["# Study A answer keys\n",
           "The four answer keys (`studyA/keys/T1` to `T4/key.json`) are laid out below without changing their text. "
           "For each key number, the key file also gives a written definition; the definitions are not reproduced here. "
           "The fields that point to how each key was made and checked (`reference`, `verification`, `source`, `sources`) and the list of "
           "verdict options (given in the briefs) are left out. The T2 key has one more field, `grader_guidance`, which is shown.\n"]
    for t in TASKS:
        k = load_json_raw(EV / "studyA" / "keys" / t / "key.json")
        if k["task"] != t:
            fail(f"key {t} has task {k['task']}")
        out.append(f"## Task {t}\n")
        out.append(f"**Question.** {esc(k['question'])}\n")
        out.append(f"**Key verdict.** {esc(k['key_verdict'])}\n")
        out.append(f"**Key conclusion.** {esc(k['key_conclusion'])}\n")
        out.append("**Acceptable alternatives.**\n")
        alts = k.get("acceptable_alternatives") or []
        if alts:
            for a in alts:
                if set(a) != {"verdict", "counts_as_correct_only_if", "reason"}:
                    fail(f"unexpected alternative fields in {t}")
                out.append(f"- *Verdict.* {esc(a['verdict'])}\n\n  *Counts as correct only if.* {esc(a['counts_as_correct_only_if'])}\n\n  *Reason.* {esc(a['reason'])}\n")
        else:
            out.append("The list in the key is empty.\n")
        if k.get("acceptable_alternatives_note") is not None:
            out.append(f"**Note on alternatives.** {esc(k['acceptable_alternatives_note'])}\n")
        out.append(f"**Key numbers** ({len(k['key_numbers'])}).\n")
        rows = [(esc(x["name"]), esc(jtext(x["value"])), esc(jtext(x["tolerance"]))) for x in k["key_numbers"]]
        out.append(small(md_table(["Name", "Value", "Tolerance"], rows, [52, 30, 18])))
        out.append(f"**Traps** ({len(k['traps'])}).\n")
        for i, tr in enumerate(k["traps"], 1):
            if set(tr) != {"id", "description", "correct_handling", "mishandling"}:
                fail(f"unexpected trap fields in {t}")
            out.append(f"*Trap {i}: {esc(tr['id'])}*\n")
            out.append(f"- *Description.* {esc(tr['description'])}\n- *Correct handling.* {esc(tr['correct_handling'])}\n- *Mishandling.* {esc(tr['mishandling'])}\n")
        out.append(f"**False-statement checks** ({len(k['false_statement_checks'])}).\n")
        for i, s in enumerate(k["false_statement_checks"], 1):
            out.append(f"{i}. {esc(s)}")
        out.append("")
        if "grader_guidance" in k:
            out.append("**Grader guidance.** This field is Markdown in the key and is shown as such.\n")
            out.append(copied(k["grader_guidance"], f"{t} grader_guidance") + "\n")
        known = {"task", "question", "verdict_options", "key_verdict", "key_conclusion", "acceptable_alternatives",
                 "acceptable_alternatives_note", "key_numbers", "traps", "false_statement_checks", "reference",
                 "verification", "grader_guidance", "source", "sources"}
        extra = set(k) - known
        if extra:
            fail(f"key {t} has fields not handled: {extra}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- section 10
def sec_studyB():
    items = json.loads(read(PROTO / "STUDYB_ITEMS.json"))
    if len(items) != 22:
        fail(f"STUDYB_ITEMS.json has {len(items)} items, expected 22")
    keys = {it["id"]: json.loads(read(EV / "studyB" / "keys" / f"{it['id']}.json")) for it in items}
    for it in items:
        k = keys[it["id"]]
        if (k["opaque_id"], k["pair_id"], k["status"]) != (it["id"], it["pair"], it["status"]):
            fail(f"item list and key disagree for {it['id']}")
        if it["status"] == "flawed" and k["error"]["in_checklist"] != it["in_checklist"]:
            fail(f"in_checklist disagrees for {it['id']}")
    pairs = {}
    for it in items:
        pairs.setdefault(it["pair"], {})[it["status"]] = it["id"]
    if len(pairs) != 11 or any(set(v) != {"flawed", "clean"} for v in pairs.values()):
        fail("Study B pairs are not 11 flawed/clean pairs")
    if "P01" in pairs:
        fail("P01 should not be in the item list")
    planted = sorted(p for p, v in pairs.items() if "(planted" in keys[v["flawed"]]["error"]["nature"])
    natural = sorted(p for p in pairs if p not in planted)
    n_in = sum(1 for p in pairs if keys[pairs[p]["flawed"]]["error"]["in_checklist"])

    for p_, v in pairs.items():  # backs the sentence "in every pair the two lists begin with the same items"
        a_, b_ = keys[v["flawed"]]["possible_false_alarms"], keys[v["clean"]]["possible_false_alarms"]
        if not (a_ and b_ and a_[0] == b_[0]):
            fail(f"pair {p_}: the two false-alarm lists do not begin with the same item")
    order = sorted(items, key=lambda x: (x["pair"], 0 if x["status"] == "flawed" else 1))
    rows = []
    for it in order:
        k = keys[it["id"]]
        where = {True: "inside", False: "outside", None: "–"}[it["in_checklist"]]
        rows.append((esc(it["id"]), esc(it["pair"]), esc(it["status"]), where, esc(k["source_project"])))

    out = ["# Study B items\n",
           f"Study B used {len(items)} analysis packages in {len(pairs)} pairs. Each pair has a flawed version with one documented "
           "error and a clean version. The table comes from `protocol/STUDYB_ITEMS.json` and the item keys in `studyB/keys/`. "
           "The column 'Error inside the checklist' says whether the documented error falls inside the checklist's ten patterns; "
           "it is empty for clean items. "
           f"{len(natural)} pairs have natural errors and {len(planted)} pairs ({', '.join(planted)}) have planted summaries that "
           f"overstate causal claims. {n_in} of the {len(pairs)} documented errors fall inside the checklist. "
           "A twelfth pair, P01 (packages B1131 and B1448), had a natural error. It failed its own verification and was excluded before freezing "
           "(Amendment 1, item 5). It is not shown.\n"]
    out.append(small(md_table(["Opaque id", "Pair", "Status", "Error inside the checklist", "Source project"], rows,
                              [10, 6, 8, 12, 64])))
    out.append("For each pair, the sections below give the analysis description, the documented error, the detection rule, and "
               "the possible false alarms listed in each of the two keys, all copied from the keys. In every pair the two lists of "
               "possible false alarms begin with the same items, word for word; these are shown once, followed by the rest of "
               "each list with its own numbering. Each key also lists known true "
               "facts about the package and, for some items, other real issues and a build record; these are not reproduced here.\n")
    for p in sorted(pairs):
        f_id, c_id = pairs[p]["flawed"], pairs[p]["clean"]
        kf, kc = keys[f_id], keys[c_id]
        out.append(f"## Pair {p} (flawed {f_id}, clean {c_id})\n")
        out.append(f"**Source project.** {esc(kf['source_project'])}\n")
        if kf["source_project"] != kc["source_project"]:
            out.append(f"**Source project of the clean version.** {esc(kc['source_project'])}\n")
        out.append(f"**Analysis.** {esc(kf['analysis'])}\n")
        if kf["analysis"] != kc["analysis"]:
            out.append(f"**Analysis, as described in the clean version's key.** {esc(kc['analysis'])}\n")
        e = kf["error"]
        if set(e) != {"nature", "location", "wrong_result", "correct_result", "pattern", "in_checklist"}:
            fail(f"unexpected error fields in {f_id}")
        out.append(f"**Documented error ({f_id}).**\n")
        out.append(f"- *Nature.* {esc(e['nature'])}\n- *Location.* {esc(e['location'])}\n"
                   f"- *Wrong result.* {esc(e['wrong_result'])}\n- *Correct result.* {esc(e['correct_result'])}\n"
                   f"- *Pattern.* {esc(e['pattern'])}\n")
        out.append(f"**Detection rule ({f_id}).** {esc(kf['detection_rule'])}\n")
        if kc["detection_rule"]:
            out.append(f"**Rule for the clean version ({c_id}).** {esc(kc['detection_rule'])}\n")
        else:
            out.append(f"**Rule for the clean version ({c_id}).** The key gives no rule (the field is empty).\n")
        if kc["error"] is not None:
            fail(f"clean item {c_id} has an error")
        fa_f, fa_c = kf["possible_false_alarms"], kc["possible_false_alarms"]
        n = 0
        while n < min(len(fa_f), len(fa_c)) and fa_f[n] == fa_c[n]:
            n += 1
        if n:
            out.append(f"**Possible false alarms, items 1 to {n} of both keys** (the same, word for word, in {f_id} and {c_id}).\n")
            for i, s in enumerate(fa_f[:n], 1):
                out.append(f"{i}. {esc(s)}")
            out.append("")
        for kid, rest, lab in ((f_id, fa_f[n:], "flawed"), (c_id, fa_c[n:], "clean")):
            if not rest:
                out.append(f"**Possible false alarms, rest of the key of {kid} ({lab}).** None.\n")
                continue
            what = "rest of the key of" if n else "key of"
            out.append(f"**Possible false alarms, {what} {kid} ({lab}).**\n")
            for i, s in enumerate(rest, n + 1):
                out.append(f"{i}. {esc(s)}")
            out.append("")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- section 11
def sec_studyC():
    path = PROTO / "STUDYC_KEY.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames
        rows = list(reader)
    print("[info] STUDYC_KEY.csv columns:", ", ".join(cols))
    if len(rows) != 68:
        fail(f"STUDYC_KEY.csv has {len(rows)} rows, expected 68")
    if any(r["layer"] != "run_outputs" or r["present_in_pre_audit_state"] != "yes" for r in rows):
        fail("STUDYC_KEY.csv has rows outside layer run_outputs or not present before the audit")
    lens = {c: sum(len(r[c]) for r in rows) / len(rows) for c in ("short_name", "where_it_lives", "deterministic_evidence", "notes")}
    if min(lens, key=lens.get) != "short_name":
        fail("short_name is not the shortest descriptive column")
    sev = {s: sum(1 for r in rows if r["severity"] == s) for s in ("critical", "major", "minor")}
    if sum(sev.values()) != 68:
        fail("unexpected severity values in STUDYC_KEY.csv")
    n_out = sum(1 for r in rows if r["audit_pattern"] == "outside checklist")
    table = [(esc(r["id"]), esc(r["severity"]), esc(r["pipeline"]), esc(r["short_name"]), esc(r["audit_pattern"])) for r in rows]
    out = ["# Study C reference set\n",
           "The Study C reference set is the frozen file `protocol/STUDYC_KEY.csv`. It has 68 rows, one per known error. "
           "All 68 rows have `layer` = run_outputs and `present_in_pre_audit_state` = yes. "
           f"{sev['critical']} are critical, {sev['major']} major and {sev['minor']} minor. "
           f"{68 - n_out} fall under one of the checklist's ten patterns and {n_out} are outside the checklist. "
           f"The file has {len(cols)} columns. "
           "The table shows five of them: `id`, `severity`, `pipeline`, `short_name` (the shortest description) and "
           "`audit_pattern` (the checklist pattern, or 'outside checklist'). The graders in Study C were given the full file.\n"]
    out.append(small(md_table(["ID", "Severity", "Pipeline", "Short description", "Pattern"], table, [6, 10, 20, 46, 18])))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- section 12
def sec_haiku():
    out = ["# Exploratory arm with Claude Haiku 4.5\n"]
    if HAIKU_SECTION.exists():
        txt = fix_md(read(HAIKU_SECTION))
        levels = [len(m.group(1)) for m in re.finditer(r"(?m)^(#{1,6})\s", txt)]
        if levels:
            shift = max(0, 2 - min(levels))
            if shift:
                txt = demote(txt, shift, unnumbered=False)
        out.append(txt.strip() + "\n")
        print(f"[info] included {HAIKU_SECTION.name}")
    else:
        out.append("[Section to be added.]\n")
        print(f"[info] {HAIKU_SECTION.name} not found; inserted the placeholder line")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- pandoc
LUA_FILTER = r"""
-- Code blocks become wrapping Verbatim (fvextra).
-- Long inline code, and long words that contain / or _, may break after / _ and .
local base = {
  ['\\'] = '\\textbackslash{}', ['{'] = '\\{', ['}'] = '\\}', ['$'] = '\\$',
  ['&'] = '\\&', ['#'] = '\\#', ['^'] = '\\textasciicircum{}',
  ['_'] = '\\_', ['%'] = '\\%', ['~'] = '\\textasciitilde{}',
  ['['] = '{[}', [']'] = '{]}',
}
local function esc(s, breaks)
  return (s:gsub('[\\{}$&#^_%%~%[%]/%.]', function(c)
    local r = base[c] or c
    if breaks and (c == '/' or c == '_' or c == '.') then r = r .. '\\allowbreak{}' end
    return r
  end))
end
return {
  {
    traverse = 'topdown',
    Header = function(el) return el, false end,
    Code = function(el)
      return pandoc.RawInline('latex', '\\texttt{' .. esc(el.text, #el.text > 20) .. '}')
    end,
    Str = function(el)
      if #el.text >= 12 and el.text:find('[/_]') then
        return pandoc.RawInline('latex', esc(el.text, true))
      end
    end,
    CodeBlock = function(el)
      local size = '\\small'
      if el.classes:includes('prompt') then size = '\\footnotesize' end
      local opts = 'breaklines,breakanywhere,fontsize=' .. size ..
                   ',frame=single,rulecolor=\\color{black!35},framesep=1.5mm'
      return pandoc.RawBlock('latex', '\\begin{Verbatim}[' .. opts .. ']\n' .. el.text .. '\n\\end{Verbatim}')
    end,
  }
}
"""

HEADER_TEX = r"""
\usepackage{tocloft}
\setlength{\cftsecnumwidth}{2.2em}
\setlength{\cftsubsecnumwidth}{3.2em}
\makeatletter
% level-4 and level-5 headings on their own line (not run into the text)
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}{2.2ex \@plus 1ex \@minus .2ex}{0.6ex}{\normalfont\normalsize\bfseries}}
\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\z@}{1.8ex \@plus .5ex \@minus .2ex}{0.5ex}{\normalfont\normalsize\bfseries\itshape}}
\makeatother
\usepackage{fvextra}
\usepackage{xcolor}
\usepackage[document]{ragged2e}
% ragged right with unlimited stretch, so a long token moves to the next line instead of sticking out
\setlength{\RaggedRightRightskip}{0pt plus 1fil}
\setlength{\emergencystretch}{3em}
"""

READER = ("markdown"
          "-raw_html-raw_tex-tex_math_dollars-smart-subscript-superscript-strikeout"
          "-citations-example_lists-fancy_lists-definition_lists-line_blocks"
          "-yaml_metadata_block-pandoc_title_block-inline_notes-footnotes-fenced_divs"
          "-bracketed_spans-native_spans-native_divs-inline_code_attributes-link_attributes"
          "-simple_tables-multiline_tables-grid_tables-implicit_figures-escaped_line_breaks-task_lists")


def run_pandoc(md_path, pdf_path):
    with tempfile.TemporaryDirectory() as td:
        lua = Path(td) / "s2_filter.lua"
        hdr = Path(td) / "s2_header.tex"
        lua.write_text(LUA_FILTER, encoding="utf-8")
        hdr.write_text(HEADER_TEX, encoding="utf-8")
        cmd = [PANDOC, str(md_path), "-f", READER, "-o", str(pdf_path),
               "--pdf-engine=xelatex", "--lua-filter", str(lua), "-H", str(hdr),
               "--toc", "--toc-depth=2", "--number-sections",
               "-M", f"title={TITLE}",
               "-V", f"mainfont={MAINFONT}", "-V", f"monofont={MONOFONT}",
               # keep straight quotes and dashes exactly as typed (no TeX quote/dash ligatures)
               "-V", "mainfontoptions=Ligatures=TeXOff",
               "-V", "fontsize=10pt", "-V", "papersize=a4", "-V", "geometry:margin=2cm",
               "-V", "colorlinks=true", "-V", "linkcolor=black", "-V", "urlcolor=black", "-V", "toccolor=black"]
        r = subprocess.run(cmd, capture_output=True, text=True)
    return r


def page_report(pdf_path):
    try:
        from pypdf import PdfReader
    except ImportError:
        print("[info] pypdf not installed; page report skipped")
        return
    rd = PdfReader(str(pdf_path))
    n = len(rd.pages)
    print(f"[result] pages: {n}")
    tops = []
    for o in rd.outline:
        if not isinstance(o, list):
            tops.append((o.title, rd.get_destination_page_number(o) + 1))
    for i, (t, p) in enumerate(tops):
        if i + 1 == len(tops):
            end = n
        else:
            nt, np_ = tops[i + 1]
            # the next section may start mid-page; then this one ends on that page
            first = next((l.strip() for l in (rd.pages[np_ - 1].extract_text() or "").split("\n") if l.strip()), "")
            starts_page = first.replace(" ", "").startswith(nt.replace(" ", "")[:25])
            end = np_ - 1 if starts_page else np_
        print(f"[result] {t}: pages {p}-{max(p, end)}")


# Local roots and the placeholders the code and data release uses for them (longest first).
# Every occurrence in this appendix sits in inline code or a code block, where pandoc keeps "<...>".
LOCAL_ROOTS = [
    ("<EVAL_ROOT>", "<EVAL_ROOT>"),
    ("<REPO_ROOT>", "<REPO_ROOT>"),
    ("<DATA_ROOT>", "<DATA_ROOT>"),
    ("<CODE_ROOT>", "<CODE_ROOT>"),
    ("<LOCAL_VOLUME>", "<LOCAL_VOLUME>"),
    ("<AGENT_TMP>", "<AGENT_TMP>"),
    ("<HOME>", "<HOME>"),
]


def local_paths_to_placeholders(md):
    for root, ph in LOCAL_ROOTS:
        md = md.replace(root, ph)
    md = re.sub(r"-(?:Volumes-Crucial-X6|Users-ihorkendiukhov)[A-Za-z0-9\-]*", "<SESSION_DIR>", md)
    left = re.findall(r"Crucial.X6|ihorkendiukhov|/Volumes/|<TMP>", md)
    if left:
        fail(f"local paths left in S2 after placeholder mapping: {left[:5]}")
    return md


def main():
    parts = [
        raw_latex("\\clearpage"),
        sec_contents(),
        sec_prereg(),
        sec_amendments(),
        sec_freeze(),
        sec_generic_checklist(),
        sec_deployed_checklist(),
        sec_prompts(),
        sec_briefs(),
        sec_keys(),
        sec_studyB(),
        sec_studyC(),
        sec_haiku(),
    ]
    md = local_paths_to_placeholders("\n".join(parts))
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"[ok] wrote {OUT_MD} ({len(md):,} characters)")
    r = run_pandoc(OUT_MD, OUT_PDF)
    if r.stdout.strip():
        print(r.stdout)
    if r.stderr.strip():
        print("[pandoc stderr]\n" + r.stderr)
    if r.returncode != 0:
        sys.exit(f"pandoc failed with code {r.returncode}")
    missing = [l for l in r.stderr.split("\n") if "Missing character" in l]
    print(f"[check] missing-character warnings: {len(missing)}")
    print(f"[ok] wrote {OUT_PDF}")
    page_report(OUT_PDF)


if __name__ == "__main__":
    main()
