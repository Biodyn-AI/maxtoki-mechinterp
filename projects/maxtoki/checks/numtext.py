"""Extract numeric claims from markdown or LaTeX text (shared by number_trace.py and verify/).

Each extracted number carries the precision it was written with, so it can be traced
"within rounding": a written value x with d decimals matches an artefact value v when
|v - x| <= 0.5 * 10^-d (scaled for 10^k notation and M/k suffixes).

Deliberately excluded (recorded with a reason, never traced):
  identifiers (L6, H65, P1, S1, MaxToki-217M, GPT-5.3, 3D, 217M), label numbers after words
  like Fig/Table/Step/Phase/layer, dates and times, and integers below 10.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

LABEL_WORDS = {
    "fig", "figs", "figure", "figures", "table", "tables", "step", "steps", "phase", "phases",
    "stage", "stages", "section", "sections", "sec", "eq", "equation", "supp", "supplementary",
    "note", "notes", "appendix", "line", "lines", "layer", "layers", "iteration", "iterations",
    "iter", "round", "rounds", "version", "opus", "sonnet", "claude", "gpt", "chapter", "part",
    "hypothesis", "hypotheses", "criterion", "criteria", "experiment", "experiments", "condition",
    "family", "pattern", "patterns", "item", "items", "row", "rows", "column", "columns", "v",
    "python", "cuda", "macos", "ios",
}
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")

NUM = re.compile(r"(?<![\w.\\])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+|\.\d+)")
SCI_AFTER = re.compile(r"^\s*(?:[×x]|\\times)\s*10\s*\^?\s*\{?\s*([-−+]?\d+)\s*\}?")
E_AFTER = re.compile(r"^[eE]([-−+]?\d+)(?![\w.])")
SUFFIX_M = re.compile(r"^(?:(M|K|k)\b(?!-)|(?:\s|~|\\,)(million|thousand|M)\b(?!-))")
PCT_AFTER = re.compile(r"^\s?(?:%|\\%|percent\b|pp\b|percentage point)")


@dataclass
class Num:
    raw: str
    value: float            # absolute value
    h: float                # half-width of the rounding interval
    decimals: int
    kind: str               # decimal | integer | sci
    pct: bool
    approx: bool
    line: int
    context: str
    trailing_zero_h: float | None = None   # relaxed half-width for integers written with trailing zeros
    excluded: str | None = None
    extra: dict = field(default_factory=dict)

    def key(self):
        return (round(self.value, 12), round(self.h, 15), self.pct)


def _line_starts(text):
    starts = [0]
    for m in re.finditer("\n", text):
        starts.append(m.end())
    return starts


def _line_of(starts, pos):
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def _blank(text, pattern, flags=0):
    """Replace matches with spaces (keeps offsets, so line numbers stay right)."""
    return re.sub(pattern, lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=flags)


def clean_markdown(text: str) -> str:
    t = _blank(text, r"^```.*?^```", re.S | re.M)                        # fenced code
    t = _blank(t, r"`[^`\n]*[A-Za-z/_=][^`\n]*`")                           # code spans with letters/paths
    t = _blank(t, r"https?://\S+")
    t = _blank(t, r"\]\([^)]*\)")                                          # link targets
    t = _blank(t, r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b")  # dates
    t = _blank(t, r"\b\d{1,2}:\d{2}(?::\d{2})?\b")                          # times
    t = _blank(t, r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b")               # hashes
    t = _blank(t, r"\bENS[A-Z]*\d{6,}\b")
    t = _blank(t, r"(?i)arxiv[:\s]*\d{4}\.\d{4,5}|\b[12]\d{3}\.\d{4,5}\b")                # arXiv ids
    t = _blank(t, r"\b[vV]\d+(?:\.\d+)+\b")                                # version strings
    t = t.replace("−", "-").translate(_SUP)
    return t


def clean_latex(text: str):
    """Return (cleaned_body, offset_line) for the document body of a LaTeX file."""
    b = text.find("\\begin{document}")
    e = text.find("\\bibliography{")
    if e < 0:
        e = text.find("\\end{document}")
    body = text[b:e] if b >= 0 else text
    first_line = text[:b].count("\n") + 1 if b >= 0 else 1
    t = _blank(body, r"(?<!\\)%[^\n]*")                                                   # comments
    t = _blank(t, r"\\(?:label|ref|eqref|pageref|cite[pt]?|url|input|include|includegraphics|"
                  r"bibliographystyle|setlength|vspace\*?|hspace\*?|arraystretch|newcommand|"
                  r"renewcommand|setcounter|addtocounter|linenumbers|documentclass|usepackage)"
                  r"(?:\[[^\]]*\])?(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})*")
    t = _blank(t, r"\\begin\{(?:longtable|tabular\*?|tabularx|table\*?|figure\*?)\}(?:\[[^\]]*\])?(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})*")
    t = _blank(t, r"[pmb]\{\s*[\d.]+\s*\\(?:linewidth|textwidth|columnwidth)\s*\}")
    # LaTeX lengths are written attached ("0.2in", "4pt"). No whitespace allowed here: with "\s*" a
    # number followed by the English word "in" (e.g. "0.190, against 0.59\nin the source study") was
    # blanked as if it were a length (fixed in the 2026-10-01 verification pass; see CHECKER_REPORT.md).
    t = _blank(t, r"(?<![\w.])[\d.]+(?:pt|cm|mm|in|em|ex|bp)\b")
    t = _blank(t, r"[\d.]+\s*\\(?:linewidth|textwidth|columnwidth|baselineskip)")
    t = _blank(t, r"\\(?:tabcolsep|arraystretch)\S*")
    t = _blank(t, r"\\[a-zA-Z]+\d+")                                                      # macros with digits
    # normalise number formatting, keeping string length where possible
    t = t.replace("{,}", ",")
    t = re.sub(r"(?<=\d)\\,(?=\d)", "", t)
    t = t.replace("\\,", " ").replace("\\!", " ").replace("{=}", "=")
    t = t.replace("$", " ").replace("~", " ")
    t = t.replace("\\times", "×     ").replace("\\%", "% ").replace("--", "– ")
    t = t.replace("\\approx", "≈      ").replace("\\sim", "~   ").replace("−", "-")
    return t, first_line


def extract(text: str, fmt: str = "markdown"):
    """Return a list of Num (included and excluded)."""
    if fmt == "latex":
        t, first_line = clean_latex(text)
    else:
        t, first_line = clean_markdown(text), 1
    starts = _line_starts(t)
    out = []
    for m in NUM.finditer(t):
        s, e = m.span(1)
        raw = m.group(1)
        before = t[max(0, s - 24):s]
        after = t[e:e + 24]
        ctx = t[max(0, s - 60):min(len(t), e + 60)].replace("\n", " ")
        ctx = re.sub(r"\s+", " ", ctx).strip()
        line = _line_of(starts, s) + first_line - 1
        num = raw.replace(",", "")
        decimals = len(num.split(".")[1]) if "." in num else 0
        val = float(num)
        exp = 0
        kind = "decimal" if decimals else "integer"
        sm = SCI_AFTER.match(after)
        em = E_AFTER.match(after)
        consumed = 0
        if sm:
            exp = int(sm.group(1).replace("−", "-"))
            kind = "sci"
            consumed = sm.end()
        elif em:
            exp = int(em.group(1).replace("−", "-"))
            kind = "sci"
            consumed = em.end()
        rest = after[consumed:]
        scale = 1.0
        mm = SUFFIX_M.match(rest)
        if mm and kind != "sci":
            scale = {"M": 1e6, "million": 1e6, "k": 1e3, "K": 1e3, "thousand": 1e3}[mm.group(1) or mm.group(2)]
            rest = rest[mm.end():]
        pct = bool(PCT_AFTER.match(rest))
        approx = bool(re.search(r"(~|≈|approx(?:imately)?|about|roughly|nearly|around)\s*$", before))
        x = val * (10.0 ** exp) * scale
        h = 0.5 * (10.0 ** -decimals) * (10.0 ** exp) * scale
        n = Num(raw=raw + (f"e{exp}" if kind == "sci" else "") + ("%" if pct else ""), value=abs(x), h=h,
                decimals=decimals, kind=kind, pct=pct, approx=approx, line=line, context=ctx)
        if kind == "integer" and scale == 1.0:
            tz = len(num) - len(num.rstrip("0"))
            if tz >= 1 and val >= 10:
                n.trailing_zero_h = 0.5 * 10 ** tz
        # ---- exclusions
        if re.search(r"(?:^|[^A-Za-z0-9])[A-Za-z][A-Za-z0-9]*-$", before):
            n.excluded = "identifier (letter-hyphen before)"
        elif consumed == 0 and scale == 1.0 and re.match(r"^[A-Za-z_]", after) and not re.match(r"^[x×](?![A-Za-z])", after):
            n.excluded = "identifier (letter after)"
        elif re.match(r"^\.\d", after) or re.match(r"^\d", after):
            n.excluded = "part of a longer token"
        else:
            w = re.findall(r"([A-Za-z]+)\.?\s*$", before)
            if w and w[-1].lower() in LABEL_WORDS:
                n.excluded = f"label number after '{w[-1]}'"
            elif re.search(r"(?:Figs?|Tables?|Steps?|Phases?|Layers?|Stages?|L|H|P|E|S)\s?\d+\s?[–-]\s?$", before):
                n.excluded = "label range"
            elif kind == "integer" and scale == 1.0 and not pct and val < 10:
                n.excluded = "integer below 10"
        out.append(n)
    return out


def dedupe(nums):
    seen = {}
    for n in nums:
        if n.excluded:
            continue
        k = n.key()
        if k not in seen:
            seen[k] = n
            n.extra["occurrences"] = [n.line]
        else:
            seen[k].extra["occurrences"].append(n.line)
    return list(seen.values())
