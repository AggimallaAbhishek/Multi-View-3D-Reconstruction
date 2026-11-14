"""Text of a PDF written by ReportLab without page compression, decoded as a viewer's
search or copy would: through each font's ``/ToUnicode`` map. Embedded TrueType subsets
draw glyph codes, not characters, so reading the raw strings would miss non-ASCII text."""

from __future__ import annotations

import re


def pdf_text(pdf: bytes) -> str:
    """The strings the pages draw, in order, one per line."""
    raw = pdf.decode("latin-1")
    objects = {int(n): body for n, body in re.findall(r"(?s)(\d+) 0 obj\s*(.*?)\s*endobj", raw)}
    to_unicode = {}  # font resource name (F1, F3+0, ...) -> {code: text}, or None: as is
    for name, ref in re.findall(r"/(F\d+(?:\+\d+)?) (\d+) 0 R", raw):
        font = objects.get(int(ref), "")
        cmap = re.search(r"/ToUnicode (\d+) 0 R", font)
        to_unicode[name] = _cmap(objects[int(cmap.group(1))]) if cmap else None

    lines, font = [], None
    for token in re.finditer(r"/(F\d+(?:\+\d+)?) [\d.]+ Tf|\(((?:\\.|[^\\)])*)\)\s*Tj", raw):
        if token.group(1):
            font = token.group(1)
            continue
        codes = _unescape(token.group(2))
        mapping = to_unicode.get(font)
        lines.append(codes if mapping is None else "".join(mapping.get(c, c) for c in codes))
    return "\n".join(lines)


def _unescape(s: str) -> str:
    s = re.sub(r"\\([0-7]{1,3})", lambda m: chr(int(m.group(1), 8)), s)
    return re.sub(r"\\(.)", r"\1", s)


def _cmap(stream: str) -> dict[str, str]:
    pairs = re.findall(r"<([0-9A-Fa-f]{2})>\s*<([0-9A-Fa-f]{4,})>", stream)
    return {
        chr(int(code, 16)): "".join(chr(int(text[k : k + 4], 16)) for k in range(0, len(text), 4))
        for code, text in pairs
        if int(text, 16)  # <0000>: unused code
    }
