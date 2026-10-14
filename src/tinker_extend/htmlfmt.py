"""Render chunk content as safe HTML.

Extend chunks mix Markdown-style headers with raw HTML tables. This
module escapes everything, then re-allows a whitelist of structural
tags, and converts Markdown headers/bold to HTML. Tables render as
real tables instead of visible markup.
"""
from __future__ import annotations

import html
import re

_ALLOWED_TAGS = (
    "table", "thead", "tbody", "tfoot", "caption", "colgroup", "col",
    "tr", "td", "th", "br", "p", "b", "strong", "i", "em", "u",
    "ul", "ol", "li", "h1", "h2", "h3", "h4", "figure", "figcaption",
    "blockquote", "hr", "span", "div",
)

# After html.escape, a whitelisted tag looks like
#   &lt;td rowspan=&quot;2&quot;&gt;  — restore it, keeping attributes.
# The lookahead after the tag name prevents prefix matches: without it,
# the "i" alternative would match the start of "&lt;img ...&gt;".
_TAG_RE = re.compile(
    r"&lt;(/?)((?:" + "|".join(_ALLOWED_TAGS) + r"))(?=[\s/]|&gt;)"
    r"((?:[^&]|&(quot|amp|#\d+|lt|gt);)*?)&gt;",
    re.IGNORECASE,
)


def _restore_tag(m: re.Match) -> str:
    attrs = m.group(3)
    attrs = attrs.replace("&quot;", '"').replace("&#39;", "'").replace("&#x27;", "'")
    attrs = attrs.replace("&amp;", "&")
    return f"<{m.group(1)}{m.group(2)}{attrs}>"

_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)$")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def to_safe_html(text: str) -> str:
    """Convert chunk text to sanitized HTML."""
    if not text:
        return ""
    out_lines = []
    for line in html.escape(text).splitlines():
        m = _HEADER_RE.match(line.strip())
        if m:
            level = len(m.group(1))
            out_lines.append(f"<h{level + 1}>{m.group(2)}</h{level + 1}>")
            continue
        out_lines.append(line)
    out = "<br>".join(out_lines)
    out = _BOLD_RE.sub(r"<strong>\1</strong>", out)
    out = _TAG_RE.sub(_restore_tag, out)
    return out
