"""Page images and provenance: render PDF pages and map chunks to pages.

Extend layout blocks carry ``metadata.page.number`` and a ``boundingBox``
in the page's own coordinate space, so every chunk can be traced back to
the exact page region it came from. The web UI renders those pages and
draws the highlight boxes client-side.
"""
from __future__ import annotations

import json
import os


def page_boxes(blocks_json: str) -> dict[int, dict]:
    """Map chunk blocks to {page_number: {width, height, boxes:[...]}}.

    Coordinates are in the block coordinate space; the caller scales
    them to the rendered image size.
    """
    try:
        blocks = json.loads(blocks_json or "[]")
    except (json.JSONDecodeError, TypeError):
        return {}
    pages: dict[int, dict] = {}
    for b in blocks:
        if not isinstance(b, dict):
            continue
        meta = b.get("metadata") or {}
        page = meta.get("page") or {}
        num = page.get("number")
        bbox = b.get("boundingBox")
        if not isinstance(num, int) or not isinstance(bbox, dict):
            continue
        entry = pages.setdefault(num, {
            "width": page.get("width"), "height": page.get("height"),
            "boxes": [],
        })
        try:
            entry["boxes"].append({
                "left": float(bbox["left"]), "top": float(bbox["top"]),
                "right": float(bbox["right"]), "bottom": float(bbox["bottom"]),
                "type": b.get("type", ""),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return pages


def render_pages(pdf_path: str, out_dir: str, width: int = 900) -> list[str]:
    """Render every page of a PDF to PNG. Returns the written filenames."""
    import pymupdf  # local import: only the web service needs rendering

    os.makedirs(out_dir, exist_ok=True)
    written = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            target = os.path.join(out_dir, f"p{i + 1}.png")
            if os.path.exists(target):
                written.append(target)
                continue
            zoom = width / page.rect.width
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
            pix.save(target)
            written.append(target)
    return written


def resolve_pdf(doc_source: str, seed_dir: str) -> str | None:
    """Find a readable PDF for a document: original path, else seed copy."""
    if doc_source and os.path.exists(doc_source):
        return doc_source
    base = os.path.basename(doc_source or "")
    if not base or not os.path.isdir(seed_dir):
        return None
    # Exact basename match first, then any PDF sharing a long stem.
    for name in sorted(os.listdir(seed_dir)):
        if name == base:
            return os.path.join(seed_dir, name)
    stem = base[:30]
    for name in sorted(os.listdir(seed_dir)):
        if name.lower().endswith(".pdf") and stem in name:
            return os.path.join(seed_dir, name)
    pdfs = [n for n in sorted(os.listdir(seed_dir)) if n.lower().endswith(".pdf")]
    return os.path.join(seed_dir, pdfs[0]) if len(pdfs) == 1 else None
