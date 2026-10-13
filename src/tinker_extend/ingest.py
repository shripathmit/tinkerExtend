"""Collect candidate input files: expand directories, hash for delta updates."""
from __future__ import annotations

import hashlib
import os

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp",
    ".txt", ".md", ".csv", ".tsv", ".html", ".htm",
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iter_candidate_files(sources: list[str]) -> list[str]:
    """Expand files/directories into a sorted list of supported file paths."""
    out: list[str] = []
    for src in sources:
        if os.path.isfile(src):
            out.append(src)
        elif os.path.isdir(src):
            for root, _dirs, files in os.walk(src):
                for name in sorted(files):
                    p = os.path.join(root, name)
                    if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
                        out.append(p)
        else:
            raise FileNotFoundError(f"Not a file or directory: {src}")
    # de-dupe, keep order
    seen: set[str] = set()
    unique = []
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            unique.append(ap)
    return unique
