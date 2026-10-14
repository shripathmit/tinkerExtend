"""Pluggable text embeddings. The KB works without them (FTS search only);
with an embedder configured, search becomes hybrid (FTS + vectors).

The default embedder runs a small local model through fastembed (ONNX, no
torch, no API key). Configure via environment:
  TINKER_EMBEDDINGS   - "on" (default) or "off"
  TINKER_EMBED_MODEL  - default BAAI/bge-small-en-v1.5 (384 dims, ~67MB)
  TINKER_EMBED_CACHE  - model download dir; put it on a persistent volume
                        so deploys don't re-download the model
"""
from __future__ import annotations

import os
import threading

import numpy as np

from .chunking import strip_markup

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def normalize(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def to_blob(vec) -> bytes:
    return normalize(vec).tobytes()


class Embedder:
    """Interface: embed passages and queries into normalized float32 vectors.

    ``model`` names the vector space; vectors from different models are
    never compared.
    """

    model: str | None = None

    def embed(self, texts: list[str]) -> list[np.ndarray | None]:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray | None:
        return self.embed([text])[0]


class NullEmbedder(Embedder):
    """No embeddings: chunks are stored without vectors, search is FTS only."""

    def embed(self, texts: list[str]) -> list[np.ndarray | None]:
        return [None] * len(texts)

    def embed_query(self, text: str) -> np.ndarray | None:
        return None


class FastEmbedEmbedder(Embedder):
    """Local ONNX embedding model via fastembed, loaded on first use.

    One instance is shared by the web app and wave worker threads; a lock
    serializes model calls.
    """

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str | None = None,
                 batch_size: int = 32):
        self.model = model
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._impl = None
        self._lock = threading.Lock()

    def _load(self):
        if self._impl is None:
            from fastembed import TextEmbedding
            self._impl = TextEmbedding(model_name=self.model, cache_dir=self.cache_dir)
        return self._impl

    def embed(self, texts: list[str]) -> list[np.ndarray | None]:
        if not texts:
            return []
        clean = [strip_markup(t) for t in texts]
        with self._lock:
            impl = self._load()
            vecs = list(impl.embed(clean, batch_size=self.batch_size))
        return [normalize(v) for v in vecs]

    def embed_query(self, text: str) -> np.ndarray | None:
        with self._lock:
            impl = self._load()
            vec = next(iter(impl.query_embed(strip_markup(text))))
        return normalize(vec)


_SHARED: Embedder | None = None
_SHARED_LOCK = threading.Lock()


def embedder_from_env() -> Embedder:
    """The process-wide embedder from env; NullEmbedder if disabled or unavailable.

    Never raises: a missing package or model download failure logs a
    warning and leaves search on FTS only.
    """
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is not None:
            return _SHARED
        if os.environ.get("TINKER_EMBEDDINGS", "on").strip().lower() in ("off", "0", "false", "no"):
            _SHARED = NullEmbedder()
            return _SHARED
        emb = FastEmbedEmbedder(
            model=os.environ.get("TINKER_EMBED_MODEL", DEFAULT_MODEL),
            cache_dir=os.environ.get("TINKER_EMBED_CACHE") or None,
        )
        try:
            emb._load()
            _SHARED = emb
        except Exception as exc:  # noqa: BLE001 - ImportError, download, ONNX errors
            print(f"tinkerExtend: embeddings disabled ({type(exc).__name__}: {exc})",
                  flush=True)
            _SHARED = NullEmbedder()
        return _SHARED
