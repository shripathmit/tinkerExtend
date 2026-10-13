"""Pluggable text embeddings. The KB works without them (FTS search);
configure an embedder to enable vector search later."""
from __future__ import annotations


class Embedder:
    """Interface: embed a batch of texts -> one vector (or None) per text."""

    def embed(self, texts: list[str]) -> list[list[float] | None]:
        raise NotImplementedError


class NullEmbedder(Embedder):
    """Default: no embeddings. Chunks are stored with embedding_json = NULL."""

    def embed(self, texts: list[str]) -> list[list[float] | None]:
        return [None] * len(texts)
