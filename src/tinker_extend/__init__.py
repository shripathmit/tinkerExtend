"""tinker_extend: Extend-powered document ingestion → SQLite knowledge base."""

from .config import Config, load_config
from .extend_client import ExtendClient, ExtendAPIError, ExtendRunFailed
from .models import Document, Chunk
from .pipeline import KnowledgeBasePipeline
from .store import KnowledgeBase

__all__ = [
    "Config",
    "load_config",
    "ExtendClient",
    "ExtendAPIError",
    "ExtendRunFailed",
    "Document",
    "Chunk",
    "KnowledgeBasePipeline",
    "KnowledgeBase",
]
