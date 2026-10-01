"""Local, free embedding function (sentence-transformers) wired into Chroma's
EmbeddingFunction interface, so Chroma calls it automatically on add()/query().

No paid embedding API (e.g. OpenAI embeddings) is used anywhere.
"""
from __future__ import annotations

from chromadb import Documents, EmbeddingFunction, Embeddings

from app.config import settings

_model = None  # lazy singleton - avoids loading the model until first use


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(settings.embedding_model_name)
    return _model


class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        model = _get_model()
        vectors = model.encode(list(input), convert_to_numpy=True)
        return vectors.tolist()

    def name(self) -> str:
        return f"local:{settings.embedding_model_name}"
