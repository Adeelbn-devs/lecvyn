import numpy as np
from . import config

_model = None


def embed(texts: list[str]) -> np.ndarray:
    """Normalised sentence embeddings (so dot product = cosine similarity)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBED_MODEL)
    v = _model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return v.astype("float32")
