"""Step 4: Store on-topic chunks + vectors in FAISS and search them (used by the doubt solver)."""
import numpy as np


class ChunkStore:
    def __init__(self, chunks: list[dict]):
        import faiss

        self.chunks = chunks
        self.index = None
        if chunks:
            vecs = np.stack([c["vec"] for c in chunks]).astype("float32")
            self.index = faiss.IndexFlatIP(vecs.shape[1])  # inner product == cosine (vectors normalised)
            self.index.add(vecs)

    def search(self, query_vec: np.ndarray, k: int = 3):
        if self.index is None:
            return []
        k = min(k, len(self.chunks))
        scores, ids = self.index.search(query_vec.reshape(1, -1).astype("float32"), k)
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
