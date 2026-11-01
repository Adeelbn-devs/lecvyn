"""Step 3: Off-topic filter using multilingual embeddings + cosine similarity."""
import numpy as np


def filter_chunks(chunks, embed_fn, topic: str = "", threshold: float = 0.25):
    """Adds 'score' and 'on_topic' to every chunk.

    - If a topic is given: score = cosine(chunk, topic).
    - If not: score = cosine(chunk, centre of the whole lecture).
    """
    if not chunks:
        return chunks
    vecs = embed_fn([c["text"] for c in chunks])
    if topic.strip():
        ref = embed_fn([topic])[0]
    else:
        ref = vecs.mean(axis=0)
        ref = ref / (np.linalg.norm(ref) + 1e-9)
    scores = vecs @ ref
    for c, v, s in zip(chunks, vecs, scores):
        c["vec"] = v
        c["score"] = float(s)
        c["on_topic"] = bool(s >= threshold)
    return chunks
