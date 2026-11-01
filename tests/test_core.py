import numpy as np
from lecvyn.chunker import make_chunks, text_to_chunks
from lecvyn.offtopic import filter_chunks
from lecvyn.store import ChunkStore


def fake_embed(texts):
    # tiny fake: "database" words -> axis 0, everything else -> axis 1
    out = []
    for t in texts:
        v = np.array([1.0, 0.0]) if any(w in t.lower() for w in ["database", "table", "sql", "normal"]) else np.array([0.0, 1.0])
        out.append(v)
    return np.array(out, dtype="float32")


def test_chunker():
    segs = [{"start": i * 10, "end": i * 10 + 10, "text": f"s{i}"} for i in range(13)]
    ch = make_chunks(segs, 60)
    assert len(ch) == 3 and ch[0]["start"] == 0 and ch[1]["id"] == 1


def test_text_chunks():
    assert len(text_to_chunks("word " * 250, 120)) == 3


def test_filter_and_store():
    ch = text_to_chunks("database table normal forms", 10) + text_to_chunks("did you watch the cricket match yesterday", 10)
    ch = filter_chunks(ch, fake_embed, topic="database normalization", threshold=0.5)
    assert [c["on_topic"] for c in ch] == [True, False]
    kept = [c for c in ch if c["on_topic"]]
    hits = ChunkStore(kept).search(fake_embed(["sql table"])[0], k=2)
    assert hits and hits[0][0]["text"].startswith("database")


def test_filter_without_topic_runs():
    ch = filter_chunks(text_to_chunks("database table " * 30, 10), fake_embed, topic="", threshold=0.1)
    assert all("score" in c for c in ch)
