"""Step 2: Group ASR segments into ~1 minute chunks."""


def make_chunks(segments: list[dict], seconds: int = 60) -> list[dict]:
    chunks, cur, cur_start = [], [], None
    for s in segments:
        if not s["text"].strip():
            continue
        if cur_start is None:
            cur_start = s["start"]
        cur.append(s)
        if s["end"] - cur_start >= seconds:
            chunks.append(_pack(cur, cur_start))
            cur, cur_start = [], None
    if cur:
        chunks.append(_pack(cur, cur_start))
    for i, c in enumerate(chunks):
        c["id"] = i
    return chunks


def _pack(segs, start):
    return {"start": start, "end": segs[-1]["end"], "text": " ".join(x["text"] for x in segs)}


def text_to_chunks(text: str, words_per_chunk: int = 120) -> list[dict]:
    """For pasted transcripts (no timestamps)."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunks.append({"id": len(chunks), "start": 0.0, "end": 0.0,
                       "text": " ".join(words[i:i + words_per_chunk])})
    return chunks
