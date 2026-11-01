"""Step 1: Speech to text. Returns a list of segments: {"start", "end", "text"}."""
from . import config


def transcribe(audio_path: str, language: str | None = None) -> list[dict]:
    """language: None = auto-detect, or "en", "hi", "kn"."""
    if config.ASR_BACKEND == "local":
        return _local(audio_path, language)
    return _groq(audio_path, language)


def _groq(audio_path, language):
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)
    with open(audio_path, "rb") as f:
        kwargs = dict(
            file=(audio_path, f.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    if language:
        kwargs["language"] = language
    res = client.audio.transcriptions.create(**kwargs)
    segs = res.segments if hasattr(res, "segments") else res["segments"]
    out = []
    for s in segs:
        g = (lambda k: s[k] if isinstance(s, dict) else getattr(s, k))
        out.append({"start": float(g("start")), "end": float(g("end")), "text": g("text").strip()})
    return out


_local_model = None


def _local(audio_path, language):
    global _local_model
    from faster_whisper import WhisperModel

    if _local_model is None:
        _local_model = WhisperModel(config.LOCAL_WHISPER_SIZE, device="cpu", compute_type="int8")
    segments, _ = _local_model.transcribe(audio_path, language=language, vad_filter=True)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
