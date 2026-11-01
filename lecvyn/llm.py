"""Step 5: Notes, quiz, flashcards and doubt answers using an LLM."""
import json
import re
from . import config

LANG_NOTE = ("The lecture may mix English, Hindi and Kannada. Write the output in simple English "
             "unless the user asks for another language. Keep technical terms in English.")


def _chat(system: str, user: str, temperature: float = 0.3) -> str:
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)
    r = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return r.choices[0].message.content.strip()


def make_notes(text: str) -> str:
    return _chat(
        "You are a careful study-notes writer. " + LANG_NOTE +
        " Use ONLY the lecture text. Do not add facts that are not in it.",
        "Make clean notes in Markdown from this lecture: a title, key points as bullets, "
        "important definitions, and a 3-line summary.\n\nLECTURE:\n" + text,
    )


def _json(s: str):
    m = re.search(r"(\[.*\]|\{.*\})", s, re.S)
    return json.loads(m.group(1) if m else s)


def make_quiz(text: str, n: int = 5) -> list[dict]:
    out = _chat(
        "You write exam questions. " + LANG_NOTE + " Use ONLY the lecture text. Reply with JSON only.",
        f"Create {n} multiple-choice questions from this lecture. Return a JSON list of objects with keys: "
        '"question", "options" (list of 4 strings), "answer" (index 0-3), "explanation".\n\nLECTURE:\n' + text,
        temperature=0.4,
    )
    return _json(out)


def make_flashcards(text: str, n: int = 8) -> list[dict]:
    out = _chat(
        "You make flashcards. " + LANG_NOTE + " Use ONLY the lecture text. Reply with JSON only.",
        f"Create {n} flashcards. Return a JSON list of objects with keys \"front\" and \"back\".\n\nLECTURE:\n" + text,
        temperature=0.4,
    )
    return _json(out)


def answer_doubt(question: str, context_chunks: list[str]) -> str:
    ctx = "\n\n".join(f"[Part {i+1}] {c}" for i, c in enumerate(context_chunks))
    return _chat(
        "You are a teaching assistant. " + LANG_NOTE +
        " Answer ONLY from the lecture parts given. If the answer is not there, say "
        "'This was not covered in the lecture' and then give a short general hint, clearly marked as general.",
        f"LECTURE PARTS:\n{ctx}\n\nSTUDENT QUESTION: {question}\n\nAnswer simply and mention which Part you used.",
    )


def video_queries(text: str, n: int = 3) -> list[str]:
    out = _chat(
        "You write YouTube search queries for students. Reply with a JSON list of strings only.",
        f"Give {n} short YouTube search queries (English) to learn the main topics of this lecture.\n\nLECTURE:\n" + text[:6000],
    )
    return _json(out)
