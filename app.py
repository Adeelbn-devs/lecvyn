"""LECVYN - AI Lecture Assistant. Run:  streamlit run app.py"""
import tempfile
import streamlit as st

from lecvyn import config, chunker, offtopic, llm, videos, dashboard
from lecvyn.store import ChunkStore

st.set_page_config(page_title="LECVYN", page_icon="🎓", layout="wide")
st.title("🎓 LECVYN: AI Lecture Assistant")
st.caption("Lecture → transcript → off-topic filter → notes, quiz, doubts, videos (English / Hindi / Kannada)")

if not config.GROQ_API_KEY:
    st.warning("GROQ_API_KEY is missing. Copy .env.example to .env and add your key, then restart.")

# ---------- Sidebar: input ----------
with st.sidebar:
    st.header("1. Lecture input")
    mode = st.radio("Source", ["Upload audio/video", "Paste transcript"])
    topic = st.text_input("Lecture topic (helps the off-topic filter)", placeholder="e.g. Database normalization")
    lang = st.selectbox("Language", ["Auto-detect", "English", "Hindi", "Kannada"])
    threshold = st.slider("Off-topic threshold", 0.0, 0.8, 0.25, 0.01,
                          help="Higher = stricter. Chunks below this score are removed.")
    audio = st.file_uploader("Audio / video file", type=["mp3", "wav", "m4a", "mp4", "webm", "ogg"]) \
        if mode == "Upload audio/video" else None
    pasted = st.text_area("Transcript text", height=200) if mode == "Paste transcript" else ""
    go = st.button("Process lecture", type="primary")

LANG_CODE = {"Auto-detect": None, "English": "en", "Hindi": "hi", "Kannada": "kn"}


@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def get_embed():
    from lecvyn.embedder import embed
    embed(["warm up"])
    return embed


if go:
    try:
        with st.spinner("Step 1/3: speech to text..."):
            if mode == "Upload audio/video":
                if audio is None:
                    st.error("Please upload a file."); st.stop()
                from lecvyn import asr
                with tempfile.NamedTemporaryFile(delete=False, suffix="_" + audio.name) as f:
                    f.write(audio.read())
                segs = asr.transcribe(f.name, LANG_CODE[lang])
                chunks = chunker.make_chunks(segs, config.CHUNK_SECONDS)
            else:
                if not pasted.strip():
                    st.error("Please paste a transcript."); st.stop()
                chunks = chunker.text_to_chunks(pasted)
        with st.spinner("Step 2/3: off-topic filter..."):
            embed = get_embed()
            chunks = offtopic.filter_chunks(chunks, embed, topic, threshold)
        st.session_state.update(chunks=chunks, embed=embed, notes=None, quiz=None, cards=None, vids=None, chat=[])
    except Exception as e:
        st.error(f"Something went wrong: {e}")

chunks = st.session_state.get("chunks")
if not chunks:
    st.info("Upload a lecture (or paste a transcript) in the sidebar and click **Process lecture**.")
    st.stop()

kept = [c for c in chunks if c["on_topic"]]
text = "\n".join(c["text"] for c in kept)
store = ChunkStore(kept)
dashboard.render(chunks, threshold)

t_tr, t_notes, t_quiz, t_ask, t_vid = st.tabs(["Transcript", "Notes", "Quiz & Flashcards", "Ask a doubt", "Videos"])

with t_tr:
    st.write("Green = kept, red = removed as off-topic. You can tune the threshold in the sidebar and process again.")
    for c in chunks:
        icon = "🟢" if c["on_topic"] else "🔴"
        st.markdown(f"{icon} **score {c['score']:.2f}** — {c['text']}")

with t_notes:
    if st.button("Generate notes"):
        with st.spinner("Writing notes..."):
            st.session_state.notes = llm.make_notes(text)
    if st.session_state.get("notes"):
        st.markdown(st.session_state.notes)
        st.download_button("Download notes (.md)", st.session_state.notes, "lecvyn_notes.md")

with t_quiz:
    c1, c2 = st.columns(2)
    if c1.button("Generate quiz"):
        with st.spinner("Making quiz..."):
            try: st.session_state.quiz = llm.make_quiz(text)
            except Exception as e: st.error(f"Quiz failed, try again: {e}")
    if c2.button("Generate flashcards"):
        with st.spinner("Making flashcards..."):
            try: st.session_state.cards = llm.make_flashcards(text)
            except Exception as e: st.error(f"Flashcards failed, try again: {e}")
    for i, q in enumerate(st.session_state.get("quiz") or []):
        st.markdown(f"**Q{i+1}. {q['question']}**")
        pick = st.radio("Choose", q["options"], index=None, key=f"q{i}", label_visibility="collapsed")
        if pick is not None:
            ok = q["options"].index(pick) == int(q["answer"])
            (st.success if ok else st.error)(("Correct! " if ok else f"Wrong. Answer: {q['options'][int(q['answer'])]}. ") + q.get("explanation", ""))
    for i, cd in enumerate(st.session_state.get("cards") or []):
        with st.expander(f"Card {i+1}: {cd['front']}"):
            st.write(cd["back"])

with t_ask:
    for role, msg in st.session_state.get("chat", []):
        st.chat_message(role).write(msg)
    q = st.chat_input("Ask a doubt about this lecture (any language)")
    if q:
        st.chat_message("user").write(q)
        hits = store.search(st.session_state.embed([q])[0], k=3)
        ans = llm.answer_doubt(q, [h[0]["text"] for h in hits])
        st.chat_message("assistant").write(ans)
        st.session_state.chat += [("user", q), ("assistant", ans)]

with t_vid:
    if st.button("Suggest videos"):
        with st.spinner("Finding videos..."):
            res = []
            for qq in llm.video_queries(text):
                res.append((qq, videos.search_videos(qq)))
            st.session_state.vids = res
    for qq, vs in st.session_state.get("vids") or []:
        st.markdown(f"**{qq}**")
        for v in vs:
            st.markdown(f"- [{v['title']}]({v['url']})")
