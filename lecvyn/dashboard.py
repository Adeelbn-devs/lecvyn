"""Dashboard: summary numbers + off-topic score chart."""
import pandas as pd
import altair as alt
import streamlit as st


def render(chunks: list[dict], threshold: float):
    kept = [c for c in chunks if c["on_topic"]]
    removed = len(chunks) - len(kept)
    total_min = (chunks[-1]["end"] - chunks[0]["start"]) / 60 if chunks and chunks[-1]["end"] > 0 else None
    avg = sum(c["score"] for c in chunks) / len(chunks)
    words_all = sum(len(c["text"].split()) for c in chunks)
    words_kept = sum(len(c["text"].split()) for c in kept)

    a, b, c_, d = st.columns(4)
    a.metric("Total chunks", len(chunks))
    b.metric("On-topic kept", len(kept))
    c_.metric("Removed (off-topic)", removed)
    d.metric("Lecture length", f"{total_min:.1f} min" if total_min else "n/a")

    e, f, g = st.columns(3)
    e.metric("Kept", f"{100 * len(kept) / len(chunks):.0f}%")
    f.metric("Text reduced by", f"{100 * (1 - words_kept / max(words_all, 1)):.0f}%")
    g.metric("Average score", f"{avg:.2f}")

    df = pd.DataFrame({
        "Chunk": [c["id"] + 1 for c in chunks],
        "Score": [round(c["score"], 3) for c in chunks],
        "Status": ["Kept" if c["on_topic"] else "Removed" for c in chunks],
    })
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X("Chunk:O", title="Chunk number"),
        y=alt.Y("Score:Q", title="Similarity to topic"),
        color=alt.Color("Status:N", scale=alt.Scale(domain=["Kept", "Removed"], range=["#2e9e5b", "#d64545"])),
        tooltip=["Chunk", "Score", "Status"],
    )
    line = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(strokeDash=[6, 4]).encode(y="t:Q")
    st.altair_chart((bars + line).properties(height=260), width="stretch")
    st.caption("Dashed line = threshold. Bars below it are removed as off-topic.")
