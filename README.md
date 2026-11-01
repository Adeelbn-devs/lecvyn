# LECVYN: AI Lecture Assistant (starter code)

Lecture (audio / pasted text) → speech to text → 1-minute chunks → off-topic filter → notes, quiz, flashcards, doubt solver, YouTube videos. Web app in Streamlit.

## Setup (once)
1. Install Python 3.10 or newer (python.org). Tick "Add to PATH".
2. Open a terminal in this folder and run:
   ```
   python -m venv venv
   venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (Mac/Linux)
   pip install -r requirements.txt
   ```
3. Get a free Groq key at https://console.groq.com → API Keys. Copy `.env.example` to `.env` and paste the key.
   (Optional) YouTube key: Google Cloud Console → enable "YouTube Data API v3" → create API key. Without it, the Videos tab gives YouTube search links.
4. Run: `streamlit run app.py`  (opens in the browser).

## First test (no audio needed)
Sidebar → "Paste transcript" → paste `sample_transcript.txt` → topic: `Database normalization` → Process lecture.
The cricket part should turn red (removed). Then try Notes, Quiz, Ask, Videos.
First run downloads the embedding model (LaBSE ~1.8 GB). For a faster, smaller test set `EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` in `.env`.

## Run the tests
`pip install pytest` then `pytest -q tests`

## Files
| File | Job |
|---|---|
| `lecvyn/asr.py` | speech to text (Whisper via Groq, or local faster-whisper) |
| `lecvyn/chunker.py` | split into ~1 minute chunks |
| `lecvyn/offtopic.py` | off-topic filter (embeddings + cosine similarity) |
| `lecvyn/store.py` | FAISS store + search |
| `lecvyn/llm.py` | notes, quiz, flashcards, doubt answers, video queries |
| `lecvyn/videos.py` | YouTube suggestions |
| `app.py` | Streamlit web app |

## Never upload `.env` to GitHub (it has your keys).
