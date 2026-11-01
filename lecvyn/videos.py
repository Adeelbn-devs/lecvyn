"""Step 6: YouTube video suggestions."""
from urllib.parse import quote_plus
import requests
from . import config


def search_videos(query: str, n: int = 3) -> list[dict]:
    """With a YouTube API key: real results. Without: a search link (still useful)."""
    if not config.YOUTUBE_API_KEY:
        return [{"title": f"Search YouTube: {query}", "url": "https://www.youtube.com/results?search_query=" + quote_plus(query)}]
    r = requests.get(
        "https://www.googleapis.com/youtube/v3/search",
        params={"part": "snippet", "q": query, "type": "video", "maxResults": n, "key": config.YOUTUBE_API_KEY},
        timeout=15,
    )
    r.raise_for_status()
    return [{"title": it["snippet"]["title"], "url": "https://www.youtube.com/watch?v=" + it["id"]["videoId"]}
            for it in r.json().get("items", [])]
