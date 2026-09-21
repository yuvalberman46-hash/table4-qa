"""
מודול משותף לפרויקט Table4 Q&A.
משמש גם את הסקריפטים שרצים מהמחשב (fetch_episodes.py, build_index.py, ask.py)
וגם את שרת האפליקציה (app.py) - כדי לא לשכפל לוגיקה בין שני המקומות.
"""

import re
import glob
import tempfile
import time

import yt_dlp
from voyageai.error import (
    RateLimitError,
    APIConnectionError,
    APIError,
    Timeout,
    ServerError,
    ServiceUnavailableError,
)

CHANNEL_HANDLE = "Table4pod"
HEBREW_LANGUAGE_CODES = ["iw", "he"]
EPISODE_TITLE_PATTERN = re.compile(r"פרק\s*\d+", re.IGNORECASE)

EMBED_MODEL = "voyage-4"
CLAUDE_MODEL = "claude-sonnet-5"
CHUNK_WORDS = 350
CHUNK_OVERLAP = 50
EMBED_BATCH_SIZE = 5             # מוקטן בגלל מגבלת 10K TPM של Voyage בחינמי
SLEEP_BETWEEN_EMBED_CALLS = 21   # בגלל מגבלת 3 קריאות בדקה של Voyage בחינמי
MATCH_COUNT = 8


# ---------------------------------------------------------------------------
# יוטיוב: רשימת פרקים
# ---------------------------------------------------------------------------

def get_uploads_playlist_id(youtube, handle: str = CHANNEL_HANDLE) -> str:
    response = youtube.channels().list(part="contentDetails", forHandle=handle).execute()
    items = response.get("items", [])
    if not items:
        raise RuntimeError(f"לא נמצא ערוץ עבור הכינוי {handle}")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def list_all_videos(youtube, playlist_id: str) -> list[dict]:
    videos = []
    next_page_token = None
    while True:
        response = youtube.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token,
        ).execute()
        for item in response.get("items", []):
            snippet = item["snippet"]
            video_id = snippet["resourceId"]["videoId"]
            videos.append({
                "video_id": video_id,
                "title": snippet["title"],
                "published_at": snippet["publishedAt"],
            })
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return videos


def list_episodes(youtube, channel_handle: str = CHANNEL_HANDLE) -> list[dict]:
    playlist_id = get_uploads_playlist_id(youtube, channel_handle)
    all_videos = list_all_videos(youtube, playlist_id)
    return [v for v in all_videos if EPISODE_TITLE_PATTERN.search(v["title"])]


# ---------------------------------------------------------------------------
# פענוח VTT -> טקסט + חותמות זמן
# ---------------------------------------------------------------------------

def _clean_vtt_line(line: str) -> str:
    return re.sub(r"<[^>]+>", "", line).strip()


def _parse_vtt_timestamp(ts: str) -> float:
    """ '00:12:34.560' או '12:34.560' -> שניות (float) """
    ts = ts.strip()
    parts = ts.split(":")
    parts = [p.replace(",", ".") for p in parts]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = "0"
        m, s = parts
    else:
        return 0.0
    return int(h) * 3600 + int(m) * 60 + float(s)


def vtt_to_segments(vtt_content: str) -> list[dict]:
    """
    מפענח VTT (כתוביות אוטומטיות מתגלגלות של יוטיוב) לרשימת קטעים
    {"start": שניות, "text": טקסט}, תוך מיזוג החפיפות בין כתוביות עוקבות
    (יוטיוב שולח כל כתובית עם חלק מהמילים של הקודמת) - כל מילה משויכת
    לזמן ההופעה הראשון שלה בלבד.
    """
    lines = vtt_content.splitlines()
    cues = []  # list of (start_seconds, [text_lines])
    current_start = None
    current_lines = []

    for line in lines:
        if "-->" in line:
            if current_lines and current_start is not None:
                cues.append((current_start, current_lines))
            current_lines = []
            start_str = line.split("-->")[0].strip()
            current_start = _parse_vtt_timestamp(start_str)
            continue
        if line.strip() == "" or line.strip().upper() == "WEBVTT" or line.strip().isdigit():
            if current_lines and current_start is not None:
                cues.append((current_start, current_lines))
                current_lines = []
                current_start = None
            continue
        current_lines.append(_clean_vtt_line(line))

    if current_lines and current_start is not None:
        cues.append((current_start, current_lines))

    cue_records = []
    for start, cue_lines in cues:
        text = " ".join(l for l in cue_lines if l)
        if text:
            cue_records.append((start, text))

    if not cue_records:
        return []

    segments = []
    merged_words = cue_records[0][1].split()
    if merged_words:
        segments.append({"start": cue_records[0][0], "text": " ".join(merged_words)})

    for start, cue_text in cue_records[1:]:
        words = cue_text.split()
        max_overlap = min(len(merged_words), len(words), 15)
        overlap_len = 0
        for k in range(max_overlap, 0, -1):
            if merged_words[-k:] == words[:k]:
                overlap_len = k
                break
        new_words = words[overlap_len:]
        merged_words.extend(new_words)
        if new_words:
            segments.append({"start": start, "text": " ".join(new_words)})

    return segments


def segments_to_text(segments: list[dict]) -> str:
    return " ".join(seg["text"] for seg in segments)


# ---------------------------------------------------------------------------
# יוטיוב: הורדת תמלול
# ---------------------------------------------------------------------------

def fetch_hebrew_transcript(video_id: str, proxy_url: str | None = None) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        import os
        outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": HEBREW_LANGUAGE_CODES,
            "subtitlesformat": "vtt",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "retries": 3,
        }
        if proxy_url:
            ydl_opts["proxy"] = proxy_url

        url = f"https://www.youtube.com/watch?v={video_id}"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        vtt_files = glob.glob(os.path.join(tmpdir, f"{video_id}*.vtt"))
        if not vtt_files:
            return {"has_transcript": False, "transcript": "", "segments": []}

        with open(vtt_files[0], "r", encoding="utf-8") as f:
            vtt_content = f.read()

        segments = vtt_to_segments(vtt_content)
        text = segments_to_text(segments)
        return {"has_transcript": bool(text), "transcript": text, "segments": segments}


# ---------------------------------------------------------------------------
# חלוקה לקטעים (chunking) עם / בלי חותמות זמן
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """חלוקה ישנה, בלי חותמות זמן - שימוש כגיבוי כשאין segments."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_words - overlap)
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_words]))
        if start + chunk_words >= len(words):
            break
        start += step
    return chunks


def chunk_segments(segments: list[dict], chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """חלוקה לקטעים תוך שמירת חותמת הזמן של תחילת כל קטע."""
    words_with_time = []
    for seg in segments:
        for w in seg["text"].split():
            words_with_time.append((w, seg["start"]))

    if not words_with_time:
        return []

    chunks = []
    step = max(1, chunk_words - overlap)
    start = 0
    while start < len(words_with_time):
        window = words_with_time[start:start + chunk_words]
        text = " ".join(w for w, _ in window)
        start_seconds = window[0][1] if window else None
        chunks.append({"text": text, "start_seconds": start_seconds})
        if start + chunk_words >= len(words_with_time):
            break
        start += step
    return chunks


def chunks_for_video(video: dict, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """מחזיר [{"text":..., "start_seconds": float|None}, ...] בין אם יש segments ובין אם רק טקסט שטוח."""
    segments = video.get("segments")
    if segments:
        return chunk_segments(segments, chunk_words, overlap)
    text = video.get("transcript", "")
    return [{"text": t, "start_seconds": None} for t in chunk_text(text, chunk_words, overlap)]


# ---------------------------------------------------------------------------
# תצוגת זמן / קישורים
# ---------------------------------------------------------------------------

def format_timestamp(seconds) -> str | None:
    if seconds is None:
        return None
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def youtube_timestamp_url(video_id: str, start_seconds) -> str:
    if video_id is None:
        return ""
    if start_seconds is None:
        return f"https://www.youtube.com/watch?v={video_id}"
    return f"https://www.youtube.com/watch?v={video_id}&t={int(start_seconds)}s"


# ---------------------------------------------------------------------------
# Voyage embeddings, עם ניסיונות חוזרים
# ---------------------------------------------------------------------------

def embed_with_retry(voyage_client, batch: list[str], model: str = EMBED_MODEL,
                      input_type: str = "document", max_retries: int = 5):
    for attempt in range(1, max_retries + 1):
        try:
            return voyage_client.embed(batch, model=model, input_type=input_type)
        except RateLimitError:
            if attempt == max_retries:
                raise
            print("    (מגבלת קצב - ממתין 60 שניות ומנסה שוב...)")
            time.sleep(60)
        except (APIConnectionError, Timeout) as e:
            if attempt == max_retries:
                raise
            print(f"    (בעיית חיבור/רשת זמנית - {e}. ממתין 30 שניות, ניסיון {attempt}/{max_retries}...)")
            time.sleep(30)
        except (APIError, ServerError, ServiceUnavailableError) as e:
            if attempt == max_retries:
                raise
            print(f"    (שגיאת שרת זמנית - {e}. ממתין 30 שניות, ניסיון {attempt}/{max_retries}...)")
            time.sleep(30)


# ---------------------------------------------------------------------------
# Supabase: כתיבה
# ---------------------------------------------------------------------------

def episode_already_indexed(supabase, video_id: str) -> bool:
    result = supabase.table("episodes").select("id").eq("video_id", video_id).execute()
    if not result.data:
        return False
    episode_id = result.data[0]["id"]
    chunks_result = (
        supabase.table("transcript_chunks")
        .select("id", count="exact")
        .eq("episode_id", episode_id)
        .execute()
    )
    return (chunks_result.count or 0) > 0


def episode_exists(supabase, video_id: str) -> bool:
    result = supabase.table("episodes").select("id").eq("video_id", video_id).execute()
    return bool(result.data)


def upsert_episode(supabase, video: dict) -> int:
    existing = supabase.table("episodes").select("id").eq("video_id", video["video_id"]).execute()
    if existing.data:
        return existing.data[0]["id"]
    inserted = supabase.table("episodes").insert({
        "video_id": video["video_id"],
        "title": video["title"],
        "published_at": video.get("published_at"),
    }).execute()
    return inserted.data[0]["id"]


def index_episode(supabase, voyage_client, video: dict,
                   chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP,
                   embed_batch_size: int = EMBED_BATCH_SIZE,
                   sleep_between_embed: float = SLEEP_BETWEEN_EMBED_CALLS,
                   log=print) -> int:
    """מפצל, עושה embedding ומעלה ל-Supabase עבור פרק אחד. מחזיר כמות הקטעים שהועלו."""
    if not video.get("has_transcript") or not video.get("transcript"):
        log("    אין תמלול - מדלג.")
        return 0

    episode_id = upsert_episode(supabase, video)
    chunks = chunks_for_video(video, chunk_words, overlap)
    if not chunks:
        log("    אין תוכן לפיצול - מדלג.")
        return 0

    log(f"    {len(chunks)} קטעים לעיבוד.")
    all_rows = []
    for batch_start in range(0, len(chunks), embed_batch_size):
        batch = chunks[batch_start:batch_start + embed_batch_size]
        batch_num = batch_start // embed_batch_size + 1
        total_batches = (len(chunks) + embed_batch_size - 1) // embed_batch_size
        log(f"    מבצע embedding לקבוצה {batch_num}/{total_batches}...")

        texts = [c["text"] for c in batch]
        result = embed_with_retry(voyage_client, texts)
        for offset, embedding in enumerate(result.embeddings):
            chunk_index = batch_start + offset
            chunk = chunks[chunk_index]
            all_rows.append({
                "episode_id": episode_id,
                "chunk_index": chunk_index,
                "content": chunk["text"],
                "embedding": embedding,
                "start_seconds": chunk["start_seconds"],
            })

        time.sleep(sleep_between_embed)

    if all_rows:
        supabase.table("transcript_chunks").insert(all_rows).execute()
    log(f"    הושלם - {len(all_rows)} קטעים נשמרו.")
    return len(all_rows)


def fetch_and_index_new_episodes(youtube, supabase, voyage_client,
                                  channel_handle: str = CHANNEL_HANDLE,
                                  proxy_url: str | None = None,
                                  max_new: int = 3,
                                  sleep_between_fetch: float = 5.0,
                                  log=print) -> list[dict]:
    """בודק אם יש פרקים חדשים בערוץ שעוד לא במסד הנתונים, ומוסיף אותם."""
    episodes = list_episodes(youtube, channel_handle)
    new_videos = [v for v in episodes if not episode_exists(supabase, v["video_id"])]
    new_videos = new_videos[:max_new]

    results = []
    for video in new_videos:
        log(f"פרק חדש נמצא: {video['title']}")
        try:
            transcript_data = fetch_hebrew_transcript(video["video_id"], proxy_url=proxy_url)
            video_record = dict(video)
            video_record.update(transcript_data)
            if not video_record.get("has_transcript"):
                log("    אין תמלול עברי זמין עדיין - מדלג (ינוסה שוב בשבוע הבא).")
                results.append({"video_id": video["video_id"], "title": video["title"], "status": "no_transcript"})
                continue
            n_chunks = index_episode(supabase, voyage_client, video_record, log=log)
            results.append({"video_id": video["video_id"], "title": video["title"], "status": "indexed", "chunks": n_chunks})
        except Exception as e:
            log(f"    שגיאה בעיבוד הפרק: {e}")
            results.append({"video_id": video["video_id"], "title": video["title"], "status": "error", "error": str(e)})
        time.sleep(sleep_between_fetch)

    return results


# ---------------------------------------------------------------------------
# חיפוש סמנטי + תשובה
# ---------------------------------------------------------------------------

def find_relevant_chunks(supabase, voyage_client, question: str,
                          match_count: int = MATCH_COUNT, embed_model: str = EMBED_MODEL) -> list[dict]:
    query_embedding = voyage_client.embed([question], model=embed_model, input_type="query").embeddings[0]
    result = supabase.rpc("match_chunks", {
        "query_embedding": query_embedding,
        "match_count": match_count,
    }).execute()
    return result.data


def build_prompt(question: str, chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        ts = format_timestamp(c.get("start_seconds"))
        header = f"[פרק: {c['episode_title']}"
        if ts:
            header += f", דקה: {ts}"
        header += "]"
        parts.append(f"{header}\n{c['content']}")
    context = "\n\n---\n\n".join(parts)

    return f"""להלן קטעים מתוך תמלולי פרקים של פודקאסט, עם ציון הפרק ולעיתים גם הדקה שבה הקטע מתחיל.
ענה על השאלה בעברית, רק על סמך הקטעים האלה.
עבור כל פרט בתשובה, ציין בסוגריים מאיזה פרק הוא מגיע, ואם צוינה דקה - ציין גם אותה, למשל: "(פרק 42, דקה 12:34)".
אם התשובה לא נמצאת בקטעים - אמור זאת בפירוש, אל תמציא.

קטעים:
{context}

שאלה: {question}
"""


def ask_claude(claude_client, question: str, chunks: list[dict], model: str = CLAUDE_MODEL) -> str:
    prompt = build_prompt(question, chunks)
    response = claude_client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    # response.content יכול לכלול גם בלוקים אחרים (כמו "thinking") לפני
    # בלוק הטקסט בפועל - לא תמיד content[0] הוא הטקסט. מחפשים את בלוק
    # הטקסט הראשון בפירוש במקום להניח שהוא הראשון ברשימה.
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def build_sources(chunks: list[dict]) -> list[dict]:
    seen = set()
    sources = []
    for c in chunks:
        key = (c.get("episode_title"), c.get("start_seconds"))
        if key in seen:
            continue
        seen.add(key)
        ts = format_timestamp(c.get("start_seconds"))
        video_id = c.get("video_id")
        url = youtube_timestamp_url(video_id, c.get("start_seconds")) if video_id else None
        sources.append({
            "episode_title": c.get("episode_title"),
            "timestamp": ts,
            "url": url,
        })
    return sources
