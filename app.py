import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import voyageai
from supabase import create_client
import anthropic
from googleapiclient.discovery import build as build_youtube

from table4_common import (
    find_relevant_chunks,
    ask_claude,
    build_sources,
    fetch_and_index_new_episodes,
    MATCH_COUNT,
)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
UPDATE_SECRET = os.environ.get("UPDATE_SECRET")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
voyage = voyageai.Client()
claude = anthropic.Anthropic()

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Table4 Q&A")


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    episode_title: str
    timestamp: Optional[str] = None
    url: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="שאלה ריקה")

    chunks = find_relevant_chunks(supabase, voyage, question, match_count=MATCH_COUNT)
    if not chunks:
        return AskResponse(answer="לא נמצאו קטעים רלוונטיים בפרקים כדי לענות על השאלה הזו.", sources=[])

    answer = ask_claude(claude, question, chunks)
    sources = build_sources(chunks)
    return AskResponse(answer=answer, sources=sources)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/update-index")
def update_index(secret: str = ""):
    """
    נקודת קצה שבודקת אם יש פרק חדש בערוץ שעדיין לא במאגר, ואם כן - שולפת
    את התמלול שלו ומוסיפה אותו. מיועדת להיקרא אוטומטית פעם בשבוע (למשל
    בכל שבת), לא ידנית מהדפדפן.
    """
    if not UPDATE_SECRET:
        raise HTTPException(status_code=500, detail="UPDATE_SECRET לא מוגדר בשרת")
    if secret != UPDATE_SECRET:
        raise HTTPException(status_code=403, detail="קוד סודי שגוי")
    if not YOUTUBE_API_KEY:
        raise HTTPException(status_code=500, detail="YOUTUBE_API_KEY לא מוגדר בשרת")

    youtube = build_youtube("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    results = fetch_and_index_new_episodes(youtube, supabase, voyage)
    return {"checked": True, "new_episodes": results}


# חייב להיות אחרון - כל נתיב שלא נתפס למעלה מוגש כקובץ סטטי (index.html, manifest.json וכו')
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
