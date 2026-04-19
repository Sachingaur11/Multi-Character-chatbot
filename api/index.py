import asyncio
import json
import os
import pathlib
import uuid
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from groq import Groq, AsyncGroq
from pydantic import BaseModel

# Resolve path to the frontend HTML (works from any cwd)
_HTML = pathlib.Path(__file__).parent.parent / "public" / "index.html"

load_dotenv()

# =========================
# Groq Clients
# =========================

# Sync client for character generation (one-shot, blocking is fine)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# Async client for streaming chat responses
async_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# =========================
# App Setup
# =========================

app = FastAPI(title="Multi-Character Chatbot Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Timezone
# =========================

IST = timezone(timedelta(hours=5, minutes=30))


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


# =========================
# Database Setup
# =========================
# Supports two backends:
#   - PostgreSQL (production / Vercel): set DATABASE_URL env var
#   - SQLite (local dev): no DATABASE_URL needed

DATABASE_URL = os.getenv("DATABASE_URL")
USE_POSTGRES = bool(DATABASE_URL)

if not USE_POSTGRES:
    import sqlite3
    DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chat_history.db")


def get_db():
    if USE_POSTGRES:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
        return conn
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn


def init_db():
    """Create tables if they don't exist. Safe to call on every cold start."""
    conn = get_db()
    if USE_POSTGRES:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id    TEXT PRIMARY KEY,
                character     TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                created_at    TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          SERIAL PRIMARY KEY,
                session_id  TEXT NOT NULL REFERENCES sessions(session_id),
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                timestamp   TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            SERIAL PRIMARY KEY,
                session_id    TEXT NOT NULL REFERENCES sessions(session_id),
                name          TEXT NOT NULL,
                email         TEXT NOT NULL,
                character     TEXT NOT NULL,
                ip            TEXT,
                city          TEXT,
                region        TEXT,
                country       TEXT,
                isp           TEXT,
                chat_time_ist TEXT NOT NULL
            )
        """)
        conn.commit()
        cur.close()
    else:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id    TEXT PRIMARY KEY,
                character     TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT NOT NULL,
                name          TEXT NOT NULL,
                email         TEXT NOT NULL,
                character     TEXT NOT NULL,
                ip            TEXT,
                city          TEXT,
                region        TEXT,
                country       TEXT,
                isp           TEXT,
                chat_time_ist TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
        """)
        conn.commit()
    conn.close()


init_db()


# =========================
# DB Helpers
# =========================

def db_save_session(session_id: str, character: str, system_prompt: str):
    conn = get_db()
    cur = conn.cursor()
    if USE_POSTGRES:
        cur.execute(
            """INSERT INTO sessions (session_id, character, system_prompt, created_at)
               VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING""",
            (session_id, character, system_prompt, now_utc()),
        )
    else:
        cur.execute(
            """INSERT OR IGNORE INTO sessions (session_id, character, system_prompt, created_at)
               VALUES (?, ?, ?, ?)""",
            (session_id, character, system_prompt, now_utc()),
        )
    conn.commit()
    cur.close()
    conn.close()


def db_save_user(
    session_id: str,
    name: str,
    email: str,
    character: str,
    ip: str,
    location: Dict,
    chat_time_ist: str,
):
    conn = get_db()
    cur = conn.cursor()
    if USE_POSTGRES:
        cur.execute(
            """INSERT INTO users
               (session_id, name, email, character, ip, city, region, country, isp, chat_time_ist)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                session_id, name, email, character, ip,
                location.get("city", ""),
                location.get("region", ""),
                location.get("country", ""),
                location.get("isp", ""),
                chat_time_ist,
            ),
        )
    else:
        cur.execute(
            """INSERT INTO users
               (session_id, name, email, character, ip, city, region, country, isp, chat_time_ist)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, name, email, character, ip,
                location.get("city", ""),
                location.get("region", ""),
                location.get("country", ""),
                location.get("isp", ""),
                chat_time_ist,
            ),
        )
    conn.commit()
    cur.close()
    conn.close()


def db_save_message(session_id: str, role: str, content: str):
    conn = get_db()
    cur = conn.cursor()
    if USE_POSTGRES:
        cur.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (%s, %s, %s, %s)",
            (session_id, role, content, now_utc()),
        )
    else:
        cur.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now_utc()),
        )
    conn.commit()
    cur.close()
    conn.close()


def db_get_session(session_id: str):
    conn = get_db()
    cur = conn.cursor()
    if USE_POSTGRES:
        cur.execute(
            "SELECT session_id, character, system_prompt, created_at FROM sessions WHERE session_id = %s",
            (session_id,),
        )
    else:
        cur.execute(
            "SELECT session_id, character, system_prompt, created_at FROM sessions WHERE session_id = ?",
            (session_id,),
        )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return dict(row) if row else None


def db_get_messages(session_id: str, limit: int = 20) -> List[Dict]:
    """Return the most recent `limit` messages, in chronological order."""
    conn = get_db()
    cur = conn.cursor()
    if USE_POSTGRES:
        cur.execute(
            """SELECT role, content FROM (
                 SELECT role, content, id FROM messages
                 WHERE session_id = %s ORDER BY id DESC LIMIT %s
               ) sub ORDER BY id ASC""",
            (session_id, limit),
        )
    else:
        cur.execute(
            """SELECT role, content FROM (
                 SELECT role, content, id FROM messages
                 WHERE session_id = ? ORDER BY id DESC LIMIT ?
               ) ORDER BY id ASC""",
            (session_id, limit),
        )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


# =========================
# In-Memory Caches
# =========================

HISTORY_LIMIT = 20

_session_cache: Dict[str, Dict] = {}
_messages_cache: Dict[str, deque] = {}


def get_session_cached(session_id: str):
    if session_id not in _session_cache:
        session = db_get_session(session_id)
        if session:
            _session_cache[session_id] = session
    return _session_cache.get(session_id)


def get_messages_cached(session_id: str) -> List[Dict]:
    if session_id not in _messages_cache:
        rows = db_get_messages(session_id, limit=HISTORY_LIMIT)
        _messages_cache[session_id] = deque(rows, maxlen=HISTORY_LIMIT)
    return list(_messages_cache[session_id])


def append_message_to_cache(session_id: str, role: str, content: str):
    if session_id not in _messages_cache:
        _messages_cache[session_id] = deque(maxlen=HISTORY_LIMIT)
    _messages_cache[session_id].append({"role": role, "content": content})


# =========================
# IP & Geolocation Helpers
# =========================

def get_client_ip(request: Request) -> str:
    """Extract real client IP, respecting common proxy headers."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"


_PRIVATE_IP_PREFIXES = (
    "127.", "10.", "192.168.",
    "172.16.", "172.17.", "172.18.", "172.19.",
    "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.",
    "172.28.", "172.29.", "172.30.", "172.31.",
    "::1", "localhost", "unknown",
)


async def get_location(ip: str) -> Dict:
    """
    Lookup geolocation via ip-api.com (free, no API key).
    Returns empty strings for private/local IPs or on failure.
    """
    if any(ip.startswith(p) for p in _PRIVATE_IP_PREFIXES):
        return {"city": "Local", "region": "Local", "country": "Local", "isp": "Local Network"}

    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(
                f"http://ip-api.com/json/{ip}",
                params={"fields": "status,city,regionName,country,isp"},
            )
            d = r.json()
            if d.get("status") == "success":
                return {
                    "city": d.get("city", ""),
                    "region": d.get("regionName", ""),
                    "country": d.get("country", ""),
                    "isp": d.get("isp", ""),
                }
    except Exception:
        pass

    return {"city": "", "region": "", "country": "", "isp": ""}


# =========================
# Character Engine
# =========================

def generate_character_profile(character: str) -> str:
    # Kept intentionally short: this profile is sent to Groq on every chat turn,
    # so every extra token costs TTFT latency.
    prompt = (
        f"Write a concise character profile for {character} in under 80 words. "
        "Cover: speaking style, personality, key beliefs, and one signature quirk. "
        "No headers, plain prose only."
    )
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=120,
    )
    return response.choices[0].message.content


# =========================
# Request Models
# =========================

class CreateSessionRequest(BaseModel):
    name: str
    email: str
    character: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


# =========================
# API Routes
# =========================

router = APIRouter(prefix="/api")


@router.post("/create-session")
async def create_session(req: CreateSessionRequest, request: Request):
    ip = get_client_ip(request)
    chat_time_ist = now_ist()

    # Run character profile generation and IP geolocation concurrently
    profile, location = await asyncio.gather(
        asyncio.to_thread(generate_character_profile, req.character),
        get_location(ip),
    )

    session_id = str(uuid.uuid4())

    system_prompt = f"""You are roleplaying as {req.character}.

Character Profile:
{profile}

Rules:
- Fully stay in character at all times
- Speak according to the personality and style above
- Be immersive and human-like
- Do NOT break character
- Keep responses under 50 words"""

    # Persist session and user details
    db_save_session(session_id, req.character, system_prompt)
    db_save_user(session_id, req.name, req.email, req.character, ip, location, chat_time_ist)

    # Prime caches — first chat message hits neither DB
    _session_cache[session_id] = {
        "session_id": session_id,
        "character": req.character,
        "system_prompt": system_prompt,
    }
    _messages_cache[session_id] = deque(maxlen=HISTORY_LIMIT)

    return {
        "status": "created",
        "session_id": session_id,
        "character": req.character,
    }


@router.post("/chat")
async def chat(req: ChatRequest):
    session = get_session_cached(req.session_id)
    if not session:
        return {"error": "Session not found"}

    # Served entirely from RAM — no DB query on the hot path
    history = get_messages_cached(req.session_id)

    messages = [{"role": "system", "content": session["system_prompt"]}]
    messages.extend(history)
    messages.append({"role": "user", "content": req.message})

    # Update RAM cache immediately (no DB round-trip before LLM call)
    append_message_to_cache(req.session_id, "user", req.message)

    # Persist to DB in background — does NOT block the stream from starting
    asyncio.create_task(
        asyncio.to_thread(db_save_message, req.session_id, "user", req.message)
    )

    async def generate():
        full_reply: List[str] = []
        try:
            stream = await async_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.8,
                max_tokens=150,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_reply.append(delta)
                    yield f"data: {json.dumps({'token': delta})}\n\n"

            complete_reply = "".join(full_reply)
            append_message_to_cache(req.session_id, "assistant", complete_reply)
            asyncio.create_task(
                asyncio.to_thread(db_save_message, req.session_id, "assistant", complete_reply)
            )
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/sessions")
def list_sessions():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT session_id FROM sessions ORDER BY created_at DESC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {"sessions": [dict(r)["session_id"] for r in rows]}


@router.get("/users")
def list_users():
    """Return all tracked user records."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, session_id, name, email, character, ip, city, region, country, isp, chat_time_ist
           FROM users ORDER BY id DESC"""
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {"users": [dict(r) for r in rows]}


@router.get("/history")
def list_history():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT session_id, character, created_at FROM sessions ORDER BY created_at DESC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {"sessions": [dict(r) for r in rows]}


@router.get("/history/{session_id}")
def get_history(session_id: str):
    session = db_get_session(session_id)
    if not session:
        return {"error": "Session not found"}

    history = db_get_messages(session_id, limit=100)
    return {
        "session_id": session["session_id"],
        "character": session["character"],
        "created_at": session["created_at"],
        "messages": history,
    }


@app.get("/")
def serve_frontend():
    return FileResponse(_HTML)


app.include_router(router)
