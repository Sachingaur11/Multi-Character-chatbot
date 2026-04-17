import os
import sqlite3
import json
from datetime import datetime, timezone
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
from dotenv import load_dotenv
from groq import Groq
import time

# Load env
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="Multi-Character Chatbot Agent")

# =========================
# Database Setup
# =========================

DB_PATH = os.path.join(os.path.dirname(__file__), "chat_history.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            character    TEXT NOT NULL,
            created_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT NOT NULL,
            role         TEXT NOT NULL,
            content      TEXT NOT NULL,
            timestamp    TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
    """)
    conn.commit()
    conn.close()

init_db()

def db_save_session(session_id: str, character: str):
    conn = get_db()
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, character, created_at) VALUES (?, ?, ?)",
        (session_id, character, now())
    )
    conn.commit()
    conn.close()

def db_save_message(session_id: str, role: str, content: str):
    conn = get_db()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, now())
    )
    conn.commit()
    conn.close()

def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# =========================
# Memory Store (in-memory)
# =========================

chat_sessions: Dict[str, List[Dict[str, str]]] = {}

# =========================
# Character Engine
# =========================

def generate_character_profile(character: str) -> str:
    prompt = f"""
Provide a detailed character profile for {character}.

Include:
- Background / history
- Personality traits
- Speaking style (tone, vocabulary, quirks)
- Behavior patterns
- Beliefs and motivations

Keep it structured and realistic.
"""

    response = client.chat.completions.create(
        # model="llama-3.1-8b-instant",
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content


def build_system_prompt(character: str) -> str:
    profile = generate_character_profile(character)

    return f"""
You are roleplaying as {character}.

Here is your full character profile:
{profile}

Rules:
- Fully stay in character
- Speak exactly according to the personality and speaking style above
- Maintain consistency with background and beliefs
- Be immersive and human-like
- Do NOT break character
"""

# =========================
# LLM Call
# =========================

def call_llm(messages: List[Dict[str, str]]):
    start_time = time.time()
    response = client.chat.completions.create(
        # model="llama-3.3-70b-versatile",  # Free Groq model
        model="llama-3.1-8b-instant",  # Free Groq model
        messages=messages,
        temperature=0.8,
        max_tokens=1024
    )
    latency = time.time() - start_time
    return response.choices[0].message.content, latency

# =========================
# Request Models
# =========================

class CreateSessionRequest(BaseModel):
    session_id: str
    character: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

# =========================
# API Endpoints
# =========================

@app.post("/create-session")
def create_session(req: CreateSessionRequest):
    profile = generate_character_profile(req.character)

    system_prompt = f"""
    You are roleplaying as {req.character}.

    Character Profile:
    {profile}

    Rules:
    - Fully stay in character
    - Speak exactly according to the personality and speaking style above
    - Maintain consistency with background and beliefs
    - Be immersive and human-like
    - Do NOT break character
    - Do NOT respond more than 50 words per response
    """

    chat_sessions[req.session_id] = [
        {"role": "system", "content": system_prompt}
    ]

    db_save_session(req.session_id, req.character)

    return {"status": "created", "session_id": req.session_id}


@app.post("/chat")
def chat(req: ChatRequest):
    if req.session_id not in chat_sessions:
        return {"error": "Session not found"}

    messages = chat_sessions[req.session_id]

    # Add user message
    messages.append({"role": "user", "content": req.message})
    db_save_message(req.session_id, "user", req.message)

    # Call LLM
    reply, latency = call_llm(messages)

    # Save assistant reply
    messages.append({"role": "assistant", "content": reply})
    db_save_message(req.session_id, "assistant", reply)

    print(f"Latency: {latency} seconds")

    return {"response": reply}


@app.get("/sessions")
def list_sessions():
    return {"sessions": list(chat_sessions.keys())}


@app.get("/history")
def list_history():
    """List all persisted sessions with metadata."""
    conn = get_db()
    rows = conn.execute(
        "SELECT session_id, character, created_at FROM sessions ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return {
        "sessions": [
            {"session_id": r["session_id"], "character": r["character"], "created_at": r["created_at"]}
            for r in rows
        ]
    }


@app.get("/history/{session_id}")
def get_history(session_id: str):
    """Return full chat history for a session as a structured JSON log."""
    conn = get_db()
    session = conn.execute(
        "SELECT session_id, character, created_at FROM sessions WHERE session_id = ?",
        (session_id,)
    ).fetchone()

    if not session:
        conn.close()
        return {"error": "Session not found"}

    messages = conn.execute(
        "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    ).fetchall()
    conn.close()

    return {
        "session_id": session["session_id"],
        "character": session["character"],
        "created_at": session["created_at"],
        "messages": [
            {
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"]
            }
            for row in messages
        ]
    }


# =========================
# CLI Interface (Optional)
# =========================

if __name__ == "__main__":
    print("=== Multi-Character Chatbot CLI ===")
    session_id = input("Enter session id: ")
    character = input("Enter character (e.g. Elon Musk, Pirate, Anime villain): ")

    # Initialize session
    system_prompt = build_system_prompt(character)
    chat_sessions[session_id] = [
        {"role": "system", "content": system_prompt}
    ]
    db_save_session(session_id, character)

    print(f"\nChatting with {character}. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        chat_sessions[session_id].append({"role": "user", "content": user_input})
        db_save_message(session_id, "user", user_input)

        reply, latency = call_llm(chat_sessions[session_id])
        chat_sessions[session_id].append({"role": "assistant", "content": reply})
        db_save_message(session_id, "assistant", reply)

        print("\n")
        print(f"{character}: {reply} (Latency: {latency} seconds)")
        print("\n")
