import json

import httpx
import streamlit as st

API_BASE = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="AI Character Chatbot", layout="wide")
st.title("🎭 Multi-Character AI Chatbot")

# =========================
# Sidebar — Session Setup
# =========================
st.sidebar.header("Start Chatting")

name      = st.sidebar.text_input("Your Name",  placeholder="e.g. Alex")
email     = st.sidebar.text_input("Your Email", placeholder="e.g. alex@example.com")
character = st.sidebar.text_input("Character",  placeholder="e.g. Elon Musk, Pirate…")

if st.sidebar.button("Start Chat"):
    if not (name and email and character):
        st.sidebar.warning("Please fill in all three fields.")
    else:
        try:
            res = httpx.post(
                f"{API_BASE}/create-session",
                json={"name": name, "email": email, "character": character},
                timeout=30.0,
            )
            data = res.json()
            if res.status_code == 200 and data.get("status") == "created":
                st.session_state.session_id = data["session_id"]
                st.session_state.character  = data["character"]
                st.session_state.user_name  = name
                st.session_state.messages   = []
                st.sidebar.success(f"Hi {name}! Session started with {character}.")
            else:
                st.sidebar.error(data.get("detail", "Failed to create session."))
        except Exception as e:
            st.sidebar.error(f"Network error: {e}")

# =========================
# SSE Streaming Helper
# =========================

def stream_reply(session_id: str, message: str):
    """
    Generator that connects to the /api/chat SSE endpoint and
    yields text tokens one by one as they arrive from Groq.
    Compatible with st.write_stream().
    """
    with httpx.Client(timeout=30.0) as client:
        with client.stream(
            "POST",
            f"{API_BASE}/chat",
            json={"session_id": session_id, "message": message},
        ) as response:
            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk
                lines = buffer.split("\n")
                buffer = lines.pop()          # keep any incomplete line for next iteration
                for line in lines:
                    if not line.startswith("data: "):
                        continue
                    try:
                        payload = json.loads(line[6:])
                        if "token" in payload:
                            yield payload["token"]
                        elif "error" in payload:
                            yield f"\n\n⚠️ {payload['error']}"
                    except json.JSONDecodeError:
                        pass

# =========================
# Chat UI
# =========================

if "session_id" in st.session_state:
    st.subheader(f"{st.session_state.user_name} × {st.session_state.character}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render existing history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your message…")

    if user_input:
        # Show user bubble immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Stream assistant reply token-by-token; st.write_stream returns full text
        with st.chat_message("assistant"):
            reply = st.write_stream(
                stream_reply(st.session_state.session_id, user_input)
            )

        st.session_state.messages.append({"role": "assistant", "content": reply})

else:
    st.info("Fill in your details in the sidebar to start chatting.")
