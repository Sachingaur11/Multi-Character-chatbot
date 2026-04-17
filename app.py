import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Character Chatbot", layout="wide")
st.title("🎭 Multi-Character AI Chatbot")

# =========================
# Sidebar (Session Setup)
# =========================
st.sidebar.header("Create Session")
session_id = st.sidebar.text_input("Session ID")
character = st.sidebar.text_input("Character")

if st.sidebar.button("Start Chat"):
    if session_id and character:
        res = requests.post(
            f"{API_URL}/create-session",
            json={"session_id": session_id, "character": character}
        )
        if res.status_code == 200:
            st.session_state.session_id = session_id
            st.session_state.messages = []
            st.success(f"Session started with {character}")
        else:
            st.error("Failed to create session")

# =========================
# Chat UI
# =========================
if "session_id" in st.session_state:
    st.subheader(f"Chat Session: {st.session_state.session_id}")

    # Display chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Type your message...")

    if user_input:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Send to backend
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "session_id": st.session_state.session_id,
                "message": user_input
            }
        )

        if response.status_code == 200:
            bot_reply = response.json().get("response", "Error")
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            with st.chat_message("assistant"):
                st.markdown(bot_reply)
        else:
            st.error("Error communicating with backend")

else:
    st.info("Create a session from the sidebar to start chatting.")