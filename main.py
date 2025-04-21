# main.py
import os
import streamlit as st
import torch
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIG ===
EMBEDDING_FILE = "embeddings.npy"
CHUNK_FILE = "chunks.json"
GEMINI_API_KEY = "AIzaSyCfyvK6MhIyr4Dc0BrJg3T2C1N05EB1Wy0"
GEMINI_MODEL = "gemini-2.0-flash"
TOP_K = 3
SIM_THRESHOLD = 0.4
CUSTOM_PROMPT = (
    "You are a helpful maintenance assistant. Use the provided context to answer clearly and simply. "
    "If you don't know, say so. Give step-by-step solutions to problems. Keep answers precise and helpful."
)

# === LOAD MODELS AND DATA ===
embedder = SentenceTransformer("thenlper/gte-large")
if not (os.path.exists(EMBEDDING_FILE) and os.path.exists(CHUNK_FILE)):
    st.error("Required data files (embeddings.npy or chunks.json) are missing. Please upload them.")
    st.stop()
chunks = json.load(open(CHUNK_FILE, "r", encoding="utf-8"))
embeddings = np.load(EMBEDDING_FILE)
client = genai.Client(api_key=GEMINI_API_KEY)

# === SEARCH CHUNKS ===
def search_chunks(query):
    query_embedding = embedder.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = scores.argsort()[-TOP_K:][::-1]
    return [chunks[i] for i in top_indices], max(scores)

# === GEMINI CALL ===
def answer_with_gemini(question, context=None):
    parts = [types.Part.from_text(text=f"System Prompt:\n{CUSTOM_PROMPT}")]
    if context:
        parts.append(types.Part.from_text(text=f"Context:\n{context}"))
    parts.append(types.Part.from_text(text=question))

    contents = [types.Content(role="user", parts=parts)]
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        response_mime_type="text/plain"
    )

    try:
        response = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=contents,
            config=config,
        )
        return "".join(chunk.text for chunk in response).strip()
    except Exception as e:
        return f"Error: {e}"

# === STREAMLIT APP ===
st.set_page_config(page_title="🧠 GMS Maintenance Assistant", layout="wide")
st.title("🔧 Maintenance Chatbot")

# === Init session state ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "👋 Hello! I’m your GMS maintenance assistant. Ask me anything about inspections, issues, or fixes."}
    ]


# === Display chat history ===
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === Input box ===
user_input = st.chat_input("Ask a maintenance question...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Retrieve relevant context
    top_chunks, score = search_chunks(user_input)
    context = "\n\n".join(top_chunks)

    # Generate response
    if score > SIM_THRESHOLD:
        answer = answer_with_gemini(user_input, context=context)
        response_display = f"**✅ Based on uploaded data:**\n\n{answer}"
    else:
        answer = "🤖 I couldn't find a good match in the uploaded data."
        response_display = answer

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(response_display)

    # Save response in history
    st.session_state.chat_history.append({"role": "assistant", "content": response_display})

    # Save last Q for fallback
    st.session_state["last_question"] = user_input

# === Optional Gemini Web Search Button ===
st.markdown("---")
if st.button("🌐 Want to try web search?"):
    if "last_question" in st.session_state:
        with st.chat_message("assistant"):
            st.markdown("🔍 Searching on web ...")
            web_answer = answer_with_gemini(st.session_state["last_question"])
            st.markdown(f"**🌐Web Answer:**\n\n{web_answer}")
        st.session_state.chat_history.append({"role": "assistant", "content": f"🌐 Web Answer:\n\n{web_answer}"})
