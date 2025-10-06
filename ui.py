import streamlit as st
import requests

st.set_page_config(page_title="AI Agent UI (Flask)", layout="wide")

st.title("🧑‍💻 Meeting Agent ")

# Sidebar for setup + file upload
st.sidebar.header("⚙️ Setup")
api_base = st.sidebar.text_input("Flask API base URL", value="http://127.0.0.1:5000")
st.sidebar.info("All requests will be sent to this Flask backend.")

uploaded = st.sidebar.file_uploader("Upload transcript or audio (wav/mp3/m4a/txt/docx)")

# Store transcript in session
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if uploaded:
    data = uploaded.read()
    filename = uploaded.name or "uploaded"
    ext = filename.split(".")[-1].lower()

    try:
        spinner_container = st.sidebar.empty()
        spinner_container.info("Processing file...")
        files = {"file": (filename, data)}
        if ext in ("wav", "mp3", "m4a"):
            resp = requests.post(f"{api_base}/transcribe", files=files)
        else:
            resp = requests.post(f"{api_base}/read_transcript", files=files)
        
        spinner_container.empty()
        if resp.status_code == 200:
                transcript = resp.json().get("transcript")
                if transcript:
                    st.session_state.transcript = transcript
                    st.sidebar.success("✅ Transcript loaded!")
                else:
                    st.sidebar.error("No transcript found in the response")
        else:
                error_msg = resp.json().get("error", str(resp.status_code))
                st.sidebar.error(f"Failed to process file: {error_msg}")
    except Exception as e:
        st.sidebar.error(f"Error processing file: {str(e)}")

    # Index build
    st.sidebar.subheader("📑 Indexing")
    chunk_size = st.sidebar.number_input(
        "Chunk size (characters)", value=500, min_value=100, max_value=5000, step=50
    )
    if st.sidebar.button("Build Index"):
        resp = requests.post(
            f"{api_base}/index/build",
            json={"transcript": st.session_state.transcript, "chunk_size": chunk_size},
        )
        if resp.status_code == 200:
            st.sidebar.success("Index built successfully!")
        else:
            st.sidebar.error(f"Failed: {resp.status_code} - {resp.text}")

# ---- Chat Section ----
st.subheader("💬 Chat with AI")

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input (sticky at bottom)
if st.session_state.transcript:
    user_query = st.chat_input("Ask about this meeting...")
    if user_query:
        k = 10  # fixed for simplicity, could move to sidebar if you want
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Build context
        resp = requests.post(f"{api_base}/search", json={"query": user_query, "k": k})
        if resp.status_code == 200:
            matches = resp.json().get("matches", [])
            context = "\n\n".join(m.get("chunk", "") for m in matches)
        else:
            context = st.session_state.transcript

        contextualized_query = (
            f"Context from meeting:\n{context}\n\nUser asked: {user_query}\nRespond conversationally and clearly."
        )

        # Get all previous messages and add current context
        messages = []
        for prev_msg in st.session_state.messages[:-1]:  # Exclude the last user message we just added
            if prev_msg["role"] == "user":
                messages.append({"role": "user", "content": prev_msg["content"]})
            else:
                messages.append({"role": "assistant", "content": prev_msg["content"]})
        
        # Add the current contextualized query
        messages.append({"role": "user", "content": contextualized_query})
        
        # Call chat API with full conversation history
        resp = requests.post(
            f"{api_base}/chat",
            json={"messages": messages},
        )
        if resp.status_code == 200:
            answer = resp.json().get("response", "")
        else:
            answer = f"Chat error: {resp.status_code} - {resp.text}"

        # Append assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
else:
    st.info("👆 Upload a transcript or audio file in the sidebar to start chatting.")
