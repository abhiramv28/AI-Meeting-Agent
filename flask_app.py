from flask import Flask, request, jsonify
import os
import numpy as np
import pickle

from src.transcription import transcribe_audio, read_transcript_file
from src.enrichment import enrich_transcript
from src.embedd import chunk_text, embed_text, build_faiss_index
from src.chat import chat_with_gemini

app = Flask(__name__)

INDEX_FILE = "index_store.pkl"

def save_index():
    """Save INDEX_STORE to disk using pickle."""
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(INDEX_STORE, f)


def load_index():
    """Load INDEX_STORE from disk if available."""
    global INDEX_STORE
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "rb") as f:
            INDEX_STORE = pickle.load(f)


# Load index at startup (if exists)
load_index()


@app.route("/", methods=["GET"])
def root():
    """Simple root endpoint listing available routes."""
    return jsonify({
        "message": "Flask AI Agent API",
        "endpoints": {
            "/transcribe": "POST (form-data file=...)",
            "/read_transcript": "POST (form-data file=...)",
            "/enrich": "POST (json {transcript: ...})",
            "/index/build": "POST (json {transcript: ..., chunk_size?: int})",
            "/index/stats": "GET",
            "/search": "POST (json {query: ..., k?: int})",
            "/chat": "POST (json {messages: [...]})"
        }
    })

@app.route("/index/stats", methods=["GET"])
def index_stats():
    """Return basic diagnostics about the stored FAISS index."""
    if not INDEX_STORE["index"]:
        return jsonify({
            "built": False,
            "chunks": len(INDEX_STORE["chunks"]),
            "ntotal": 0,
            "dim": INDEX_STORE["dim"]
        })
    return jsonify({
        "built": True,
        "chunks": len(INDEX_STORE["chunks"]),
        "ntotal": int(INDEX_STORE["index"].ntotal),
        "dim": INDEX_STORE["dim"]
    })

@app.route("/transcribe", methods=["POST"])
def api_transcribe():
    """Accepts raw audio bytes in the request body (form-data file "file") and returns transcript."""
    if "file" not in request.files:
        return jsonify({"error": "missing file form-data field 'file'"}), 400
    f = request.files["file"]
    data = f.read()
    try:
        text = transcribe_audio(data)
        return jsonify({"transcript": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/read_transcript", methods=["POST"])
def api_read_transcript():
    """Accepts transcript file in form-data 'file' and returns plain text. Supports txt/docx."""
    if "file" not in request.files:
        return jsonify({"error": "missing file form-data field 'file'"}), 400
    f = request.files["file"]
    ext = f.filename.split(".")[-1].lower()
    data = f.read()
    try:
        text = read_transcript_file(data, ext)
        return jsonify({"transcript": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/enrich", methods=["POST"])
def api_enrich():
    """Accepts JSON {"transcript": "..."} and returns Gemini-enriched JSON."""
    payload = request.get_json(force=True)
    transcript = payload.get("transcript")
    if not transcript:
        return jsonify({"error": "missing 'transcript' in JSON body"}), 400
    enriched = enrich_transcript(transcript)
    return jsonify({"enriched": enriched})


@app.route("/index/build", methods=["POST"])
def api_index_build():
    """Build and persist a FAISS index from provided transcript text.

    Body JSON: {"transcript": "...", "chunk_size": optional int}
    Returns: status and the number of chunks.
    """
    payload = request.get_json(force=True)
    transcript = payload.get("transcript")
    if not transcript:
        return jsonify({"error": "missing 'transcript' in JSON body"}), 400
    chunk_size = payload.get("chunk_size", 500)
    chunks = chunk_text(transcript, size=chunk_size)
    embeddings = embed_text(chunks)
    print(f"Built {len(chunks)} chunks. Embedding shape={embeddings[0].shape}, dtype={embeddings[0].dtype}")
    dim = embeddings[0].shape[0]
    index = build_faiss_index(dim)
    index.add(np.vstack(embeddings))
    print("FAISS index ntotal:", index.ntotal)

    INDEX_STORE.update({"index": index, "chunks": chunks, "dim": dim})

    # Persist to file
    save_index()

    return jsonify({"status": "ok", "chunks": len(chunks)})


@app.route("/search", methods=["POST"])
def api_search():
    """Search the persisted FAISS index.

    Body JSON: {"query": "...", "k": optional int}
    Returns: list of matched chunks and distances.
    """
    payload = request.get_json(force=True)
    query = payload.get("query")
    if not query:
        return jsonify({"error": "missing 'query' in JSON body"}), 400

    if not INDEX_STORE["index"]:
        load_index()
        if not INDEX_STORE["index"]:
            return jsonify({"error": "index not built"}), 400

    k = int(payload.get("k", 5))
    q_emb = embed_text([query])[0]
    dists, idxs = INDEX_STORE["index"].search(np.array([q_emb]), k=k)
    matches = []
    for dist, idx in zip(dists[0], idxs[0]):
        matches.append({"chunk": INDEX_STORE["chunks"][int(idx)], "dist": float(dist)})
    return jsonify({"matches": matches})


@app.route("/chat", methods=["POST"])
def api_chat():
    """Proxy chat messages to Gemini. Accepts JSON {"messages": [...] } and returns Gemini response.
    Messages should contain the full conversation history to maintain context."""
    payload = request.get_json(force=True)
    messages = payload.get("messages")
    if not messages or not isinstance(messages, list):
        return jsonify({"error": "missing or invalid 'messages' (list) in JSON body"}), 400
    
    # Process the full conversation history
    chat_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ["user", "assistant"]:
            chat_messages.append({"role": role, "content": content})
    
    # Send the complete conversation to maintain context
    resp = chat_with_gemini(chat_messages)
    return jsonify({"response": resp})


if __name__ == "__main__":
    # Allow running directly for local development
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)