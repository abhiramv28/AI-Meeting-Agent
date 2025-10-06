import google.generativeai as gen

def _normalize_messages(messages):
    """
    Convert OpenAI-style messages into Gemini's expected format.
    Input:  [{"role": "user", "content": "hi"}]
    Output: [{"role": "user", "parts": [{"text": "hi"}]}]
    """
    normalized = []
    for m in messages:
        if "content" in m:
            normalized.append({
                "role": m["role"],
                "parts": [{"text": m["content"]}]
            })
        else:
            normalized.append(m)  # assume already in correct format
    return normalized

def chat_with_gemini(messages, model="gemini-1.5-flash"):
    """
    Multi-turn conversation with Gemini.
    `messages` should be a list of dicts like:
    [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]
    """
    try:
        gemini_model = gen.GenerativeModel(model)

        # Convert into Gemini format
        history = _normalize_messages(messages[:-1])
        latest = _normalize_messages([messages[-1]])[0]

        chat_session = gemini_model.start_chat(history=history)
        resp = chat_session.send_message(latest["parts"][0]["text"])
        return resp.text.strip()
    except Exception as e:
        return f"⚠️ Gemini API Error: {e}"
