import os
import json
from google import genai

# Initialize Gemini client
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def enrich_transcript(transcript: str):
    """
    Send transcript to Gemini for enrichment (summary, action items, key points).
    Ensures strict JSON output.
    """
    prompt = f"""
    You are a meeting assistant.
    Given the transcript below, extract and structure the following:

    - summary: A concise summary of the meeting.
    - action_items: A list of action items with owners if possible.
    - key_points: Key discussion points.

    Return the result as a STRICT JSON object only.
    No extra text, no explanation, no markdown.

    Transcript:
    {transcript}
    """

    response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents=[{"parts": [{"text": prompt}]}],
    config={"response_mime_type": "application/json"}
)

    raw_output = response.text.strip()

    # Debug log (helpful for dev)
    print("🔎 Gemini raw output:", raw_output)

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        # Fallback to safe structure if Gemini gives invalid JSON
        return {
            "summary": raw_output if raw_output else "No summary generated.",
            "action_items": [],
            "key_points": []
        }