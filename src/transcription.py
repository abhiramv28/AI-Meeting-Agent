import tempfile
import os
import whisper
import librosa
import soundfile as sf
from docx import Document

# Load Whisper model once (change model name if needed)
model = whisper.load_model("tiny")

def transcribe_audio(file_bytes, chunk_duration=60):
    """
    Save incoming audio bytes to a temp file, split into smaller chunks,
    transcribe each chunk with Whisper, then combine results.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name

    # Load audio
    audio, sr = librosa.load(tmp_path, sr=16000)

    # Calculate samples per chunk (default 60 seconds)
    samples_per_chunk = chunk_duration * sr
    transcripts = []

    # Process in chunks
    for i in range(0, len(audio), samples_per_chunk):
        chunk = audio[i : i + samples_per_chunk]
        chunk_path = f"{tmp_path}_{i}.wav"
        sf.write(chunk_path, chunk, sr)

        # Transcribe
        result = model.transcribe(chunk_path, fp16=False)
        transcripts.append(result["text"])

        os.remove(chunk_path)

    os.remove(tmp_path)
    return " ".join(transcripts)

def read_transcript_file(file_bytes, file_type):
    """
    Decode .txt or parse .docx bytes to return plain text.
    """
    if file_type == "txt":
        return file_bytes.decode("utf-8")
    elif file_type == "docx":
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            tmp_path = tmp.name

        doc = Document(tmp_path)
        os.remove(tmp_path)
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Unsupported transcript format: {file_type}")
