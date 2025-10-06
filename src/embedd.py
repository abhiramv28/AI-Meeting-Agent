from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load once (global) to avoid reloading on every request
st_model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, size, overlap=50):
    """
    Split text into overlapping chunks.
    
    Args:
        text (str): The input text.
        size (int): Maximum characters in each chunk.
        overlap (int): Number of characters to overlap between chunks.
    
    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)

        # Move start ahead by size - overlap
        start += size - overlap
        if start >= text_length:
            break

    return chunks


def embed_text(texts, normalize=True):
    """
    Generate embeddings for a list of texts via Sentence-Transformers.
    Returns a single 2D numpy array (n, dim), dtype float32.
    """
    embeddings = st_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embeddings.astype("float32")  # always float32 for FAISS

def build_faiss_index(dim, use_cosine=True):
    """
    Create a new FAISS index.
    If use_cosine=True → use IndexFlatIP (cosine similarity with normalized vectors).
    Else → use IndexFlatL2 (Euclidean distance).
    """
    if use_cosine:
        return faiss.IndexFlatIP(dim)  # cosine sim if vectors normalized
    else:
        return faiss.IndexFlatL2(dim)
