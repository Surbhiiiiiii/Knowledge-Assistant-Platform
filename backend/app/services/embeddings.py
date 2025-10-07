import google.generativeai as genai
import numpy as np
from typing import List, Tuple
from app.config import settings
from app.services.preprocessing import clean_text
import re

genai.configure(api_key=settings.GEMINI_API_KEY)

def chunk_text(text: str, chunk_size_words: int = 180, overlap: int = 40) -> List[str]:
    """
    Split text into overlapping word chunks.
    Additionally, respects sections based on headings.
    """
    # Normalize headings (lines in ALL CAPS or ending with colon)
    lines = text.split("\n")
    sections = []
    current_section = ""
    for line in lines:
        if re.match(r'^[A-Z ]{3,}$', line.strip()) or line.strip().endswith(":"):
            if current_section.strip():
                sections.append(current_section.strip())
            current_section = line.strip() + "\n"
        else:
            current_section += line.strip() + " "
    if current_section.strip():
        sections.append(current_section.strip())

    # Break sections into overlapping word chunks
    chunks = []
    for sec in sections:
        words = sec.split()
        if len(words) <= chunk_size_words:
            chunks.append(sec)
        else:
            step = chunk_size_words - overlap
            for i in range(0, len(words), step):
                chunk = " ".join(words[i:i + chunk_size_words])
                if chunk.strip():
                    chunks.append(chunk)
    print(f"[DEBUG] chunk_text created {len(chunks)} chunks")
    return chunks

def get_embedding(text: str):
    text = clean_text(text)
    if not text:
        return None
    try:
        response = genai.embed_content(model=settings.GEMINI_EMBEDDING_MODEL, content=text)
        # inspect possible shapes
        vector = None
        if isinstance(response, dict):
            # some SDKs put embedding under 'embedding' or 'embeddings'
            vector = response.get("embedding") or response.get("embeddings")
            if isinstance(vector, list) and len(vector)>0 and isinstance(vector[0], (int, float)):
                # vector is the embedding list
                pass
            elif isinstance(vector, list) and len(vector)>0 and isinstance(vector[0], list):
                # maybe embeddings is list of lists; choose first
                vector = vector[0]
        elif hasattr(response, "embedding"):
            vector = response.embedding
        elif hasattr(response, "embeddings"):
            vector = response.embeddings

        if vector is None:
            return None
        arr = np.array(vector, dtype="float32")
        if arr.size == 0 or np.all(arr == 0):
            return None
        return arr
    except Exception as e:
        print("[ERROR] Embedding failed:", repr(e))
        return None
def embed_document(document_text: str, chunk_size: int = 180, overlap: int = 40) -> Tuple[List[str], List[np.ndarray]]:
    """
    Embed a full document with section-aware chunking.
    Returns chunks and embeddings.
    """
    chunks = chunk_text(document_text, chunk_size_words=chunk_size, overlap=overlap)
    embeddings = []
    for i, chunk in enumerate(chunks):
        vec = get_embedding(chunk)
        embeddings.append(vec)
        if vec is not None:
            print(f"[DEBUG] Embedded chunk {i+1}/{len(chunks)}")
        else:
            print(f"[WARN] Failed to embed chunk {i+1}/{len(chunks)}")
    print(f"[DEBUG] Total chunks: {len(chunks)} | Successful embeddings: {sum(e is not None for e in embeddings)}")
    return chunks, embeddings
