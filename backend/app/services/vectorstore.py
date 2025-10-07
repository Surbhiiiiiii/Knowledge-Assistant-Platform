# app/services/vectorstore.py
import faiss
import os
import pickle
import numpy as np
import re
from datetime import datetime
from threading import Lock
from app.services.embeddings import get_embedding
from app.config import settings

# use settings paths
INDEX_FILE = settings.VECTOR_PATH
META_FILE = settings.METADATA_PATH

# simple in-memory lock (works for single-process; for multi-process use file locking)
faiss_lock = Lock()

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def _ensure_dirs():
    p = os.path.dirname(INDEX_FILE)
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def load_or_create_index(dim: int = None):
    _ensure_dirs()
    if dim is None:
        test_vec = get_embedding("hello world")
        if test_vec is None:
            dim = 768
        else:
            dim = int(test_vec.shape[0])

    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            metadata = pickle.load(f)
        print(f"[DEBUG] Loaded FAISS index with {index.ntotal} vectors and {len(metadata)} metadata entries")
    else:
        index = faiss.IndexFlatIP(dim)  # inner product for cosine if normalized
        metadata = []
        print("[DEBUG] Created new FAISS index (IP) with dim", dim)

    return index, metadata

def save_index(index, metadata):
    _ensure_dirs()
    with faiss_lock:
        faiss.write_index(index, INDEX_FILE)
        with open(META_FILE, "wb") as f:
            pickle.dump(metadata, f)
    print(f"[DEBUG] FAISS index saved. Total vectors: {index.ntotal}")

def add_to_index(chunks, index, metadata, embeddings=None, filename=None):
    now = datetime.now().timestamp()
    to_add = []
    added_meta = []

    for i, chunk in enumerate(chunks):
        vec = None
        if embeddings and i < len(embeddings):
            vec = embeddings[i]
        if vec is None:
            vec = get_embedding(chunk)
        if vec is None:
            print(f"[WARN] Skipping chunk {i+1}/{len(chunks)} from {filename} because embedding failed.")
            continue

        arr = np.array(vec, dtype="float32")
        arr = normalize_vec(arr)
        to_add.append(arr)
        added_meta.append({"filename": filename, "text": chunk, "timestamp": now})

    if not to_add:
        print("[WARN] No vectors to add to index for", filename)
        return

    arr_stack = np.stack(to_add, axis=0)
    with faiss_lock:
        index.add(arr_stack)
        metadata.extend(added_meta)
        faiss.write_index(index, INDEX_FILE)
        with open(META_FILE, "wb") as f:
            pickle.dump(metadata, f)

    print(f"[DEBUG] Added {len(to_add)} vectors for {filename}. New total: {index.ntotal}")

# The search_index implementation you had looks good; reuse it here
# (Copy/paste your search_index + helpers or import them). Example:
def _token_overlap_score(query: str, text: str) -> float:
    q_toks = re.findall(r"\w+", query.lower())
    if not q_toks:
        return 0.0
    text_toks = set(re.findall(r"\w+", text.lower()))
    matches = sum(1 for t in q_toks if t in text_toks)
    return matches / len(q_toks)

def _recency_score_from_timestamp(timestamp: float, recency_halflife_days: float = 7.0) -> float:
    now = datetime.now().timestamp()
    age_seconds = max(0.0, now - timestamp)
    age_days = age_seconds / 86400.0
    return 1.0 / (1.0 + (age_days / max(recency_halflife_days, 1e-6)))

def search_index(query: str, index, metadata, top_k: int = 5, alpha: float = 0.5, beta: float = 0.2, gamma: float = 0.3, recency_halflife_days: float = 7.0):
    if index.ntotal == 0 or not metadata:
        print("[DEBUG] FAISS index or metadata is empty")
        return []

    vec = get_embedding(query)
    if vec is None:
        print("[WARN] Query embedding failed.")
        return []

    qvec = normalize_vec(np.array(vec, dtype="float32"))
    search_k = min(max(top_k, 1), index.ntotal)
    D, I = index.search(np.array([qvec], dtype="float32"), search_k)

    results = []
    now = datetime.now().timestamp()
    for sim, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        timestamp = meta.get("timestamp", now)
        recency = _recency_score_from_timestamp(timestamp, recency_halflife_days)
        match_score = _token_overlap_score(query, meta.get("text", ""))
        final_score = alpha * float(sim) + beta * recency + gamma * match_score
        results.append({
            "final_score": final_score,
            "similarity": float(sim),
            "recency": recency,
            "match_score": match_score,
            "text": meta.get("text", ""),
            "filename": meta.get("filename", "unknown"),
            "timestamp": timestamp
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    print(f"[DEBUG] search_index returned {len(results)} hits (requested top_k={top_k})")
    return results[:top_k]
