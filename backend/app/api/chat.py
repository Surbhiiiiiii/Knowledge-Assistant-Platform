# app/api/chat.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.vectorstore import search_index, load_or_create_index
from app.config import settings
import google.generativeai as genai
import logging

router = APIRouter()
genai.configure(api_key=settings.GEMINI_API_KEY)

class QueryRequest(BaseModel):
    query: str

def build_prompt(context: str, query: str) -> str:
    return f"""
You are a helpful, factual AI assistant. Answer in 2–4 sentences, clearly and politely.

Only use the information provided in the context below.
If the answer is not found there, reply: "I'm sorry, I couldn’t find that in the uploaded documents."

Context:
{context}

Question: {query}

Answer factually and conversationally.
End with: Sources: [file1, file2]
"""

@router.post("/chat")
def chat(req: QueryRequest):
    try:
        logging.info(f"User query: {req.query}")
        index, metadata = load_or_create_index()
        if index is None or index.ntotal == 0:
            return {"answer": "No indexed documents available.", "sources": []}

        hits = search_index(req.query, index, metadata, top_k=min(settings.TOP_K, 10))
        if not hits:
            return {"answer": "No relevant content found.", "sources": []}

        # deduplicate
        seen_texts = set()
        unique_hits = []
        for h in hits:
            t = h["text"].strip()
            if t and t not in seen_texts:
                unique_hits.append(h)
                seen_texts.add(t)

        # Build context prioritizing recency (search_index already uses recency in final_score)
        context_parts = []
        for h in unique_hits[:settings.TOP_K]:
            context_parts.append(f"[Source: {h['filename']} | ts: {h['timestamp']}]\n{h['text']}")

        context = "\n\n---\n".join(context_parts)
        prompt = build_prompt(context, req.query)
        logging.info(f"Prompt length: {len(prompt)} chars")

        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        response = model.generate_content(prompt)
        answer_text = ""
        if response:
            # adapt depending on response shape
            if hasattr(response, "text"):
                answer_text = response.text.strip()
            elif isinstance(response, dict) and "candidates" in response:
                # fallback: pick first candidate text
                answer_text = response["candidates"][0].get("content", "").strip()
            else:
                answer_text = str(response)[:1000]
        else:
            answer_text = "No response from the model."

        sources = [{"filename": h["filename"], "preview": h["text"][:400], "timestamp": h["timestamp"]} for h in unique_hits[:settings.TOP_K]]
        return {"answer": answer_text, "sources": sources}

    except Exception as e:
        logging.exception("Chat endpoint failed")
        raise HTTPException(status_code=500, detail=f"Chat endpoint failed: {type(e).__name__}: {e}")
