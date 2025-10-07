# app/services/process_document.py
import os
from app.services import ocr, preprocessing, extraction, embeddings, vectorstore
from app.models.db_models import insert_document
from app.config import settings

def process_document(file_path: str, filename: str):
    try:
        if filename.lower().endswith(".pdf"):
            text = ocr.extract_text_from_pdf(file_path)
        else:
            text = ocr.extract_text_from_image(file_path)
        clean_text = preprocessing.clean_text(text)
        entities = extraction.extract_entities(clean_text)
        patterns = extraction.extract_custom_patterns(clean_text)
        metadata = {**entities, **patterns}

        # chunk in a way consistent with embed_document
        chunks, embeddings_list = embeddings.embed_document(clean_text, chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
        valid_embs = [e for e in embeddings_list if e is not None]
        if not valid_embs:
            print("[WARN] No valid embeddings for", filename)
        index, meta = vectorstore.load_or_create_index()
        vectorstore.add_to_index(chunks, index, meta, embeddings=embeddings_list, filename=filename)

        doc = {
            "filename": filename,
            "status": "done",
            "metadata": metadata,
            "text": clean_text,
            "chunk_count": len(chunks)
        }
        insert_document(doc)
    except Exception as e:
        print("Error processing document:", e)
