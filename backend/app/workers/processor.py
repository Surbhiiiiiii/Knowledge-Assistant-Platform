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
        chunks = [clean_text[i:i+settings.CHUNK_SIZE] for i in range(0, len(clean_text), settings.CHUNK_SIZE) if len(clean_text[i:i+settings.CHUNK_SIZE])>20]
        vectors = [embeddings.get_embedding(c) for c in chunks]
        # remove any empty vectors
        vectors = [v for v in vectors if v]
        vectorstore.add_vectors(vectors)
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
