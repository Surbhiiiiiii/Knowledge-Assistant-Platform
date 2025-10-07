# app/api/upload.py
from fastapi import APIRouter, UploadFile, File, HTTPException
import os
from app.config import settings
from app.services.embeddings import embed_document
from app.services.ocr import extract_text_from_pdf, extract_text_from_image
from app.services.vectorstore import load_or_create_index, add_to_index, save_index
import traceback

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs(settings.STORAGE_PATH, exist_ok=True)
        os.makedirs(os.path.dirname(settings.VECTOR_PATH) or ".", exist_ok=True)

        file_path = os.path.join(settings.STORAGE_PATH, file.filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        ext = file.filename.lower()
        if ext.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif ext.endswith((".png", ".jpg", ".jpeg")):
            text = extract_text_from_image(file_path)
        else:
            try:
                text = content.decode(errors="ignore")
            except Exception:
                text = ""

        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="No extractable text in file.")

        print(f"[DEBUG] Extracted {len(text)} chars from {file.filename}")

        index, metadata = load_or_create_index()
        chunks, embeddings = embed_document(text, chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks created from document.")

        # keep embeddings aligned with chunks; embeddings may include None
        valid_embeds = [e for e in embeddings if e is not None]
        if not valid_embeds:
            raise HTTPException(status_code=400, detail="Embedding failed for all chunks.")

        add_to_index(chunks, index, metadata, embeddings=embeddings, filename=file.filename)
        # save_index is called inside add_to_index; but call again to be safe
        save_index(index, metadata)

        return {"message": "File uploaded and indexed successfully", "filename": file.filename, "chunks_indexed": len(valid_embeds)}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {type(e).__name__}: {e}")
