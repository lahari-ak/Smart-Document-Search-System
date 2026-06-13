from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from dotenv import load_dotenv
import fitz
import shutil
import uuid
import os
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer


app = FastAPI(title="DocuMind Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

documents = {}
chunks_store = []

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dimension = 384
faiss_index = faiss.IndexFlatL2(embedding_dimension)


@app.get("/")
def home():
    return {
        "message": "DocuMind backend is running",
        "semantic_search": True,
        "gemini_enabled": gemini_model is not None
    }


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    document_id = str(uuid.uuid4())
    safe_filename = file.filename.replace(" ", "_")
    file_path = UPLOAD_DIR / f"{document_id}_{safe_filename}"

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pages = extract_text_from_pdf(file_path)

    full_text = "\n\n".join([page["text"] for page in pages])

    if not full_text.strip():
        raise HTTPException(status_code=400, detail="No readable text found in PDF")

    chunks = create_chunks_with_pages(pages)

    documents[document_id] = {
        "id": document_id,
        "filename": file.filename,
        "path": str(file_path),
        "text": full_text,
        "chunks": chunks
    }

    rebuild_faiss_index()

    return {
        "message": "PDF uploaded, text extracted, chunks created, and embeddings stored successfully",
        "document_id": document_id,
        "filename": file.filename,
        "total_characters": len(full_text),
        "total_chunks": len(chunks),
        "preview": full_text[:700]
    }


@app.get("/documents")
def get_documents():
    return [
        {
            "id": doc["id"],
            "filename": doc["filename"],
            "total_chunks": len(doc["chunks"])
        }
        for doc in documents.values()
    ]


@app.get("/documents/{document_id}")
def get_document(document_id: str):
    if document_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = documents[document_id]

    return {
        "id": doc["id"],
        "filename": doc["filename"],
        "total_chunks": len(doc["chunks"]),
        "preview": doc["text"][:1000],
        "chunks": doc["chunks"]
    }


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    if document_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = Path(documents[document_id]["path"])

    if file_path.exists():
        file_path.unlink()

    del documents[document_id]
    rebuild_faiss_index()

    return {
        "message": "Document deleted successfully",
        "document_id": document_id
    }


@app.post("/search")
def semantic_search(query: str):
    if faiss_index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No documents uploaded")

    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    top_k = min(5, faiss_index.ntotal)
    distances, indexes = faiss_index.search(query_embedding, top_k)

    results = []

    for distance, index in zip(distances[0], indexes[0]):
        if index == -1:
            continue

        chunk = chunks_store[index]

        results.append({
            "document_id": chunk["document_id"],
            "filename": chunk["filename"],
            "chunk_number": chunk["chunk_number"],
            "page_number": chunk["page_number"],
            "score": float(distance),
            "content": chunk["content"]
        })

    context_chunks = [result["content"] for result in results[:3]]
    answer = generate_ai_answer(query, context_chunks)

    return {
        "query": query,
        "answer": answer,
        "note": "Lower score means better semantic match",
        "sources": [
            {
                "filename": result["filename"],
                "chunk_number": result["chunk_number"],
                "page_number": result["page_number"],
                "score": result["score"]
            }
            for result in results[:3]
        ],
        "results": results
    }


@app.get("/stats")
def get_stats():
    return {
        "total_documents": len(documents),
        "total_chunks": len(chunks_store),
        "gemini_enabled": gemini_model is not None,
        "semantic_search": True
    }


def extract_text_from_pdf(file_path: Path):
    pages = []
    pdf = fitz.open(file_path)

    for page_number, page in enumerate(pdf, start=1):
        page_text = page.get_text().strip()

        if page_text:
            pages.append({
                "page_number": page_number,
                "text": page_text
            })

    pdf.close()
    return pages


def create_chunks_with_pages(pages, chunk_size=250, overlap=50):
    chunks = []

    for page in pages:
        words = page["text"].split()
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])

            if chunk_text.strip():
                chunks.append({
                    "page_number": page["page_number"],
                    "content": chunk_text
                })

            start += chunk_size - overlap

    return chunks


def rebuild_faiss_index():
    global faiss_index, chunks_store

    faiss_index = faiss.IndexFlatL2(embedding_dimension)
    chunks_store = []

    all_chunk_texts = []

    for document_id, doc in documents.items():
        for index, chunk in enumerate(doc["chunks"]):
            chunks_store.append({
                "document_id": document_id,
                "filename": doc["filename"],
                "chunk_number": index + 1,
                "page_number": chunk["page_number"],
                "content": chunk["content"]
            })

            all_chunk_texts.append(chunk["content"])

    if all_chunk_texts:
        embeddings = embedding_model.encode(all_chunk_texts)
        embeddings = np.array(embeddings).astype("float32")
        faiss_index.add(embeddings)


def generate_ai_answer(question, context_chunks):
    if not context_chunks:
        return "I could not find this information in the uploaded documents."

    if gemini_model is None:
        return context_chunks[0]

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are DocuMind, an AI document assistant.

Answer the user's question using only the document context below.

Rules:
1. Do not use outside knowledge.
2. If the answer is not present in the context, say:
   "I could not find this information in the uploaded documents."
3. Keep the answer clear and concise.
4. Do not mention that you are an AI model.

Question:
{question}

Document context:
{context}

Answer:
"""

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return f"Gemini answer generation failed. Showing best source chunk instead: {context_chunks[0]}"