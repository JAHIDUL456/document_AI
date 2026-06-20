# pyrefly: ignore [missing-import]
from fastapi import FastAPI
from parser.extract import extract_pdf
from parser.cleaner import clean_documents
from config import *
from pathlib import Path
from chunker.section_chunker import build_chunks
app=FastAPI(title="Policy AI",description="Policy Analysis API", version="1.0.0")

pdf_path="data/policy.pdf"

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "RAG pipeline test server running"
    }


@app.get("/extract")
def extract():
    pages = extract_pdf(pdf_path)

    return {
        "total_pages": len(pages),
        "sample": pages[:2]  # preview first 2 pages only
    }

@app.get("/clean")
def clean():
    pages = extract_pdf(pdf_path)
    cleaned = clean_documents(pages)

    return {
        "total_pages": len(cleaned),
        "sample": cleaned[:2]
    }


@app.get("/chunks")
def chunks():

    pages = extract_pdf(pdf_path)
    cleaned = clean_documents(pages)

    final_chunks = build_chunks(cleaned)

    return {
        "total_chunks": len(final_chunks),
        "sample": final_chunks[:5]
    }


@app.post("/ingest")
def ingest():
    try:
        from embeddings.embedder import embed_batch
        from vectordb.qdrant_store import init_collection, upsert_chunks
        pages = extract_pdf(pdf_path)
        cleaned = clean_documents(pages)
        final_chunks = build_chunks(cleaned)
        texts_to_embed = [c["text"] for c in final_chunks]
        embeddings = embed_batch(texts_to_embed)
        init_collection(recreate=True)
        upsert_chunks(final_chunks, embeddings)
        return {
            "status": "success",
            "message": f"Successfully ingested {len(final_chunks)} chunks into Qdrant collection '{COLLECTION_NAME}'."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/retrieve")
def retrieve(query: str, k: int = TOP_K):
    try:
        from retriever.retriever import retrieve_hybrid
        results = retrieve_hybrid(query, k=k)
        return {
            "status": "success",
            "query": query,
            "results": results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }