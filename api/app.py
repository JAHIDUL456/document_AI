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