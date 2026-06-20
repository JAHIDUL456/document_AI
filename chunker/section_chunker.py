from typing import List, Dict, Any
# pyrefly: ignore [missing-import]
import tiktoken
import re

ENCODER = tiktoken.get_encoding("cl100k_base")


# -----------------------------
# CONFIG
# -----------------------------
class ChunkConfig:
    def __init__(self):
        self.max_tokens = 220     # tight for embeddings
        self.overlap_sentences = 2  # semantic overlap (NOT token overlap)


# -----------------------------
# CLEAN
# -----------------------------
def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u200b", "")
    return text.strip()


# -----------------------------
# TOKEN COUNT
# -----------------------------
def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


# -----------------------------
# SENTENCE SPLIT (IMPORTANT FIX)
# -----------------------------
def split_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.?!])\s+", text)


# -----------------------------
# MAIN CHUNKER
# -----------------------------
def build_chunks(pages: List[Dict], config: ChunkConfig = ChunkConfig()):

    chunks = []
    chunk_id = 0

    buffer_sentences = []
    buffer_pages = set()

    def flush():
        nonlocal buffer_sentences, buffer_pages, chunk_id

        if not buffer_sentences:
            return

        text = clean(" ".join(buffer_sentences))

        chunks.append({
            "chunk_id": chunk_id,
            "text": text,
            "pages": sorted(list(buffer_pages))
        })

        chunk_id += 1

        # 🔥 SMART OVERLAP (last N sentences)
        overlap = buffer_sentences[-config.overlap_sentences:] if len(buffer_sentences) > config.overlap_sentences else buffer_sentences

        buffer_sentences = overlap
        buffer_pages = set(buffer_pages)

    for p in pages:

        text = clean(p.get("text", ""))
        page = p.get("page") or p.get("page_number")

        if not text:
            continue

        sentences = split_sentences(text)

        for sent in sentences:
            if not sent.strip():
                continue

            buffer_sentences.append(sent)
            buffer_pages.add(page)

            # check size
            current_text = " ".join(buffer_sentences)

            if count_tokens(current_text) >= config.max_tokens:
                flush()

    flush()

    return chunks