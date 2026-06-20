from typing import List, Dict, Any
import tiktoken
import re

ENCODER = tiktoken.get_encoding("cl100k_base")


# -----------------------------
# CONFIG
# -----------------------------
class ChunkConfig:
    def __init__(self):
        self.max_tokens = 250
        self.overlap_tokens = 60


# -----------------------------
# TOKEN UTIL
# -----------------------------
def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u200b", "")
    return text.strip()


# -----------------------------
# FLATTEN INPUT (IMPORTANT)
# -----------------------------
def flatten_pages(pages: List[Dict]) -> List[Dict]:
    """Turn pages into sequential text blocks"""
    blocks = []

    for p in pages:
        text = clean(p.get("text", ""))
        page = p.get("page") or p.get("page_number")

        if not text:
            continue

        blocks.append({
            "text": text,
            "page": page
        })

    return blocks


# -----------------------------
# STREAMING CHUNKER (CORE FIX)
# -----------------------------
def build_chunks(pages: List[Dict], config: ChunkConfig = ChunkConfig()) -> List[Dict]:

    blocks = flatten_pages(pages)

    chunks = []
    current_tokens = []
    current_pages = set()
    chunk_id = 0

    def flush():
        nonlocal current_tokens, current_pages, chunk_id

        if not current_tokens:
            return

        text = ENCODER.decode(current_tokens)

        chunks.append({
            "chunk_id": chunk_id,
            "text": clean(text),
            "pages": sorted(list(current_pages))
        })

        chunk_id += 1

        # overlap logic (KEEP LAST TOKENS)
        overlap = current_tokens[-config.overlap_tokens:] if len(current_tokens) > config.overlap_tokens else current_tokens

        current_tokens = overlap
        current_pages = set(current_pages)

    for block in blocks:

        tokens = ENCODER.encode(block["text"])
        page = block["page"]

        for t in tokens:
            current_tokens.append(t)
            current_pages.add(page)

            if len(current_tokens) >= config.max_tokens:
                flush()

    flush()

    return chunks