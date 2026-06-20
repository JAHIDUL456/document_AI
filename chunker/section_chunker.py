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
# HEADING DETECTOR
# -----------------------------
def detect_heading(line: str):
    line = line.strip()
    h1_match = re.match(r"^(\d+)\.\s*(.*)$", line)
    if h1_match:
        num, text = h1_match.groups()
        if text.isupper() and len(text.strip()) > 0:
            return 1, f"{num}. {text.strip()}"
            
    h2_match = re.match(r"^([A-Z])\.\s*(.*)$", line)
    if h2_match:
        letter, text = h2_match.groups()
        if text and text[0].isupper():
            return 2, f"{letter}. {text.strip()}"
            
    return 0, None


# -----------------------------
# MAIN CHUNKER
# -----------------------------
def build_chunks(pages: List[Dict], config: ChunkConfig = ChunkConfig()) -> List[Dict[str, Any]]:
    # 1. Line extraction and basic filtering
    all_lines = []
    for p in pages:
        page_num = p.get("page") or p.get("page_number") or 1
        text = p.get("text", "")
        lines = text.split("\n")
        
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            # Filter out running page numbers at the bottom/top
            if re.match(r"^\d+$", line_str):
                continue
            
            all_lines.append({
                "text": line_str,
                "pages": {page_num}
            })
            
    # 2. Pre-merge structural heading/list prefixes
    merged_lines = []
    marker_pats = [r"^\d+\.$", r"^[A-Z]\.$", r"^[a-z]\)$"]
    
    for item in all_lines:
        line_str = item["text"]
        pages_set = item["pages"]
        
        if merged_lines and any(re.match(pat, merged_lines[-1]["text"]) for pat in marker_pats):
            merged_lines[-1]["text"] = f"{merged_lines[-1]['text']} {line_str}"
            merged_lines[-1]["pages"].update(pages_set)
        else:
            merged_lines.append({
                "text": line_str,
                "pages": set(pages_set)
            })

    # 3. Assemble lines into Paragraph blocks
    paragraphs = []
    bullet_pat = r"^[a-z]\)\s+"
    
    for item in merged_lines:
        line_str = item["text"]
        pages_set = item["pages"]
        
        h_level, h_val = detect_heading(line_str)
        is_heading = h_level > 0
        
        if not paragraphs:
            paragraphs.append({
                "text": line_str,
                "pages": set(pages_set),
                "is_heading": is_heading,
                "heading_level": h_level,
                "heading_val": h_val
            })
            continue
            
        prev = paragraphs[-1]
        
        is_new_block = (
            is_heading or 
            re.match(bullet_pat, line_str) or 
            prev["is_heading"] or
            prev["text"].endswith((".", ":", ";", "?", "!"))
        )
        
        if is_new_block:
            paragraphs.append({
                "text": line_str,
                "pages": set(pages_set),
                "is_heading": is_heading,
                "heading_level": h_level,
                "heading_val": h_val
            })
        else:
            prev["text"] = f"{prev['text']} {line_str}"
            prev["pages"].update(pages_set)
            
    # 4. Hierarchical Context Tracking
    current_h1 = None
    current_h2 = None
    
    structured_paragraphs = []
    for p in paragraphs:
        if p["is_heading"]:
            if p["heading_level"] == 1:
                current_h1 = p["heading_val"]
                current_h2 = None
            elif p["heading_level"] == 2:
                current_h2 = p["heading_val"]
        
        structured_paragraphs.append({
            "text": p["text"],
            "pages": sorted(list(p["pages"])),
            "h1": current_h1,
            "h2": current_h2,
            "is_heading": p["is_heading"]
        })
        
    # 5. Group into Chunks
    chunks = []
    chunk_id = 0
    
    current_chunk_blocks = []
    current_tokens = 0
    current_h1_section = None
    current_h2_section = None
    
    def get_context_header(h1, h2):
        if h1 and h2:
            return f"[Section: {h1} > {h2}]\n"
        elif h1:
            return f"[Section: {h1}]\n"
        return ""

    def flush_chunk():
        nonlocal chunk_id, current_chunk_blocks, current_tokens
        if not current_chunk_blocks:
            return
            
        h1 = current_chunk_blocks[-1]["h1"]
        h2 = current_chunk_blocks[-1]["h2"]
        context_prefix = get_context_header(h1, h2)
        
        chunk_body = " ".join([b["text"] for b in current_chunk_blocks])
        full_text = f"{context_prefix}{chunk_body}"
        
        pages_spanned = set()
        for b in current_chunk_blocks:
            pages_spanned.update(b["pages"])
            
        chunks.append({
            "chunk_id": chunk_id,
            "text": clean(full_text),
            "pages": sorted(list(pages_spanned))
        })
        chunk_id += 1
        current_chunk_blocks = []
        current_tokens = 0

    max_toks = config.max_tokens

    for block in structured_paragraphs:
        if not block["text"].strip():
            continue
            
        block_tokens = count_tokens(block["text"])
        section_changed = (block["h1"] != current_h1_section or block["h2"] != current_h2_section)
        
        has_only_headings = all(b["is_heading"] for b in current_chunk_blocks) if current_chunk_blocks else False
        
        if section_changed and current_chunk_blocks and not has_only_headings:
            flush_chunk()
            
        current_h1_section = block["h1"]
        current_h2_section = block["h2"]
        
        context_prefix = get_context_header(block["h1"], block["h2"])
        prefix_tokens = count_tokens(context_prefix)
        
        if current_tokens + block_tokens + 1 <= max_toks:
            current_chunk_blocks.append(block)
            current_tokens += block_tokens + 1
        else:
            if current_chunk_blocks:
                flush_chunk()
                
            if block_tokens + prefix_tokens > max_toks:
                sentences = split_sentences(block["text"])
                sub_blocks = []
                sub_tokens = 0
                for sent in sentences:
                    sent_tokens = count_tokens(sent)
                    if sub_tokens + sent_tokens + 1 > max_toks - prefix_tokens:
                        if sub_blocks:
                            chunks.append({
                                "chunk_id": chunk_id,
                                "text": clean(f"{context_prefix}{' '.join(sub_blocks)}"),
                                "pages": block["pages"]
                            })
                            chunk_id += 1
                        sub_blocks = [sent]
                        sub_tokens = sent_tokens
                    else:
                        sub_blocks.append(sent)
                        sub_tokens += sent_tokens + 1
                if sub_blocks:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": clean(f"{context_prefix}{' '.join(sub_blocks)}"),
                        "pages": block["pages"]
                    })
                    chunk_id += 1
            else:
                current_chunk_blocks.append(block)
                current_tokens = block_tokens + prefix_tokens
                
    flush_chunk()
    return chunks