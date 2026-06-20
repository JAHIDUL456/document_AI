import re
from typing import List, Dict


STRUCTURE_PATTERNS = [
    r"^\d+\.",
    r"^[A-Z]\.",
    r"^[a-z]\)",
    r"^\(?i+\)?\.",
]


def is_structure_line(line: str) -> bool:
    line = line.strip()
    return any(re.match(p, line) for p in STRUCTURE_PATTERNS)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def preserve_bullets(text: str) -> str:
    text = re.sub(r"\n([a-z]\))", r"\n\1", text)
    text = re.sub(r"\n([A-Z]\.)", r"\n\1", text)
    return text


def is_raw_marker(line: str) -> bool:
    line = line.strip()
    return bool(
        re.match(r"^\d+\.$", line) or
        re.match(r"^[A-Z]\.$", line) or
        re.match(r"^[a-z]\)$", line) or
        re.match(r"^\(?i+\)?\.$", line)
    )


def is_header(line: str) -> bool:
    line = line.strip()
    return bool(re.match(r"^\d+\.\s+\S+", line) or re.match(r"^[A-Z]\.\s+\S+", line))


def smart_line_merge(text: str) -> str:
    text = text.replace("\u200b", "").replace("\u00a0", " ")
    lines = text.split("\n")
    merged = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if is_structure_line(line):
            merged.append(line)
            continue

        if merged and is_header(merged[-1]):
            merged.append(line)
            continue

        if merged and (is_raw_marker(merged[-1]) or not merged[-1].endswith((".", ":", ";"))):
            merged[-1] += " " + line
        else:
            merged.append(line)

    return "\n".join(merged)


def clean_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = preserve_bullets(text)
    text = smart_line_merge(text)
    return text


def clean_documents(pages: List[Dict]) -> List[Dict]:
    cleaned = []

    for p in pages:

        page_number = p.get("page_number") or p.get("page")

        cleaned.append({
            "page": page_number,   # FIXED HERE
            "text": clean_text(p.get("text", "")),
            "char_count": len(p.get("text", "")),
            "word_count": len(p.get("text", "").split()),
            "has_text": bool(p.get("text", "").strip())
        })

    return cleaned