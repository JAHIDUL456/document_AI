import requests
from config import OLLAMA_URL, LLM_MODEL

def generate_answer(query: str, chunks: list) -> str:
    """
    Generates a smart, concise, and context-constrained response using Ollama.
    """
    if not chunks:
        return "I cannot find any relevant policy documents to answer this question."

    # Format the context chunks cleanly
    context_parts = []
    for idx, chunk in enumerate(chunks, 1):
        pages = ", ".join(map(str, chunk.get("pages", [])))
        text = chunk.get("text", "").strip()
        context_parts.append(f"[{idx}] (Page(s): {pages})\n{text}")

    context_str = "\n\n".join(context_parts)

    # Prompt optimized to act like a Lead AI Engineer, strictly grounded in the context
    prompt = f"""You are a Lead AI Engineer explaining a company policy directly to a colleague.
Explain the answer using ONLY the retrieved context below.

Rules:
1. Rely strictly on the retrieved context. Do not use external knowledge or make assumptions.
2. If the context does not contain the answer, reply: "I cannot find the answer in the provided documents."
3. Do not use robotic preambles like "Based on the context" or "According to the document". Answer directly.
4. Format the response as a single, continuous, friendly paragraph. Do not use numbered lists, bullet points, or newlines.

Retrieved Context:
{context_str}

Question: {query}
Answer (as a single conversational paragraph, no lists, no bullet points, no newlines):"""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to connect to Ollama at {OLLAMA_URL}. Ensure Ollama is running. ({e})"
