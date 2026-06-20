# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
# pyrefly: ignore [missing-import]
import torch
from config import EMBED_MODEL

_model = None

def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # Determine execution provider
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load local embedding model
        _model = SentenceTransformer(EMBED_MODEL, device=device)
    return _model
