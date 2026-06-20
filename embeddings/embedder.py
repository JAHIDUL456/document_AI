from typing import List
from embeddings.model_loader import get_embedding_model

def embed_text(text: str) -> List[float]:
    """
    Generates a dense vector embedding for a single text query.
    """
    model = get_embedding_model()
    return model.encode(text, convert_to_numpy=True).tolist()

def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Generates dense vector embeddings for a list of texts (batch optimized).
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    return embeddings.tolist()
