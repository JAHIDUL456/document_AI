# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
# pyrefly: ignore [missing-import]
from qdrant_client.http import models
# pyrefly: ignore [missing-import]
from config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
from typing import List, Dict, Any

_client = None

def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client

def init_collection(recreate: bool = False):
    """
    Ensures the target collection exists.
    If recreate is True, forces a fresh rebuild of the index.
    """
    client = get_qdrant_client()
    
    if recreate:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384,  # BGE-small dimension
                distance=models.Distance.COSINE
            )
        )
    else:
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                )
            )

def upsert_chunks(chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
    """
    Upserts chunks and their embeddings into the Qdrant collection in batch.
    """
    client = get_qdrant_client()
    points = []
    
    for chunk, vector in zip(chunks, embeddings):
        points.append(
            models.PointStruct(
                id=chunk["chunk_id"],
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "pages": chunk["pages"]
                }
            )
        )
        
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
