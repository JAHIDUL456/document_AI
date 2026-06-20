import math
from typing import List, Dict, Any
# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
from embeddings.embedder import embed_text
from vectordb.qdrant_store import get_qdrant_client
from config import COLLECTION_NAME, TOP_K

class SimpleBM25:
    """
    Lightweight, highly optimized BM25 keyword ranker.
    """
    def __init__(self, corpus: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.avg_doc_len = sum(len(doc["text"].split()) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 0
        self.doc_freqs = []
        self.idf = {}
        
        # Calculate frequencies
        for doc in corpus:
            words = doc["text"].lower().split()
            freq = {}
            for w in words:
                freq[w] = freq.get(w, 0) + 1
            self.doc_freqs.append(freq)
            
            # Count document frequencies for IDF
            for w in set(words):
                self.idf[w] = self.idf.get(w, 0) + 1
                
        # Calculate IDF
        for w, df in self.idf.items():
            self.idf[w] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            
    def score(self, query: str, index: int, doc_len: int) -> float:
        query_words = query.lower().split()
        score = 0.0
        freq = self.doc_freqs[index]
        for w in query_words:
            if w not in freq:
                continue
            f = freq[w]
            idf = self.idf.get(w, 0.0)
            numerator = idf * f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += numerator / denominator
        return score

def reciprocal_rank_fusion(dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], c: int = 60) -> List[Dict[str, Any]]:
    """
    Applies Reciprocal Rank Fusion (RRF) to merge and rerank dense and sparse lists.
    """
    rrf_scores = {}
    point_details = {}
    
    # Process dense rank
    for rank, point in enumerate(dense_results, start=1):
        pid = point["id"]
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (c + rank))
        point_details[pid] = point
        
    # Process sparse rank
    for rank, point in enumerate(sparse_results, start=1):
        pid = point["id"]
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (c + rank))
        point_details[pid] = point
        
    # Sort by score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    ranked_results = []
    for pid in sorted_ids:
        point = point_details[pid]
        point["rrf_score"] = rrf_scores[pid]
        ranked_results.append(point)
        
    return ranked_results

def retrieve_hybrid(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Performs a lead-level hybrid retrieval combining dense semantic search (Qdrant)
    with sparse keyword search (BM25) reranked via RRF.
    """
    client = get_qdrant_client()
    
    # 1. DENSE SEMANTIC RETRIEVAL
    query_vector = embed_text(query)
    dense_candidates = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k * 2
    ).points
    
    dense_results = []
    for res in dense_candidates:
        dense_results.append({
            "id": res.id,
            "text": res.payload.get("text", ""),
            "pages": res.payload.get("pages", []),
            "score": res.score,
            "type": "dense"
        })
        
    # 2. SPARSE KEYWORD RETRIEVAL (Local BM25)
    scroll_res, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=True,
        with_vectors=False
    )
    
    if not scroll_res:
        return dense_results[:k]
        
    corpus = []
    for point in scroll_res:
        corpus.append({
            "id": point.id,
            "text": point.payload.get("text", ""),
            "pages": point.payload.get("pages", [])
        })
        
    # Initialize BM25 ranker
    bm25 = SimpleBM25(corpus)
    
    # Score all candidates
    sparse_candidates = []
    for i, doc in enumerate(corpus):
        doc_len = len(doc["text"].split())
        score = bm25.score(query, i, doc_len)
        if score > 0.0:  # Only count documents with keyword overlap
            sparse_candidates.append({
                "id": doc["id"],
                "text": doc["text"],
                "pages": doc["pages"],
                "score": score,
                "type": "sparse"
            })
            
    # Sort sparse results
    sparse_results = sorted(sparse_candidates, key=lambda x: x["score"], reverse=True)[:k * 2]
    
    # 3. RECIPROCAL RANK FUSION
    rrf_results = reciprocal_rank_fusion(dense_results, sparse_results)
    
    # Return top K
    return rrf_results[:k]
