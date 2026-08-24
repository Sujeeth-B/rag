from typing import List, Tuple, Callable, Optional
from vector_store import VectorStore
from ingestion import Document


class Retriever:
    def __init__(self, vector_store: VectorStore, top_k: int = 3, rerank_fn: Optional[Callable] = None):
        self.vector_store = vector_store
        self.top_k = top_k
        self.rerank_fn = rerank_fn

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        results = self.vector_store.similarity_search(query, top_k=self.top_k)
        if self.rerank_fn is not None:
            results = self.rerank_fn(query, results)
        return results
