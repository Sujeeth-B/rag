import numpy as np
from typing import List, Tuple
from ingestion import Document


class VectorStore:
    def __init__(self, embedding_model=None):
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_model = embedding_model

    def _get_embedding(self, text: str) -> np.ndarray:
        if self.embedding_model is not None:
            emb = self.embedding_model.encode(text)
            return np.array(emb, dtype=np.float32)
        # Fallback dummy embedding (bag of character frequencies normalized)
        vec = np.zeros(128, dtype=np.float32)
        for ch in text:
            vec[ord(ch) % 128] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def add_documents(self, docs: List[Document]):
        new_embeddings = [self._get_embedding(d.text) for d in docs]
        new_emb_arr = np.vstack(new_embeddings)
        if self.embeddings is None:
            self.embeddings = new_emb_arr
        else:
            self.embeddings = np.vstack([self.embeddings, new_emb_arr])
        self.documents.extend(docs)

    def similarity_search(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        if not self.documents or self.embeddings is None:
            return []
        q_emb = self._get_embedding(query)
        # Cosine similarity
        norm_q = np.linalg.norm(q_emb)
        if norm_q > 0:
            q_emb = q_emb / norm_q
        norms_doc = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms_doc[norms_doc == 0] = 1e-10
        norm_docs = self.embeddings / norms_doc
        scores = np.dot(norm_docs, q_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[idx], float(scores[idx])) for idx in top_indices]
