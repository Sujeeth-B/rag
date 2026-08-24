from typing import List, Dict, Any, Callable, Optional


class Document:
    def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(text_len={len(self.text)}, metadata={self.metadata})"


class TextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= len(text):
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunked_docs = []
        for doc in documents:
            chunks = self.split_text(doc.text)
            for idx, chunk in enumerate(chunks):
                meta = doc.metadata.copy()
                meta["chunk_id"] = idx
                chunked_docs.append(Document(text=chunk, metadata=meta))
        return chunked_docs
