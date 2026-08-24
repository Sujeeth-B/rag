from typing import List, Dict, Any, Optional, Callable
from ingestion import Document, TextSplitter
from vector_store import VectorStore
from retriever import Retriever
from generator import IntrospectiveGenerator
from introspection import ModelIntrospector
from injection_detector import IntrospectiveInjectionDetector


class FlexibleRAGPipeline:
    def __init__(
        self,
        model_name: str = "gpt2",
        chunk_size: int = 300,
        chunk_overlap: int = 30,
        top_k: int = 2,
        prompt_template: Optional[Callable[[str, List[Document]], str]] = None,
        embedding_model=None
    ):
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = VectorStore(embedding_model=embedding_model)
        self.retriever = Retriever(vector_store=self.vector_store, top_k=top_k)
        self.generator = IntrospectiveGenerator(model_name=model_name)
        self.prompt_template = prompt_template or self.default_prompt_template
        self.injection_detector = IntrospectiveInjectionDetector()

    @staticmethod
    def default_prompt_template(query: str, retrieved_docs: List[Document]) -> str:
        context = "\n---\n".join([doc.text for doc in retrieved_docs])
        return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    def ingest_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        raw_docs = []
        for idx, text in enumerate(texts):
            meta = metadatas[idx] if metadatas and idx < len(metadatas) else {"source_id": idx}
            raw_docs.append(Document(text=text, metadata=meta))
        chunked_docs = self.splitter.split_documents(raw_docs)
        self.vector_store.add_documents(chunked_docs)

    def query(self, query_text: str, max_new_tokens: int = 30) -> Dict[str, Any]:
        # 1. Retrieve
        retrieved_results = self.retriever.retrieve(query_text)
        retrieved_docs = [doc for doc, score in retrieved_results]

        # 2. Build Prompt
        prompt = self.prompt_template(query_text, retrieved_docs)

        # 3. Generate with Introspection
        gen_results = self.generator.generate_with_introspection(prompt, max_new_tokens=max_new_tokens)

        # 4. Introspect hidden representations
        hidden_stats = ModelIntrospector.inspect_hidden_representations(gen_results["hidden_states"])

        # 5. Evaluate Indirect Prompt Injection & Topic Transition Risk
        injection_evaluation = self.injection_detector.evaluate_injection_risk(
            attentions=gen_results["attentions"],
            hidden_states=gen_results["hidden_states"],
            input_tokens=gen_results["input_tokens"]
        )

        return {
            "query": query_text,
            "retrieved_documents": retrieved_results,
            "prompt": prompt,
            "answer": gen_results["generated_text"],
            "input_tokens": gen_results["input_tokens"],
            "attentions": gen_results["attentions"],
            "hidden_states": gen_results["hidden_states"],
            "hidden_stats": hidden_stats,
            "injection_evaluation": injection_evaluation
        }
