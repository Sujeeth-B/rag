import os
import sys
from rag_pipeline import FlexibleRAGPipeline
from introspection import ModelIntrospector


def load_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def interactive_chat():
    print("=" * 60)
    print("      RAG Chat & Model Introspection System")
    print("=" * 60)
    print("Commands:")
    print("  /upload <file_path>  : Load and ingest a text file into RAG memory")
    print("  /add <text>          : Ingest a raw text snippet")
    print("  /params              : Display model architecture & parameters summary")
    print("  /heatmap             : Generate & save attention heatmap for last response")
    print("  /exit                : Quit chat session")
    print("=" * 60)

    model_name = "gpt2"
    print(f"Initializing RAG Pipeline with model: {model_name}...")
    pipeline = FlexibleRAGPipeline(model_name=model_name, top_k=2)
    last_result = None

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("Goodbye!")
            break

        elif user_input.startswith("/upload "):
            file_path = user_input[8:].strip()
            try:
                content = load_file(file_path)
                pipeline.ingest_texts([content], metadatas=[{"source": file_path}])
                print(f"[System] Successfully uploaded and ingested '{file_path}' ({len(content)} chars).")
            except Exception as e:
                print(f"[Error] Failed to load file: {e}")

        elif user_input.startswith("/add "):
            text_snippet = user_input[5:].strip()
            if text_snippet:
                pipeline.ingest_texts([text_snippet], metadatas=[{"source": "user_input"}])
                print(f"[System] Ingested text snippet into vector store.")

        elif user_input.lower() == "/params":
            ModelIntrospector.print_parameter_summary(pipeline.generator.model)

        elif user_input.lower() == "/heatmap":
            if last_result is None:
                print("[System] No query has been run yet.")
            else:
                save_path = "latest_attention_heatmap.png"
                ModelIntrospector.plot_attention_heatmap(
                    attentions=last_result["attentions"],
                    tokens=last_result["input_tokens"],
                    layer_idx=-1,
                    save_path=save_path
                )
                print(f"[System] Saved attention heatmap to {save_path}")

        else:
            # RAG Query
            print("\nProcessing query...")
            last_result = pipeline.query(user_input, max_new_tokens=40)

            print("\n--- Retrieved Context ---")
            if not last_result["retrieved_documents"]:
                print("(No documents retrieved. Querying model directly...)")
            for doc, score in last_result["retrieved_documents"]:
                source = doc.metadata.get("source", "document")
                print(f"[{source} | Similarity Score: {score:.4f}]")
                print(f"  {doc.text[:200]}...")

            print("\n--- Model Response ---")
            print(last_result["answer"])

            print("\n--- Introspection Summary ---")
            h_stats = last_result["hidden_stats"]
            print(f"Hidden State Layers: {h_stats['num_layers']} | Token Count: {len(last_result['input_tokens'])}")
            print(f"Attention Layers: {len(last_result['attentions'])} (Layer 0 shape: {last_result['attentions'][0].shape})")
            print("Tip: Run '/heatmap' to visualize attention weights for this query.")


if __name__ == "__main__":
    interactive_chat()
