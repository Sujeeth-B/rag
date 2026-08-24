from rag_pipeline import FlexibleRAGPipeline
from introspection import ModelIntrospector


def main():
    print("Initializing Flexible RAG Pipeline with Model Introspection...")
    pipeline = FlexibleRAGPipeline(model_name="gpt2", top_k=2)

    # 1. Ingest Knowledge Base
    documents = [
        "Retrieval-Augmented Generation (RAG) is an AI framework for improving the quality of LLM responses by grounding the model on external sources of knowledge.",
        "Attention mechanisms in Transformer architectures compute dynamic weight matrices (Query, Key, Value) to represent token interactions across layers.",
        "Hidden states in deep neural networks represent learned intermediate embeddings that preserve context and semantic representations at each layer."
    ]
    pipeline.ingest_texts(documents)

    # 2. View Model Parameters
    print("\n" + "="*50)
    print("1. MODEL PARAMETER INSPECTION")
    print("="*50)
    ModelIntrospector.print_parameter_summary(pipeline.generator.model)

    # 3. Query RAG Pipeline
    query = "What is Retrieval-Augmented Generation?"
    print("\n" + "="*50)
    print(f"2. RUNNING RAG QUERY: '{query}'")
    print("="*50)
    result = pipeline.query(query, max_new_tokens=20)

    print("\n[Retrieved Documents]:")
    for doc, score in result["retrieved_documents"]:
        print(f" - Score: {score:.4f} | Text: {doc.text}")

    print("\n[Generated Output]:")
    print(result["answer"])

    # 4. View Model Hidden Representations
    print("\n" + "="*50)
    print("3. HIDDEN REPRESENTATIONS ANALYSIS")
    print("="*50)
    hidden_stats = result["hidden_stats"]
    print(f"Number of Hidden Layers (including embeddings): {hidden_stats['num_layers']}")
    for l_info in hidden_stats["layer_stats"][:3]:  # Print first 3 layers
        print(f"Layer {l_info['layer']}: Shape = {l_info['shape']} | Mean = {l_info['mean']:.4f} | Std = {l_info['std']:.4f}")

    # 5. View Model Attention Weights & Save Heatmap
    print("\n" + "="*50)
    print("4. ATTENTION MAP INSPECTION & HEATMAP GENERATION")
    print("="*50)
    attentions = result["attentions"]
    print(f"Number of Attention Layers: {len(attentions)}")
    print(f"Attention Layer 0 Tensor Shape: {attentions[0].shape}")

    # Save Heatmap plot
    save_path = "attention_heatmap.png"
    ModelIntrospector.plot_attention_heatmap(
        attentions=attentions,
        tokens=result["input_tokens"],
        layer_idx=-1,  # Last layer
        save_path=save_path
    )


if __name__ == "__main__":
    main()
