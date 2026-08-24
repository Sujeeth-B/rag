# Flexible RAG Pipeline with Model Introspection

A fully flexible Retrieval-Augmented Generation (RAG) framework built from modular components, with comprehensive model parameter inspection, attention map visualization, and hidden representation analysis.

## Features
- **Flexible Document Ingestion & Chunking**: Configurable chunk sizes and overlaps.
- **Customizable Vector Store & Retriever**: Similarity search with support for custom embeddings and reranking functions.
- **Introspective Generation Module**: Captures layer-wise attention weights and hidden states during text generation.
- **Model Parameter & Introspection Tools**:
  - Full parameter summary (layer shapes, parameter counts, trainability).
  - Layer-wise hidden state statistics (mean, std, min, max, norms).
  - Attention heatmaps per layer/head saved as images.

## Architecture
- `ingestion.py`: Text chunking and Document wrapper.
- `vector_store.py`: In-memory vector store for embedding-based similarity search.
- `retriever.py`: Modular retriever supporting custom reranking functions.
- `generator.py`: HuggingFace transformer wrapper with attention and hidden state output enabled.
- `introspection.py`: Tools to inspect model parameters, plot attention maps, and analyze hidden representations.
- `rag_pipeline.py`: Pipeline orchestrator linking retriever, prompt builder, generator, and introspector.
- `chat.py`: Interactive chat CLI supporting document/file uploads, query answering, and model introspection commands.
- `main.py`: Interactive demo script.

## Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Interactive Chat
```bash
python chat.py
```

Inside `chat.py`:
- `/upload <file_path>` to ingest any file or document into RAG memory.
- `/add <text>` to add raw text snippets.
- `/params` to view layer parameters.
- `/heatmap` to save the last query's attention heatmap.
- `/exit` to quit.

### 3. Run Static Demo Script
```bash
python main.py
```
