# LLM & RAG Model

An end-to-end Retrieval-Augmented Generation (RAG) system built to learn how LLMs and vector search work together. Combines semantic search over a knowledge base with local LLM inference to answer questions grounded in custom documents.

## Quick Start

### Prerequisites
- **Python 3.10+**
- **Ollama** running locally with `phi3` model pulled
  ```bash
  ollama pull phi3
  ```

### Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:aadarshaAB/LLM_and_RAG_model.git
   cd LLM_and_RAG_model
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or: source .venv/bin/activate  # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run the RAG System

```bash
python rag.py
```

You'll see:
```
Total chunks: 82
Embeddings shape: (82, 384)
FAISS index size: 82 vectors, dimension 384

============================================================
RAG Q&A System Ready. Ask your questions (empty line to quit)
============================================================

Q: What is retrieval augmented generation?

Generating answer...

A: Retrieval-augmented generation is a method to improve domain-specific responses...
```

Ask any question relevant to your knowledge base, or press Enter to quit.

## How It Works

### The RAG Pipeline

1. **Chunking** — Documents are split into ~150-word chunks (configurable via `CHUNK_SIZE`). This produces 82 chunks from the 4 `.txt` knowledge base files.

2. **Embedding** — Each chunk is encoded into a 384-dimensional vector using `sentence-transformers` (`all-MiniLM-L6-v2`). Chunks with similar meaning end up close together in this vector space.

3. **Indexing** — Embeddings are stored in a FAISS `IndexFlatL2` index for fast similarity search. This is exact brute-force search (fine for ~100 vectors; would upgrade to approximate indexes like IVF at scale).

4. **Retrieval** — When you ask a question:
   - The question is embedded with the same model
   - FAISS finds the 3 closest chunk embeddings (lowest L2 distance)
   - Chunks are retrieved and ranked by relevance

5. **Prompting** — The top 3 chunks, plus the question, are assembled into a structured prompt that instructs the LLM to answer only from the provided context.

6. **Generation** — The prompt is sent to a local Ollama server running `phi3`, which generates an answer grounded in the retrieved context.

### Key Concepts

- **Retrieval Distance**: Measured in L2 (Euclidean) distance between embedding vectors. Lower = more relevant. Example: 0.73 (very close) vs. 1.2 (less similar). Used to rank retrieved chunks.

- **Chunk Size**: Trade-off between precision and context. Smaller chunks (150 words) retrieve more precisely but carry less surrounding text. Larger chunks (500+ words) include more context but may average over multiple ideas and lose precision. Tested at 5, 100, 150, and 500 words.

- **Ollama Integration**: A local LLM server runs `phi3` for answer generation. Keeps everything offline and under your control (no API calls or rate limits).

## Project Structure

```
LLM_and_RAG_model/
├── rag.py                          # Main RAG pipeline
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
├── CLAUDE.md                       # Development notes & architecture
├── Rag_&_LLM_QnA.md               # Q&A learning log
└── knowledge_base/
    ├── LLM.txt                     # Notes on Large Language Models
    ├── RAG.txt                     # Notes on Retrieval-Augmented Generation
    ├── vector_database.txt         # Notes on vector databases & FAISS
    └── langchain.txt               # Notes on LangChain framework
```

## Configuration

Edit these constants in `rag.py`:

```python
CHUNK_SIZE = 150              # Words per chunk (lower = more precise, higher = more context)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence-transformers model (fast, CPU-friendly)
OLLAMA_URL = "http://localhost:11434/api/generate"  # Local Ollama server
OLLAMA_MODEL = "phi3"         # Model to use (must be pulled with `ollama pull phi3`)
```

## Learning & Documentation

- **`CLAUDE.md`** — High-level architecture, pipeline stages, and development decisions.
- **`Rag_&_LLM_QnA.md`** — Q&A log covering:
  - What `CHUNK_SIZE` does and how to tune it
  - How to identify the "perfect" chunk size for your use case
  - What retrieval distance means and why it matters

Designed for learning: each Q&A explains *why*, not just *what*.

## Examples

### In-Domain Question (Answer from Retrieved Chunks)
```
Q: What is retrieval augmented generation?

A: Retrieval-augmented generation is a technique that enables large language models 
(LLMs) to retrieve and incorporate new information from external data sources. This 
method uses an external document database to supplement the information available to 
the LLM from its own training data...
```

### Out-of-Domain Question (Proper Refusal)
```
Q: How does FAISS work internally?

A: I'm sorry, but I cannot provide information on how FAISS works based on the given 
context. The provided context does not contain information about FAISS...
```

The system correctly refuses to hallucinate when context doesn't cover the question.

## Technical Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| **Embeddings** | `sentence-transformers` | Convert text chunks → 384-dim vectors |
| **Vector Search** | `faiss-cpu` | Fast nearest-neighbor search over embeddings |
| **LLM Inference** | `ollama` + `phi3` | Local answer generation (no API calls) |
| **Text Processing** | `Python` stdlib | Document loading, chunking, text manipulation |

## Known Limitations & Future Work

1. **No persistence**: Embeddings/index are rebuilt on every run. For production, save the FAISS index to disk.

2. **No distance threshold**: FAISS always returns top-k chunks even if none are truly relevant. A guardrail (reject matches with distance > 2.0) would prevent weak retrievals.

3. **Resume PDF not ingested**: The resume PDF in `knowledge_base/` needs text extraction before it can be indexed.

4. **Single LLM**: Only phi3 is tested. Larger models (Llama, Mistral) would likely follow instructions better.

5. **Exact search only**: `IndexFlatL2` is brute-force. At 10K+ vectors, upgrade to `IndexIVFFlat` (approximate nearest neighbors).

## Contributing

This is a learning project. Feel free to:
- Experiment with different chunk sizes and embedding models
- Add new documents to `knowledge_base/`
- Test with different LLMs via Ollama
- Measure retrieval quality and refine the pipeline

## References

- [Retrieval-Augmented Generation Paper](https://arxiv.org/abs/2005.11401)
- [Sentence-Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.ai/)

---

**Learning Goal**: Understand how RAG systems work by building one from scratch, step by step, rather than using a high-level framework.
