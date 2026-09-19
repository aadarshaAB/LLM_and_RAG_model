from pathlib import Path

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

KNOWLEDGE_BASE_DIR = Path("knowledge_base")
CHUNK_SIZE = 150  # target words per chunk
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"



def load_documents(directory: Path) -> dict[str, str]:
    """Read every .txt file in the directory into {filename: full_text}."""
    docs = {}
    for file_path in sorted(directory.glob("*.txt")):
        docs[file_path.name] = file_path.read_text(encoding="utf-8")
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split text into chunks of ~chunk_size words, no overlap."""
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


def chunk_documents(directory: Path) -> list[dict]:
    """Turn every document in the directory into a flat list of chunk records."""
    chunks = []
    for filename, text in load_documents(directory).items():
        for i, chunk in enumerate(chunk_text(text)):
            chunks.append({"source": filename, "chunk_id": i, "text": chunk})
    return chunks

def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    """Embed every chunk's text, returning a (num_chunks, embedding_dim) array."""
    texts = [c["text"] for c in chunks]
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a flat (exact search) FAISS index over the embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def search(
    question: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatL2,
    chunks: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Embed the question, search the FAISS index, and return the top_k matching chunks."""
    query_embedding = model.encode([question], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for distance, idx in zip(distances[0], indices[0]):
        results.append({**chunks[idx], "distance": float(distance)})
    return results


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """Combine instructions, retrieved context, and the question into one prompt."""
    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in retrieved_chunks)
    return (
        "Answer the question using only the context below. "
        "If the context does not contain the answer, say you don't know — "
        "do not make anything up.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def ask_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Send the prompt to a local Ollama server and return the generated answer."""
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
    )
    response.raise_for_status()
    return response.json()["response"]


def answer_question(
    question: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatL2,
    chunks: list[dict],
    top_k: int = 3,
) -> str:
    """Full RAG pipeline: retrieve top_k chunks, build the prompt, generate the answer."""
    retrieved = search(question, model, index, chunks, top_k)
    prompt = build_prompt(question, retrieved)
    return ask_ollama(prompt)


if __name__ == "__main__":
    chunks = chunk_documents(KNOWLEDGE_BASE_DIR)
    print(f"Total chunks: {len(chunks)}\n")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embed_chunks(chunks, model)
    print(f"Embeddings shape: {embeddings.shape}")

    index = build_faiss_index(embeddings)
    print(f"FAISS index size: {index.ntotal} vectors, dimension {index.d}\n")

    print("=" * 60)
    print("RAG Q&A System Ready. Ask your questions (empty line to quit)")
    print("=" * 60)

    while True:
        question = input("\nQ: ").strip()
        if not question:
            break

        print("\nGenerating answer...\n")
        answer = answer_question(question, model, index, chunks)
        print(f"A: {answer}")
        print("-" * 60)
