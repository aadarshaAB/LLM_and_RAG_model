# RAG & LLM Q&A Log

Questions asked during this learning project and the answers given, kept for reference.

## Q: What is CHUNK_SIZE, and what does increasing or decreasing it do exactly?

`CHUNK_SIZE` is the target number of words per chunk when splitting a document before embedding (see `chunk_text` in `rag.py`). Each chunk becomes one vector in the FAISS index, so this setting decides how finely the knowledge base gets sliced before anything else in the pipeline happens.

**Increasing it** (e.g. 500+ words):
- Fewer, larger chunks → smaller index, faster to build.
- Each chunk's embedding is an average over more content. If a chunk covers several sub-topics, the embedding gets "smeared" across all of them, so it matches any single question less precisely.
- Retrieved chunks carry more surrounding context for the LLM to read, which can help reasoning — but also pull in more irrelevant filler alongside the relevant part.

**Decreasing it** (e.g. 150 → 50 → 5 words):
- More, smaller chunks → bigger index (still trivial to search at this scale).
- Each embedding represents a narrower, more specific idea, so retrieval matches more precisely. Measured directly in this project: dropping `CHUNK_SIZE` from 500 to 150 improved the top match's distance from 0.9028 to 0.7296 for the same test question.
- Push it too far (e.g. 5 words) and chunks stop being coherent text — a short fragment like `"companies (bottom right). The majority"` doesn't carry real meaning, so its embedding becomes noisy rather than more precise. Tested at `CHUNK_SIZE = 5`: 26 chunks became 2,397, and retrieval quality degraded into near-random fragments instead of improving further.

The practical trade-off: shrink chunk size to sharpen retrieval precision, but stop once chunks are still complete, coherent thoughts (a sentence or two) — going smaller than that adds noise, not precision.

---

## Q: How do I identify the perfect CHUNK_SIZE that fits my RAG model?

There's no one "perfect" size — it depends on your documents and use case. But here's how to find a good one:

**1. Start with your document type:**
- Dense academic/technical docs (lots of specialized concepts close together) → smaller chunks (100–150 words) catch more precision.
- Conversational or narrative docs (ideas are more spread out) → larger chunks (200–300 words) preserve context.
- Your knowledge base is technical notes on LLM/RAG/vector DBs → start with 100–150 words.

**2. Test a few sizes empirically:**
Run the same test questions with different `CHUNK_SIZE` values and watch two things:
- **Retrieval distance**: lower is better. If the top match's distance drops when you shrink chunks, that size is more precise.
- **Retrieved text quality**: do the top results actually contain relevant info, or are they noisy? Display the first 200 chars (like the script does) and eyeball them.

Example from this project:
- `CHUNK_SIZE = 500` → top match distance = 0.9028
- `CHUNK_SIZE = 150` → top match distance = 0.7296 ← better precision
- `CHUNK_SIZE = 5` → 2,397 fragments, retrieval became noise ← too small

**3. Check the chunk count:**
- If you went from 26 chunks to 78 (500 → 150), that's a healthy increase, still manageable.
- If you jump to 2,000+ chunks, you've probably gone too small.

**4. Real-world test:**
The best validation is end-to-end: ask your actual use-case questions, look at the top 3 retrieved chunks, and ask "would an LLM have enough context to answer this correctly?" If yes, the size works. If chunks are irrelevant or too fragmented, adjust up or down.

**Rule of thumb for this project:** Start at 100–150 words, test with your actual questions, then shift up if chunks feel too small or down if they're too broad. You're looking for the point where retrieval finds *relevant* chunks, not just *close* vectors.

---

## Q: What is retrieval distance?

Retrieval distance is a **number that measures how far apart two vectors (embeddings) are in vector space** — think of it like measuring distance between two points on a map, but in 384-dimensional space instead of 2D.

**How embeddings work:**
When you embed text with `sentence-transformers`, it converts that text into a vector of numbers (384 dimensions in this project). Two chunks with similar *meaning* end up close to each other in this space; chunks with different meanings are far apart.

**The distance score:**
Our code uses **L2 distance** (Euclidean distance, the "as-the-crow-flies" distance between two points). FAISS's `IndexFlatL2` computes this for every stored chunk and ranks them.

**Lower distance = more relevant:**
- Distance `0.7296` (top match for "what is RAG?") = very close, high semantic similarity
- Distance `0.8751` (second match) = still close but less similar
- Distance `0.9147` (third match) = even farther away, but still the best FAISS could find

**In practice (from your project output):**
```
[rag.txt chunk 0] distance=0.7296   ← closest match, best answer
[rag.txt chunk 1] distance=0.8751   ← next closest
[vector_database.txt chunk 2] distance=0.9147   ← third closest
```

The top result has the lowest distance because its embedding is nearest to your question's embedding in vector space — both are "talking about RAG" so they're positioned near each other.

**Why it matters for RAG:**
A low distance means the retrieved chunk is semantically relevant to your question. If distances are all high (like 2.0+), FAISS still returns top-k results, but they're all poor matches — that's when you'd want to add a distance threshold to reject weak matches and say "I don't know" instead (that's the guardrail we noted earlier when phi-3 answered "Paris" despite no France context — all distances were >0.8 because nothing actually matched).


