# local-rag

Retrieval Augmented Generation running on a local model.

## Architecture

```
User Question
     |
     v
[1. EMBED the question]        <-- sentence-transformers (converts text to vectors)
     |
     v
[2. SEARCH your documents]     <-- cosine similarity (find most similar chunks)
     |
     v
[3. BUILD a prompt]            <-- combine question + retrieved context
     |
     v
[4. SEND to Ollama Gemma3:4b]  <-- local LLM (generates the answer)
     |
     v
Answer
```

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Convert text to dense vectors |
| Similarity search | `scikit-learn` | Cosine similarity ranking |
| Environment config | `python-dotenv` | Load API tokens and model names |
| Local LLM | Ollama (Gemma3:4b) | Answer generation |

## Project Structure

```
local-rag/
├── embedding.py       # Generate embeddings from data.txt and save to pickle_dup.pkl
├── retrival.py        # Load embeddings, retrieve top-k similar chunks for a query
├── main.py            # Entry point — full pipeline orchestration (in progress)
├── data.txt           # Source documents to index
├── pickle_dup.pkl     # Pre-computed embeddings cache
└── .env               # HF_TOKEN and SENTENCE_TRANSFORMER_MODEL
```

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) with Gemma3:4b pulled (`ollama pull gemma3:4b`)
- HuggingFace token (set `HF_TOKEN` in `.env`)

## Setup & Usage

```bash
# Install dependencies
uv sync

# Generate embeddings from data.txt (run once, or when data changes)
uv run python embedding.py

# Run similarity retrieval for a query
uv run python retrival.py
```
