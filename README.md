# RAG Chroma Repo

This repository implements a Retrieval-Augmented Generation (RAG) pipeline using:
- ChromaDB for vector storage
- OpenAI for embeddings and LLM
- S3 for chunk text storage
- FastAPI for the API
- Re-ranking with batched parallel LLM scoring
- Streaming, pagination, and conversational memory

Quick start:
1. Copy .env.example to .env and fill keys.
2. Start local services:
   docker compose up --build
3. Ingest example:
   python scripts/ingest_example.py
4. Query:
   curl -X POST "http://localhost:8000/search" -H "Content-Type: application/json" -d '{"query":"Explain the main point","k":4,"rerank":true}'
