# Architecture

## System Overview

The Enterprise Hybrid RAG Knowledge Assistant implements a complete Retrieval-Augmented Generation pipeline with three retrieval strategies: vector-only, hybrid (dense + BM25), and hybrid with cross-encoder reranking.

## Ingestion Pipeline

```
Documents (PDF/TXT/MD)
    ↓
Document Parsing (page-by-page for PDFs)
    ↓
Text Cleaning (normalize whitespace, line endings)
    ↓
Chunking (RecursiveCharacterTextSplitter, ~500 characters, 80 character overlap)
    ↓
Metadata Creation (chunk_id, doc_id, page_number, chunk_index)
    ↓
Embedding Generation (BAAI/bge-small-en-v1.5)
    ↓
Qdrant Upsert (must succeed first)
    ↓
Chunk Registry Persist (JSONL)
    ↓
BM25 Index Rebuild
```

## Query Pipeline

```
User Question
    ↓
Query Processing
    ↓
┌─────────────────┬──────────────────┐
│                 │                  │
Dense Retrieval   BM25 Retrieval
(Qdrant)          (rank_bm25)
│                 │
└────────┬────────┘
         ↓
  RRF Hybrid Fusion
         ↓
  Candidate Chunks (~10-20)
         ↓
  Cross-Encoder Reranking (optional)
         ↓
  Top Context Chunks (4-5)
         ↓
  Grounded LLM Prompt (with [S1], [S2] labels)
         ↓
  Answer + Source Citations
```

## Key Design Decisions

1. **Page-independent chunking**: PDFs are chunked per-page to preserve page metadata accurately
2. **RRF over score addition**: Dense similarity and BM25 scores are on different scales; RRF normalizes via rank
3. **Lazy model loading**: Embedding and reranker models load on first use to keep startup fast
4. **JSONL chunk registry**: Simple persistent storage without requiring a database
5. **BM25 rebuild on ingest**: Acceptable for portfolio-scale, mentioned as a limitation
6. **Character-based chunking**: Uses Python `len()` for chunk size measurement (not token-based)
7. **Safe persistence order**: Qdrant upsert must succeed before chunk registry is written, preventing BM25/Qdrant corpus drift
8. **Stable citation labels**: LLM context uses [S1], [S2] labels to prevent invented references
9. **Sync route handlers**: CPU-heavy model inference runs in FastAPI's threadpool, not blocking the async event loop
