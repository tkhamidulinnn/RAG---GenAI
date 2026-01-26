# RAG Practice: Foundational to Advanced RAG System

A text-only RAG system that extracts text from a PDF, chunks it, stores embeddings in FAISS and Qdrant vector stores, retrieves relevant chunks via similarity search, and generates answers using Gemini.

**Practices Implemented:**
- Practice 1: Foundational Text-Based RAG (Naive RAG)
- Practice 2: RAG Evaluation Pipeline (RAGAS metrics)
- Practice 3: Advanced Retrieval Techniques

## Setup (Step by Step)

### Step 1: Install Google Cloud CLI

**macOS:**
```bash
brew install google-cloud-sdk
```

**Windows:** Download from https://cloud.google.com/sdk/docs/install

**Linux:**
```bash
curl https://sdk.cloud.google.com | bash
```

After install, restart terminal.

### Step 2: Create GCP Project

1. Go to https://console.cloud.google.com
2. Click dropdown at top left (next to "Google Cloud")
3. Click "New Project"
4. Enter name (e.g., `my-rag-project`)
5. Click "Create"
6. Wait 30 seconds, then select your new project from dropdown

### Step 3: Enable Vertex AI API

1. In GCP Console, go to: https://console.cloud.google.com/apis/library/aiplatform.googleapis.com
2. Make sure your project is selected at the top
3. Click "Enable"
4. Wait until it says "API enabled"

### Step 4: Login to GCP from Terminal

```bash
gcloud auth application-default login
```

This opens a browser. Sign in with your Google account and allow access.

### Step 5: Set Environment Variables

Find your project ID:
- In GCP Console, click the project dropdown at top
- Copy the ID (not the name) — looks like `my-rag-project` or `my-rag-project-123456`

Set the variables:
```bash
export GCP_PROJECT=your-project-id
export GCP_LOCATION=us-central1
```

### Step 6: Create Virtual Environment and Install Dependencies

```bash
# Create venv (if not exists or broken)
python3 -m venv .venv

# Activate venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 7: Verify Setup

```bash
python -c "from rag.settings import load_settings; print('OK:', load_settings().gcp_project)"
```

Should print: `OK: your-project-id`

## Ingestion

Extract text from PDF and build vector stores:

```bash
# Basic ingestion (FAISS only, quick start)
python ingest.py --skip-qdrant

# Basic ingestion (FAISS + Qdrant)
python ingest.py

# With Practice 3 features (metadata + BM25)
python ingest.py --extract-metadata --build-bm25

# Full ingestion with all features
python ingest.py --extract-metadata --build-bm25 --skip-qdrant
```

This creates:
- `vectorstore_ifc/` — FAISS index
- `qdrant_local/` — Qdrant local storage
- `bm25_index.json` — BM25 sparse index (if `--build-bm25`)

**Chunking**: 1000 characters per chunk, 150 character overlap.

## Run the UI

```bash
python -m streamlit run app.py
```

Open http://localhost:8501, select FAISS or Qdrant, enter a question.

**Note:** Stop Streamlit (Ctrl+C) before re-running `ingest.py` — Qdrant local storage cannot be accessed by two processes simultaneously.

## FAISS vs Qdrant

| Aspect | FAISS | Qdrant |
|--------|-------|--------|
| Setup | Local files, no server | Local files (or Docker) |
| Persistence | Disk files (`index.faiss`, `index.pkl`) | Local folder or Docker volume |
| Concurrency | Multiple processes OK | Single process only (local mode) |
| Scalability | Single-node, in-memory | Distributed (server mode) |
| Best for | Prototyping, small datasets | Production, larger datasets |

---

## Practice 3: Advanced Retrieval

Practice 3 improves retrieval quality with optional, modular techniques:

### Features

| Feature | Description | Flag |
|---------|-------------|------|
| **Re-ranking** | Cross-encoder two-stage retrieval | UI checkbox / `--rerank` |
| **Metadata Filtering** | Filter by page range or section | UI sidebar |
| **BM25 Sparse Retrieval** | Lexical keyword matching | `--build-bm25` at ingest |
| **Hybrid Retrieval** | Dense + Sparse with RRF fusion | UI dropdown |

### Retrieval Modes

- **Dense (baseline)**: Embedding-based semantic search
- **Sparse**: BM25 lexical/keyword search
- **Hybrid**: Combined dense + sparse using Reciprocal Rank Fusion (RRF)

### Re-ranking

Two-stage retrieval:
1. Retrieve Top-N candidates (default: 20) using dense/hybrid
2. Re-score using cross-encoder model
3. Return Top-K final results

Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` by default.

### Setup for Practice 3

```bash
# Re-ingest with metadata and BM25
python ingest.py --extract-metadata --build-bm25 --skip-qdrant

# Install cross-encoder (if not already installed)
pip install sentence-transformers
```

### Architecture

```
rag/
├── retrieval_config.py    # Configuration dataclasses
├── reranker.py            # Cross-encoder re-ranking
├── sparse_retrieval.py    # BM25 implementation
├── hybrid_retrieval.py    # RRF fusion
├── advanced_retriever.py  # Unified interface
└── metadata_extractor.py  # Section/content type detection
```

### References

- [Pinecone: Advanced RAG Techniques](https://www.pinecone.io/learn/advanced-rag-techniques/)
- [Qdrant: Hybrid Search Tutorial](https://qdrant.tech/documentation/tutorials/hybrid-search/)
- [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [HuggingFace: Sentence Transformers](https://www.sbert.net/docs/pretrained_cross-encoders.html)

---

## Evaluation (Practice 2)

Run the evaluation pipeline:

```bash
# Quick test (k=5 only)
python run_evaluation.py --quick

# Standard evaluation (k=3,5,8)
python run_evaluation.py

# Full evaluation with LLM judge
python run_evaluation.py --full

# With Practice 3 features
python run_evaluation.py --modes dense hybrid --test-reranking
```

Results are saved to `eval/results/`:
- `eval_samples_*.json` — Raw evaluation samples
- `experiments_*.json` — Aggregated metrics
- `evaluation_report.md` — Markdown report

### Metrics (RAGAS-style)

| Metric | Description |
|--------|-------------|
| Faithfulness | Answer grounded in retrieved context (no hallucination) |
| Answer Relevance | Answer directly addresses the question |
| Context Precision | Retrieved contexts are relevant (low noise) |
| Context Recall | Retrieved contexts cover ground truth information |

### Comparing Retrieval Modes

```bash
# Compare dense baseline vs hybrid vs reranked
python run_evaluation.py --modes dense sparse hybrid --test-reranking --quick
```

This runs experiments across all combinations and generates a comparison report.
