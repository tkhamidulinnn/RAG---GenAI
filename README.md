# RAG Practice: Foundational to Advanced RAG System

A comprehensive RAG system that extracts content from PDFs (text, tables, images), stores embeddings in FAISS and Qdrant vector stores, retrieves relevant content via similarity search, and generates answers using Gemini.

**Practices Implemented:**
- Practice 1: Foundational Text-Based RAG (Naive RAG)
- Practice 2: RAG Evaluation Pipeline (RAGAS metrics)
- Practice 3: Advanced Retrieval Techniques
- Practice 4: System Optimization (Semantic Caching, Multi-hop Retrieval)
- Practice 5: Multimodal RAG (Tables + Charts/Graphs)

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

# Multimodal ingestion (Practice 5)
python ingest.py --multimodal --skip-qdrant

# Full multimodal with all features
python ingest.py --extract-metadata --build-bm25 --multimodal --skip-qdrant
```

This creates:
- `vectorstore_ifc/` — FAISS index (text chunks)
- `qdrant_local/` — Qdrant local storage
- `bm25_index.json` — BM25 sparse index (if `--build-bm25`)
- `table_index/` — FAISS index for tables (if `--multimodal`)
- `image_index/` — FAISS index for images (if `--multimodal`)
- `artifacts/tables_extracted/` — Extracted table data
- `artifacts/images_extracted/` — Extracted images with VLM descriptions

**Chunking**: 1000 characters per chunk, 150 character overlap.

## Run the UI

```bash
# Text-based RAG (Practices 1-4)
python -m streamlit run app.py

# Multimodal RAG (Practice 5)
python -m streamlit run app_multimodal.py
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

## Practice 4: System Optimization

Practice 4 adds performance optimizations to reduce latency and handle complex questions:

### Features

| Feature | Description | UI Control |
|---------|-------------|------------|
| **Semantic Caching** | Cache answers for similar questions | Checkbox + threshold slider |
| **Multi-hop Retrieval** | Decompose complex questions into sub-queries | Checkbox |

### Semantic Caching

Caches query-answer pairs using embeddings-based similarity lookup:
- Computes embedding for each query
- Compares against cached query embeddings using cosine similarity
- Returns cached answer if similarity exceeds threshold (default: 0.92)
- Persists cache to disk (`semantic_cache.json`)

**Benefits:**
- Faster response for repeated/similar questions
- Reduced API costs (no LLM call on cache hit)
- Observable hit/miss statistics

**Configuration:**
- `Similarity threshold`: Higher = stricter matching, fewer cache hits
- `Max entries`: Maximum cached query-answer pairs (default: 500)

### Multi-hop Retrieval

Decomposes complex questions requiring information from multiple document sections:

1. **Complexity Detection**: Analyzes query for multi-part indicators
2. **Query Decomposition**: Breaks into focused sub-queries
3. **Sequential Retrieval**: Retrieves documents for each sub-query
4. **Context Aggregation**: Combines results using round-robin selection

**Triggers decomposition when:**
- Query contains conjunctions ("and", "also", "both")
- Query asks for comparisons ("difference between", "compare")
- Query has multiple question marks
- Query exceeds minimum word count (10 words)

### Architecture

```
rag/
├── semantic_cache.py      # Embeddings-based caching
└── multihop_retrieval.py  # Query decomposition
```

### Usage

Enable in the Streamlit sidebar under "Optimization (Practice 4)":
1. Check "Enable Semantic Cache" for caching
2. Adjust similarity threshold (0.80-0.99)
3. Check "Enable Multi-hop Retrieval" for complex questions

### References

- [Anthropic: Caching in RAG](https://www.anthropic.com/news/prompt-caching)
- [LangChain: Query Decomposition](https://python.langchain.com/docs/tutorials/query_analysis/)

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

---

## Practice 5: Multimodal RAG

Practice 5 adds multimodal retrieval for tables and charts/graphs:

### Features

| Feature | Description | Flag/Control |
|---------|-------------|--------------|
| **Table Extraction** | Extract tables with structure preservation | `--multimodal` or `--tables-only` |
| **Image Extraction** | Extract charts/graphs with VLM descriptions | `--multimodal` or `--images-only` |
| **Multimodal Retrieval** | Search across text, tables, and images | UI mode selector |
| **Smart Routing** | Auto-detect query intent and route to relevant modalities | "Auto" mode |

### Table Processing (5.1)

Each table is extracted with dual representations:
- **R1 (Retrieval)**: Markdown-formatted text for semantic search
- **R2 (Structured)**: JSON with headers, rows, and normalized numeric values

**Extraction pipeline:**
1. Detect tables using PyMuPDF (fallback: pdfplumber)
2. Extract table structure (rows, columns, headers)
3. Detect captions from surrounding text
4. Normalize numeric values (currency, percentages)
5. Create both representations and index

**Table QA prompts** force:
- Step-by-step reasoning
- Exact cell value extraction
- Calculation with shown inputs
- Evidence citations (page, table_id, row/column)

### Image/Chart Processing (5.2)

Charts and graphs are processed using VLM (Vision Language Model):
1. Extract embedded images from PDF
2. Filter by size (skip icons/logos)
3. Send to Gemini VLM for description
4. Extract: chart type, axes, values, trends
5. Index textual description for retrieval

**VLM extracts:**
- Chart type (bar, line, pie, area, scatter)
- Axis labels and units
- Key data points and values
- Trends and insights
- Confidence level

### Setup for Practice 5

```bash
# Install dependencies (pdfplumber optional but recommended)
pip install pdfplumber

# Run multimodal ingestion
python ingest.py --multimodal --skip-qdrant

# Or extract tables only
python ingest.py --tables-only --skip-qdrant

# Or extract images only
python ingest.py --images-only --skip-qdrant

# Full ingestion with all features
python ingest.py --extract-metadata --build-bm25 --multimodal --skip-qdrant
```

### Running Multimodal UI

```bash
python -m streamlit run app_multimodal.py
```

**UI Controls:**
- **Mode selector**: Auto, Text only, Tables only, Images only, Mixed
- **Top-K sliders**: Separate K for text, tables, images
- **Display toggles**: Show/hide text chunks, tables, images
- **Raw data view**: Inspect structured table JSON

### Retrieval Modes

| Mode | Description |
|------|-------------|
| **Auto** | Detects query intent using keywords and routes accordingly |
| **Text only** | Baseline text chunk retrieval |
| **Tables only** | Search only table representations |
| **Images only** | Search only image descriptions |
| **Mixed** | Search all modalities and merge results |

**Intent detection keywords:**
- Tables: "table", "row", "column", "value", "percent", "FY", "$", "revenue", etc.
- Images: "chart", "graph", "figure", "trend", "visualization", etc.

### Demo Queries

**Table queries:**
- "What was the total revenue in FY24?"
- "Compare IFC's assets between 2023 and 2024"
- "What percentage of disbursements went to climate finance?"

**Chart queries:**
- "What trends are shown in the investment growth chart?"
- "Describe the regional distribution of projects"
- "What does the portfolio composition figure show?"

**Combined queries:**
- "Explain the financial performance and how it's visualized"
- "What are the key metrics and their trends over time?"

### Architecture

```
rag/
├── table_extraction.py      # Table extraction with dual representation
├── image_extraction.py      # Image extraction with VLM descriptions
├── multimodal_indexing.py   # Unified indexing and retrieval
├── multimodal_prompts.py    # Table QA, Visual QA, Combined prompts
app_multimodal.py            # Streamlit UI for multimodal RAG
```

### Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Table extraction errors | Dual extraction (PyMuPDF + pdfplumber fallback) |
| Poor image descriptions | Confidence scoring, caption detection |
| Hallucination | Strict grounding prompts, evidence citations required |
| Missing evidence | Explicit refusal when data not found |
| Slow VLM processing | Cached descriptions, batch processing |

### References

- [Google Cloud: Multimodal RAG](https://cloud.google.com/blog/products/ai-machine-learning/multimodal-rag-with-gemini)
- [LangChain: Multi-Vector Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector)
