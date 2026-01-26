"""
Streamlit UI for IFC Annual Report RAG System

Supports both baseline (Practice 1) and advanced (Practice 3) retrieval modes:
- Dense: Vector similarity search (baseline)
- Sparse: BM25 lexical search
- Hybrid: Dense + Sparse with RRF fusion
- Re-ranking: Cross-encoder second stage
- Metadata filtering: Page range, section
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP conflict on macOS

import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore

from rag.settings import load_settings
from rag.gemini_client import build_client
from rag.embeddings import GeminiEmbeddings
from rag.rag_pipeline import retrieve, answer_stream, answer_structured
from rag.observability import build_langfuse, start_trace, end_trace


settings = load_settings()
client = build_client(settings)
langfuse = build_langfuse(settings)

# Paths for advanced retrieval
BM25_INDEX_PATH = settings.faiss_dir.parent / "bm25_index.json"


@st.cache_resource(show_spinner=True)
def load_faiss() -> FAISS:
    if not settings.faiss_dir.exists():
        raise FileNotFoundError(f"Vectorstore not found: {settings.faiss_dir}")
    embeddings = GeminiEmbeddings(client=client, model=settings.embedding_model)
    return FAISS.load_local(
        str(settings.faiss_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource(show_spinner=True)
def load_qdrant() -> QdrantVectorStore:
    if not settings.qdrant_path.exists():
        raise FileNotFoundError(f"Qdrant store not found: {settings.qdrant_path}")
    embeddings = GeminiEmbeddings(client=client, model=settings.embedding_model)
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        path=str(settings.qdrant_path),
        collection_name=settings.qdrant_collection,
    )


@st.cache_resource(show_spinner=True)
def load_bm25():
    """Load BM25 index for sparse/hybrid retrieval."""
    if not BM25_INDEX_PATH.exists():
        return None
    from rag.sparse_retrieval import load_bm25_index
    return load_bm25_index(BM25_INDEX_PATH)


@st.cache_resource(show_spinner=True)
def load_advanced_retriever(store_choice: str):
    """Load advanced retriever with all components."""
    from rag.advanced_retriever import create_advanced_retriever

    dense = load_faiss() if store_choice == "FAISS" else load_qdrant()
    sparse = load_bm25()

    return create_advanced_retriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )


def get_available_sections() -> list:
    """Get list of available sections from indexed documents."""
    return [
        "Introduction",
        "Executive Summary",
        "Letter to Stakeholders",
        "Financial Highlights",
        "Financial Statements",
        "Balance Sheet",
        "Income Statement",
        "Cash Flow Statement",
        "Notes to Financial Statements",
        "Risk Management",
        "Management Discussion",
        "Auditor's Report",
    ]


# Page config
st.set_page_config(page_title="IFC RAG System", layout="wide")
st.title("IFC Annual Report 2024 — RAG (Gemini)")
st.caption("Gemini 2.0 Flash with Vertex AI, FAISS/Qdrant retrieval, Langfuse tracing")

# Sidebar for retrieval configuration
with st.sidebar:
    st.header("Retrieval Settings")

    # Vector store selection
    store_choice = st.selectbox("Vector Store", ["FAISS", "Qdrant"])

    # Retrieval mode
    bm25_available = BM25_INDEX_PATH.exists()
    mode_options = ["Dense (baseline)"]
    if bm25_available:
        mode_options.extend(["Sparse (BM25)", "Hybrid (Dense + BM25)"])

    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        mode_options,
        help="Dense = embedding similarity, Sparse = BM25 keyword, Hybrid = combined"
    )

    # Top K
    top_k = st.slider("Top K", min_value=3, max_value=15, value=5, step=1)

    # Re-ranking option
    st.subheader("Re-ranking")
    enable_reranking = st.checkbox(
        "Enable Cross-Encoder Re-ranking",
        help="Two-stage retrieval: retrieve more candidates, then re-rank with cross-encoder"
    )

    if enable_reranking:
        rerank_top_n = st.slider(
            "Candidates to retrieve",
            min_value=10, max_value=50, value=20,
            help="Number of candidates before re-ranking"
        )
    else:
        rerank_top_n = top_k

    # Metadata filtering
    st.subheader("Metadata Filters")
    enable_page_filter = st.checkbox("Filter by page range")

    if enable_page_filter:
        col1, col2 = st.columns(2)
        with col1:
            page_start = st.number_input("Start page", min_value=1, value=1)
        with col2:
            page_end = st.number_input("End page", min_value=1, value=100)
    else:
        page_start = None
        page_end = None

    enable_section_filter = st.checkbox("Filter by section")
    if enable_section_filter:
        sections = get_available_sections()
        selected_sections = st.multiselect("Sections", sections)
    else:
        selected_sections = None

    # Hybrid weights
    if "Hybrid" in retrieval_mode:
        st.subheader("Hybrid Weights")
        dense_weight = st.slider("Dense weight", 0.0, 1.0, 0.7, 0.1)
        sparse_weight = 1.0 - dense_weight
        st.caption(f"Sparse weight: {sparse_weight:.1f}")
    else:
        dense_weight = 0.7
        sparse_weight = 0.3

    # Display options
    st.subheader("Display")
    show_context = st.checkbox("Show retrieved chunks", value=True)
    show_scores = st.checkbox("Show retrieval scores", value=False)

# Main area - Query input
q = st.text_input("Ask a question about the PDF")

if q:
    # Build retrieval config
    from rag.retrieval_config import RetrievalConfig, MetadataFilter, RerankerConfig, HybridConfig

    # Determine mode
    if "Sparse" in retrieval_mode:
        mode = "sparse"
    elif "Hybrid" in retrieval_mode:
        mode = "hybrid"
    else:
        mode = "dense"

    # Build config
    config = RetrievalConfig(
        mode=mode,
        top_k=top_k,
        reranker=RerankerConfig(
            enabled=enable_reranking,
            top_n=rerank_top_n,
            top_k=top_k,
        ),
        metadata_filter=MetadataFilter(
            page_start=page_start if enable_page_filter else None,
            page_end=page_end if enable_page_filter else None,
            sections=selected_sections if enable_section_filter and selected_sections else None,
        ) if (enable_page_filter or (enable_section_filter and selected_sections)) else None,
        hybrid=HybridConfig(
            enabled=("Hybrid" in retrieval_mode),
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        ),
        preserve_scores=show_scores,
    )

    # Load retriever
    try:
        if mode == "dense" and not enable_reranking and config.metadata_filter is None:
            # Use simple baseline retrieval for maximum compatibility
            db = load_faiss() if store_choice == "FAISS" else load_qdrant()
            docs = retrieve(db, q, k=top_k)
            scores = None
            retrieval_details = None
        else:
            # Use advanced retriever
            retriever = load_advanced_retriever(store_choice)
            result = retriever.retrieve(q, config)
            docs = result.documents
            scores = result.scores if show_scores else None
            retrieval_details = {
                "mode": result.mode,
                "reranked": result.reranked,
                "metadata_filtered": result.metadata_filtered,
            }

    except FileNotFoundError as exc:
        st.error(f"{exc}. Run: python ingest.py")
        st.stop()
    except Exception as exc:
        st.error(f"Retrieval error: {exc}")
        st.stop()

    # Build trace metadata
    retrieved_chunks = [
        {
            "page": d.metadata.get("page"),
            "section": d.metadata.get("section", ""),
            "content": d.page_content[:200],
        }
        for d in docs
    ]

    trace_metadata = {
        "query": q,
        "store": store_choice,
        "mode": mode,
        "reranking": enable_reranking,
        "retrieved_chunks": retrieved_chunks,
    }
    trace = start_trace(langfuse, "rag_query", trace_metadata)

    # Display retrieval info
    if retrieval_details:
        with st.expander("Retrieval Details", expanded=False):
            st.json(retrieval_details)

    # Generate and stream answer
    st.subheader("Answer (streaming)")
    answer_box = st.empty()
    streamed = ""
    for token in answer_stream(client, settings.gemini_model, q, docs):
        streamed += token
        answer_box.markdown(streamed)

    # Structured answer
    st.subheader("Answer (structured)")
    structured = answer_structured(client, settings.gemini_model, q, docs)
    st.json(structured)

    end_trace(trace, {"answer": structured, "raw_answer": streamed})

    # Show retrieved context
    if show_context:
        st.subheader("Retrieved Chunks")

        for i, d in enumerate(docs, 1):
            page = d.metadata.get("page", "?")
            section = d.metadata.get("section", "")
            content_type = d.metadata.get("content_type", "")

            # Build header
            header_parts = [f"Match {i}", f"page {page}"]
            if section:
                header_parts.append(section)
            if content_type:
                header_parts.append(f"[{content_type}]")

            # Add score if available
            if scores and i <= len(scores):
                header_parts.append(f"score: {scores[i-1]:.4f}")

            st.markdown(f"### {' — '.join(header_parts)}")
            st.write(d.page_content)
            st.divider()

# Info section when no query
if not q:
    st.info("""
    **How to use:**
    1. Configure retrieval settings in the sidebar
    2. Enter your question about the IFC Annual Report 2024
    3. View the AI-generated answer and retrieved context

    **Retrieval Modes (Practice 3):**
    - **Dense**: Embedding-based semantic search (baseline)
    - **Sparse**: BM25 keyword matching
    - **Hybrid**: Combined dense + sparse with RRF fusion

    **Advanced Features:**
    - **Re-ranking**: Cross-encoder second stage for better precision
    - **Metadata filtering**: Filter by page range or document section

    Run `python ingest.py --build-bm25 --extract-metadata` to enable all features.
    """)
