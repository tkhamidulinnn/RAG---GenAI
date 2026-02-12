"""
Multimodal RAG UI for IFC Annual Report (Practice 5)

Supports retrieval across:
- Text chunks (baseline)
- Tables (extracted with structure)
- Images/Charts (VLM-described)

Run with: python -m streamlit run app_multimodal.py
"""
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st

from rag.settings import load_settings
from rag.gemini_client import build_client
from rag.embeddings import GeminiEmbeddings
from rag.observability import build_langfuse, start_trace, end_trace

# Multimodal imports
from rag.multimodal_indexing import create_multimodal_retriever, load_tables, load_images
from rag.multimodal_prompts import (
    answer_multimodal_qa_stream,
    answer_multimodal_structured,
    build_table_qa_prompt,
    build_visual_qa_prompt,
)
from rag.table_extraction import load_tables as load_extracted_tables
from rag.image_extraction import load_images as load_extracted_images


settings = load_settings()
client = build_client(settings)
langfuse = build_langfuse(settings)

# Paths
TEXT_INDEX_DIR = settings.faiss_dir
TABLE_INDEX_DIR = settings.faiss_dir.parent / "table_index"
IMAGE_INDEX_DIR = settings.faiss_dir.parent / "image_index"
TABLES_DIR = settings.artifacts_dir / "tables_extracted"
IMAGES_DIR = settings.artifacts_dir / "images_extracted"


def check_multimodal_available():
    """Check which multimodal indices are available."""
    return {
        "text": TEXT_INDEX_DIR.exists(),
        "tables": TABLE_INDEX_DIR.exists(),
        "images": IMAGE_INDEX_DIR.exists(),
    }


@st.cache_resource(show_spinner=True)
def load_multimodal_retriever():
    """Load multimodal retriever with all available indices."""
    embeddings = GeminiEmbeddings(client=client, model=settings.embedding_model)

    return create_multimodal_retriever(
        text_index_dir=TEXT_INDEX_DIR if TEXT_INDEX_DIR.exists() else None,
        table_index_dir=TABLE_INDEX_DIR if TABLE_INDEX_DIR.exists() else None,
        image_index_dir=IMAGE_INDEX_DIR if IMAGE_INDEX_DIR.exists() else None,
        tables_dir=TABLES_DIR if TABLES_DIR.exists() else None,
        images_dir=IMAGES_DIR if IMAGES_DIR.exists() else None,
        embeddings=embeddings,
    )


@st.cache_data
def get_table_data():
    """Load extracted table data for display."""
    if TABLES_DIR.exists():
        return load_extracted_tables(TABLES_DIR)
    return []


@st.cache_data
def get_image_data():
    """Load extracted image data for display."""
    if IMAGES_DIR.exists():
        return load_extracted_images(IMAGES_DIR)
    return []


# Page config
st.set_page_config(page_title="IFC RAG - Multimodal", layout="wide")
st.title("IFC Annual Report 2024 — Multimodal RAG")
st.caption("Practice 5: Text + Tables + Charts/Graphs retrieval")

# Check availability
available = check_multimodal_available()

# Sidebar
with st.sidebar:
    st.header("Multimodal Settings")

    # Show availability status
    st.subheader("Available Indices")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Text", "✓" if available["text"] else "✗")
    with col2:
        st.metric("Tables", "✓" if available["tables"] else "✗")
    with col3:
        st.metric("Images", "✓" if available["images"] else "✗")

    if not any(available.values()):
        st.error("No indices found. Run: python ingest.py --multimodal")
        st.stop()

    # Retrieval mode
    st.subheader("Retrieval Mode")
    mode_options = ["Auto (Smart routing)"]
    if available["text"]:
        mode_options.append("Text only")
    if available["tables"]:
        mode_options.append("Tables only")
    if available["images"]:
        mode_options.append("Images only")
    if available["text"] and (available["tables"] or available["images"]):
        mode_options.append("Mixed (all modalities)")

    retrieval_mode = st.selectbox(
        "Mode",
        mode_options,
        help="Auto detects query intent and routes accordingly"
    )

    # Top K settings
    st.subheader("Retrieval Settings")
    top_k_text = st.slider("Text chunks (K)", 1, 10, 5) if available["text"] else 0
    top_k_tables = st.slider("Tables (K)", 1, 5, 3) if available["tables"] else 0
    top_k_images = st.slider("Images (K)", 1, 5, 2) if available["images"] else 0

    # Display options
    st.subheader("Display")
    show_text_chunks = st.checkbox("Show text chunks", value=True)
    show_tables = st.checkbox("Show retrieved tables", value=True)
    show_images = st.checkbox("Show retrieved images", value=True)
    show_raw_data = st.checkbox("Show raw table data (JSON)", value=False)

# Main area
q = st.text_input("Ask a question about the IFC Annual Report 2024")

# Demo queries
if not q:
    st.info("""
    **Multimodal RAG Demo**

    Try these example queries:

    **Table queries:**
    - "What was the total revenue in FY24?"
    - "Compare IFC's assets between 2023 and 2024"
    - "What percentage of disbursements went to climate finance?"

    **Chart/Image queries:**
    - "What trends are shown in the investment growth chart?"
    - "Describe the regional distribution of projects"
    - "What does the portfolio composition figure show?"

    **Combined queries:**
    - "Explain the financial performance and how it's visualized"
    - "What are the key metrics and their trends over time?"

    Run `python ingest.py --multimodal` to enable table and image retrieval.
    """)

if q:
    # Map mode to retriever mode
    mode_map = {
        "Auto (Smart routing)": "auto",
        "Text only": "text",
        "Tables only": "tables",
        "Images only": "images",
        "Mixed (all modalities)": "mixed",
    }
    mode = mode_map.get(retrieval_mode, "auto")

    # Load retriever and retrieve
    try:
        retriever = load_multimodal_retriever()
        results = retriever.retrieve(
            query=q,
            mode=mode,
            top_k_text=top_k_text,
            top_k_tables=top_k_tables,
            top_k_images=top_k_images,
        )

        text_docs = results.get("text", [])
        table_docs = results.get("tables", [])
        image_docs = results.get("images", [])

    except Exception as e:
        st.error(f"Retrieval error: {e}")
        st.stop()

    # Show retrieval summary
    with st.expander("Retrieval Summary", expanded=False):
        st.json({
            "mode": mode,
            "text_docs": len(text_docs),
            "table_docs": len(table_docs),
            "image_docs": len(image_docs),
        })

    # Build trace
    trace_metadata = {
        "query": q,
        "mode": mode,
        "text_count": len(text_docs),
        "table_count": len(table_docs),
        "image_count": len(image_docs),
    }
    trace = start_trace(langfuse, "multimodal_rag_query", trace_metadata)

    # Generate answer
    st.subheader("Answer")

    # Show retrieved relevant image (only if we have image docs — do not auto-display all)
    if image_docs:
        raw_path = (image_docs[0].metadata.get("image_path") or image_docs[0].metadata.get("file_path") or "").strip()
        primary_image_path = Path(raw_path) if raw_path else None
        if primary_image_path and not primary_image_path.exists() and IMAGES_DIR.exists():
            primary_image_path = IMAGES_DIR / primary_image_path.name
        if primary_image_path and primary_image_path.exists():
            st.caption("Retrieved figure")
            st.image(str(primary_image_path), use_container_width=True)
            st.caption("")

    answer_box = st.empty()
    streamed = ""

    for token in answer_multimodal_qa_stream(
        client=client,
        model=settings.gemini_model,
        query=q,
        text_docs=text_docs,
        table_docs=table_docs,
        image_docs=image_docs,
    ):
        streamed += token
        answer_box.markdown(streamed)

    # Structured answer (include image_path per Practice 5 requirement)
    structured = answer_multimodal_structured(
        client=client,
        model=settings.gemini_model,
        query=q,
        text_docs=text_docs,
        table_docs=table_docs,
        image_docs=image_docs,
    )
    image_paths = [d.metadata.get("image_path") or d.metadata.get("file_path") for d in image_docs if (d.metadata.get("image_path") or d.metadata.get("file_path"))]
    if image_paths:
        structured["image_paths"] = image_paths
    with st.expander("Structured Answer", expanded=False):
        st.json(structured)

    end_trace(trace, {"answer": streamed, "structured": structured})

    # Show retrieved content
    st.divider()
    st.subheader("Retrieved Evidence")

    # Create tabs for different modalities
    tabs = []
    tab_names = []
    if text_docs and show_text_chunks:
        tab_names.append(f"Text ({len(text_docs)})")
    if table_docs and show_tables:
        tab_names.append(f"Tables ({len(table_docs)})")
    if image_docs and show_images:
        tab_names.append(f"Images ({len(image_docs)})")

    if tab_names:
        tabs = st.tabs(tab_names)
        tab_idx = 0

        # Text tab
        if text_docs and show_text_chunks:
            with tabs[tab_idx]:
                for i, doc in enumerate(text_docs, 1):
                    page = doc.metadata.get("page", "?")
                    section = doc.metadata.get("section", "")
                    st.markdown(f"**Match {i}** — Page {page}" + (f" — {section}" if section else ""))
                    content = doc.page_content
                    max_len = 600
                    if len(content) > max_len:
                        cut = content[:max_len].rfind(" ")
                        content = (content[:cut] if cut > 400 else content[:max_len]) + "..."
                    st.text(content)
                    st.divider()
            tab_idx += 1

        # Tables tab
        if table_docs and show_tables:
            with tabs[tab_idx]:
                for i, doc in enumerate(table_docs, 1):
                    page = doc.metadata.get("page", "?")
                    table_id = doc.metadata.get("table_id", "unknown")
                    caption = doc.metadata.get("caption", "")

                    st.markdown(f"**{table_id}** — Page {page}")
                    if caption:
                        st.caption(caption)

                    # Try to render as markdown table
                    content = doc.page_content
                    if "|" in content:
                        st.markdown(content)
                    else:
                        st.text(content)

                    # Show raw data if enabled
                    if show_raw_data:
                        table_data = retriever.get_table_data(table_id)
                        if table_data:
                            with st.expander("Raw Table Data"):
                                st.json(table_data.structured_data)

                    st.divider()
            tab_idx += 1

        # Images tab
        if image_docs and show_images:
            with tabs[tab_idx]:
                for i, doc in enumerate(image_docs, 1):
                    page = doc.metadata.get("page", "?")
                    figure_id = doc.metadata.get("figure_id", "unknown")
                    chart_type = doc.metadata.get("chart_type", "")
                    caption = doc.metadata.get("caption", "")
                    file_path = doc.metadata.get("file_path", "")

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.markdown(f"**{figure_id}** — Page {page}")
                        if chart_type:
                            st.caption(f"Type: {chart_type}")
                        if caption:
                            st.caption(caption)

                        # Try to show image
                        if file_path and Path(file_path).exists():
                            st.image(file_path, use_container_width=True)

                    with col2:
                        st.markdown("**Description:**")
                        st.text(doc.page_content)

                    st.divider()
    else:
        st.info("No evidence retrieved. Try adjusting the retrieval mode or K values.")
