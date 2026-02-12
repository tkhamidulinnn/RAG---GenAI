"""
Practice 6: ColPali-style Visual RAG UI

Implements Docs/ Practice 6: Multimodal RAG with ColPali-like approach.

End-to-end visual document retrieval and QA:
- Visual patch-based retrieval (no text extraction)
- Late interaction scoring (MaxSim-style)
- Visual context generation with VLM
- Comparison with classic RAG pipeline

Run with: python -m streamlit run app_colpali.py
"""
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import io

from rag.settings import load_settings
from rag.gemini_client import build_client
from rag.embeddings import GeminiEmbeddings
from rag.observability import build_langfuse, start_trace, end_trace


settings = load_settings()
client = build_client(settings)
langfuse = build_langfuse(settings)

# Paths for ColPali
COLPALI_DIR = settings.artifacts_dir / "colpali"
PAGES_DIR = COLPALI_DIR / "pages"
PATCHES_DIR = COLPALI_DIR / "patches"
EMBEDDINGS_DIR = COLPALI_DIR / "embeddings"

# Classic RAG paths
TEXT_INDEX_DIR = settings.faiss_dir


def check_colpali_available() -> dict:
    """Check which ColPali components are available."""
    return {
        "pages": PAGES_DIR.exists() and any(PAGES_DIR.glob("*.png")),
        "patches": PATCHES_DIR.exists() and any(PATCHES_DIR.glob("*.json")),
        "embeddings": EMBEDDINGS_DIR.exists() and (EMBEDDINGS_DIR / "patch_embeddings.json").exists(),
        "faiss_index": (EMBEDDINGS_DIR / "patch_index.faiss").exists(),
        "classic_index": TEXT_INDEX_DIR.exists(),
    }


@st.cache_resource(show_spinner=True)
def load_colpali_retriever():
    """Load ColPali retriever."""
    from rag.colpali import LateInteractionRetriever

    return LateInteractionRetriever(
        embeddings_dir=EMBEDDINGS_DIR,
        client=client,
        use_clip=False,
        use_faiss=True,
        verbose=False,
    )


@st.cache_resource(show_spinner=True)
def load_classic_retriever():
    """Load classic FAISS retriever."""
    from langchain_community.vectorstores import FAISS

    if not TEXT_INDEX_DIR.exists():
        return None

    embeddings = GeminiEmbeddings(client=client, model=settings.embedding_model)
    return FAISS.load_local(
        str(TEXT_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# Page config
st.set_page_config(page_title="IFC RAG - ColPali (Practice 6)", layout="wide")
st.title("IFC Annual Report 2024 — ColPali Visual RAG")
st.caption("Practice 6: ColPali-style visual patch-based retrieval")

# Check availability
available = check_colpali_available()

# Sidebar
with st.sidebar:
    st.header("ColPali Settings")

    # Show availability status
    st.subheader("Pipeline Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pages", "✓" if available["pages"] else "✗")
        st.metric("Patches", "✓" if available["patches"] else "✗")
    with col2:
        st.metric("Embeddings", "✓" if available["embeddings"] else "✗")
        st.metric("Classic Index", "✓" if available["classic_index"] else "✗")

    if not available["embeddings"]:
        st.error("ColPali embeddings not found. Run: python ingest.py --colpali")
        st.stop()

    # Mode selection
    st.subheader("Mode")
    mode_options = ["ColPali Only"]
    if available["classic_index"]:
        mode_options.extend(["Classic Only", "Side-by-Side Comparison"])

    mode = st.selectbox(
        "Pipeline Mode",
        mode_options,
        help="Compare ColPali visual retrieval with classic text retrieval"
    )

    # Retrieval settings
    st.subheader("Retrieval Settings")
    top_k_pages = st.slider("Top K Pages", 1, 10, 5)
    top_k_patches = st.slider("Top K Patches", 10, 100, 50)

    aggregation = st.selectbox(
        "Score Aggregation",
        ["max", "mean", "sum", "weighted"],
        help="How to aggregate patch scores to page level"
    )

    # Display options
    st.subheader("Display")
    show_page_images = st.checkbox("Show page images", value=True)
    show_patch_scores = st.checkbox("Show patch scores", value=False)
    show_attribution = st.checkbox("Show attribution heatmap", value=True)
    show_raw_scores = st.checkbox("Show raw score data", value=False)

# Main area
q = st.text_input("Ask a question about the IFC Annual Report 2024")

# Demo queries
if not q:
    st.info("""
    **Practice 6: ColPali Visual RAG Demo**

    This pipeline implements Docs/ Practice 6: Multimodal RAG with ColPali-like approach.
    It retrieves document pages using visual similarity, without requiring text extraction.
    Documents are understood as images, preserving layout and visual information.

    **How it works:**
    1. Documents are split into visual patches (grid regions)
    2. Patches are embedded using vision-language models
    3. Queries are matched against patches (late interaction)
    4. Top pages are sent to VLM for answer generation

    **Try these queries:**
    - "What are the key financial highlights for 2024?"
    - "Show me information about climate investments"
    - "What does the organizational chart look like?"
    - "Explain the regional distribution of projects"

    Run `python ingest.py --colpali` to enable this pipeline.
    """)

if q:
    trace_metadata = {"query": q, "mode": mode}
    trace = start_trace(langfuse, "colpali_rag_query", trace_metadata)

    if mode == "ColPali Only":
        # Run ColPali pipeline
        try:
            from rag.colpali import (
                LateInteractionRetriever,
                collect_visual_context,
                generate_visual_answer_stream,
                generate_attribution,
                attribution_to_dict,
            )

            retriever = load_colpali_retriever()
            result = retriever.retrieve(
                query=q,
                top_k_pages=top_k_pages,
                top_k_patches=top_k_patches,
                aggregation=aggregation,
            )

            # Show retrieval summary
            with st.expander("Retrieval Summary", expanded=False):
                st.json({
                    "total_patches_searched": result.total_patches_searched,
                    "total_pages": result.total_pages,
                    "pages_found": len(result.page_scores),
                    "top_pages": [ps.page for ps in result.page_scores],
                })

            # Collect visual context
            contexts = collect_visual_context(result, PAGES_DIR, max_pages=top_k_pages)

            # Generate answer
            st.subheader("Answer")
            answer_box = st.empty()
            streamed = ""

            for token in generate_visual_answer_stream(
                client=client,
                model=settings.gemini_model,
                query=q,
                contexts=contexts,
            ):
                streamed += token
                answer_box.markdown(streamed)

            # Show attribution
            if show_attribution:
                attribution = generate_attribution(
                    result, PAGES_DIR, max_pages=top_k_pages
                )

                st.divider()
                st.subheader("Source Attribution")
                st.text(attribution.summary)

                if show_raw_scores:
                    with st.expander("Attribution Data"):
                        st.json(attribution_to_dict(attribution))

            # Show page images
            if show_page_images and result.page_scores:
                st.divider()
                st.subheader("Retrieved Pages")

                for ps in result.page_scores[:top_k_pages]:
                    page_image = PAGES_DIR / f"document_page_{ps.page}.png"
                    if not page_image.exists():
                        page_image = PAGES_DIR / f"page_{ps.page:03d}.png"
                    if not page_image.exists():
                        page_image = PAGES_DIR / f"page_{ps.page}.png"

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.markdown(f"**Page {ps.page}**")
                        st.metric("Score", f"{ps.score:.3f}")
                        st.caption(f"Mean: {ps.mean_patch_score:.3f}, Max: {ps.max_patch_score:.3f}")
                        st.caption(f"{ps.num_relevant_patches} relevant patches")

                    with col2:
                        if page_image.exists():
                            st.image(str(page_image), use_container_width=True)
                        else:
                            st.warning(f"Page image not found: {page_image.name}")

                    # Show patch scores if enabled
                    if show_patch_scores:
                        with st.expander(f"Patch Scores (Page {ps.page})"):
                            for patch in ps.top_patches[:5]:
                                st.text(f"  {patch.patch_id}: {patch.score:.3f} at ({patch.x:.2f}, {patch.y:.2f})")

                    st.divider()

            end_trace(trace, {"answer": streamed, "pages": [ps.page for ps in result.page_scores]})

        except Exception as e:
            st.error(f"ColPali pipeline error: {e}")
            import traceback
            st.code(traceback.format_exc())

    elif mode == "Classic Only":
        # Run classic pipeline
        try:
            from rag.rag_pipeline import retrieve, answer_stream

            db = load_classic_retriever()
            if db is None:
                st.error("Classic index not found. Run: python ingest.py")
                st.stop()

            docs = retrieve(db, q, k=top_k_pages)

            st.subheader("Answer")
            answer_box = st.empty()
            streamed = ""

            for token in answer_stream(client, settings.gemini_model, q, docs):
                streamed += token
                answer_box.markdown(streamed)

            st.divider()
            st.subheader("Retrieved Chunks")

            for i, doc in enumerate(docs, 1):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**Match {i}** — Page {page}")
                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                st.divider()

            end_trace(trace, {"answer": streamed, "mode": "classic"})

        except Exception as e:
            st.error(f"Classic pipeline error: {e}")

    elif mode == "Side-by-Side Comparison":
        # Run both pipelines
        try:
            from rag.colpali import (
                run_comparison,
                comparison_to_dict,
            )

            classic_retriever = load_classic_retriever()
            if classic_retriever is None:
                st.error("Classic index not found for comparison.")
                st.stop()

            with st.spinner("Running both pipelines..."):
                comparison = run_comparison(
                    query=q,
                    classic_retriever=classic_retriever,
                    embeddings_dir=EMBEDDINGS_DIR,
                    pages_dir=PAGES_DIR,
                    client=client,
                    model=settings.gemini_model,
                    classic_top_k=top_k_pages,
                    colpali_top_k_pages=top_k_pages,
                )

            # Show comparison summary
            st.subheader("Comparison Result")

            pref_color = "green" if comparison.preferred_pipeline == "colpali" else "blue"
            st.markdown(f"**Preferred Pipeline:** :{pref_color}[{comparison.preferred_pipeline.upper()}]")
            st.caption(comparison.reason)

            # Metrics
            with st.expander("Performance Metrics", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Classic Total (ms)",
                        f"{comparison.metrics['total_classic_ms']:.0f}"
                    )
                with col2:
                    st.metric(
                        "ColPali Total (ms)",
                        f"{comparison.metrics['total_colpali_ms']:.0f}"
                    )
                with col3:
                    st.metric(
                        "Answer Similarity",
                        f"{comparison.metrics['answer_similarity']:.1%}"
                    )

            # Side-by-side answers
            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Classic RAG Answer")
                st.markdown(comparison.classic.answer)

                with st.expander("Retrieved Pages"):
                    pages = [d.metadata.get("page") for d in comparison.classic.documents]
                    st.write(f"Pages: {pages}")

            with col2:
                st.subheader("ColPali Answer")
                st.markdown(comparison.colpali.answer)

                with st.expander("Retrieved Pages"):
                    pages = [ps["page"] for ps in comparison.colpali.page_scores]
                    st.write(f"Pages: {pages}")

            # Page overlap analysis
            st.divider()
            with st.expander("Page Coverage Analysis"):
                st.write(f"**Classic pages:** {comparison.metrics['classic_pages']}")
                st.write(f"**ColPali pages:** {comparison.metrics['colpali_pages']}")
                st.write(f"**Overlap:** {comparison.metrics['page_overlap']} pages")
                st.write(f"**Classic only:** {comparison.metrics['classic_only_pages']}")
                st.write(f"**ColPali only:** {comparison.metrics['colpali_only_pages']}")

            if show_raw_scores:
                with st.expander("Raw Comparison Data"):
                    st.json(comparison_to_dict(comparison))

            end_trace(trace, {
                "preferred": comparison.preferred_pipeline,
                "answer_similarity": comparison.metrics["answer_similarity"],
            })

        except Exception as e:
            st.error(f"Comparison error: {e}")
            import traceback
            st.code(traceback.format_exc())

# Footer
st.divider()
st.caption("""
**Practice 6**: ColPali-style visual retrieval. [ColPali paper](https://arxiv.org/abs/2407.01449) · Docs/ Practice 6.
""")
