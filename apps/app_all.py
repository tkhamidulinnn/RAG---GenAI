"""
Unified launcher: one app for all practices (1–6).

Run after full ingest: python scripts/ingest.py --all --skip-qdrant
Then: streamlit run apps/app_all.py

Use the sidebar to open:
- 1 Text RAG (Practices 1–4)
- 2 Multimodal (Practice 5)
- 3 ColPali (Practice 6)
"""
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import streamlit as st

st.set_page_config(page_title="RAG Practice — All Modes", layout="wide")
st.title("RAG Practice: All Practices (1–6)")
st.caption("Single workflow: ingest once with `ingest.py --all`, then use the sidebar to switch practice.")

st.markdown("""
**Select a practice from the sidebar** (1 Text RAG, 2 Multimodal, 3 ColPali).

- **1 Text RAG** — Practices 1–4: dense/sparse/hybrid retrieval, re-ranking, semantic cache, multi-hop.
- **2 Multimodal** — Practice 5: tables + images/charts retrieval and QA.
- **3 ColPali** — Practice 6: visual patch-based retrieval and comparison with classic RAG.

Make sure you ran full ingest first:
```bash
python scripts/ingest.py --all --skip-qdrant
```
""")

# Show what's available
from rag.settings import load_settings
settings = load_settings()
faiss = settings.faiss_dir.exists()
table_ix = (settings.faiss_dir.parent / "table_index").exists()
image_ix = (settings.faiss_dir.parent / "image_index").exists()
colpali_ix = (settings.artifacts_dir / "colpali" / "embeddings").exists()

with st.expander("Index status", expanded=False):
    st.write(f"- Text (FAISS): **{'OK' if faiss else 'missing'}**")
    st.write(f"- Table index (P5): **{'OK' if table_ix else 'missing'}**")
    st.write(f"- Image index (P5): **{'OK' if image_ix else 'missing'}**")
    st.write(f"- ColPali index (P6): **{'OK' if colpali_ix else 'missing'}**")
    if not (faiss and table_ix and image_ix and colpali_ix):
        st.warning("Run `python scripts/ingest.py --all --skip-qdrant` to build all indexes.")
