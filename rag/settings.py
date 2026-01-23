from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    gcp_project: str
    gcp_location: str
    gemini_model: str
    embedding_model: str
    pdf_path: Path
    artifacts_dir: Path
    faiss_dir: Path
    qdrant_path: Path
    qdrant_collection: str
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_host: str | None


def load_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[1]
    gcp_project = os.environ.get("GCP_PROJECT", "").strip()
    gcp_location = os.environ.get("GCP_LOCATION", "us-central1").strip()
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip()
    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-004").strip()

    pdf_path = Path(os.environ.get("PDF_PATH", str(repo_root / "data" / "ifc-annual-report-2024-financials.pdf")))
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", str(repo_root / "artifacts")))
    faiss_dir = Path(os.environ.get("FAISS_DIR", str(repo_root / "vectorstore_ifc")))
    qdrant_path = Path(os.environ.get("QDRANT_PATH", str(repo_root / "qdrant_local")))
    qdrant_collection = os.environ.get("QDRANT_COLLECTION", "ifc_annual_report").strip()

    langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    langfuse_host = os.environ.get("LANGFUSE_HOST")

    if not gcp_project:
        raise RuntimeError("GCP_PROJECT must be set. See README for setup instructions.")

    return Settings(
        gcp_project=gcp_project,
        gcp_location=gcp_location,
        gemini_model=gemini_model,
        embedding_model=embedding_model,
        pdf_path=pdf_path.expanduser().resolve(),
        artifacts_dir=artifacts_dir.expanduser().resolve(),
        faiss_dir=faiss_dir.expanduser().resolve(),
        qdrant_path=qdrant_path.expanduser().resolve(),
        qdrant_collection=qdrant_collection,
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_host=langfuse_host,
    )
