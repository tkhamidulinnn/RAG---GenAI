"""Practice 1–4: Text RAG (dense/sparse/hybrid, re-ranking, cache, multi-hop)."""
import runpy
from pathlib import Path

_apps_dir = Path(__file__).resolve().parents[1]
runpy.run_path(str(_apps_dir / "app.py"), run_name="__main__")
