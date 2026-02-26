"""Practice 6: ColPali-style visual RAG."""
import runpy
from pathlib import Path

_apps_dir = Path(__file__).resolve().parents[1]
runpy.run_path(str(_apps_dir / "app_colpali.py"), run_name="__main__")
