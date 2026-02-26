#!/bin/sh
set -e
# Optional: run full ingest for all practices (text + P3 + P5 + P6) then start app.
# RUN_FULL_INGEST=0 → never run on start; RUN_FULL_INGEST=1 → always run; unset → run if index, ColPali, or Qdrant missing.
TEXT_INDEX="/app/vectorstore_ifc/index.faiss"
COLPALI_EMBEDDINGS="/app/artifacts/colpali/embeddings/patch_embeddings.json"
QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-ifc_annual_report}"
export QDRANT_URL QDRANT_COLLECTION

# Check if Qdrant has the collection (wait up to ~15s for Qdrant to be ready)
qdrant_has_collection() {
  i=0
  while [ $i -lt 6 ]; do
    if python3 -c "
import urllib.request
import os
url = os.environ.get('QDRANT_URL', 'http://qdrant:6333') + '/collections/' + os.environ.get('QDRANT_COLLECTION', 'ifc_annual_report')
try:
    r = urllib.request.urlopen(urllib.request.Request(url), timeout=3)
    exit(0 if r.status == 200 else 1)
except Exception:
    exit(1)
" 2>/dev/null; then
      return 0
    fi
    i=$((i + 1))
    [ $i -lt 6 ] && sleep 2
  done
  return 1
}

NEED_INGEST=0
if [ "${RUN_FULL_INGEST}" = "1" ]; then
  NEED_INGEST=1
elif [ ! -f "$TEXT_INDEX" ] || [ ! -f "$COLPALI_EMBEDDINGS" ]; then
  NEED_INGEST=1
elif [ -n "$QDRANT_URL" ] && ! qdrant_has_collection; then
  echo "Qdrant collection '$QDRANT_COLLECTION' not found; will run full ingest."
  NEED_INGEST=1
fi

if [ "$NEED_INGEST" = "1" ]; then
  echo "Running full ingest (all practices, including ColPali and Qdrant)..."
  python scripts/ingest.py --all
  echo "Ingest done."
fi
exec streamlit run apps/app_all.py --server.port=8501 --server.address=0.0.0.0
