# RAG Practice (Vertex AI)

A learning project: a question-answering system over PDFs using a neural network (RAG on Google Cloud). The repo contains a report in PDF; you ask a question — the app finds relevant passages and generates an answer via Vertex AI (Gemini).

Practices 1–6 are implemented: text RAG, evaluation, advanced retrieval, semantic cache and multi-hop, multimodal RAG (tables and charts), and visual RAG (ColPali-style).

---

# Quick start (after you have Docker and gcloud)

If Docker and Google Cloud CLI are already installed and you have a GCP project with Vertex AI enabled:

```bash
cd rag-practice
cp .env.example .env
# Edit .env: set GCP_PROJECT=your-project-id and GCP_LOCATION=us-central1
gcloud auth application-default login
docker compose up --build
```

Then open http://localhost:8501. On first run, full ingestion (text, multimodal, ColPali, and Qdrant) runs automatically — wait until you see “Ingest done.” in the logs (about 15–40 minutes). In the app you can choose **Vector Store: FAISS** or **Qdrant**. Details and troubleshooting: see “How to run” below.

---

# How to run (step by step, from scratch)

Below is a single way to run everything via Docker. If you don’t have something yet, follow the steps in order.

---

## Step 0. What you need on your machine

- A **computer** with macOS, Windows, or Linux.
- **Internet access** (for installation and for Google Cloud).
- A **Google account** and access to a Google Cloud project with Vertex AI enabled (e.g. a corporate project like `gd-gcp-gridu-genai` or your own project).

You’ll need to install two things: Docker and Google Cloud CLI.

---

## Step 1. Install Docker

Docker runs the app in a “container” with all dependencies, without installing Python or libraries on your system.

1. Go to: https://docs.docker.com/get-docker/
2. Choose your OS (Mac, Windows, Linux) and download **Docker Desktop** (or Docker Engine on Linux).
3. Install and start it.
4. Wait until the system tray/menu shows that Docker is running (e.g. “Docker Desktop is running”).
5. Check: open **Terminal** (Mac/Linux) or **PowerShell / Command Prompt** (Windows) and run:
   ```bash
   docker --version
   ```
   You should see a version line (e.g. `Docker version 24.0.7`). If the command is not found, restart the terminal after installing Docker.

---

## Step 2. Install Google Cloud CLI (gcloud)

This tool is used to sign in to Google Cloud and let the app use Vertex AI (Gemini).

1. Go to: https://cloud.google.com/sdk/docs/install
2. Select your OS and follow the install instructions (e.g. script or Homebrew on Mac).
3. After installing, open the terminal again and run:
   ```bash
   gcloud --version
   ```
   The gcloud version should be printed.

---

## Step 3. Get a Google Cloud project

- If you already have a **corporate or course project** (e.g. `gd-gcp-gridu-genai`) — get its **Project ID** and go to Step 4.
- To create your own project:
  1. Go to https://console.cloud.google.com
  2. Open the project dropdown (top left) → “New project” → enter a name and note the **Project ID** (e.g. `my-rag-project`).
  3. **Vertex AI API** must be enabled in the project. Usually: “APIs & Services” → “Enable APIs” → search for “Vertex AI API” and enable it. If the project was given by an instructor or company, the API may already be enabled.

---

## Step 4. Prepare the project folder

1. Download or clone the repo (e.g. unzip an archive from Google Drive).
2. Open the terminal and go to the project folder:
   ```bash
   cd path/to/rag-practice
   ```
   For example: `cd ~/Downloads/rag-practice` or `cd /Users/jane/Desktop/rag-practice`. Replace `path/to/rag-practice` with your path.

3. Check that the folder contains `docker-compose.yml`, `Dockerfile`, and directories `apps`, `rag`, `data`. The `data` folder should contain a PDF (e.g. `ifc-annual-report-2024-financials.pdf`).

---

## Step 5. Create the settings file (.env)

The `.env` file holds your Google Cloud project ID. It is **not** in the archive (it’s not in the repo), so you must create it once.

1. In the `rag-practice` folder, find **`.env.example`**.
2. Copy it and name the copy **`.env`**:
   - In the terminal (from `rag-practice`):
     ```bash
     cp .env.example .env
     ```
   - Or manually: copy `.env.example` and rename the copy to `.env`.
3. Open `.env` in any editor and set:
   - **GCP_PROJECT** — your Project ID (e.g. `gd-gcp-gridu-genai` or your project).
   - **GCP_LOCATION** — region, usually `us-central1`.
   Example:
   ```
   GCP_PROJECT=gd-gcp-gridu-genai
   GCP_LOCATION=us-central1
   ```
4. Save the file.

---

## Step 6. Sign in to Google Cloud from the terminal

You need to log in once so the app in Docker can use your account for Vertex AI.

1. In the terminal run:
   ```bash
   gcloud auth application-default login
   ```
2. A browser window will open. Sign in with your Google account and allow access.
3. After success you’ll see something like “Credentials saved” in the terminal. You don’t need to repeat this until the token expires or you switch accounts.

---

## Step 7. Run the app with Docker

1. Make sure Docker is running (Docker Desktop is up).
2. In the terminal you should be in the `rag-practice` folder. If not, run `cd path/to/rag-practice` again.
3. Run this single command:
   ```bash
   docker compose up --build
   ```
---

## Step 8. Open the app in the browser

1. Open a browser (Chrome, Firefox, Safari, etc.).
2. In the address bar type:
   ```
   http://localhost:8501
   ```
   and press Enter.
3. The RAG Practice app page should open.

---

## Step 9. Using the interface

1. **Home page (app_all)** — short description and index status. In the **sidebar on the left** you’ll see:
   - **1_Text_RAG** — Practices 1–4: text Q&A, semantic search, hybrid retrieval, cache, multi-hop.
   - **2_Multimodal** — Practice 5: questions over tables and charts from the PDF.
   - **3_ColPali** — Practice 6: visual search over page “patches”.

2. Click the practice you want (e.g. **1_Text_RAG**).
3. On **1_Text_RAG** you can select **Vector Store: FAISS** or **Qdrant** in the sidebar (both are populated on first run).
4. Enter a question about the PDF (e.g. “What was IFC's Net Income for the fiscal year ending June 30, 2024?”) and press Enter or the submit button.
5. The app will find relevant passages and generate an answer. Below you can see the retrieved text chunks (sources).
6. To switch to another practice, choose it again from the sidebar.

---

## Running again (second and later times)

1. Start Docker Desktop if you stopped it.
2. In the terminal go to `rag-practice`: `cd path/to/rag-practice`.
3. Run:
   ```bash
   docker compose up --build
   ```
   Indexes (FAISS, Qdrant, ColPali) are already built. Full ingestion runs only if the **text index** or **ColPali embeddings** are missing; if both exist, the app starts quickly. Open http://localhost:8501 in the browser again.

---

# For reviewers (short)

If you downloaded an archive (e.g. from Google Drive) or cloned the repo:

1. Unzip the archive and go to the `rag-practice` folder.
2. Install Docker and Google Cloud CLI (gcloud).
3. Create `.env`: `cp .env.example .env` and set your `GCP_PROJECT` and `GCP_LOCATION` (project with Vertex AI enabled).
4. Run: `gcloud auth application-default login`.
5. Run: `docker compose up --build`.
6. Open http://localhost:8501 in the browser and pick a practice from the sidebar.

On first run the container will run full ingestion (a few minutes). The PDF is already in the `data/` folder in the repo.

---

# Additional

- **Run without Docker (local):** You need Python 3.11+, install dependencies from `requirements.txt`, create `.env`, and run `gcloud auth application-default login`. Then run once: `python scripts/ingest.py --all` (or `--all --skip-qdrant` if you don't use Qdrant). Then: `streamlit run apps/app_all.py`. Open http://localhost:8501.
- **Docker environment variables:**
  - `RUN_FULL_INGEST=1` — force full ingestion on every container start (FAISS, Qdrant, ColPali, etc.).
  - `RUN_FULL_INGEST=0` — never run ingestion on start (only start the app; all indexes, including ColPali and Qdrant, must already exist).
  - If neither is set, ingestion runs when the text index or ColPali embeddings are missing (first run or after deleting `artifacts/colpali`).

---

# Links

- [Vertex AI](https://cloud.google.com/vertex-ai)
- [Gemini API](https://ai.google.dev/gemini-api/docs)
- [Docker — install](https://docs.docker.com/get-docker/)
- [Google Cloud CLI — install](https://cloud.google.com/sdk/docs/install)
