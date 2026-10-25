"""Central settings. Change values here (or in .env) and re-run to experiment."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vectordb"
COLLECTION_NAME = "clinic_docs"

# --- Chunking -----------------------------------------------------------------
# LESSON: chunk size is a trade-off.
#   Too big  -> each chunk covers many topics, its embedding gets "blurry",
#               and you waste prompt tokens on irrelevant text.
#   Too small -> a chunk loses the context needed to answer (e.g. a price row
#               without the table header saying what it's a price *of*).
# Overlap repeats a bit of text across neighbouring chunks so a fact that sits
# on a boundary isn't cut in half.
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200

# --- Retrieval ----------------------------------------------------------------
# How many chunks to hand to the LLM. More = better recall, but more noise/cost.
TOP_K = 4

# --- Generation ---------------------------------------------------------------
# LESSON: the "G" in RAG is swappable. Retrieval doesn't care which LLM reads
# the chunks, so you can run a free local model or a hosted one.
#   "ollama" -> local model on your laptop via Ollama (free, private, offline)
#   "claude" -> Anthropic API (needs ANTHROPIC_API_KEY, much stronger model)
LLM_PROVIDER = os.getenv("CLINICBOT_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("CLINICBOT_OLLAMA_MODEL", "qwen3:4b")
CLAUDE_MODEL = os.getenv("CLINICBOT_MODEL", "claude-opus-5-5")
