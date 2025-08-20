import os

# Embeddings
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local")  # "local" | "openai"
EMBED_LOCAL_MODEL = os.getenv("EMBED_LOCAL_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_OPENAI_MODEL = os.getenv("EMBED_OPENAI_MODEL", "text-embedding-3-small")

# Generation
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")       # "mock" | "openai"
DRY_RUN_LLM = os.getenv("DRY_RUN_LLM", "true").lower() == "true"  # don't actually call LLM

# Storage (local dev)
DB_URL = os.getenv("DB_URL", "sqlite:///./local.db")
INDEX_PATH = os.getenv("INDEX_PATH", "./faiss.index")

# CORS/frontend
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")