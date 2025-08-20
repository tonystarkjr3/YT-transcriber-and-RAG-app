import os
import json
import hashlib
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import faiss

from sqlalchemy import (
    create_engine, String, Integer, Float, Text, DateTime, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import (
    declarative_base, Mapped, mapped_column, relationship, sessionmaker, Session
)

# -----------------------------
# SQLAlchemy base / engine
# -----------------------------
Base = declarative_base()

# -----------------------------
# ORM Models
# -----------------------------

class Video(Base):
    __tablename__ = "videos"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # YouTube ID
    title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    channel: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    lang: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, default="en")
    duration_sec: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending|ready|failed
    added_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    last_ingested_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)

    chunks: Mapped[List["VectorChunk"]] = relationship("VectorChunk", back_populates="video", lazy="selectin")


class VectorChunk(Base):
    __tablename__ = "vector_chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Content & provenance
    text: Mapped[str] = mapped_column(Text)
    source: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)  # e.g., "youtube"
    url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)   # canonical content URL

    # Optional article-style fields (kept for compatibility)
    published_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)

    # YouTube specifics
    video_pk: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("videos.id"), nullable=True, index=True)
    start_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dur_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Free-form metadata (JSON serialized)
    meta_data: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Idempotency (unique per chunk content)
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    video: Mapped[Optional[Video]] = relationship("Video", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_vector_chunks_idem"),
    )


# --- Admin Traces (debug-only) ---
class AdminTrace(Base):
    __tablename__ = "admin_query_traces"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    query: Mapped[str] = mapped_column(Text)
    top_k: Mapped[int] = mapped_column(Integer)
    hit_ids_json: Mapped[str] = mapped_column(Text)   # JSON list[int]
    scores_json: Mapped[str] = mapped_column(Text)    # JSON list[float]
    confidence_json: Mapped[str] = mapped_column(Text)
    prompt_preview: Mapped[str] = mapped_column(Text, default="")
    answer_preview: Mapped[str] = mapped_column(Text, default="")
    providers_json: Mapped[str] = mapped_column(Text) # {"embed":"local","llm":"mock"}
    embed_dim: Mapped[int] = mapped_column(Integer, default=0)


def save_trace(session: Session, **data) -> str:
    # Generate an id if not provided
    trace_id = data.get("id")
    if not trace_id:
        trace_id = hashlib.sha1(
            f"{data['query']}-{data['top_k']}-{dt.datetime.utcnow().isoformat()}".encode("utf-8")
        ).hexdigest()[:40]

    t = AdminTrace(
        id=trace_id,
        query=data["query"],
        top_k=int(data["top_k"]),
        hit_ids_json=json.dumps(data["hit_ids"]),
        scores_json=json.dumps(data["scores"]),
        confidence_json=json.dumps(data["confidence"]),
        prompt_preview=data.get("prompt_preview", ""),
        answer_preview=data.get("answer_preview", ""),
        providers_json=json.dumps(data["providers"]),
        embed_dim=int(data.get("embed_dim", 0)),
    )
    session.add(t)
    session.commit()
    return t.id

# -----------------------------
# Lightweight DTOs
# -----------------------------
@dataclass
class ChunkInput:
    text: str
    source: Optional[str] = "youtube"
    url: Optional[str] = None
    chunk_id: Optional[str] = None  # optional external id
    published_at: Optional[dt.datetime] = None
    meta_data: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None

    # YouTube extras
    video_pk: Optional[int] = None
    start_sec: Optional[float] = None
    dur_sec: Optional[float] = None


@dataclass
class SearchHit:
    id: int
    score: float
    text: str
    source: Optional[str]
    url: Optional[str]
    published_at: Optional[dt.datetime]
    meta_data: Dict[str, Any]
    video_pk: Optional[int]
    start_sec: Optional[float]
    dur_sec: Optional[float]

# -----------------------------
# FAISS helpers
# -----------------------------
def _faiss_dim(idx) -> int:
    if hasattr(idx, "d"):
        return int(idx.d)
    if hasattr(idx, "index"):          # IndexIDMap / IndexIDMap2
        return _faiss_dim(idx.index)
    if hasattr(idx, "inner_index"):    # some builds
        return _faiss_dim(idx.inner_index)
    raise ValueError("Cannot determine FAISS index dimension")

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    # x: (n, d)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / norms).astype("float32")

# -----------------------------
# Vector store
# -----------------------------
class VectorDB:
    """
    Local SQLite + FAISS (IndexFlatIP wrapped by IndexIDMap2).
    - SQLite stores text/metadata rows (VectorChunk), plus Video catalog and AdminTrace.
    - FAISS stores embeddings aligned by VectorChunk.id.
    """
    def __init__(
        self,
        db_url: str = "sqlite:///./local.db",
        index_path: str = "./faiss.index",
        dim: Optional[int] = None,
        embedding_fn: Optional[Any] = None,
        echo_sql: bool = False,
        embed_provider: str = "unknown",
    ):
        # Normalize db_url for relative sqlite paths
        if db_url.startswith("sqlite:///./"):
            abs_db = os.path.abspath(db_url.replace("sqlite:///", ""))
            db_url = f"sqlite:///{abs_db}"

        self.db_url = db_url
        self.index_path = os.path.abspath(index_path)
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)

        self.embedding_fn = embedding_fn
        self.engine = create_engine(self.db_url, echo=echo_sql, future=True)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

        self.index: Optional[faiss.Index] = None
        self.dim: Optional[int] = dim
        self._load_or_create_index()

        # (Optional) persist provider/dim later via a Settings table if you like.

    # ---------- index lifecycle ----------
    def _create_new_index(self, dim: int) -> faiss.Index:
        base = faiss.IndexFlatIP(dim)
        return faiss.IndexIDMap2(base)

    def _load_or_create_index(self):
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                if self.dim is None:
                    self.dim = _faiss_dim(self.index)
            except Exception as e:
                print(f"[VectorDB] Warning: failed to read FAISS index ({self.index_path}): {e}. Recreating.")
                self.index = None
        if self.index is None and self.dim is not None:
            self.index = self._create_new_index(self.dim)

    def _persist_index(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)

    # ---------- utilities ----------
    def count(self) -> int:
        with Session(self.engine) as s:
            return s.query(VectorChunk).count()

    # ---------- write path ----------
    def add_chunks(self, chunks: Sequence[ChunkInput], embeddings: np.ndarray) -> List[int]:
        """
        Insert new VectorChunk rows and add their vectors to FAISS.
        - Uses idempotency_key to avoid duplicates.
        - embeddings must align with filtered 'new' chunks only.
        Returns the DB ids of newly inserted rows.
        """
        if len(chunks) == 0:
            return []

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        # Normalize so IP ≈ cosine
        embeddings = _l2_normalize(embeddings)

        new_ids: List[int] = []
        with Session(self.engine) as s:
            # Filter out existing by idempotency_key; collect rows to insert
            to_insert: List[Tuple[VectorChunk, np.ndarray]] = []

            # Preload existing keys to avoid N roundtrips (optional micro-opt)
            keys = [self._ensure_idem_key(c) for c in chunks]
            existing = set(
                k for (k,) in s.query(VectorChunk.idempotency_key)
                .filter(VectorChunk.idempotency_key.in_(keys)).all()
            )

            emb_rows: List[np.ndarray] = []
            ids_for_faiss: List[int] = []

            for c, key, emb in zip(chunks, keys, embeddings):
                if key in existing:
                    continue

                meta_json = json.dumps(c.meta_data or {})
                row = VectorChunk(
                    text=c.text,
                    source=c.source or "youtube",
                    url=c.url,
                    published_at=c.published_at,
                    meta_data=meta_json,
                    idempotency_key=key,
                    video_pk=c.video_pk,
                    start_sec=c.start_sec,
                    dur_sec=c.dur_sec,
                )
                s.add(row)
                s.flush()  # get row.id
                new_ids.append(row.id)
                emb_rows.append(emb)
                ids_for_faiss.append(row.id)

            if emb_rows:
                X = np.vstack(emb_rows).astype("float32")
                # Ensure index created with proper dim
                d = X.shape[1]
                if self.index is None:
                    self.index = self._create_new_index(d)
                    self.dim = d
                elif self.dim != d:
                    raise ValueError(f"Embedding dimension mismatch: index={self.dim}, new={d}")

                ids_np = np.array(ids_for_faiss, dtype=np.int64)
                self.index.add_with_ids(X, ids_np)
                self._persist_index()

            s.commit()

        return new_ids

    def _ensure_idem_key(self, c: ChunkInput) -> str:
        if c.idempotency_key:
            return c.idempotency_key
        payload = f"{c.source}|{c.url}|{c.video_pk}|{c.start_sec}|{(c.text or '')[:200]}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    # ---------- read path ----------
    def search(self, query_vec: np.ndarray, k: int = 5) -> List[SearchHit]:
        """
        Search top-k by inner product (cosine if vectors are L2-normalized).
        """
        if self.index is None:
            return []
        q = query_vec.astype("float32")
        q = _l2_normalize(q[None, :])
        scores, ids = self.index.search(q, k)
        ids = ids[0].tolist()
        scrs = scores[0].tolist()

        # Filter out -1 ids (not found)
        results: List[SearchHit] = []
        with Session(self.engine) as s:
            rows = s.query(VectorChunk).filter(VectorChunk.id.in_([i for i in ids if i >= 0])).all()
            row_by_id = {r.id: r for r in rows}
            for i, sc in zip(ids, scrs):
                if i < 0:
                    continue
                r = row_by_id.get(i)
                if not r:
                    continue
                md = {}
                try:
                    if r.meta_data:
                        md = json.loads(r.meta_data)
                except Exception:
                    md = {}
                results.append(
                    SearchHit(
                        id=r.id,
                        score=float(sc),
                        text=r.text or "",
                        source=r.source,
                        url=r.url,
                        published_at=r.published_at,
                        meta_data=md,
                        video_pk=r.video_pk,
                        start_sec=r.start_sec,
                        dur_sec=r.dur_sec,
                    )
                )
        return results