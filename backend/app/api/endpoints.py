import os
import json
import random
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.models.rag_model import RAGModel
from app.db.vector_db import VectorChunk, AdminTrace, save_trace

# -------------------------
# Config / toggles
# -------------------------
LOG_QUERIES = os.getenv("LOG_QUERIES", "true").lower() == "true"
LOG_SAMPLE_RATE = float(os.getenv("LOG_SAMPLE_RATE", "1.0"))  # 0.0 - 1.0

# -------------------------
# Router + Model
# -------------------------
router = APIRouter(prefix="/api", tags=["api"])
rag_model = RAGModel()

# -------------------------
# Schemas
# -------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language question")
    top_k: int = Field(5, ge=1, le=20, description="How many chunks to retrieve")
    debug: bool = False
    video_ids: Optional[List[int]] = Field(default=None, description="Limit retrieval to these Video PKs")

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    debug: Optional[Dict[str, Any]] = None  # confidence + internals (prompt_preview, trace_id, etc.)

class IngestResponse(BaseModel):
    espn_indexed: int = 0
    skysports_indexed: int = 0
    total_vectors: int

# ---- YouTube: Add/List ----
class AddVideoRequest(BaseModel):
    urlOrId: str = Field(..., description="YouTube URL or ID")
    lang: str = Field("en", description="Transcript language (default: en)")
    window_size: int = Field(8, ge=2, le=20)
    overlap: int = Field(3, ge=0, le=10)

class AddVideoResponse(BaseModel):
    video_pk: int
    yt_id: str
    chunks_added: int
    status: str

class VideoListItem(BaseModel):
    id: int
    video_id: str
    title: Optional[str] = None
    channel: Optional[str] = None
    url: Optional[str] = None
    lang: Optional[str] = None
    duration_sec: Optional[int] = None
    status: str
    last_ingested_at: Optional[str] = None

# -------------------------
# Health
# -------------------------
@router.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# Query
# -------------------------
@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Accepts a natural-language query and returns an answer plus ranked sources.
    Optional: limit retrieval to specific videos via `video_ids`.
    If debug=true (and LOG_QUERIES on), a compact admin trace is stored.
    """
    try:
        answer, sources, dbg = rag_model.run(
            req.query, top_k=req.top_k, debug=req.debug, video_ids=req.video_ids
        )

        # Normalize sources to plain dicts
        norm_sources = [dict(s) for s in (sources or [])]

        # Optionally log an admin trace (sampled), only if debug requested
        if LOG_QUERIES and req.debug and random.random() <= LOG_SAMPLE_RATE:
            try:
                # Map sources back to VectorChunk ids by URL (best-effort). We also carry "id" now.
                hit_ids: List[int] = []
                scores: List[float] = []
                with Session(rag_model.vector_db.engine) as s:
                    for src in norm_sources:
                        scores.append(float(src.get("score", 0.0)))
                        # Prefer the explicit chunk id if present
                        cid = src.get("id")
                        if isinstance(cid, int) and cid > 0:
                            hit_ids.append(cid)
                            continue
                        url = src.get("url")
                        row_id = -1
                        if url:
                            row = s.query(VectorChunk.id).filter(VectorChunk.url == url).first()
                            row_id = row[0] if row else -1
                        hit_ids.append(int(row_id))

                    trace_id = save_trace(
                        s,
                        query=req.query,
                        top_k=req.top_k,
                        hit_ids=hit_ids,
                        scores=scores,
                        confidence=dbg or {"confidence": 0.0},
                        prompt_preview=(dbg or {}).get("prompt_preview", ""),
                        answer_preview=answer,
                        providers={
                            "embed": os.getenv("EMBED_PROVIDER", "local"),
                            "llm": os.getenv("LLM_PROVIDER", "mock"),
                        },
                        embed_dim=int(rag_model.vector_db.dim or 0),
                    )
                    if dbg is not None:
                        dbg = dict(dbg)
                        dbg["trace_id"] = trace_id
            except Exception:
                # Logging must never break the main request
                pass

        return QueryResponse(answer=answer, sources=norm_sources, debug=dbg)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"query failed: {e}")

# -------------------------
# YouTube: Add video (open to all)
# -------------------------
@router.post("/videos/add", response_model=AddVideoResponse)
def add_video(req: AddVideoRequest):
    """
    Add a YouTube video by URL or ID: fetch transcript, chunk, embed, index.
    """
    try:
        result = rag_model.add_youtube_video(
            url_or_id=req.urlOrId,
            lang=req.lang,
            window_size=req.window_size,
            overlap=req.overlap,
        )
        return AddVideoResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"add_video failed: {e}")

# -------------------------
# YouTube: List videos (for combobox/library)
# -------------------------
@router.get("/videos", response_model=List[VideoListItem])
def list_videos(status: Optional[str] = Query(None), search: Optional[str] = Query(None), limit: int = Query(100, ge=1, le=500)):
    """
    List videos with optional status filter (ready|pending|failed) and text search.
    """
    try:
        items = rag_model.list_videos(status=status, search=search, limit=limit)
        # Fill missing fields with None as needed (Pydantic will coerce)
        return [VideoListItem(**i) for i in items]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"list_videos failed: {e}")

# -------------------------
# Admin: traces
# -------------------------
@router.get("/admin/traces")
def list_traces(limit: int = 50):
    """
    Lightweight list for timeline view. Newest first.
    """
    try:
        with Session(rag_model.vector_db.engine) as s:
            rows = (
                s.query(AdminTrace)
                .order_by(AdminTrace.ts.desc())
                .limit(int(limit))
                .all()
            )
            return [
                {
                    "id": r.id,
                    "ts": r.ts.isoformat(),
                    "query": r.query,
                    "top_k": r.top_k,
                    "confidence": json.loads(r.confidence_json).get("confidence", 0.0),
                }
                for r in rows
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"list_traces failed: {e}")

@router.get("/admin/traces/{trace_id}")
def get_trace(trace_id: str):
    """
    Full trace detail plus hydrated hits (joins VectorChunk by id).
    """
    try:
        with Session(rag_model.vector_db.engine) as s:
            r = s.get(AdminTrace, trace_id)
            if not r:
                raise HTTPException(status_code=404, detail="trace not found")

            hit_ids = json.loads(r.hit_ids_json)
            scores = json.loads(r.scores_json)
            by_id: Dict[int, VectorChunk] = {}

            valid_ids = [i for i in hit_ids if isinstance(i, int) and i > 0]
            if valid_ids:
                rows = s.query(VectorChunk).filter(VectorChunk.id.in_(valid_ids)).all()
                by_id = {row.id: row for row in rows}

            hits = []
            for i, sc in zip(hit_ids, scores):
                row = by_id.get(i)
                hits.append(
                    {
                        "id": i,
                        "score": sc,
                        "source": getattr(row, "source", None),
                        "url": getattr(row, "url", None),
                        "published_at": getattr(row, "published_at", None),
                        "snippet": (getattr(row, "text", "") or "")[:240],
                        # YouTube extras if present
                        "video_pk": getattr(row, "video_pk", None),
                        "start_sec": getattr(row, "start_sec", None),
                        "dur_sec": getattr(row, "dur_sec", None),
                    }
                )

            return {
                "id": r.id,
                "ts": r.ts.isoformat(),
                "query": r.query,
                "top_k": r.top_k,
                "confidence": json.loads(r.confidence_json),
                "prompt_preview": r.prompt_preview,
                "answer_preview": r.answer_preview,
                "providers": json.loads(r.providers_json),
                "embed_dim": r.embed_dim,
                "hits": hits,
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"get_trace failed: {e}")
