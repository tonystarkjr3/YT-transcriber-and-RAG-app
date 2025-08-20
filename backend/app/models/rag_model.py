import os
import math
import datetime as dt
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

from app.db.vector_db import VectorDB, ChunkInput, SearchHit, Video

load_dotenv()

# -----------------------------
# Env toggles (defaults: FREE)
# -----------------------------
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local")        # "local" | "openai"
EMBED_LOCAL_MODEL = os.getenv("EMBED_LOCAL_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_OPENAI_MODEL = os.getenv("EMBED_OPENAI_MODEL", "text-embedding-3-small")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")             # "mock" | "openai"

DB_URL = os.getenv("DB_URL", "sqlite:///./local.db")
INDEX_PATH = os.getenv("INDEX_PATH", "./faiss.index")

# -----------------------------
# Embeddings
# -----------------------------
_local_embedder = None
_embed_dim_cache: Optional[int] = None

def _load_local_embedder():
    global _local_embedder
    if _local_embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(
                "Local embeddings selected but sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from e
        _local_embedder = SentenceTransformer(EMBED_LOCAL_MODEL)
    return _local_embedder

def embed_texts(texts: List[str]) -> np.ndarray:
    """Return float32 embeddings; cost-free if EMBED_PROVIDER=local."""
    global _embed_dim_cache
    if EMBED_PROVIDER == "local":
        model = _load_local_embedder()
        vecs = model.encode(texts, normalize_embeddings=False, convert_to_numpy=True).astype("float32")
        _embed_dim_cache = vecs.shape[1]
        return vecs
    elif EMBED_PROVIDER == "openai":
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("OpenAI selected but openai package not installed. pip install openai") from e
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.embeddings.create(model=EMBED_OPENAI_MODEL, input=texts)
        mat = np.vstack([np.array(d.embedding, dtype="float32") for d in resp.data])
        _embed_dim_cache = mat.shape[1]
        return mat
    else:
        raise RuntimeError(f"Unknown EMBED_PROVIDER={EMBED_PROVIDER}")

def embedding_dim() -> int:
    if _embed_dim_cache is not None:
        return _embed_dim_cache
    if EMBED_PROVIDER == "local":
        return 384  # MiniLM-L6 / many small models
    if EMBED_PROVIDER == "openai":
        return 1536  # text-embedding-3-small
    return 384

# -----------------------------
# Prompt (for debug/LLM)
# -----------------------------
def build_prompt(query: str, sources: List[Dict[str, Any]], token_budget_chars=5000) -> str:
    header = (
        "You are answering questions about YouTube videos using retrieved transcript snippets.\n"
        "Make a best effort to ground your answer in the context below.\n"
        "Prefer quotes or specifics when helpful, but keep the answer concise.\n"
        "If the context seems insufficient or off-topic, say so briefly and note what is missing.\n"
        "Cite snippets using bracket numbers like [1], [2].\n\n"
    )
    q = f"Question: {query}\n\n"
    ctx = "Context (top-k snippets):\n"
    used = len(header) + len(q) + len(ctx)
    lines = []
    for i, s in enumerate(sources, 1):
        when = s.get("timecode", "")
        vid = s.get("video_id", "?")
        url = s.get("url", "")
        line = f"[{i}] ({vid} @ {when}) {url} :: {s.get('snippet','')}\n"
        if used + len(line) > token_budget_chars:
            break
        used += len(line)
        lines.append(line)
    instr = (
        "\nGuidelines:\n"
        "- Answer succinctly, grounded in the provided snippets.\n"
        "- If uncertain or context is insufficient, say so and avoid hallucinating.\n"
        "- Include bracket citations like [1], [2] when you reference specifics.\n"
    )
    return header + q + ctx + "".join(lines) + instr

# -----------------------------
# Simple “confidence” (optional)
# -----------------------------
HALF_LIFE_DAYS = 365.0  # transcripts don't age fast; keep decay very mild
def _confidence(hits, k: int) -> Dict[str, Any]:
    if not hits: return {"confidence": 0.0, "explain": "no hits"}
    sims = [max(0.0, min(1.0, float(h.score))) for h in hits]
    sim_avg = sum(sims)/len(sims)
    # recency: not very meaningful here, but keep structure
    recency = 1.0
    src_div = len(set([h.video_pk or -1 for h in hits])) / float(max(1,k))
    return {
        "confidence": round(sim_avg * recency * src_div, 4),
        "sim_avg": round(sim_avg, 4),
        "recency": round(recency, 4),
        "src_diversity": round(src_div, 4),
        "explain": "confidence = sim_avg * recency * source_diversity",
    }

# -----------------------------
# YouTube helpers
# -----------------------------
def parse_yt_id(url_or_id: str) -> str:
    u = url_or_id.strip()
    if "youtu.be/" in u:
        return u.split("youtu.be/")[1].split("?")[0].split("&")[0]
    if "youtube.com/watch" in u and "v=" in u:
        # support playlists etc. by taking v=
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(u).query)
        return qs.get("v", [""])[0]
    return u  # assume already an id

def _secs_to_timecode(sec: Optional[float]) -> str:
    if not sec and sec != 0: return "-"
    s = int(sec)
    h = s // 3600
    m = (s % 3600) // 60
    s2 = s % 60
    return f"{h}:{m:02d}:{s2:02d}" if h else f"{m}:{s2:02d}"

def fetch_youtube_oembed(yt_id: str) -> Dict[str, str]:
    import json
    from urllib.request import urlopen
    from urllib.parse import urlencode
    import ssl, certifi

    watch_url = f"https://www.youtube.com/watch?v={yt_id}"
    oe_url = "https://www.youtube.com/oembed?" + urlencode({"url": watch_url, "format": "json"})
    ctx = ssl.create_default_context(cafile=certifi.where())
    try:
        with urlopen(oe_url, context=ctx, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8", errors="ignore"))
        return {
            "title": data.get("title", ""),
            "channel": data.get("author_name", ""),
            "thumbnail_url": data.get("thumbnail_url", ""),
        }
    except Exception as e:
        print(f"[oembed] failed for {yt_id}: {e}")
        return {"title": "", "channel": "", "thumbnail_url": ""}


def window_segments(segments: List[Dict[str, Any]], window_size=8, overlap=3) -> List[Dict[str, Any]]:
    """
    Slide a window over transcript segments to form chunk text with start/end.
    segments: [{text, start, duration}]
    returns: [{text, start_sec, dur_sec, seg_start_idx, seg_end_idx}]
    """
    if not segments: return []
    out = []
    n = len(segments)
    i = 0
    while i < n:
        j = min(i + window_size, n)
        window = segments[i:j]
        text = " ".join([w["text"].strip() for w in window]).strip()
        start_sec = float(window[0]["start"])
        end_sec = float(window[-1]["start"]) + float(window[-1].get("duration", 0))
        out.append({
            "text": text,
            "start_sec": start_sec,
            "dur_sec": max(0.0, end_sec - start_sec),
            "seg_start_idx": i,
            "seg_end_idx": j-1,
        })
        if j >= n: break
        i = max(i + window_size - overlap, 0)
        if i <= 0 and j >= n: break
    return out

def _to_dict_segments(transcript_iterable):
    """
    Normalize youtube-transcript-api results to a list of dicts
    with keys: text, start, duration.

    Works for versions that yield dicts OR objects like FetchedTranscriptSnippet.
    """
    out = []
    for item in transcript_iterable:
        # Already a dict?
        if isinstance(item, dict):
            # Ensure required keys exist (fallbacks if missing)
            out.append({
                "text": item.get("text", "") or "",
                "start": float(item.get("start", 0.0) or 0.0),
                "duration": float(item.get("duration", 0.0) or 0.0),
            })
            continue

        # Object style: try attribute access
        text = getattr(item, "text", "")
        start = getattr(item, "start", 0.0)
        dur = getattr(item, "duration", 0.0)

        # Some versions wrap values differently; be defensive
        try:
            start = float(start)
        except Exception:
            start = 0.0
        try:
            dur = float(dur)
        except Exception:
            dur = 0.0

        out.append({"text": text or "", "start": start, "duration": dur})
    return out


# -----------------------------
# Model
# -----------------------------
class RAGModel:
    def __init__(self, model_path="yt-transcripts", vector_db: Optional[VectorDB]=None, dim: Optional[int]=None):
        dim = dim or embedding_dim()
        self.vector_db = vector_db or VectorDB(
            db_url=DB_URL,
            index_path=INDEX_PATH,
            dim=dim,
            embed_provider=EMBED_PROVIDER,
        )

    # ---- Public: add a YouTube video ----
    def add_youtube_video(self, url_or_id: str, lang: str = "en", window_size: int = 8, overlap: int = 3) -> Dict[str, Any]:
        """
        Fetch transcript, chunk, embed, and index.
        Returns {video_pk, yt_id, chunks_added, status}
        """
        yt_id = parse_yt_id(url_or_id)
        from sqlalchemy.orm import Session
        from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

        with Session(self.vector_db.engine) as s:
            # Ensure a Video row exists
            v = s.query(Video).filter(Video.video_id == yt_id).first()
            if v is None:
                v = Video(video_id=yt_id, url=f"https://youtu.be/{yt_id}", lang=lang, status="pending")
                s.add(v); s.commit(); s.refresh(v)
            else:
                # Early out if already indexed for this lang
                if v.status == "ready" and (v.lang or "en") == lang:
                    return {"video_pk": v.id, "yt_id": yt_id, "chunks_added": 0, "status": "already_indexed"}
                # Ensure we're marked pending while we (re)index
                if v.status != "pending":
                    v.status = "pending"; s.commit()

            # Best-effort: fetch oEmbed metadata (no API key)
            meta = fetch_youtube_oembed(yt_id)
            print('HELLO, getting metadata for video', yt_id, 'it is', meta)
            changed = False
            if meta.get("title") and not v.title:
                v.title = meta["title"]; changed = True
            if meta.get("channel") and not v.channel:
                v.channel = meta["channel"]; changed = True
            if changed:
                s.commit()

            # Fetch transcript (your version uses instance .fetch)
            try:
                raw = YouTubeTranscriptApi().fetch(yt_id, languages=[lang])
                transcript = _to_dict_segments(raw)  # normalize objects → dicts
            except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
                v.status = "failed"; s.commit()
                return {"video_pk": v.id, "yt_id": yt_id, "chunks_added": 0, "status": f"failed: {type(e).__name__}"}

            # Window segments
            windows = window_segments(transcript, window_size=window_size, overlap=overlap)

            # Build ChunkInputs
            chunks: List[ChunkInput] = []
            texts: List[str] = []
            for w in windows:
                t = w["text"]
                if not t or len(t) < 10:
                    continue
                texts.append(t)
                chunks.append(ChunkInput(
                    text=t,
                    source="youtube",
                    url=f"https://youtu.be/{yt_id}",
                    meta_data={"yt_id": yt_id, "lang": lang},
                    video_pk=v.id,
                    start_sec=w["start_sec"],
                    dur_sec=w["dur_sec"],
                ))

            if not texts:
                v.status = "failed"; s.commit()
                return {"video_pk": v.id, "yt_id": yt_id, "chunks_added": 0, "status": "no_chunks"}

            # Embed & add
            embs = embed_texts(texts)
            new_ids = self.vector_db.add_chunks(chunks, embeddings=embs)

            # Mark ready
            v.status = "ready"
            v.last_ingested_at = dt.datetime.utcnow()
            s.commit()

            print('adding chunks:', len(new_ids))

            return {"video_pk": v.id, "yt_id": yt_id, "chunks_added": len(new_ids), "status": "ready"}

    # ---- Public: list videos (for combobox) ----
    def list_videos(self, status: Optional[str] = None, search: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        from sqlalchemy.orm import Session
        from sqlalchemy import or_
        out: List[Dict[str, Any]] = []
        with Session(self.vector_db.engine) as s:
            q = s.query(Video)
            if status:
                q = q.filter(Video.status == status)
            if search:
                like = f"%{search}%"
                q = q.filter(or_(Video.title.ilike(like), Video.channel.ilike(like), Video.video_id.ilike(like)))
            q = q.order_by(Video.last_ingested_at.desc().nullslast()).limit(limit)
            for v in q.all():
                out.append({
                    "id": v.id,
                    "video_id": v.video_id,
                    "title": v.title,
                    "channel": v.channel,
                    "url": v.url or f"https://youtu.be/{v.video_id}",
                    "lang": v.lang,
                    "duration_sec": v.duration_sec,
                    "status": v.status,
                    "last_ingested_at": v.last_ingested_at.isoformat() if v.last_ingested_at else None,
                })
        return out

    # ---- Query (with optional scoping by video_ids) ----
    def run(self, query: str, top_k: int = 5, debug: bool = False, video_ids: Optional[List[int]] = None
            ) -> Tuple[str, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        q_vec = embed_texts([query])[0]
        hits = self.vector_db.search(q_vec, k=top_k * 2)  # get extra, we'll filter
        # Optional filtering by selected videos
        if video_ids:
            vid_set = set(video_ids)
            hits = [h for h in hits if (h.video_pk in vid_set)]
        hits = hits[:top_k]

        sources: List[Dict[str, Any]] = []
        for h in hits or []:
            yt_id = None
            if h.meta_data: yt_id = h.meta_data.get("yt_id")
            if (not yt_id) and h.url and "youtu" in h.url:
                yt_id = parse_yt_id(h.url)
            watch_url = f"https://youtu.be/{yt_id}?t={int(h.start_sec or 0)}" if yt_id is not None else (h.url or "")
            sources.append({
                "id": h.id,
                "score": round(float(h.score), 4),
                "snippet": (h.text or "").strip().replace("\n", " ")[:240],
                "video_pk": h.video_pk,
                "video_id": yt_id,
                "timecode": _secs_to_timecode(h.start_sec),
                "start_sec": h.start_sec,
                "dur_sec": h.dur_sec,
                "url": watch_url,  # point directly to timestamp
            })

        # No-cost path (default)
        if LLM_PROVIDER != "openai":
            print('using FREE path for LLM inference', LLM_PROVIDER)
            if not hits:
                ans = "No matching video snippets yet. Add a YouTube video and try again."
            else:
                snips = [s["snippet"] for s in sources[:min(3, len(sources))]]
                ans = "Top results:\n" + "\n".join([f"- {t}" for t in snips]) + "\n\n(LLM dry-run / mock mode)"
            dbg = _confidence(hits, top_k) if debug else None
            if debug:
                dbg = dbg or {}
                dbg["prompt_preview"] = build_prompt(query, sources)
            return ans, sources, dbg

        # Paid path: real LLM call
        try:
            from openai import OpenAI
            print('using PAID path for LLM inference')
            theKey = os.getenv("OPENAI_API_KEY")
            print('using key', theKey)
            client = OpenAI(api_key=theKey)
            prompt = build_prompt(query, sources)
            comp = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            ans = comp.choices[0].message.content.strip()
            dbg = _confidence(hits, top_k) if debug else None
            if debug:
                dbg = dbg or {}
                dbg["prompt_preview"] = prompt
            return ans, sources, dbg
        except Exception as e:
            # >>> surface the reason to help debug <<<
            err = f"{type(e).__name__}: {e}"
            print(f"[LLM ERROR] {err}")
            snips = [s["snippet"] for s in sources[:min(3, len(sources))]]
            ans = "LLM call failed; falling back to snippets.\n" + "\n".join([f"- {t}..." for t in snips])
            dbg = _confidence(hits, top_k) if debug else None
            if debug:
                dbg = dbg or {}
                dbg["llm_error"] = err
                dbg["prompt_preview"] = build_prompt(query, sources)
            return ans, sources, dbg
