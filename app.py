# app.py - RAG + Groq + Enrichment + Suggestions + Rating
import io
import os
import uuid
import time
import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple, List as List_

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

import numpy as np
import requests

# Optional PDF extraction
try:
    import PyPDF2
    HAVE_PYPDF2 = True
except Exception:
    HAVE_PYPDF2 = False

# sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Groq client
try:
    from groq import Groq
except Exception:
    Groq = None

logger = logging.getLogger("app")
logging.basicConfig(level=logging.INFO)

# ---------- config ----------
MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB
CHUNK_SIZE = 800
DEFAULT_TOP_K = 5  # Updated to match UI (no top_k selector)
ENRICH_SIM_THRESHOLD = 0.35  # Lowered for better enrichment trigger

# ---------- in-memory stores ----------
DOC_STORE: Dict[str, Dict[str, Any]] = {}
INDEX = {
    "model": None,
    "chunk_texts": [],
    "chunk_doc_ids": [],
    "embeddings": None
}

QUERY_LOG: Dict[str, Dict[str, Any]] = {}
RATINGS: List[Dict[str, Any]] = []

# ---------- embedding model ----------
def init_embedding_model():
    if INDEX["model"] is None:
        if SentenceTransformer is None:
            raise RuntimeError("pip install sentence-transformers")
        INDEX["model"] = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded embedding model: all-MiniLM-L6-v2")

# ---------- index rebuild ----------
def rebuild_index():
    init_embedding_model()
    texts: List[str] = []
    ids: List[str] = []
    for doc_id, entry in DOC_STORE.items():
        full = entry.get("text", "")
        if not full:
            continue
        paragraphs = [p.strip() for p in full.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [full]
        for p in paragraphs:
            if len(p) <= CHUNK_SIZE:
                texts.append(p)
                ids.append(doc_id)
            else:
                for i in range(0, len(p), CHUNK_SIZE):
                    texts.append(p[i:i+CHUNK_SIZE])
                    ids.append(doc_id)
    if not texts:
        INDEX["chunk_texts"] = []
        INDEX["chunk_doc_ids"] = []
        INDEX["embeddings"] = None
        return
    model = INDEX["model"]
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb_norm = emb / norms
    INDEX["chunk_texts"] = texts
    INDEX["chunk_doc_ids"] = ids
    INDEX["embeddings"] = emb_norm
    logger.info("Index rebuilt: %d chunks, %d docs", len(texts), len(set(ids)))

# ---------- search ----------
def search(query: str, top_k: int = DEFAULT_TOP_K):
    if INDEX["embeddings"] is None or len(INDEX["chunk_texts"]) == 0:
        return []
    init_embedding_model()
    q_emb = INDEX["model"].encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    sims = np.dot(INDEX["embeddings"], q_emb[0])
    top_idx = np.argsort(-sims)[:top_k]
    results = []
    for idx in top_idx:
        results.append({
            "doc_id": INDEX["chunk_doc_ids"][idx],
            "score": float(sims[idx]),
            "text": INDEX["chunk_texts"][idx],
            "meta": DOC_STORE.get(INDEX["chunk_doc_ids"][idx], {}).get("meta", {})
        })
    return results

# ---------- text extraction ----------
def extract_text(raw: bytes, filename: str):
    if not raw:
        return ""
    if raw[:4] == b"%PDF" and HAVE_PYPDF2:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(raw))
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n\n".join(pages).strip()
        except Exception:
            logger.debug("PDF failed: %s", filename)
    try:
        return raw.decode("utf-8")
    except Exception:
        try:
            return raw.decode("latin-1")
        except Exception:
            return ""

# ---------- FastAPI ----------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root_index():
    return FileResponse("static/index.html")

# ---------- upload ----------
class UploadResponse(BaseModel):
    doc_ids: List[str]
    status: str

@app.post("/upload_document", response_model=UploadResponse)
async def upload_document(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files")
    stored_ids = []
    for file in files:
        raw = await file.read()
        if not raw or len(raw) > MAX_UPLOAD_BYTES:
            continue
        text = extract_text(raw, file.filename) or f"<failed: {file.filename}>"
        doc_id = str(uuid.uuid4())
        DOC_STORE[doc_id] = {
            "text": text,
            "meta": {"filename": file.filename, "content_type": file.content_type or "", "source": "document"}
        }
        stored_ids.append(doc_id)
    rebuild_index()
    return {"doc_ids": stored_ids, "status": "stored"}

@app.get("/status")
def status():
    return {"docs_count": len(DOC_STORE)}

# ---------- debug ----------
class DebugQuery(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K

@app.post("/debug_retrieve")
def debug_retrieve(req: DebugQuery):
    results = search(req.query, top_k=req.top_k)
    return {"query": req.query, "retrieved": results}

# ---------- Groq ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY")
if Groq is None:
    raise RuntimeError("pip install groq")
groq_client = Groq(api_key=GROQ_API_KEY)

RAG_PROMPT = """You are a precise assistant. Use ONLY the provided context to answer.
If the context does not directly state the answer, reply exactly with "INSUFFICIENT_INFO".
Return ONLY a JSON object with keys:
- answer (string),
- confidence (number between 0 and 1),
- missing_info (array of strings).

Context:
{context}

Question:
{question}

Return only the JSON object.
"""

def _render_prompt(context: str, question: str) -> str:
    return RAG_PROMPT.format(context=context, question=question)

def call_groq_model(prompt_text: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.0, max_tokens: int = 600) -> str:
    try:
        resp = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.exception("Groq failed")
        raise HTTPException(status_code=502, detail=f"Groq error: {e}")

# ---------- helpers ----------
def compute_confidence_from_sims(sims: List[float]) -> float:
    return round(float(np.mean(sims)), 3) if sims else 0.0

def _aggregate_unique(retrieved: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    agg = {}
    for r in retrieved:
        doc_id = r["doc_id"]
        score = r["score"]
        meta = r.get("meta", {})
        if doc_id not in agg or score > agg[doc_id]["best_score"]:
            agg[doc_id] = {"best_score": score, "meta": meta}
    return sorted(
        [{"doc_id": d, "score": v["best_score"], "meta": v["meta"]} for d, v in agg.items()],
        key=lambda x: -x["score"]
    )[:top_k]

# ---------- Wikipedia Enrichment (FIXED) ----------
def _wikipedia_summary(page_title: str) -> Optional[str]:
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(page_title)}"
        headers = {"User-Agent": "RAG-Demo/1.0"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            return r.json().get("extract")
    except Exception as e:
        logger.debug("Wiki summary error: %s", e)
    return None

def attempt_enrichment(query: str) -> Tuple[Optional[str], Optional[str]]:
    candidates = [query, " ".join(query.split()[:6])]
    for c in candidates:
        try:
            params = {
                "action": "query",
                "list": "search",
                "srsearch": c,
                "limit": 1,
                "format": "json",
                "origin": "*"
            }
            r = requests.get("https://en.wikipedia.org/w/api.php", params=params, timeout=10)
            if r.status_code != 200:
                continue
            data = r.json()
            results = data.get("query", {}).get("search", [])
            if not results:
                continue
            title = results[0]["title"]
            extract = _wikipedia_summary(title)
            if extract and len(extract) > 50:
                return f"[ENRICHMENT from Wikipedia – {title}]\n\n{extract}", title
        except Exception as e:
            logger.warning("Enrich failed for '%s': %s", c, e)
    return None, None

def _log_enrichment_attempt(query: str, enriched_text: Optional[str]):
    logger.info("Enrichment %s for query='%s'", "SUCCESS" if enriched_text else "NONE", query)

# ---------- SUGGESTIONS (SIMPLIFIED) ----------
def _make_suggestions(missing: List_[str], query: str) -> List_[Dict[str, str]]:
    """
    Return simple text suggestions for missing info.
    No URLs, just "Search Wikipedia or arXiv for: X".
    """
    suggestions = []
    for item in missing:
        term = item.strip(" .,;\"'").strip()
        if not term:
            continue
        suggestions.append({
            "title": term,
            "detail": f"The answer could not be found because “{term}” is not in the uploaded documents.",
            "action": f"Search Wikipedia or arXiv for: {term}"
        })
    return suggestions

# ---------- Models ----------
class QueryRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K

class QueryResponse(BaseModel):
    query_id: str
    answer: str
    confidence: float
    missing_info: List[str]
    retrieved: List[Dict[str, Any]]
    enriched: bool = False
    enriched_from: Optional[str] = None
    parse_error: bool = False
    suggestions: List_[Dict[str, str]] = []

class RateRequest(BaseModel):
    query_id: str
    rating: int
    feedback: Optional[str] = None

# ---------- Rating ----------
@app.post("/rate_answer")
def rate_answer(req: RateRequest):
    if not 1 <= req.rating <= 5:
        raise HTTPException(status_code=400, detail="Rating 1–5")
    record = QUERY_LOG.get(req.query_id)
    if not record:
        raise HTTPException(status_code=404, detail="query_id not found")
    RATINGS.append({
        "query_id": req.query_id,
        "rating": req.rating,
        "feedback": req.feedback,
        "timestamp": time.time(),
        "query": record.get("query"),
        "answer": record.get("answer")
    })
    return {"status": "stored"}

@app.get("/ratings")
def get_ratings():
    return {"count": len(RATINGS), "ratings": RATINGS}

# ---------- main /query ----------
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    q = req.query.strip()
    top_k = int(req.top_k or DEFAULT_TOP_K)
    query_id = str(uuid.uuid4())

    # 1. Retrieve
    retrieved_chunks = search(q, top_k=top_k * 3)
    sims = [r["score"] for r in retrieved_chunks]

    # 2. Context
    context_parts = [f"[score={r['score']:.3f}] {r['text'][:2000]}" for r in retrieved_chunks[:top_k * 2]]
    context = "\n\n".join(context_parts)

    # 3. LLM Call
    llm_text = call_groq_model(_render_prompt(context, q))

    # 4. Parse JSON
    parse_error = False
    parsed = None
    try:
        cleaned = llm_text.strip().strip("`")
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first != -1 and last > first:
            cleaned = cleaned[first:last+1]
        parsed = json.loads(cleaned)
    except Exception:
        parse_error = True

    # 5. Enrichment
    need_enrich = False
    if parsed and isinstance(parsed, dict):
        if parsed.get("answer") == "INSUFFICIENT_INFO":
            need_enrich = True
        if parsed.get("missing_info"):
            need_enrich = True
    if sims and max(sims) < ENRICH_SIM_THRESHOLD:
        need_enrich = True

    enriched = False
    enrichment_source = None
    if need_enrich:
        enrichment_text, enrichment_source = attempt_enrichment(q)
        _log_enrichment_attempt(q, enrichment_text)
        if enrichment_text:
            enriched = True
            context_enriched = context + "\n\n" + enrichment_text
            llm_text2 = call_groq_model(_render_prompt(context_enriched, q))
            try:
                cleaned2 = llm_text2.strip().strip("`")
                first2 = cleaned2.find("{")
                last2 = cleaned2.rfind("}")
                if first2 != -1 and last2 > first2:
                    cleaned2 = cleaned2[first2:last2+1]
                parsed = json.loads(cleaned2)
                llm_text = llm_text2
            except Exception:
                pass

    # 6. Suggestions (always)
    suggestions: List_[Dict[str, str]] = []
    if enriched and enrichment_source:
        suggestions.append({
            "title": "Enriched from Wikipedia",
            "detail": f"Answer supplemented from Wikipedia page “{enrichment_source}”.",
            "action": f"Search Wikipedia or arXiv for: {enrichment_source}"
        })

    final_mi = parsed.get("missing_info", []) if parsed and isinstance(parsed, dict) else []
    if final_mi:
        suggestions.extend(_make_suggestions(final_mi, q))

    # 7. Final answer
    if parsed and isinstance(parsed, dict):
        answer = str(parsed.get("answer", ""))
        confidence = float(parsed.get("confidence", compute_confidence_from_sims(sims)))
        missing_info = parsed.get("missing_info", []) or []
    else:
        answer = llm_text
        confidence = compute_confidence_from_sims(sims)
        missing_info = ["LLM failed to return valid JSON"] if parse_error else []

    # 8. Log
    unique_results = _aggregate_unique(retrieved_chunks, top_k)
    QUERY_LOG[query_id] = {"query": q, "answer": answer, "confidence": confidence, "retrieved": unique_results}

    # 9. Retrieved meta
    retrieved_meta = [
        {
            "doc_id": r["doc_id"],
            "score": round(r["score"], 3),
            "meta": r["meta"],
            "source_filename": r["meta"].get("filename")
        }
        for r in unique_results
    ]

    return {
        "query_id": query_id,
        "answer": answer,
        "confidence": round(confidence, 3),
        "missing_info": missing_info,
        "retrieved": retrieved_meta,
        "enriched": enriched,
        "enriched_from": enrichment_source,
        "parse_error": parse_error,
        "suggestions": suggestions
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)