# Wand AI Challenge 2: AI-Powered Knowledge Base Search & Enrichment


---

## Overview

This is a **RAG system** that:

1. **Uploads** PDFs/TXTs  
2. **Indexes** content semantically  
3. **Answers** natural language queries  
4. **Detects** missing info  
5. **Suggests** enrichment from **Wikipedia or arXiv**  
6. **Supports** user ratings  

**All stretch goals achieved** — **auto-enrichment** + **rating**.

---

## Core Features

| Feature | Implementation |
|-------|----------------|
| Document Upload | `POST /upload_document` |
| Semantic Search | `all-MiniLM-L6-v2` + cosine similarity |
| RAG Pipeline | Groq `llama-3.1-8b-instant` |
| Completeness Check | `INSUFFICIENT_INFO` + `missing_info` |
| Enrichment | Wikipedia API on low confidence |
| Suggestions | "Search Wikipedia or arXiv for: X" |
| Structured Output | JSON: `answer`, `confidence`, `missing_info` |
| Rating System | `POST /rate_answer` |
| UI | Dark mode, file names, feeback, suggestions |

---


## Design Decisions

| Decision | Rationale |
|--------|---------|
| In-memory store for docs & index | 24h sprint — fastest to code. No Redis/Pinecone setup. Works great for demo. In production, I'd use Redis or Pinecone for persistence. |
| `numpy` + `cosine` | No vector DB needed |
| Groq `llama-3.1-8b-instant` + JSON mode | Tried OpenAI first — too slow/expensive. Groq gives **<100ms latency**, JSON output is reliable. No parsing issues. |
| Wikipedia | Trusted academic source |
| No Docker | My laptop (8GB RAM) crashed with Docker. Focused on `pip install` → anyone can run in 10s. |

---

## Trade-offs (24h Constraint)

| Trade-off | Choice |
|---------|--------|
| Persistence | In-memory → faster dev. Lost on restart. Real app → Redis + disk backup. |
| Vector DB | `numpy` arrays. Zero setup |
| Enrichment | Wikipedia only. Wikipedia was reliable + fast. Prioritized working stretch goal. |
| Error Handling | Basic try/except | Covers PDF fail, LLM timeout. More logging = next sprint.|
| Chunking | Fixed 800 chars | Simple, fast. Could do overlap/semantic splits — but time ran out. |

---

## How to Run

```bash
# 1. Clone
git clone https://github.com/krvvk2/wand.ai-challenge
cd wand-ai-challenge

# 2. Install
pip install -r requirements.txt

# 3. Set API key
export GROQ_API_KEY=your_key_here

# 4. Run
uvicorn app:app --reload
