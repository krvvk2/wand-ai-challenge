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
| UI | Dark mode, file names, no clutter |

---

## Architecture


[User] → [FastAPI] → [Sentence Transformers] → [Vector Index]
↓
[Groq LLM] ← RAG Context
↓
[Answer + Confidence + Suggestions]
↓
[Wikipedia API] (if needed)
---

## Design Decisions

| Decision | Rationale |
|--------|---------|
| In-memory store | Fast prototyping |
| `numpy` + `cosine` | No vector DB needed |
| Groq LLM | Ultra-low latency |
| Wikipedia | Trusted academic source |
| Dark mode | Professional look |

---

## Trade-offs (24h Constraint)

| Trade-off | Choice |
|---------|--------|
| Persistence | In-memory → faster dev |
| Vector DB | `numpy` → no setup |
| Auth | None → focus on core |
| Multi-user | Single session → demo scope |

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