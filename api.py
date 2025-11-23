# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any, Dict

from core_rag import answer_question

app = FastAPI(title="Thailand Brain API")

class ChatRequest(BaseModel):
    question: str
    top_k: int | None = 5

class Source(BaseModel):
    score: float | None = None
    text: str | None = None
    url: str | None = None
    city: str | None = None
    tags: List[str] | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer, contexts = answer_question(req.question, top_k=req.top_k or 5)

    sources = []
    for c in contexts:
        sources.append(
            Source(
                score=float(c["score"]) if c["score"] is not None else None,
                text=c["text"],
                url=c["url"],
                city=c["city"],
                tags=c["tags"] or [],
            )
        )

    return ChatResponse(answer=answer, sources=sources)
