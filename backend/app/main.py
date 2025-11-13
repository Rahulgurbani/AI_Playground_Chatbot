
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from backend.app.embeddings import get_embedding
from backend.app.vector_db import store_doc, query_docs
from backend.app.llm import generate_response



app = FastAPI(title="AI Playground Backend", version="1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    llm: str = "gpt-j"
    embedding: str = "minilm"
    vectordb: str = "weaviate"


@app.get("/")
def home():
    return {"message": "AI Playground Backend is running 🚀"}


@app.post("/store")
def store(document_id: str = Query(...), text: str = Query(...)):
    try:
        emb = get_embedding(text)
        store_doc(document_id, text, emb, db_name="weaviate")
        return {"stored_id": document_id, "status": "success"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        
        query_emb = get_embedding(req.query)

        
        docs = query_docs(query_emb, db_name=req.vectordb, top_k=3)
        context = "\n".join([d["text"] for d in docs]) if docs else "No context."

        
        answer = generate_response(req.query, context, llm_name=req.llm)

        return {
            "query": req.query,
            "context": context,
            "response": answer,
            "retrieved_docs": docs
        }

    except Exception as e:
        return {"error": str(e)}
