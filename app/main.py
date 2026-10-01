import os, uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()
from app.utils import load_text, chunk_text, make_metadata
from app.s3_store import upload_chunk_text, download_chunk_text
from app.chroma_connector import get_collection, upsert_documents, query_index
from app.reranker import rerank_with_llm_parallel
from app.chat_memory import ChatMemory
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

collection = get_collection()
app = FastAPI()
memory = ChatMemory()

class IngestRequest(BaseModel):
    path: str
    title: str
    source: str
    extra_metadata: Optional[Dict] = None

class QueryRequest(BaseModel):
    query: str
    k: int = 5
    filter: Optional[Dict] = None
    rerank: bool = True
    generate_answer: bool = True
    session_id: Optional[str] = None
    page: int = 1
    page_size: int = 5
    stream: bool = False
    llm_instructions: str = ""

@app.post("/ingest")
def ingest(req: IngestRequest):
    text = load_text(req.path)
    chunks = chunk_text(text)
    ids, metadatas, texts = [], [], []
    for i, chunk in enumerate(chunks):
        uid = str(uuid.uuid4())
        s3_key = f"chunks/{req.title.replace(' ','_')}/{uid}.txt"
        upload_chunk_text(chunk, s3_key)
        meta = make_metadata(req.title, req.source, i, req.extra_metadata)
        meta["s3_key"] = s3_key
        ids.append(uid)
        metadatas.append(meta)
        texts.append(chunk)
    upsert_documents(collection, texts, metadatas, ids)
    return {"status":"ok", "chunks": len(chunks)}

@app.post("/search")
def search(req: QueryRequest):
    hits = query_index(collection, req.query, top_k=req.k*3, where=req.filter)
    candidates = []
    for h in hits:
        text = download_chunk_text(h["metadata"]["s3_key"])
        candidates.append({"text": text, "metadata": h["metadata"], "score": h.get("score")})
    if req.rerank:
        candidates = rerank_with_llm_parallel(req.query, candidates, top_n=req.k, batch_size=8, max_workers=8)
    start = (req.page - 1) * req.page_size
    page_items = candidates[start:start + req.page_size]
    if req.generate_answer:
        context = "\n\n---\n\n".join([f"Source: {c['metadata'].get('title')}\nText: {c['text']}" for c in page_items])
        prompt = f"You are an assistant. Use only the context to answer. Context:\n{context}\n\nUser question:\n{req.query}\n\n{req.llm_instructions}"
        if req.stream:
            from fastapi.responses import StreamingResponse
            def event_stream():
                resp = openai.ChatCompletion.create(model=os.getenv("RERANK_MODEL"), messages=[{"role":"user","content":prompt}], stream=True)
                for chunk in resp:
                    if 'choices' in chunk:
                        delta = chunk['choices'][0].get('delta', {})
                        content = delta.get('content')
                        if content:
                            yield content
            return StreamingResponse(event_stream(), media_type="text/plain")
        else:
            resp = openai.ChatCompletion.create(model=os.getenv("RERANK_MODEL"), messages=[{"role":"user","content":prompt}], max_tokens=512, temperature=0.0)
            answer = resp.choices[0].message.content
            return {"answer": answer, "sources": [c["metadata"] for c in page_items], "page": req.page}
    return {"results": page_items, "page": req.page}
