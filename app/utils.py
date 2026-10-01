import os, time, uuid
from typing import List, Dict
from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def make_metadata(title: str, source: str, chunk_index: int, extra: Dict = None) -> Dict:
    meta = {
        "doc_id": str(uuid.uuid4()),
        "title": title,
        "source": source,
        "chunk_index": chunk_index,
        "chunk_id": f"{int(time.time())}-{chunk_index}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    if extra:
        meta.update(extra)
    return meta

def embed_texts(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    embeddings = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = openai.Embeddings.create(model=model, input=batch)
        for item in resp.data:
            embeddings.append(item.embedding)
    return embeddings
