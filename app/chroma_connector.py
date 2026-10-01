import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict
from app.utils import embed_texts
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
COLLECTION_NAME = "documents"

def get_collection():
    names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in names:
        return client.get_collection(COLLECTION_NAME)
    return client.create_collection(name=COLLECTION_NAME)

def upsert_documents(collection, texts: List[str], metadatas: List[Dict], ids: List[str]):
    embeddings = embed_texts(texts)
    collection.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)

def query_index(collection, query_text: str, top_k: int = 5, where: Dict = None):
    q_emb = embed_texts([query_text])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=top_k, where=where)
    hits = []
    for i, doc in enumerate(results["documents"][0]):
        hits.append({"text": doc, "metadata": results["metadatas"][0][i], "score": results["distances"][0][i]})
    return hits
