import os, json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4o-mini")
openai.api_key = os.getenv("OPENAI_API_KEY")

def llm_score_pair(query: str, doc_text: str) -> float:
    prompt = (
        "Rate how relevant the following document excerpt is to the query on a scale 0 to 1.\n\n"
        f"Query: {query}\n\nDocument excerpt:\n{doc_text}\n\n"
        "Return only a JSON object like: {\"score\": 0.87}"
    )
    resp = openai.ChatCompletion.create(
        model=RERANK_MODEL,
        messages=[{"role":"user","content":prompt}],
        max_tokens=20,
        temperature=0.0
    )
    text = resp.choices[0].message.content.strip()
    try:
        obj = json.loads(text)
        return float(obj.get("score", 0.0))
    except Exception:
        import re
        m = re.search(r"([0-9]*\.?[0-9]+)", text)
        return float(m.group(1)) if m else 0.0

def rerank_with_llm_parallel(query: str, candidates: List[Dict], top_n: int = 5, batch_size: int = 8, max_workers: int = 8) -> List[Dict]:
    scored = []
    def score_item(c):
        text = c["text"][:2000]
        score = llm_score_pair(query, text)
        c2 = c.copy()
        c2["rerank_score"] = score
        return c2

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(score_item, c) for c in candidates]
        for fut in as_completed(futures):
            try:
                scored.append(fut.result())
            except Exception:
                pass
    scored.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return scored[:top_n]
