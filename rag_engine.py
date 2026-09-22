"""
rag_engine.py
--------------
Minimal RAG engine for the Meef malware-analysis add-on.
"""

import os
import json
import math
from pathlib import Path

import requests

HERE = Path(__file__).parent
KB_PATH = HERE / "knowledge_base.json"
CACHE_PATH = HERE / "embeddings_cache.json"

EMBED_MODEL = "models/gemini-embedding-2"
EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/{EMBED_MODEL}:embedContent"


def _get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Run: export GEMINI_API_KEY='your-key-here'"
        )
    return key


def embed_text(text: str, api_key: str | None = None) -> list[float]:
    """Embed a single string via the Gemini embeddings API. Returns a list of floats."""
    api_key = api_key or _get_api_key()
    resp = requests.post(
        EMBED_URL,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={
            "model": EMBED_MODEL,
            "content": {"parts": [{"text": text}]},
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["embedding"]["values"]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class RagEngine:
    def __init__(self, kb_path: Path = KB_PATH, cache_path: Path = CACHE_PATH, api_key: str | None = None):
        self.kb_path = kb_path
        self.cache_path = cache_path
        self.api_key = api_key or _get_api_key()
        self.kb = self._load_kb()
        self._ensure_embeddings()

    def _load_kb(self):
        with open(self.kb_path, "r") as f:
            return json.load(f)

    def _load_cache(self):
        if self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self, cache):
        with open(self.cache_path, "w") as f:
            json.dump(cache, f)

    def _ensure_embeddings(self):
        cache = self._load_cache()
        changed = False
        for entry in self.kb:
            if entry["id"] in cache:
                entry["embedding"] = cache[entry["id"]]
            else:
                print(f"[rag_engine] embedding {entry['id']} ({entry['title']}) ...")
                emb = embed_text(entry["text"], self.api_key)
                entry["embedding"] = emb
                cache[entry["id"]] = emb
                changed = True
        if changed:
            self._save_cache(cache)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        query_emb = embed_text(query, self.api_key)
        scored = []
        for entry in self.kb:
            score = _cosine(query_emb, entry["embedding"])
            scored.append({**{k: v for k, v in entry.items() if k != "embedding"}, "score": round(score, 4)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


if __name__ == "__main__":
    engine = RagEngine()
    test_query = "sample uses process injection and reaches out over the network repeatedly"
    print(f"\nQuery: {test_query}\n")
    for r in engine.retrieve(test_query, top_k=3):
        print(f"- [{r['score']}] {r['title']} ({r['mitre_id']})")
