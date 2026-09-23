"""
rag_engine.py
--------------
Minimal RAG engine for the Meef malware-analysis add-on.

What it does:
1. Loads the curated knowledge base (knowledge_base.json).
2. Embeds every KB entry using Gemini's embedding model (real embeddings,
   not TF-IDF -- this is genuine semantic retrieval, and it needs no local
   model download since it's a single API call per chunk).
3. Caches those embeddings to disk (embeddings_cache.json) so you don't
   re-embed the same 12 KB entries on every run -- keeps demo/dev fast and
   avoids burning API quota.
4. Given a query string, embeds it and returns the top-k most similar KB
   entries by cosine similarity.

Set your key once:
    export GEMINI_API_KEY="your-key-here"

Usage:
    from rag_engine import RagEngine
    engine = RagEngine()
    results = engine.retrieve("process injection and network beaconing detected", top_k=3)
    for r in results:
        print(r["title"], r["score"])
"""

import os
import json
import math
import time
from pathlib import Path

import requests

from observability import tracer

HERE = Path(__file__).parent
KB_PATH = HERE / "knowledge_base.json"
CACHE_PATH = HERE / "embeddings_cache.json"

EMBED_MODEL = "models/gemini-embedding-2"
EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/{EMBED_MODEL}:embedContent"

# Transient errors worth retrying: 429 (rate limited), 500/502/503/504 (server-side,
# usually a momentarily overloaded model). Anything else (400, 401, 403, 404) is a
# real problem with the request itself -- retrying won't fix a bad key or bad model
# name, so we fail fast on those instead of wasting time looping.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 4
BASE_DELAY_SECONDS = 1.5


def _post_with_retry(url: str, headers: dict, json_body: dict, timeout: int) -> requests.Response:
    """POST with exponential backoff on transient errors (429/5xx).

    Non-retryable errors (bad key, bad model name, malformed request) raise
    immediately -- no point waiting around for those to "fix themselves".
    """
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, json=json_body, timeout=timeout)
            if resp.status_code in RETRYABLE_STATUS_CODES:
                last_exc = requests.exceptions.HTTPError(
                    f"{resp.status_code} (attempt {attempt + 1}/{MAX_RETRIES})", response=resp
                )
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_DELAY_SECONDS * (2 ** attempt)
                    print(f"[rag_engine] {resp.status_code} from API, retrying in {delay:.1f}s ...")
                    time.sleep(delay)
                    continue
                resp.raise_for_status()  # exhausted retries -- raise the real HTTPError
            resp.raise_for_status()  # non-retryable 4xx (other than 429) raises immediately
            return resp
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # network blip or a slow/overloaded model not responding in time --
            # both worth a retry, same backoff. (Timeout is NOT a subclass of
            # ConnectionError -- catching only ConnectionError, as an earlier
            # version of this code did, let a real 60s read-timeout on Streamlit
            # Cloud crash straight through instead of retrying/falling back.)
            last_exc = e
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY_SECONDS * (2 ** attempt)
                print(f"[rag_engine] connection error, retrying in {delay:.1f}s ...")
                time.sleep(delay)
                continue
            raise
    raise last_exc


def _get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Run: export GEMINI_API_KEY='your-key-here'"
        )
    return key


def embed_text(text: str, api_key: str | None = None) -> list[float]:
    """Embed a single string via the Gemini embeddings API. Returns a list of floats.

    Note: newer Google API keys (the "AQ."-prefixed format) are rejected with a
    404 if passed as a `?key=` query parameter -- they must go in the
    `x-goog-api-key` header instead. The header works for both old (AIzaSy...)
    and new (AQ...) key formats, so we always use the header.
    """
    api_key = api_key or _get_api_key()
    resp = _post_with_retry(
        EMBED_URL,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json_body={
            "model": EMBED_MODEL,
            "content": {"parts": [{"text": text}]},
        },
        timeout=30,
    )
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
        """Embed any KB entries not already in the cache; attach embeddings to self.kb."""
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
        """Return the top_k KB entries most similar to the query, each with a 'score' field."""
        with tracer.start_as_current_span("rag.retrieve") as span:
            span.set_attribute("query_length_chars", len(query))
            span.set_attribute("kb_size", len(self.kb))
            span.set_attribute("top_k", top_k)

            query_emb = embed_text(query, self.api_key)
            scored = []
            for entry in self.kb:
                score = _cosine(query_emb, entry["embedding"])
                scored.append({**{k: v for k, v in entry.items() if k != "embedding"}, "score": round(score, 4)})
            scored.sort(key=lambda x: x["score"], reverse=True)
            top_results = scored[:top_k]

            span.set_attribute("top_score", top_results[0]["score"] if top_results else 0.0)
            span.set_attribute("results_returned", len(top_results))
            return top_results


if __name__ == "__main__":
    # quick standalone smoke test
    engine = RagEngine()
    test_query = "sample uses process injection and reaches out over the network repeatedly"
    print(f"\nQuery: {test_query}\n")
    for r in engine.retrieve(test_query, top_k=3):
        print(f"- [{r['score']}] {r['title']} ({r['mitre_id']})")
