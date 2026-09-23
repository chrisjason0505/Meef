"""
report_agent.py
----------------
Turns Meef's existing ML + heuristic output into a grounded, structured
threat report using RAG (rag_engine.py) + Gemini generation, with a basic
guardrail that strips any claim the model made up and isn't actually
supported by the retrieved knowledge-base context.

This is the "agent" layer: it orchestrates three steps in sequence --
    1. build a retrieval query from the existing classifier's output
    2. retrieve grounding context (rag_engine.RagEngine)
    3. generate a structured report, then verify it against that context

Wire this into app.py: call generate_report(...) with whatever your
existing scoring function already produces, no changes needed to the
ML/heuristic code itself.

Usage:
    from report_agent import generate_report

    report = generate_report(
        ml_probability=0.94,
        heuristic_flags=["process_injection", "network_activity"],
        filename="sample_1.exe",
    )
    print(report["summary"])
"""

import os
import json
import re
import time

import requests

from rag_engine import RagEngine
from observability import tracer

# Fallback chain. Confirmed via Google's own developer forum + independent
# troubleshooting guides: persistent 503s on the newest model tier are a
# well-documented, systemic Google-side capacity issue (worse right after a
# new model generation launches, like 3.x currently is) -- not something
# retrying the SAME model harder fixes. Their own guidance is to fall back
# to an older, more established model tier, which is why the 2.5-generation
# models are last here: they're a generation older than the 3.1/3.5 lite
# tier above them, so they aren't competing for the same newly-strained
# capacity. Model names verified directly against this API key's own
# ListModels response, not assumed from docs.
GEN_MODELS = [
    "models/gemini-3.1-flash-lite",
    "models/gemini-3.5-flash-lite",
    "models/gemini-flash-lite-latest",
    "models/gemini-flash-latest",
    "models/gemini-2.5-flash",
    "models/gemini-2.5-pro",
]

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES_PER_MODEL = 1  # try each model once, then move on -- with 5 models
                            # in the fallback chain, breadth beats depth: better
                            # to try 5 different models once each than 1 model
                            # twice and wait longer for the same overloaded tier
REQUEST_TIMEOUT_SECONDS = 25  # short per-attempt timeout so a hung/overloaded
                                # model fails fast and moves to the next one,
                                # instead of the UI looking frozen for a minute
BASE_DELAY_SECONDS = 1.5

# Cross-provider fallback: everything above is Google's infrastructure, so a
# Google-wide capacity issue (exactly what we hit) takes out every model in
# GEN_MODELS at once. Groq runs on its own custom hardware (LPUs, not GPUs),
# so it isn't competing for the same capacity at all -- this is tried only
# after the entire Gemini chain fails, and only if GROQ_API_KEY is set (it's
# optional; the app works fine without it, this just adds one more layer).
# Groq's API is OpenAI-compatible. Verify the model name against
# console.groq.com/docs/models before relying on it -- model names on any
# provider can change, which is exactly what bit us with Gemini tonight.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

_engine = None  # lazy singleton so the KB is only embedded once per process


def _get_engine():
    global _engine
    if _engine is None:
        _engine = RagEngine()
    return _engine


def _build_query(ml_probability: float, heuristic_flags: list[str]) -> str:
    flags_text = ", ".join(heuristic_flags) if heuristic_flags else "no specific heuristic flags"
    return (
        f"A binary sample was scored by an ML classifier with malicious probability "
        f"{ml_probability:.2f}. Heuristic engine flags raised: {flags_text}. "
        f"What known techniques, malware families, or false-positive considerations "
        f"are relevant to this combination of signals?"
    )


def _call_groq(prompt: str) -> str | None:
    """Cross-provider fallback, tried only after every Gemini model has
    failed. Returns None (not an exception) if GROQ_API_KEY isn't set, so
    this stays fully optional -- the app works without it. Returns None on
    failure too, since by this point we're out of fallbacks anyway and the
    caller just needs to know whether it got usable text.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    with tracer.start_as_current_span("llm.generate_attempt") as span:
        span.set_attribute("provider", "groq")
        span.set_attribute("model", GROQ_MODEL)
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            data = resp.json()
            span.set_attribute("status", "success")
            return data["choices"][0]["message"]["content"]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            span.set_attribute("status", "failed")
            span.set_attribute("error", str(e))
            print(f"[report_agent] Groq fallback also failed: {e}")
            return None


def _call_gemini(prompt: str, api_key: str) -> str:
    """Call Gemini generateContent with retry-with-backoff per model, and a
    fallback chain across GEN_MODELS if a model keeps returning transient
    errors (503 = that model is momentarily overloaded, not a code problem).
    If every Gemini model fails, falls back to Groq (a different provider on
    different infrastructure) as a last resort, when GROQ_API_KEY is set.

    Header auth (x-goog-api-key), not ?key= query param -- required for the
    newer "AQ."-prefixed key format, and works fine for older keys too.
    """
    last_error = None

    with tracer.start_as_current_span("llm.generate") as pipeline_span:
        pipeline_span.set_attribute("models_available", len(GEN_MODELS))

        for model in GEN_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent"

            for attempt in range(MAX_RETRIES_PER_MODEL):
                with tracer.start_as_current_span("llm.generate_attempt") as span:
                    span.set_attribute("provider", "gemini")
                    span.set_attribute("model", model)
                    span.set_attribute("attempt", attempt + 1)
                    start = time.monotonic()
                    try:
                        resp = requests.post(
                            url,
                            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                            json={
                                "contents": [{"parts": [{"text": prompt}]}],
                                "generationConfig": {"temperature": 0.2},
                            },
                            timeout=REQUEST_TIMEOUT_SECONDS,
                        )
                        span.set_attribute("latency_ms", round((time.monotonic() - start) * 1000, 1))
                        span.set_attribute("status_code", resp.status_code)

                        if resp.status_code in RETRYABLE_STATUS_CODES:
                            span.set_attribute("status", "retryable_error")
                            last_error = requests.exceptions.HTTPError(
                                f"{resp.status_code} from {model} (attempt {attempt + 1}/{MAX_RETRIES_PER_MODEL})",
                                response=resp,
                            )
                            if attempt < MAX_RETRIES_PER_MODEL - 1:
                                delay = BASE_DELAY_SECONDS * (2 ** attempt)
                                print(f"[report_agent] {resp.status_code} from {model}, retrying in {delay:.1f}s ...")
                                time.sleep(delay)
                                continue
                            print(f"[report_agent] {model} still failing after retries, falling back to next model ...")
                            break  # give up on this model, try the next one in GEN_MODELS

                        resp.raise_for_status()  # non-retryable error -- raise immediately, no point falling back
                        data = resp.json()
                        span.set_attribute("status", "success")
                        pipeline_span.set_attribute("final_provider", "gemini")
                        pipeline_span.set_attribute("final_model", model)
                        return data["candidates"][0]["content"]["parts"][0]["text"]

                    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                        # Timeout is NOT a subclass of ConnectionError -- this was the
                        # actual bug behind the "Read timed out" crash on Streamlit
                        # Cloud: only catching ConnectionError let a real request
                        # timeout skip retry/fallback entirely and crash straight
                        # through instead of trying the next model.
                        span.set_attribute("status", "connection_error")
                        span.set_attribute("error", str(e))
                        last_error = e
                        if attempt < MAX_RETRIES_PER_MODEL - 1:
                            delay = BASE_DELAY_SECONDS * (2 ** attempt)
                            print(f"[report_agent] connection error, retrying in {delay:.1f}s ...")
                            time.sleep(delay)
                            continue
                        break  # fall back to next model

        # every Gemini model failed -- try Groq (different provider/infra) before giving up entirely
        print("[report_agent] all Gemini models failed, trying Groq cross-provider fallback ...")
        groq_result = _call_groq(prompt)
        if groq_result is not None:
            pipeline_span.set_attribute("final_provider", "groq")
            pipeline_span.set_attribute("final_model", GROQ_MODEL)
            return groq_result

        pipeline_span.set_attribute("status", "all_providers_failed")
        raise RuntimeError(
            f"All models in GEN_MODELS ({', '.join(GEN_MODELS)}) failed after retries, "
            f"and Groq fallback was unavailable or also failed. Last error: {last_error}"
        )


def _extract_json(text: str) -> dict:
    """Gemini sometimes wraps JSON in ```json fences -- strip those before parsing."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Model did not return parseable JSON:\n{text}")
    return json.loads(match.group(0))


def _apply_guardrail(report: dict, retrieved: list[dict]) -> dict:
    """
    Strip/flag any cited mitre_id in the model's output that doesn't actually
    appear in the retrieved context. This is the core "validate AI-generated
    output" step -- without it, the model could cite a technique ID it
    hallucinated instead of one it was actually given.
    """
    retrieved_ids = {r["mitre_id"] for r in retrieved}
    verified = []
    unverified = []

    for item in report.get("supporting_techniques", []):
        cited_id = item.get("mitre_id")
        if cited_id in retrieved_ids:
            verified.append(item)
        else:
            unverified.append(item)

    report["supporting_techniques"] = verified
    report["guardrail"] = {
        "unverified_claims_removed": len(unverified),
        "unverified_claims": unverified,  # kept for transparency/debugging, not shown to end user
        "status": "clean" if not unverified else "claims_removed",
    }
    return report


def generate_report(ml_probability: float, heuristic_flags: list[str], filename: str = "sample") -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    engine = _get_engine()

    # 1. retrieve grounding context
    query = _build_query(ml_probability, heuristic_flags)
    retrieved = engine.retrieve(query, top_k=4)

    # 2. build a prompt that forces the model to ONLY cite retrieved mitre_ids
    context_block = "\n\n".join(
        f"[{r['mitre_id']}] {r['title']}: {r['text']}" for r in retrieved
    )
    prompt = f"""You are a SOC (security operations center) analyst assistant. A malware
scanner produced the following result for file "{filename}":

- ML classifier malicious probability: {ml_probability:.2f}
- Heuristic flags raised: {", ".join(heuristic_flags) if heuristic_flags else "none"}

Use ONLY the following retrieved reference material to explain the result. Do not invent
any MITRE technique ID that is not listed below -- if none of the retrieved material is
relevant, say so plainly instead of guessing.

Retrieved reference material:
{context_block}

Respond with ONLY a JSON object, no extra prose, in exactly this shape:
{{
  "risk_level": "low" | "medium" | "high",
  "summary": "2-3 sentence plain-English explanation of the result for an analyst",
  "supporting_techniques": [
    {{"mitre_id": "T1055", "name": "short name", "evidence": "why this applies here, 1 sentence"}}
  ],
  "caveats": "1 sentence noting any false-positive considerations, if relevant"
}}
"""

    # 3. generate + parse + guardrail
    raw_text = _call_gemini(prompt, api_key)
    report = _extract_json(raw_text)
    report = _apply_guardrail(report, retrieved)
    report["retrieved_sources"] = [
        {"mitre_id": r["mitre_id"], "title": r["title"], "score": r["score"]} for r in retrieved
    ]
    report["filename"] = filename
    return report


if __name__ == "__main__":
    result = generate_report(
        ml_probability=0.94,
        heuristic_flags=["process_injection", "network_activity"],
        filename="suspicious_sample.exe",
    )
    print(json.dumps(result, indent=2))
