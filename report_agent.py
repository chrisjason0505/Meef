"""
report_agent.py
----------------
Turns Meef's existing ML + heuristic output into a grounded, structured
threat report using RAG + Gemini generation, with a guardrail.
"""

import os
import json
import re

import requests

from rag_engine import RagEngine

GEN_MODEL = "models/gemini-3.1-flash-lite"
GEN_URL = f"https://generativelanguage.googleapis.com/v1beta/{GEN_MODEL}:generateContent"

_engine = None


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


def _call_gemini(prompt: str, api_key: str) -> str:
    resp = requests.post(
        GEN_URL,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Model did not return parseable JSON:\n{text}")
    return json.loads(match.group(0))


def _apply_guardrail(report: dict, retrieved: list[dict]) -> dict:
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
        "unverified_claims": unverified,
        "status": "clean" if not unverified else "claims_removed",
    }
    return report


def generate_report(ml_probability: float, heuristic_flags: list[str], filename: str = "sample") -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    engine = _get_engine()

    query = _build_query(ml_probability, heuristic_flags)
    retrieved = engine.retrieve(query, top_k=4)

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
