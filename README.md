#  MEEF — Malware Explanation & Evaluation Framework

**Live app:** [malwareanalysis22.streamlit.app](https://malwareanalysis22.streamlit.app/)

MEEF classifies Windows binaries (`.asm` / `.exe`) as malicious or benign from static assembly features, then explains *why* — grounding that explanation in retrieved threat-intelligence reference material via a small RAG (retrieval-augmented generation) pipeline, with a guardrail that catches the model if it cites something it wasn't actually given.

It's built as two layers on purpose: a classical ML/heuristic detector doing the actual classification (the part that needs to be fast, deterministic, and auditable), and an LLM layer on top that turns a bare probability into a reasoned, cited report an analyst could actually act on (the part that needs to be grounded, not just fluent).

---

## Why this exists

Most classroom malware-detection projects stop at "here's my accuracy number." That's necessary but not the interesting part of shipping this in practice — a SOC analyst doesn't want a percentage, they want to know *which* behaviors triggered the flag and whether the explanation is trustworthy. This project is an attempt to build that second part honestly: not just wiring an LLM to a classifier and hoping it says something reasonable, but explicitly retrieving grounding context first and then checking the model's output against that context afterward.

---

## Architecture

```
 .asm / .exe file
        │
        ▼
 ┌─────────────────┐      ┌──────────────────────┐
 │  Feature         │      │  Behavioral heuristic │
 │  extraction      │      │  engine (API pattern  │
 │  (opcode/API     │      │  matching: injection,  │
 │  ratios, CFG)    │      │  network, crypto,      │
 └────────┬─────────┘      │  persistence)          │
          │                └───────────┬────────────┘
          ▼                            │
 ┌─────────────────┐                   │
 │  RandomForest    │                   │
 │  (calibrated,     │                  │
 │  SMOTE + class-   │                  │
 │  weight balanced) │                  │
 └────────┬─────────┘                   │
          │                             │
          └──────────────┬──────────────┘
                          ▼
              Weighted verdict (0.4 ML + 0.6 heuristic)
                          │
                          ▼
              ┌───────────────────────┐
              │  Heuristic flags →     │
              │  retrieval query       │
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │  RAG retrieval         │
              │  (Gemini embeddings +  │
              │  cosine similarity     │
              │  over a curated MITRE  │
              │  ATT&CK knowledge base)│
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │  Grounded generation   │
              │  (Gemini drafts a      │
              │  structured JSON       │
              │  report citing only    │
              │  retrieved technique   │
              │  IDs)                  │
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │  Guardrail             │
              │  (strips any cited     │
              │  technique ID that     │
              │  wasn't actually       │
              │  retrieved)            │
              └───────────┬───────────┘
                          ▼
              Structured, cited threat report
                  (shown in the Streamlit UI)
```

---

## What's actually happening under the hood

**Detection (deterministic, explainable by design)**
- 38 scale-invariant ratio features derived from opcode/API distributions and control-flow-graph structure, so predictions hold up regardless of file size
- `RandomForestClassifier` (200 trees), trained with `class_weight="balanced"` and SMOTE oversampling on the training split only (never on the held-out test set, to avoid leaking synthetic samples into evaluation), then wrapped in `CalibratedClassifierCV` (isotonic) so the output probability is actually meaningful, not just a ranking score
- A parallel rule-based heuristic engine scores known-risky API call patterns (process injection, network beaconing, crypto usage, persistence mechanisms), weighted against benign-pattern counter-evidence (e.g. registry use *without* injection reads as an installer, not malware)
- Final verdict blends both signals (0.4 ML / 0.6 heuristic) rather than trusting either alone

**Explanation (grounded, not just fluent)**
- The heuristic flags a given sample actually triggered become a retrieval query
- That query is embedded and matched via cosine similarity against a curated knowledge base of MITRE ATT&CK technique descriptions and malware-family archetypes (ransomware / banking-trojan / worm patterns, plus explicit false-positive considerations for legitimate software)
- Gemini drafts a structured report using *only* that retrieved material, instructed not to invent technique IDs it wasn't given
- A guardrail pass checks every cited MITRE ID against what was actually retrieved and strips anything unsupported before it reaches the UI — the report shows if this happened, rather than silently hiding it

---

## Model performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 98.52% |
| Precision | 98.88% |
| Recall    | 99.20% |
| F1-Score  | 99.04% |
| ROC-AUC   | 99.88% |

Trained on 4,045 real samples (77% malicious / 23% benign — an intentionally-preserved imbalance handled via stratified splitting + SMOTE + class-weighting, not papered over).

---

## Tech stack

| Layer | Tools |
|---|---|
| Classification | scikit-learn (RandomForest, CalibratedClassifierCV), imbalanced-learn (SMOTE) |
| Heuristic engine | Custom rule-based API-pattern scorer |
| Retrieval | Gemini embeddings (`gemini-embedding-2`), cosine similarity |
| Generation | Gemini (`gemini-3.1-flash-lite`), structured JSON output |
| Guardrail | Custom verification pass — checked, never assumed |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Project structure

```
Meef/
├── app.py                 # Streamlit UI + orchestration
├── train_model.py         # Full training pipeline (SMOTE, calibration, eval)
├── rag_engine.py           # Embedding + retrieval over the knowledge base
├── report_agent.py         # Retrieval → generation → guardrail pipeline
├── knowledge_base.json     # Curated MITRE ATT&CK / malware-family reference material
├── data/
│   ├── features_ml.csv     # Training features
│   └── models/              # Saved classifier, scaler, metadata
└── requirements.txt
```

---

## Quick start (local)

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"   # PowerShell: $env:GEMINI_API_KEY="your-key-here"

# optional: retrain from scratch
python train_model.py

# sanity-check retrieval + the full report pipeline before touching the UI
python rag_engine.py
python report_agent.py

streamlit run app.py
```

---

## Deployment (Streamlit Community Cloud)

The live app redeploys automatically on every push to the connected branch. The one manual step is setting the API key as a Streamlit secret — a local environment variable doesn't carry over to the cloud:

1. Open the app's settings on [share.streamlit.io](https://share.streamlit.io)
2. **Secrets** → add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
3. Save — it restarts automatically.

---


---

## Credits

@sriroo
