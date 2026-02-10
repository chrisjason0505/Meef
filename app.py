#!/usr/bin/env python3
"""
app.py - Streamlit Malware Detection Interface (MEEF)
=====================================================
Upload .asm or .exe files for analysis.
Uses a DUAL scoring system:
  1. ML model trained on ratio-based features from data/features_ml.csv
  2. Rule-based behavioral heuristic scoring

The final verdict combines both signals for robust classification
regardless of file size.
"""

import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import streamlit as st
import joblib

# -----------------------------------------------------------
#  Paths
# -----------------------------------------------------------
MODEL_DIR = Path("data/models")
MODEL_PATH = MODEL_DIR / "malware_classifier.pkl"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# -----------------------------------------------------------
#  Page config & custom CSS
# -----------------------------------------------------------
st.set_page_config(
    page_title="MEEF Malware Sentinel",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%); }

    .glass-card {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 1.8rem;
        backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.25); margin-bottom: 1.2rem;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .glass-card:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,0,0,0.35); }

    .result-benign {
        background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(6,78,59,0.20));
        border: 1px solid rgba(16,185,129,0.35); border-radius: 16px;
        padding: 2rem; text-align: center; animation: fadeInUp 0.6s ease-out;
    }
    .result-malware {
        background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(127,29,29,0.20));
        border: 1px solid rgba(239,68,68,0.35); border-radius: 16px;
        padding: 2rem; text-align: center; animation: fadeInUp 0.6s ease-out;
    }
    .result-unknown {
        background: linear-gradient(135deg, rgba(234,179,8,0.12), rgba(113,63,18,0.20));
        border: 1px solid rgba(234,179,8,0.35); border-radius: 16px;
        padding: 2rem; text-align: center; animation: fadeInUp 0.6s ease-out;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #e0e7ff; }

    .metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 0.6rem 0; }
    .metric-item {
        flex: 1; min-width: 120px; background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
        padding: 1rem; text-align: center;
    }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #a5b4fc; }
    .metric-label {
        font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;
        letter-spacing: 0.05em; margin-top: 0.25rem;
    }

    @keyframes fadeInUp { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
    .pulse-text { animation: pulse 2s ease-in-out infinite; }

    .hero-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #818cf8, #a78bfa, #c084fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub { color: #94a3b8; font-size: 1rem; margin-bottom: 1.6rem; }
    .section-label {
        font-size: 0.7rem; font-weight: 600; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------
#  Model loading
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    for p, name in [(MODEL_PATH, "Model"), (SCALER_PATH, "Scaler"), (METADATA_PATH, "Metadata")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found at '{p}'. Run `python train_model.py` first.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return model, scaler, metadata


# -----------------------------------------------------------
#  API keyword sets for behavioral scoring
# -----------------------------------------------------------
NETWORK_APIS = {
    "socket", "connect", "send", "recv", "bind", "listen", "accept",
    "wsastartup", "wsasocketa", "internetopena", "internetopenurla",
    "httpsendrequesta", "urldownloadtofile", "winhttp", "getaddrinfo",
    "gethostbyname", "internetconnecta", "httpopenrequesta",
}
FILEOPS_APIS = {
    "createfile", "readfile", "writefile", "deletefile", "copyfile",
    "movefile", "findfirstfile", "findnextfile", "setfilepointer",
    "getfilesize", "createfilea", "createfilew", "ntcreatefile",
}
REGISTRY_APIS = {
    "regopenkeyex", "regsetvalueex", "regqueryvalueex", "regcreatekeyex",
    "regdeletekey", "regdeletevalue", "regenumkeyex", "regenumvalue",
    "regopenkeyexa", "regsetvalueexa", "regclosekey", "regcreatekey",
}
MEMORY_APIS = {
    "virtualalloc", "virtualfree", "virtualprotect", "heapalloc",
    "heapfree", "globalalloc", "globalfree", "rtlmovememory",
    "ntreadvirtualmemory",
}
INJECTION_APIS = {
    "createremotethread", "writeprocessmemory", "readprocessmemory",
    "ntmapviewofsection", "queueuserapc", "ntunmapviewofsection",
    "openprocess", "virtualallocex", "virtualprotectex", "ntcreatethreadex",
}
CRYPTO_APIS = {
    "cryptacquirecontext", "cryptencrypt", "cryptdecrypt", "crypthashdata",
    "cryptcreatehash", "cryptgenkey", "cryptimportkey",
    "bcryptencrypt", "bcryptdecrypt",
}
PERSISTENCE_APIS = {
    "createservice", "startservice", "openscmanager",
    "regsetvalueex", "regsetvalueexa", "createservicea",
    "schtaskscreate", "copyfile", "movefileex",
}

VALID_OPCODES = {
    "mov", "push", "pop", "call", "ret", "retn", "jmp", "je", "jne", "jz", "jnz",
    "jg", "jl", "jge", "jle", "ja", "jb", "jae", "jbe", "jc", "jnc",
    "jo", "jno", "js", "jns", "add", "sub", "xor", "and", "or", "not",
    "neg", "inc", "dec", "shl", "shr", "sal", "sar", "rol", "ror",
    "test", "cmp", "lea", "nop", "int", "hlt", "cdq", "cbw", "cwde",
    "movzx", "movsx", "movsb", "movsw", "movsd", "cmovz", "cmovnz",
    "cmove", "cmovne", "cmovg", "cmovl", "cmovge", "cmovle",
    "sete", "setne", "setg", "setl", "setge", "setle", "seta", "setb",
    "imul", "idiv", "mul", "div", "rep", "repne", "repz",
    "stosb", "stosw", "stosd", "lodsb", "lodsw", "lodsd",
    "scasb", "scasw", "scasd", "cmpsb", "cmpsw", "cmpsd",
    "loop", "loope", "loopne", "enter", "leave",
    "fld", "fst", "fstp", "fadd", "fsub", "fmul", "fdiv",
    "pushf", "popf", "pushfd", "popfd", "lahf", "sahf",
    "bswap", "xchg", "cmpxchg", "lock", "wait", "fwait",
    "sbb", "adc", "bt", "bts", "btr", "btc", "bsf", "bsr",
    "movs", "lods", "stos", "scas", "cmps", "in", "out", "ins", "outs",
}
BRANCH_KEYWORDS = {
    "jmp", "je", "jne", "jz", "jnz", "jg", "jl", "jge", "jle",
    "ja", "jb", "jae", "jbe", "jc", "jnc", "jo", "jno", "js", "jns",
    "loop", "loope", "loopne",
}


# -----------------------------------------------------------
#  Parse ASM → raw stats
# -----------------------------------------------------------
def parse_asm(asm_text: str):
    """Parse ASM text and return raw statistics."""
    lines = asm_text.splitlines()

    api_pattern = re.compile(
        r"(?:call|CALL)\s+(?:dword\s+ptr\s+)?(?:ds:)?\[?([A-Za-z_][A-Za-z0-9_]*)\]?",
    )
    api_counter: dict[str, int] = {}
    for line in lines:
        m = api_pattern.search(line)
        if m:
            name = m.group(1).lower().strip("_")
            if name not in VALID_OPCODES and len(name) > 2:
                api_counter[name] = api_counter.get(name, 0) + 1

    all_api_names = set(api_counter.keys())

    label_pat = re.compile(r"^[A-Za-z_@?.][A-Za-z0-9_@?$.]*\s*:")
    dir_pat = re.compile(r"^\s*\.")

    num_blocks = 0
    num_edges = 0
    branch_count = 0
    opcode_counter: dict[str, int] = {}
    total_instructions = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(";") or stripped.startswith("#"):
            continue
        if dir_pat.match(stripped):
            continue
        if label_pat.match(stripped):
            num_blocks += 1
            after = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            if not after:
                continue
            stripped = after
        parts = stripped.split()
        if not parts:
            continue
        mnemonic = parts[0].lower().rstrip(":")
        if mnemonic in VALID_OPCODES or mnemonic.startswith("j") or mnemonic == "call":
            opcode_counter[mnemonic] = opcode_counter.get(mnemonic, 0) + 1
            total_instructions += 1
            if mnemonic in BRANCH_KEYWORDS:
                branch_count += 1
                num_edges += 2
            elif mnemonic == "call":
                num_edges += 1
            elif mnemonic in ("ret", "retn"):
                num_edges += 1

    if num_blocks == 0:
        num_blocks = 1

    mov_count = sum(v for k, v in opcode_counter.items() if k.startswith("mov"))
    push_count = sum(v for k, v in opcode_counter.items() if k.startswith("push"))
    pop_count = sum(v for k, v in opcode_counter.items() if k.startswith("pop"))

    return {
        "api_counter": api_counter,
        "all_api_names": all_api_names,
        "opcode_counter": opcode_counter,
        "total_instructions": total_instructions,
        "num_blocks": num_blocks,
        "num_edges": num_edges,
        "branch_count": branch_count,
        "mov_count": mov_count,
        "push_count": push_count,
        "pop_count": pop_count,
        "call_count": opcode_counter.get("call", 0),
        "ret_count": opcode_counter.get("ret", 0) + opcode_counter.get("retn", 0),
        "add_count": opcode_counter.get("add", 0) + opcode_counter.get("adc", 0),
        "sub_count": opcode_counter.get("sub", 0) + opcode_counter.get("sbb", 0),
        "xor_count": opcode_counter.get("xor", 0),
        "test_count": opcode_counter.get("test", 0),
        "jmp_count": opcode_counter.get("jmp", 0),
    }


# -----------------------------------------------------------
#  Ratio-based features for ML model
# -----------------------------------------------------------
def extract_ml_features(stats: dict, feature_names: list) -> np.ndarray:
    """Produce ratio-based features matching what train_model.py generates."""
    api_names = stats["all_api_names"]
    total_ops = max(stats["total_instructions"], 1)
    total_api = max(sum(stats["api_counter"].values()), 1)
    api_total = sum(stats["api_counter"].values())
    sorted_apis = sorted(stats["api_counter"].values(), reverse=True)
    top_api = [(sorted_apis[i] if i < len(sorted_apis) else 0) for i in range(10)]

    feature_map = {
        "uses_network": int(bool(api_names & NETWORK_APIS)),
        "uses_fileops": int(bool(api_names & FILEOPS_APIS)),
        "uses_registry": int(bool(api_names & REGISTRY_APIS)),
        "uses_memory": int(bool(api_names & MEMORY_APIS)),
        "uses_injection": int(bool(api_names & INJECTION_APIS)),
        "uses_crypto": int(bool(api_names & CRYPTO_APIS)),
        "uses_persist": int(bool(api_names & PERSISTENCE_APIS)),
        "cfg_branch_density": stats["branch_count"] / total_ops,
        "cfg_cyclomatic_complexity": max(float(stats["num_edges"] - stats["num_blocks"] + 2), 1.0),
        "call_ratio": stats["call_count"] / total_ops,
        "jmp_ratio": stats["jmp_count"] / total_ops,
        "api_to_opcode_ratio": api_total / total_ops,
        "num_unique_opcodes": len(stats["opcode_counter"]),
        "num_unique_apis": len(stats["api_counter"]),
        "log_total_opcodes": math.log1p(stats["total_instructions"]),
        "log_total_api_calls": math.log1p(api_total),
        "log_cfg_num_blocks": math.log1p(stats["num_blocks"]),
        "log_cfg_num_edges": math.log1p(stats["num_edges"]),
        "opcode_call_ratio": stats["call_count"] / total_ops,
        "opcode_mov_ratio": stats["mov_count"] / total_ops,
        "opcode_push_ratio": stats["push_count"] / total_ops,
        "opcode_pop_ratio": stats["pop_count"] / total_ops,
        "opcode_jmp_ratio": stats["jmp_count"] / total_ops,
        "opcode_ret_ratio": stats["ret_count"] / total_ops,
        "opcode_add_ratio": stats["add_count"] / total_ops,
        "opcode_sub_ratio": stats["sub_count"] / total_ops,
        "opcode_xor_ratio": stats["xor_count"] / total_ops,
        "opcode_test_ratio": stats["test_count"] / total_ops,
        "top_api_1_ratio": top_api[0] / total_api,
        "top_api_2_ratio": top_api[1] / total_api,
        "top_api_3_ratio": top_api[2] / total_api,
        "top_api_4_ratio": top_api[3] / total_api,
        "top_api_5_ratio": top_api[4] / total_api,
        "top_api_6_ratio": top_api[5] / total_api,
        "top_api_7_ratio": top_api[6] / total_api,
        "top_api_8_ratio": top_api[7] / total_api,
        "top_api_9_ratio": top_api[8] / total_api,
        "top_api_10_ratio": top_api[9] / total_api,
    }
    vec = np.array([float(feature_map.get(fn, 0.0)) for fn in feature_names], dtype=np.float64)
    return vec.reshape(1, -1)


# -----------------------------------------------------------
#  Rule-based behavioral scoring
# -----------------------------------------------------------
# This works as a complementary heuristic. It assigns a threat
# score from 0 (safe) to 1 (definitely malicious) based on
# which suspicious API categories are present, how concentrated
# they are, and whether the file lacks benign structure.
#
# Scoring rules (derived from training data analysis):
#   - Injection APIs (CreateRemoteThread, WriteProcessMemory):    +0.30
#   - Network exfil APIs (InternetOpen, send):                    +0.25
#   - Crypto APIs (CryptEncrypt, CryptDecrypt):                   +0.10
#   - Persistence APIs (CreateService, RegSetValueEx):            +0.10
#   - Registry manipulation:                                      +0.05
#   - High call-to-instruction ratio (>30%):                      +0.10
#   - Very few instruction types (num_unique_opcodes < 5):        +0.05
#   - Standard file/memory ops WITHOUT injection/network:         -0.15
#   - High structural complexity (many blocks, balanced ops):     -0.10

THREAT_WEIGHTS = {
    "injection": 0.30,
    "network": 0.25,
    "crypto": 0.10,
    "persistence": 0.10,
    "registry": 0.02,        # low: common in benign installers
    "high_call_ratio": 0.10,
    "low_diversity": 0.05,
}
BENIGN_WEIGHTS = {
    "standard_ops_only": -0.20,  # file/mem ops without injection/network
    "high_complexity": -0.10,
    "registry_no_inject": -0.10,  # registry without injection = installer
}


def compute_threat_score(stats: dict) -> tuple[float, dict]:
    """Compute a rule-based threat score from 0 (benign) to 1 (malicious)."""
    api_names = stats["all_api_names"]
    total_ops = max(stats["total_instructions"], 1)
    details: dict[str, float] = {}

    score = 0.0

    # Threatening categories
    if api_names & INJECTION_APIS:
        score += THREAT_WEIGHTS["injection"]
        details["injection_apis"] = THREAT_WEIGHTS["injection"]
    if api_names & NETWORK_APIS:
        score += THREAT_WEIGHTS["network"]
        details["network_apis"] = THREAT_WEIGHTS["network"]
    if api_names & CRYPTO_APIS:
        score += THREAT_WEIGHTS["crypto"]
        details["crypto_apis"] = THREAT_WEIGHTS["crypto"]
    if api_names & PERSISTENCE_APIS:
        score += THREAT_WEIGHTS["persistence"]
        details["persistence_apis"] = THREAT_WEIGHTS["persistence"]
    if api_names & REGISTRY_APIS:
        score += THREAT_WEIGHTS["registry"]
        details["registry_apis"] = THREAT_WEIGHTS["registry"]

    # High call ratio → suspicious (packed/obfuscated code)
    call_ratio = stats["call_count"] / total_ops
    if call_ratio > 0.30:
        score += THREAT_WEIGHTS["high_call_ratio"]
        details["high_call_ratio"] = THREAT_WEIGHTS["high_call_ratio"]

    # Very low opcode diversity → possibly packed
    if len(stats["opcode_counter"]) < 5 and total_ops > 10:
        score += THREAT_WEIGHTS["low_diversity"]
        details["low_opcode_diversity"] = THREAT_WEIGHTS["low_diversity"]

    # Benign indicators
    has_standard_ops = bool(api_names & (FILEOPS_APIS | MEMORY_APIS))
    has_no_dangerous = not bool(api_names & (INJECTION_APIS | NETWORK_APIS))
    if has_standard_ops and has_no_dangerous:
        bonus = BENIGN_WEIGHTS["standard_ops_only"]
        score += bonus
        details["standard_ops_only"] = bonus

    # Registry without injection/network = likely installer
    has_registry = bool(api_names & REGISTRY_APIS)
    if has_registry and has_no_dangerous:
        bonus = BENIGN_WEIGHTS["registry_no_inject"]
        score += bonus
        details["registry_no_injection"] = bonus

    if stats["num_blocks"] > 5 and len(stats["opcode_counter"]) > 8:
        bonus = BENIGN_WEIGHTS["high_complexity"]
        score += bonus
        details["high_structural_complexity"] = bonus

    # No API calls at all → neutral (neither threatening nor obviously safe)
    if not api_names:
        details["no_apis_found"] = 0.0

    score = max(0.0, min(1.0, score))
    return score, details


# -----------------------------------------------------------
#  Combined classification
# -----------------------------------------------------------
def classify_file(ml_proba: np.ndarray, threat_score: float,
                  threshold: float) -> tuple[str, float, str]:
    """Combine ML probability with rule-based threat score.

    Strategy:
      - ML model weight: 0.4 (it was trained on different features)
      - Behavioral score weight: 0.6 (directly measures suspicious APIs)

    Returns: (verdict, confidence, level)
    """
    ml_p_malicious = ml_proba[1]

    # Weighted combination
    combined = 0.4 * ml_p_malicious + 0.6 * threat_score

    if combined >= 0.50:
        verdict = "MALICIOUS"
        confidence = combined * 100
        level = "high" if combined >= 0.65 else "moderate"
    elif combined <= 0.20:
        verdict = "BENIGN"
        confidence = (1 - combined) * 100
        level = "high"
    else:
        verdict = "BENIGN"
        confidence = (1 - combined) * 100
        level = "low"

    return verdict, confidence, level


# -----------------------------------------------------------
#  Helpers
# -----------------------------------------------------------
def fmt_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# -----------------------------------------------------------
#  Sidebar
# -----------------------------------------------------------
def render_sidebar(metadata: dict | None) -> None:
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:1rem 0'>"
            "<span style='font-size:2.5rem'>&#x1F6E1;</span><br>"
            "<span style='font-size:1.2rem;font-weight:700;"
            "background:linear-gradient(90deg,#818cf8,#c084fc);"
            "-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>"
            "MEEF Malware Sentinel</span></div>",
            unsafe_allow_html=True,
        )
        st.divider()

        if metadata:
            metrics = metadata.get("metrics", {})
            st.markdown("##### Model Info")
            st.markdown(
                f"""
                | Property | Value |
                |----------|-------|
                | ML Model | RandomForest (Calibrated) |
                | Estimators | {metadata.get('n_estimators', 'N/A')} |
                | Features | {metadata.get('num_features', 'N/A')} (ratio-based) |
                | Detection | ML + Behavioral Heuristic |
                | ML Accuracy | {metrics.get('accuracy', 0)*100:.1f}% |
                | ML F1-Score | {metrics.get('f1', 0)*100:.1f}% |
                """
            )
            st.divider()
            dist = metadata.get("class_distribution", {})
            total = dist.get("benign", 0) + dist.get("malicious", 0)
            st.markdown("##### Training Data")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Benign", f"{dist.get('benign', '?')}")
            with c2:
                st.metric("Malicious", f"{dist.get('malicious', '?')}")
            st.caption(f"Total: {total} samples | SMOTE-rebalanced | Calibrated")

        st.divider()
        st.markdown("##### Detection Method")
        st.markdown(
            "**Dual scoring system:**\n\n"
            "1. **ML Model** (40%): Ratio-based features\n"
            "   - Opcode distribution ratios\n"
            "   - Log-scale structural metrics\n"
            "   - Binary behavioral flags\n\n"
            "2. **Behavioral Heuristic** (60%):\n"
            "   - Injection API detection\n"
            "   - Network exfiltration APIs\n"
            "   - Crypto / Persistence APIs\n"
            "   - Code structure analysis"
        )
        st.divider()
        st.caption("MEEF Project | Dual Scoring | Streamlit")


# -----------------------------------------------------------
#  Main
# -----------------------------------------------------------
def main() -> None:
    try:
        model, scaler, metadata = load_model()
        model_ready = True
    except FileNotFoundError as exc:
        model_ready = False
        metadata = None
        st.error(f"**Model not loaded**: {exc}\n\n"
                 "```bash\npython train_model.py\n```")

    render_sidebar(metadata)

    st.markdown(
        "<div class='glass-card' style='text-align:center'>"
        "<div class='hero-title'>&#x1F6E1; MEEF Malware Sentinel</div>"
        "<div class='hero-sub'>"
        "AI-powered assembly analysis &mdash; upload .asm or .exe to check if it's safe"
        "</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<p class='section-label'>Upload a file for analysis</p>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag & drop or click to browse (.asm or .exe)",
        type=["asm", "exe"],
        label_visibility="collapsed",
    )

    if uploaded is not None and model_ready:
        file_bytes = uploaded.read()
        file_name = uploaded.name
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        feature_names = metadata["feature_names"]
        threshold = metadata.get("optimal_threshold", 0.5)

        # File metadata
        st.markdown(
            f"""
            <div class='glass-card'>
                <div class='section-label'>File Metadata</div>
                <div class='metric-row'>
                    <div class='metric-item'>
                        <div class='metric-value'>&#128196;</div>
                        <div class='metric-label'>{file_name}</div>
                    </div>
                    <div class='metric-item'>
                        <div class='metric-value'>{fmt_size(len(file_bytes))}</div>
                        <div class='metric-label'>File Size</div>
                    </div>
                    <div class='metric-item'>
                        <div class='metric-value' style='font-size:0.85rem;word-break:break-all'>{file_hash[:16]}...</div>
                        <div class='metric-label'>SHA-256 (prefix)</div>
                    </div>
                </div>
                <details style='margin-top:0.8rem;color:#94a3b8;font-size:0.8rem'>
                    <summary>Full SHA-256</summary>
                    <code style='word-break:break-all'>{file_hash}</code>
                </details>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.spinner("Analyzing file ..."):
            time.sleep(0.5)

            if file_name.lower().endswith(".asm"):
                try:
                    asm_text = file_bytes.decode("utf-8", errors="replace")
                except Exception:
                    asm_text = file_bytes.decode("latin-1", errors="replace")
            else:
                asm_text = file_bytes.decode("latin-1", errors="replace")

            # 1. Parse ASM
            stats = parse_asm(asm_text)

            # 2. ML features + prediction
            ml_features = extract_ml_features(stats, feature_names)
            ml_scaled = scaler.transform(ml_features)
            ml_proba = model.predict_proba(ml_scaled)[0]

            # 3. Behavioral heuristic
            threat_score, threat_details = compute_threat_score(stats)

            # 4. Combined classification
            verdict, confidence, level = classify_file(ml_proba, threat_score, threshold)

        is_malware = verdict == "MALICIOUS"

        # Result card
        if is_malware:
            cls = "result-malware"
            emoji = "&#x1F534;"
            label = "MALICIOUS - THREAT DETECTED"
            color = "#ef4444"
            advice = "This file exhibits patterns consistent with malicious software. Proceed with extreme caution."
        elif level == "low":
            cls = "result-unknown"
            emoji = "&#x1F7E1;"
            label = "LIKELY BENIGN - LOW CONFIDENCE"
            color = "#eab308"
            advice = "The file does not strongly match malicious patterns, but exercise caution."
        else:
            cls = "result-benign"
            emoji = "&#x1F7E2;"
            label = "BENIGN - FILE IS SAFE"
            color = "#10b981"
            advice = "This file does not exhibit known malicious patterns."

        st.markdown(
            f"<div class='{cls}'>"
            f"<div style='font-size:3.5rem;margin-bottom:0.4rem'>{emoji}</div>"
            f"<div style='font-size:1.6rem;font-weight:800;color:{color};"
            f"letter-spacing:0.05em'>{label}</div>"
            f"<div style='color:#cbd5e1;margin-top:0.6rem;font-size:0.95rem'>{advice}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Confidence
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p class='section-label'>Confidence Score</p>", unsafe_allow_html=True)
        st.progress(min(confidence / 100, 1.0))
        st.markdown(
            f"<div style='text-align:center;font-size:1.8rem;font-weight:700;"
            f"color:{color}'>{confidence:.1f}%</div>",
            unsafe_allow_html=True,
        )

        # Scoring breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='glass-card'>"
            f"<div class='section-label'>Scoring Breakdown</div>"
            f"<div class='metric-row'>"
            f"<div class='metric-item'>"
            f"<div class='metric-value' style='color:#818cf8'>{ml_proba[1]*100:.1f}%</div>"
            f"<div class='metric-label'>ML P(Malicious) &middot; 40%</div></div>"
            f"<div class='metric-item'>"
            f"<div class='metric-value' style='color:#f59e0b'>{threat_score*100:.1f}%</div>"
            f"<div class='metric-label'>Behavioral Score &middot; 60%</div></div>"
            f"<div class='metric-item'>"
            f"<div class='metric-value' style='color:{color}'>"
            f"{(0.4*ml_proba[1]+0.6*threat_score)*100:.1f}%</div>"
            f"<div class='metric-label'>Combined Threat</div></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # Threat breakdown
        if threat_details:
            with st.expander("Behavioral analysis details"):
                import pandas as pd
                rows = [{"Indicator": k.replace("_", " ").title(),
                         "Impact": f"{v:+.0%}"}
                        for k, v in sorted(threat_details.items(), key=lambda x: -abs(x[1]))]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # File statistics
        with st.expander("View file statistics"):
            import pandas as pd
            stat_data = {
                "Metric": [
                    "Total instructions", "Unique opcodes", "CFG blocks",
                    "API calls found", "Unique APIs",
                    "MOV ratio", "TEST ratio", "PUSH ratio", "POP ratio",
                    "CALL ratio", "XOR ratio",
                ],
                "Value": [
                    stats["total_instructions"], len(stats["opcode_counter"]),
                    stats["num_blocks"],
                    sum(stats["api_counter"].values()), len(stats["api_counter"]),
                    f"{stats['mov_count']/max(stats['total_instructions'],1):.3f}",
                    f"{stats['test_count']/max(stats['total_instructions'],1):.3f}",
                    f"{stats['push_count']/max(stats['total_instructions'],1):.3f}",
                    f"{stats['pop_count']/max(stats['total_instructions'],1):.3f}",
                    f"{stats['call_count']/max(stats['total_instructions'],1):.3f}",
                    f"{stats['xor_count']/max(stats['total_instructions'],1):.3f}",
                ],
            }
            st.dataframe(pd.DataFrame(stat_data), use_container_width=True, hide_index=True)

    elif uploaded is None and model_ready:
        st.markdown(
            "<div class='glass-card' style='text-align:center;padding:3rem'>"
            "<div style='font-size:3rem;margin-bottom:0.6rem' class='pulse-text'>&#128194;</div>"
            "<div style='color:#94a3b8;font-size:1.05rem'>"
            "Upload a <b>.asm</b> or <b>.exe</b> file above to begin analysis</div>"
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
