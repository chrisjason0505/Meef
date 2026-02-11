# 🛡️ MEEF Live App

🔗 Live App: https://malwareanalysis22.streamlit.app/

AI-powered malware detection using assembly code analysis.  
Dataset  and ML engine extracted from https://github.com/srioo10/Meef
Upload `.asm` or `.exe` files and get instant classification with confidence scoring.

---

## 🚀 Features

- Dual scoring system (ML + heuristic)
- RandomForest model trained on 4,045 real samples
- 38 ratio-based opcode distribution features
- Isotonic regression calibrated probabilities
- Behavioral API pattern detection
- Clean Streamlit UI

---

## 📊 Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 98.52% |
| Precision | 98.88% |
| Recall    | 99.20% |
| F1-Score  | 99.04% |
| ROC-AUC   | 99.88% |

---

## 🧠 Detection Approach

**ML Model (40%)**
- RandomForest classifier
- Ratio-based opcode features
- SMOTE balanced training
- Calibrated probabilities

**Heuristic Engine (60%)**
Detects risky API patterns:
- Process injection (`CreateRemoteThread`, `WriteProcessMemory`)
- Network activity (`InternetOpen`, `send`, `recv`)
- Crypto usage (`CryptEncrypt`, `CryptDecrypt`)
- Persistence (`CreateService`, `RegSetValueEx`)

Final Score =  
`0.4 × ML Probability + 0.6 × Heuristic Score`

---

## ⚙️ Quick Start

```bash
pip install -r requirements.txt
python train_model.py  # optional
streamlit run app.py

