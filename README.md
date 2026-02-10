# MEEF Malware Sentinel 🛡️

AI-powered malware detection through assembly code analysis. Upload `.asm` or `.exe` files and get instant classification with confidence scoring.

## Features

- **Dual Scoring System**: Combines ML model predictions with rule-based behavioral heuristics
- **Ratio-Based ML Model**: RandomForest classifier trained on 4,045 real samples, using scale-invariant opcode distribution features
- **Behavioral Heuristic Engine**: Detects injection, network exfiltration, crypto, and persistence API patterns
- **Calibrated Probabilities**: Isotonic regression calibration ensures reliable confidence scores
- **Premium UI**: Glassmorphism design with real-time analysis feedback

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain the model
python train_model.py

# Launch the app
streamlit run app.py
```

## Project Structure

```
Meef/
├── app.py              # Streamlit web interface + feature extraction + classification
├── train_model.py      # Training pipeline (ratio-based features, SMOTE, calibration)
├── requirements.txt    # Python dependencies
├── data/
│   ├── features_ml.csv             # Training dataset (4,045 samples × 38 features)
│   └── models/
│       ├── malware_classifier.pkl  # Trained CalibratedClassifierCV model
│       ├── feature_scaler.pkl      # Fitted StandardScaler
│       └── model_metadata.json     # Feature names, metrics, threshold
└── README.md
```

## Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 98.52% |
| Precision | 98.88% |
| Recall    | 99.20% |
| F1-Score  | 99.04% |
| ROC-AUC   | 99.88% |

## Detection Method

The app uses a **dual scoring** approach:

1. **ML Model (40% weight)**: A calibrated Random Forest trained on 38 ratio-based features extracted from real malware/benign binaries. Features include opcode distribution ratios, log-scale structural metrics, and behavioral flags.

2. **Behavioral Heuristic (60% weight)**: Rule-based scoring that directly checks for dangerous API patterns:
   - Process injection (CreateRemoteThread, WriteProcessMemory)
   - Network exfiltration (InternetOpen, send, recv)
   - Cryptographic operations (CryptEncrypt, CryptDecrypt)
   - Persistence mechanisms (CreateService, RegSetValueEx)

## Deployment

### Streamlit Community Cloud (Free)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and deploy

### Docker
```bash
docker build -t meef-sentinel .
docker run -p 8501:8501 meef-sentinel
```

## Tech Stack

- Python 3.10+
- Streamlit
- scikit-learn (RandomForest + Calibration)
- imbalanced-learn (SMOTE)
- NumPy / Pandas
