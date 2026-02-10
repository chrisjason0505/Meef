#!/usr/bin/env python3
"""
train_model.py - Malware Detection Training Pipeline
=====================================================
Trains on real features from data/features_ml.csv. Converts raw
count features into scale-invariant ratios so the model works
correctly on files of any size (from tiny snippets to large binaries).

Outputs (saved to data/models/):
    malware_classifier.pkl  - trained CalibratedClassifierCV (RandomForest)
    feature_scaler.pkl      - fitted StandardScaler
    model_metadata.json     - feature names, metrics, model info
"""

import json
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE

# -----------------------------------------------------------
#  Paths & constants
# -----------------------------------------------------------
FEATURES_PATH = Path("data/features_ml.csv")
MODEL_DIR = Path("data/models")
MODEL_PATH = MODEL_DIR / "malware_classifier.pkl"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

EXCLUDE_COLS = ["sha256", "label", "label_binary"]
RANDOM_STATE = 42
TEST_SIZE = 0.20


# -----------------------------------------------------------
#  Feature engineering: create ratio-based features
# -----------------------------------------------------------
# The raw features include absolute counts (total_opcodes, opcode_mov_count,
# etc.) that depend on file size. When the Streamlit app receives a file
# that is much smaller or larger than training samples, these absolute
# counts throw off predictions. Solution: derive ratio-based features
# that capture the *distribution* of opcodes rather than absolute counts.
#
# We keep:
#   - 7 binary behavioural flags (already scale-invariant)
#   - cfg_branch_density, call_ratio, jmp_ratio, api_to_opcode_ratio
#     (already ratios)
#   - cyclomatic_complexity (structural, not count-dependent)
#   - num_unique_opcodes, num_unique_apis (cardinality, naturally bounded)
#
# We ADD ratio versions of each opcode count:
#   - opcode_mov_ratio = opcode_mov_count / total_opcodes
#   - etc.
# And ratio versions of API counts:
#   - top_api_1_ratio = top_api_1_count / max(total_api_calls, 1)
# -----------------------------------------------------------

OPCODE_COUNT_COLS = [
    "opcode_call_count", "opcode_mov_count", "opcode_push_count",
    "opcode_pop_count", "opcode_jmp_count", "opcode_ret_count",
    "opcode_add_count", "opcode_sub_count", "opcode_xor_count",
    "opcode_test_count",
]

TOP_API_COLS = [f"top_api_{i}_count" for i in range(1, 11)]


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Convert raw features into scale-invariant ratio features."""
    out = pd.DataFrame(index=df.index)

    # 1. Binary flags (keep as-is)
    for col in ["uses_network", "uses_fileops", "uses_registry",
                "uses_memory", "uses_injection", "uses_crypto", "uses_persist"]:
        out[col] = df[col]

    # 2. Existing ratios (keep as-is)
    out["cfg_branch_density"] = df["cfg_branch_density"]
    out["cfg_cyclomatic_complexity"] = df["cfg_cyclomatic_complexity"]
    out["call_ratio"] = df["call_ratio"]
    out["jmp_ratio"] = df["jmp_ratio"]
    out["api_to_opcode_ratio"] = df["api_to_opcode_ratio"]

    # 3. Cardinality features (keep as-is, naturally bounded)
    out["num_unique_opcodes"] = df["num_unique_opcodes"]
    out["num_unique_apis"] = df["num_unique_apis"]

    # 4. Log-scale total counts (compresses the huge range into
    #    something manageable while preserving ordering)
    out["log_total_opcodes"] = np.log1p(df["total_opcodes"])
    out["log_total_api_calls"] = np.log1p(df["total_api_calls"])
    out["log_cfg_num_blocks"] = np.log1p(df["cfg_num_blocks"])
    out["log_cfg_num_edges"] = np.log1p(df["cfg_num_edges"])

    # 5. Opcode distribution ratios
    total_ops = df["total_opcodes"].replace(0, 1)
    for col in OPCODE_COUNT_COLS:
        ratio_name = col.replace("_count", "_ratio")
        out[ratio_name] = df[col] / total_ops

    # 6. API distribution ratios
    total_apis = df["total_api_calls"].replace(0, 1)
    for col in TOP_API_COLS:
        ratio_name = col.replace("_count", "_ratio")
        out[ratio_name] = df[col] / total_apis

    # Replace NaN/inf
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return out, list(out.columns)


# -----------------------------------------------------------
#  Helpers
# -----------------------------------------------------------
def print_distribution(y, title):
    total = len(y)
    n_benign = int(np.sum(y == 0))
    n_malicious = int(np.sum(y == 1))
    print(f"\n  {'-'*50}")
    print(f"  {title}")
    print(f"  {'-'*50}")
    print(f"    Benign     (class 0): {n_benign:>6}  ({n_benign/total*100:5.1f}%)")
    print(f"    Malicious  (class 1): {n_malicious:>6}  ({n_malicious/total*100:5.1f}%)")
    print(f"    Total               : {total:>6}")


def find_optimal_threshold(y_true, y_proba):
    """Find threshold that maximizes F1-score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5


# -----------------------------------------------------------
#  Training pipeline
# -----------------------------------------------------------
def train():
    print("=" * 60)
    print("  MEEF MALWARE DETECTION - TRAINING PIPELINE")
    print("  (ratio-based features for scale-invariant predictions)")
    print("=" * 60)

    # 1. Load dataset
    if not FEATURES_PATH.exists():
        print(f"\n[ERROR] Feature file not found: {FEATURES_PATH}")
        sys.exit(1)

    df = pd.read_csv(FEATURES_PATH)
    print(f"\n[OK] Loaded {len(df)} samples from {FEATURES_PATH}")

    # 2. Separate labels
    y = df["label_binary"].values

    # 3. Engineer ratio-based features
    print("\n[>>] Engineering ratio-based features ...")
    raw_feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X_df, feature_cols = engineer_features(df[raw_feature_cols])
    X = X_df.values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[OK] {len(feature_cols)} engineered features (from {len(raw_feature_cols)} raw)")
    print(f"     Features: {', '.join(feature_cols[:10])} ...")
    print_distribution(y, "Original class distribution")

    if len(np.unique(y)) < 2:
        print("\n[ERROR] Need both benign and malicious samples.")
        sys.exit(1)

    # 4. Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print_distribution(y_train, "Training set (before SMOTE)")
    print_distribution(y_test, "Test set (held out)")

    # 5. SMOTE
    print("\n[>>] Applying SMOTE oversampling on training set ...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print_distribution(y_train_res, "Training set (after SMOTE)")

    # 6. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # 7. Train
    print("\n[>>] Training RandomForestClassifier (n_estimators=200) ...")
    base_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    base_clf.fit(X_train_scaled, y_train_res)
    print(f"[OK] Base model trained with {base_clf.n_estimators} trees")

    # 8. Calibrate
    print("\n[>>] Calibrating probability outputs (isotonic) ...")
    clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=5)
    clf.fit(X_train_scaled, y_train_res)
    print("[OK] Calibrated classifier ready")

    # 9. Cross-validation
    print("\n[>>] Running 5-fold cross-validation ...")
    cv_scores = cross_val_score(base_clf, X_train_scaled, y_train_res, cv=5, scoring="accuracy")
    print(f"[OK] CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

    # 10. Evaluate
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc = 0.0
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*60}")
    print("  EVALUATION METRICS (test set, threshold=0.50)")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc*100:6.2f}%")
    print(f"  Precision : {prec*100:6.2f}%")
    print(f"  Recall    : {rec*100:6.2f}%")
    print(f"  F1-Score  : {f1*100:6.2f}%")
    print(f"  ROC-AUC   : {roc*100:6.2f}%")

    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Benign  Malicious")
    print(f"  Actual Benign    {cm[0][0]:>5}     {cm[0][1]:>5}")
    print(f"       Malicious   {cm[1][0]:>5}     {cm[1][1]:>5}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Benign", "Malicious"],
                                zero_division=0))

    # 11. Optimal threshold
    optimal_threshold = find_optimal_threshold(y_test, y_proba)
    y_pred_opt = (y_proba >= optimal_threshold).astype(int)
    f1_opt = f1_score(y_test, y_pred_opt, zero_division=0)
    acc_opt = accuracy_score(y_test, y_pred_opt)
    print(f"  Optimal threshold (max F1): {optimal_threshold:.4f}")
    print(f"  F1 at optimal threshold   : {f1_opt*100:.2f}%")
    print(f"  Accuracy at opt threshold : {acc_opt*100:.2f}%")

    # 12. Per-class probability stats
    benign_probs = y_proba[y_test == 0]
    malicious_probs = y_proba[y_test == 1]
    print(f"\n  P(malicious) distribution on test set:")
    print(f"    Benign samples:    median={np.median(benign_probs):.4f}, "
          f"mean={np.mean(benign_probs):.4f}, p95={np.percentile(benign_probs, 95):.4f}")
    print(f"    Malicious samples: median={np.median(malicious_probs):.4f}, "
          f"mean={np.mean(malicious_probs):.4f}, p5={np.percentile(malicious_probs, 5):.4f}")

    # 13. Feature importance
    importances = base_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\n  Top 15 Feature Importances:")
    for rank in range(min(15, len(indices))):
        idx = indices[rank]
        print(f"    {rank+1:>2}. {feature_cols[idx]:<35s} {importances[idx]:.4f}")

    # 14. Save artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, MODEL_PATH)
    print(f"\n[SAVED] model    -> {MODEL_PATH}")

    joblib.dump(scaler, SCALER_PATH)
    print(f"[SAVED] scaler   -> {SCALER_PATH}")

    metadata = {
        "feature_names": feature_cols,
        "num_features": len(feature_cols),
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc": float(roc),
        },
        "model_type": "CalibratedClassifierCV(RandomForestClassifier)",
        "n_estimators": base_clf.n_estimators,
        "imbalance_correction": "SMOTE + class_weight=balanced + calibration",
        "total_samples": len(df),
        "class_distribution": {
            "benign": int(np.sum(y == 0)),
            "malicious": int(np.sum(y == 1)),
        },
        "optimal_threshold": optimal_threshold,
        "feature_engineering": "ratio-based (scale-invariant)",
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] metadata -> {METADATA_PATH}")

    print("\n[OK] Training pipeline complete!")
    print(f"     Model ready for predictions.")


if __name__ == "__main__":
    train()
