import sys
import numpy as np
import joblib
from pathlib import Path

# Load model
model_dir = Path('data/models')
model = joblib.load(model_dir / 'malware_classifier.pkl')
scaler = joblib.load(model_dir / 'feature_scaler.pkl')

print(f"Model type: {type(model)}")
print(f"Has feature_importances_: {hasattr(model, 'feature_importances_')}")

# Create two DIFFERENT random feature vectors (38 features)
np.random.seed(42)
vec1 = np.random.rand(1, 38)
vec2 = np.random.rand(1, 38) + 5

# Scale them
vec1_scaled = scaler.transform(vec1)
vec2_scaled = scaler.transform(vec2)

# Predict
prob1 = model.predict_proba(vec1_scaled)[0]
prob2 = model.predict_proba(vec2_scaled)[0]

print(f"\nVector 1 Probabilities: Benign={prob1[0]:.6f}, Malware={prob1[1]:.6f}")
print(f"Vector 2 Probabilities: Benign={prob2[0]:.6f}, Malware={prob2[1]:.6f}")

if np.allclose(prob1, prob2):
    print("\n❌ ISSUE: Model outputs identical probabilities!")
elif prob1[0] == 0.0 or prob1[0] == 1.0 or prob2[0] == 0.0 or prob2[0] == 1.0:
    print("\n❌ ISSUE: Model still outputs 0% or 100% probabilities!")
else:
    print("\n✅ Model outputs realistic probabilities")
