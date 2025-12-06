import os
import json
import pandas as pd
from datetime import datetime

CATALOG = "data/catalog.csv"
IR_DIR  = "output/ir_results"

# Load catalog
df = pd.read_csv(CATALOG)

# Collect all JSON file paths
all_json = []
for root, _, files in os.walk(IR_DIR):
    for f in files:
        if f.endswith(".json"):
            p = os.path.join(root, f).replace("\\", "/")
            all_json.append(p)

# Get existing IR paths
existing = set(df["ir_path"].dropna().tolist())

# Detect missing JSONs
missing = [p for p in all_json if p not in existing]

print(f"Total JSON IR found: {len(all_json)}")
print(f"In catalog: {len(existing)}")
print(f"Missing entries: {len(missing)}")

rows = []
for p in missing:
    sha = os.path.splitext(os.path.basename(p))[0]   # filename without .json

    # ---- Get notes (behavior classification) ----
    behavior_notes = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            ir_data = json.load(f)
            behavior = ir_data.get("behavior", {})

            if behavior.get("uses_network"):
                behavior_notes.append("network")
            if behavior.get("uses_fileops"):
                behavior_notes.append("fileops")
            if behavior.get("uses_injection"):
                behavior_notes.append("injection")
            if behavior.get("uses_crypto"):
                behavior_notes.append("crypto")

    except Exception as e:
        print(f"Warning: Unable to read {p}: {e}")

    notes = ", ".join(behavior_notes) if behavior_notes else "none"

    rows.append({
        "sha256": sha,
        "label": "malicious",
        "source": "local",
        "first_seen": datetime.now().isoformat(),
        "local_path": p,
        "ir_path": p,
        "notes": notes
    })

# Append
if rows:
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(CATALOG, index=False)
    print(f"✅ Added {len(rows)} malicious entries.")
else:
    print("✅ Nothing to add.")

