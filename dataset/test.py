# test.py — use only the features the scaler/model were trained on (short-term fix)
from joblib import load
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")
CSV_PATH = os.path.join(BASE_DIR, "cleaned_for_model.csv")
OUT_PATH = os.path.join(BASE_DIR, "predictions.csv")
USED_FEATURES_PATH = os.path.join(BASE_DIR, "used_features.txt")

# --- load artifacts (fail loudly with clear message if not found) ---
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: model not found at {MODEL_PATH}", file=sys.stderr); sys.exit(1)
if not os.path.exists(SCALER_PATH):
    print(f"ERROR: scaler not found at {SCALER_PATH}", file=sys.stderr); sys.exit(1)
if not os.path.exists(CSV_PATH):
    print(f"ERROR: cleaned CSV not found at {CSV_PATH}", file=sys.stderr); sys.exit(1)

model = load(MODEL_PATH)
scaler = load(SCALER_PATH)

# --- determine the exact feature names the scaler/model were fitted on ---
if hasattr(scaler, "feature_names_in_"):
    expected_features = list(scaler.feature_names_in_)
elif hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)
else:
    # fallback: read features.txt if available (but user said scaler has 63)
    ft = os.path.join(BASE_DIR, "features.txt")
    if os.path.exists(ft):
        with open(ft) as f:
            expected_features = [line.strip() for line in f]
    else:
        print("ERROR: cannot determine expected features (no feature_names_in_ and no features.txt).", file=sys.stderr)
        sys.exit(1)

print(f"Using {len(expected_features)} expected features (source: scaler/model).")

# --- load CSV ---
df = pd.read_csv(CSV_PATH)
print("Loaded CSV columns:", len(df.columns))

# If the CSV contains label, remove it for prediction
if "Label" in df.columns:
    df_features = df.drop(columns=["Label"])
else:
    df_features = df.copy()

# --- convert any matching columns to numeric (coerce errors) to avoid dtype issues ---
for c in list(df_features.columns):
    # Only convert columns that will be used or are numeric-like; safe to convert all to numeric then fill
    df_features[c] = pd.to_numeric(df_features[c], errors="coerce")

# --- find missing / extra relative to the expected features ---
missing = [c for c in expected_features if c not in df_features.columns]
extra = [c for c in df_features.columns if c not in expected_features]

print(f"Missing features (will be filled with 0): {len(missing)} -> {missing}")
print(f"Extra columns (will be dropped): {len(extra)} -> {extra}")

# --- create the X matrix that the scaler expects ---
# Fill any missing expected features with zeros
for c in missing:
    df_features[c] = 0

# Drop extras
if extra:
    df_features = df_features.drop(columns=extra)

# Reorder columns exactly to expected_features
X = df_features[expected_features]

# Ensure numeric and fill NaNs (coercion above could have produced NaNs)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

print("Final X shape (rows, cols):", X.shape)

# --- scale and predict ---
try:
    X_scaled = scaler.transform(X)
except Exception as e:
    print("ERROR while transforming with scaler:", e, file=sys.stderr)
    sys.exit(1)

try:
    preds = model.predict(X_scaled)
except Exception as e:
    print("ERROR while predicting with model:", e, file=sys.stderr)
    sys.exit(1)

# --- save predictions alongside original CSV (keeps everything) ---
out = df.copy()
out["Prediction"] = preds
out.to_csv(OUT_PATH, index=False)
print(f"✅ Predictions saved to: {OUT_PATH}")

# --- also save the exact used feature list for traceability ---
with open(USED_FEATURES_PATH, "w") as f:
    for feat in expected_features:
        f.write(feat + "\n")
print(f"Saved {len(expected_features)} used features to {USED_FEATURES_PATH}")

