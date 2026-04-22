# test_model_csv.py

import joblib
import pandas as pd

# =========================
# 1. Load Model + Scaler
# =========================
model_data = joblib.load("random-model\loan_pipeline.pkl")

model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]

print("✅ Model loaded successfully")

# =========================
# 2. Load Input CSV
# =========================
# Replace with your file name
input_file = "test_data.csv"

df = pd.read_csv(input_file)

print(f"📂 Loaded {len(df)} records")

# =========================
# 3. Align Features
# =========================
# Add missing columns
for col in features:
    if col not in df.columns:
        df[col] = 0

# Remove extra columns (optional but safer)
df = df[features]

# =========================
# 4. Apply Scaling
# =========================
input_scaled = scaler.transform(df)

# =========================
# 5. Predict
# =========================
predictions = model.predict(input_scaled)
probabilities = model.predict_proba(input_scaled)[:, 1]

# =========================
# 6. Save Results
# =========================
df["prediction"] = predictions
df["approval_probability"] = probabilities

output_file = "predictions.csv"
df.to_csv(output_file, index=False)

print(f"✅ Predictions saved to {output_file}")