# train_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

# -------------------------
# 1. Generate Synthetic Data
# -------------------------
np.random.seed(42)
N = 2000  # number of samples

pH = np.random.uniform(5.0, 9.5, N)
turbidity = np.random.uniform(0, 30, N)
tds = np.random.uniform(100, 2000, N)

disease_risks = []
for i in range(N):
    if 6.5 <= pH[i] <= 8.5 and turbidity[i] < 5 and 200 <= tds[i] <= 800:
        disease_risks.append("Safe")
    else:
        disease_risks.append(np.random.choice([
            "Diarrheal_Disease",
            "Enteric_Diseases_Typhoid_Cholera_HepA",
            "Chemical_Fluoride_Arsenic_Risk",
            "Bacterial_Risk_from_Poor_Disinfection",
            "Metal_Leaching_Risk"
        ]))

df = pd.DataFrame({
    "pH": pH,
    "turbidity": turbidity,
    "tds": tds,
    "disease_risk": disease_risks
})

# -------------------------
# 2. Train-Test Split
# -------------------------
X = df[["pH", "turbidity", "tds"]]
y = df["disease_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 3. Train RandomForest
# -------------------------
model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

# -------------------------
# 4. Save Model
# -------------------------
joblib.dump(model, "water_model.pkl")

# -------------------------
# 5. Save Advice Mapping
# -------------------------
label_to_advice = {
    "Safe": "Water is safe to drink. Maintain regular monitoring.",
    "Diarrheal_Disease": "Boil water before consumption; maintain proper sanitation.",
    "Enteric_Diseases_Typhoid_Cholera_HepA": "Chlorinate water; wash hands regularly.",
    "Chemical_Fluoride_Arsenic_Risk": "Use appropriate filters (RO/Activated Alumina).",
    "Bacterial_Risk_from_Poor_Disinfection": "Improve chlorination; avoid stagnant storage.",
    "Metal_Leaching_Risk": "Check pipelines and avoid corrosive plumbing."
}

with open("label_to_advice.json", "w") as f:
    json.dump(label_to_advice, f, indent=4)

print("✅ Model and advice mapping saved successfully!")
print("Class distribution:\n", df['disease_risk'].value_counts())
