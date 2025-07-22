import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Load saved model
MODEL_PATH = Path("best_model.joblib")

if not MODEL_PATH.exists():
    sys.exit("❌ best_model.joblib not found. Run the training pipeline first.")

model_bundle = joblib.load(MODEL_PATH)
model  = model_bundle["model"]
scaler = model_bundle["scaler"]

# Replace these mean values if your training data used different ones
mean_values = {
    "Glucose": 120.89,
    "BloodPressure": 69.11,
    "SkinThickness": 20.54,
    "Insulin": 79.80,
    "BMI": 31.99
}

# --- User Input ---
print("🔍  Enter patient values to predict diabetes risk:")
try:
    pregnancies     = float(input("Pregnancies: "))
    glucose         = float(input("Glucose: "))
    blood_pressure  = float(input("Blood Pressure: "))
    skin_thickness  = float(input("Skin Thickness: "))
    insulin         = float(input("Insulin: "))
    bmi             = float(input("BMI: "))
    dpf             = float(input("Diabetes Pedigree Function: "))
    age             = float(input("Age: "))
except ValueError:
    sys.exit("❌  Invalid entry — please provide numeric values only.")

# --- Replace 0s with training-time means (used during imputation) ---
glucose        = glucose if glucose != 0 else mean_values["Glucose"]
blood_pressure = blood_pressure if blood_pressure != 0 else mean_values["BloodPressure"]
skin_thickness = skin_thickness if skin_thickness != 0 else mean_values["SkinThickness"]
insulin        = insulin if insulin != 0 else mean_values["Insulin"]
bmi            = bmi if bmi != 0 else mean_values["BMI"]

# --- Prepare single-row DataFrame ---
input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                          columns=[
                              "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                              "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
                          ])

# --- Apply saved scaler ---
X_scaled = scaler.transform(input_data)

# --- Predict ---
prob_diabetic = model.predict_proba(X_scaled)[0][1]
prediction    = model.predict(X_scaled)[0]

# --- Display Result ---
if prediction == 1:
    print(f"\n🧠  Person is **Diabetic** with a {prob_diabetic * 100:.2f}% probability.")
else:
    print(f"\n🧠  Person is **Non‑Diabetic** with a {(1 - prob_diabetic) * 100:.2f}% probability.")
