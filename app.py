import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Load model and scaler
MODEL_PATH = Path("best_model.joblib")

if not MODEL_PATH.exists():
    st.error("❌ Model file not found. Please train the model and export it as 'best_model.joblib'.")
    st.stop()

model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
scaler = model_bundle["scaler"]

# Default mean values to replace zeros (adjust based on your training data if needed)
mean_values = {
    "Glucose": 120.89,
    "BloodPressure": 69.1,
    "SkinThickness": 20.5,
    "Insulin": 79.8,
    "BMI": 31.99
}

st.title("🧠 Diabetes Risk Prediction App")
st.markdown("Enter the patient's medical information to predict diabetes risk.")

# Input form
with st.form("diabetes_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose", min_value=0.0, step=1.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, step=1.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, step=1.0)
    insulin = st.number_input("Insulin", min_value=0.0, step=1.0)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
    age = st.number_input("Age", min_value=0, step=1)
    
    submitted = st.form_submit_button("Predict")

# Preprocess and predict
if submitted:
    # Replace zero-values with training-time means
    glucose = glucose if glucose != 0 else mean_values["Glucose"]
    blood_pressure = blood_pressure if blood_pressure != 0 else mean_values["BloodPressure"]
    skin_thickness = skin_thickness if skin_thickness != 0 else mean_values["SkinThickness"]
    insulin = insulin if insulin != 0 else mean_values["Insulin"]
    bmi = bmi if bmi != 0 else mean_values["BMI"]

    input_data = np.array([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk: The person is **Diabetic** with a probability of {probability*100:.2f}%.")
    else:
        st.success(f"✅ Low Risk: The person is **Non-Diabetic** with a probability of {(1 - probability)*100:.2f}%.")

