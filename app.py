import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict the risk of heart disease.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", ["male", "female"])
cp = st.selectbox("Chest Pain Type", ['typical_angina', 'atypical_angina', 'non_anginal_pain', 'asymptomatic'])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.selectbox("Resting ECG", ['normal', 'stt_abnormality', 'left_ventricular_hypertrophy'])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=6.0, step=0.1, value=1.0)
slope = st.selectbox("Slope of ST Segment", ['upsloping', 'flat', 'downsloping'])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", ['normal', 'fixed_defect', 'reversible_defect'])

# Convert input to DataFrame
input_dict = {
    "age": age,
    "trestbps": trestbps,
    "chol": chol,
    "thalach": thalach,
    "oldpeak": oldpeak,
    "ca": ca,
    "sex_male": 1 if sex == "male" else 0,
    "cp_asymptomatic": 1 if cp == "asymptomatic" else 0,
    "cp_atypical_angina": 1 if cp == "atypical_angina" else 0,
    "cp_non_anginal_pain": 1 if cp == "non_anginal_pain" else 0,
    "fbs_Yes": 1 if fbs == "Yes" else 0,
    "restecg_left_ventricular_hypertrophy": 1 if restecg == "left_ventricular_hypertrophy" else 0,
    "restecg_stt_abnormality": 1 if restecg == "stt_abnormality" else 0,
    "exang_Yes": 1 if exang == "Yes" else 0,
    "slope_flat": 1 if slope == "flat" else 0,
    "slope_upsloping": 1 if slope == "upsloping" else 0,
    "thal_fixed_defect": 1 if thal == "fixed_defect" else 0,
    "thal_normal": 1 if thal == "normal" else 0,
}

input_df = pd.DataFrame([input_dict])

# Reindex columns to match training features
model_features = model.feature_names_in_
input_df = input_df.reindex(columns=model_features, fill_value=0)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease ({proba:.1f}% probability)")
    else:
        st.success(f"✅ Low risk of heart disease ({proba:.1f}% probability)")
