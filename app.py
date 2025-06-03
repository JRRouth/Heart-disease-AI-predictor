import streamlit as st
import pandas as pd
import joblib  # or pickle

# Load your saved model
model = joblib.load('heart_disease_model.pkl')

# Load feature columns saved earlier (optional)
feature_columns = joblib.load('feature_columns.pkl')

st.title("Heart Disease Prediction AI")

# Input fields (customize based on your features)
age = st.number_input('Age', min_value=1, max_value=120, value=50)
sex = st.selectbox('Sex', options=[0, 1])
# Add other features as needed...

# Collect inputs into a DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    # Add other features here...
})

# Preprocess input the same way you did in training
input_data = pd.get_dummies(input_data, drop_first=True)
input_data = input_data.reindex(columns=feature_columns, fill_value=0)

if st.button('Predict'):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    st.write(f"Prediction (1 = Disease, 0 = No Disease): {prediction}")
    st.write(f"Probability of Heart Disease: {probability:.2f}")
