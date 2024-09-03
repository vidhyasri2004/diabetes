import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title for the Streamlit app
st.title("Diabetes Prediction App")

# Input features for the model
st.header("Input Features")

# Example input fields for diabetes prediction
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0.0, max_value=900.0, value=30.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=10, max_value=100, value=30)

# Function to make predictions
def predict_diabetes(features):
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)  # Standardize the features using the scaler
    return model.predict(features)[0]

# Button to make predictions
if st.button("Predict"):
    features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    prediction = predict_diabetes(features)
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    st.write(f"Prediction: {result}")