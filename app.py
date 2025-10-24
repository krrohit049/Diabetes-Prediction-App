
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os
import pickle

# -----------------------------
# Step 1: Load & Train Model (Auto once)
# -----------------------------
if not os.path.exists("diabetes_model.pkl"):
    st.write("🔄 Training model for the first time...")

    df = pd.read_csv("diabetes.csv")  # Make sure this file is in same folder
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    with open("diabetes_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

else:
    model = pickle.load(open("diabetes_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))

# -----------------------------
# Step 2: Streamlit UI
# -----------------------------
st.title("🩺 Diabetes Prediction App")
st.write("Check your Diabetes Risk based on your health data.")

# Gender
gender = st.radio("Select Gender:", ["Male", "Female"])

# Body Measurements
st.subheader("🏋️ Body Measurements")
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
height = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0)

bmi = weight / ((height / 100) ** 2)
st.markdown(f"**Calculated BMI:** {bmi:.2f} (Normal: 18.5 - 24.9)")

# Family History
st.subheader("👨‍👩‍👧 Family History of Diabetes")
family_history = st.radio("Do you have a family history of diabetes?", ["No", "Yes"])
dpf = 1 if family_history == "Yes" else 0

# Other Inputs
st.subheader("🩸 Health Data Inputs")
if gender == "Female":
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
else:
    pregnancies = 0
    st.markdown("Pregnancies: **0 (Not applicable for males)**")

glucose = st.number_input("Glucose (Normal: 70-140)", min_value=50.0, max_value=300.0, value=105.0)
bp = st.number_input("Blood Pressure (Normal: 60-80)", min_value=40.0, max_value=150.0, value=70.0)
insulin = st.number_input("Insulin (Normal: 16-166)", min_value=0.0, max_value=900.0, value=91.0)
age = st.number_input("Age (Normal: 18-60)", min_value=10, max_value=100, value=30)

# SkinThickness dummy (required for model)
skin_thickness = 20  

# -----------------------------
# Step 3: Prediction
# -----------------------------
input_data = pd.DataFrame({
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [bp],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age]
})

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]
proba = model.predict_proba(input_scaled)[0][1] * 100

# -----------------------------
# Step 4: Display Result
# -----------------------------
st.subheader("📊 Prediction Result")
if prediction == 1:
    st.error(f"⚠️ High chance of Diabetes ({proba:.2f}% probability)")
else:
    st.success(f"✅ Low chance of Diabetes ({proba:.2f}% probability)")

# -----------------------------
# Step 5: Visualization
# -----------------------------
st.subheader("📈 Your Health Summary")
features = ["Glucose", "BloodPressure", "Insulin", "BMI", "Age"]
values = [glucose, bp, insulin, bmi, age]
normal_ranges = [140, 80, 166, 25, 60]

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(features, values, color="skyblue", label="Your Values")
ax.plot(features, normal_ranges, color="green", linestyle="--", marker="o", label="Normal Range")
ax.set_ylabel("Value")
ax.set_title("Comparison with Normal Health Range")
ax.legend()
st.pyplot(fig)

st.caption("⚠️ This app is for educational use only. Consult a doctor for medical advice.")
st.markdown(
    "<h5 style='text-align: center; color: darkblue;'>BY ROHIT KUMAR</h5>", 
    unsafe_allow_html=True
)

