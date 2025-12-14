import altair as alt
import numpy as np
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Wellness Tourism Predictor")

st.title("🌿 Wellness Tourism Package Prediction")

# Load model
model = joblib.load("model.joblib")

st.subheader("Customer Details")

age = st.number_input("Age", 18, 100, 35)
income = st.number_input("Monthly Income", 1000, 200000, 50000)
duration = st.slider("Trip Duration (days)", 1, 30, 5)
family = st.slider("Family Size", 1, 10, 2)

designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "VP"])
city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

data = {
    "Age": age,
    "MonthlyIncome": income,
    "Duration": duration,
    "FamilySize": family,
    "CityTier": city_tier,
    "Designation_Executive": 0,
    "Designation_Manager": 0,
    "Designation_Senior Manager": 0,
    "Designation_VP": 0,
}

data[f"Designation_{designation}"] = 1
df = pd.DataFrame([data])

if st.button("Predict"):
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    if pred == 1:
        st.success("✅ Customer WILL purchase the package")
    else:
        st.error("❌ Customer will NOT purchase the package")

    st.metric("Purchase Probability", f"{prob:.2%}")
