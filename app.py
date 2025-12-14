import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

st.title("Wellness Tourism Package Purchase Prediction")

model_repo = "perusingh/wellness-tourism-model"
model_filename = "random_forest_model.pkl"

model_path = hf_hub_download(
    repo_id=model_repo,
    filename=model_filename,
    repo_type="model"
)

model = joblib.load(model_path)

st.write("### Enter customer details")

age = st.number_input("Age", 18, 100)
income = st.number_input("Monthly Income", 0)
trips = st.number_input("Number of Trips", 0)
passport = st.selectbox("Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])
persons = st.number_input("Persons Visiting", 1)
children = st.number_input("Children Visiting", 0)
pitch = st.slider("Pitch Satisfaction", 1, 5)
followups = st.number_input("Followups", 0)
duration = st.number_input("Pitch Duration", 1)

if st.button("Predict"):
    df = pd.DataFrame({
        "Age": [age],
        "MonthlyIncome": [income],
        "NumberOfTrips": [trips],
        "Passport": [passport],
        "OwnCar": [own_car],
        "NumberOfPersonVisiting": [persons],
        "NumberOfChildrenVisiting": [children],
        "PitchSatisfactionScore": [pitch],
        "NumberOfFollowups": [followups],
        "DurationOfPitch": [duration]
    })

    pred = model.predict(df)[0]
    st.success("Likely to Purchase" if pred == 1 else "Not Likely to Purchase")

