import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

st.title("Wellness Tourism Package Purchase Prediction")

# ---- Load model ----
model_repo = "perusingh/wellness-tourism-model"
model_filename = "random_forest_model.pkl"

model_path = hf_hub_download(
    repo_id=model_repo,
    filename=model_filename,
    repo_type="model"
)
model = joblib.load(model_path)

# ---- Load train.csv to get expected columns ----
dataset_repo_id = "perusingh/wellness-tourism-data"
train_path = hf_hub_download(
    repo_id=dataset_repo_id,
    filename="train.csv",
    repo_type="dataset"
)
train_df = pd.read_csv(train_path)
expected_cols = [c for c in train_df.columns if c != "ProdTaken"]

st.write("### Enter customer details")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
typeofcontact = st.selectbox("TypeofContact", ["Company Invited", "Self Inquiry"])
citytier = st.selectbox("CityTier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business", "Other"])
gender = st.selectbox("Gender", ["Male", "Female"])
nopv = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=10, value=2)
pps = st.selectbox("PreferredPropertyStar", [3, 4, 5])
marital = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
notrips = st.number_input("NumberOfTrips", min_value=0, max_value=50, value=3)
passport = st.selectbox("Passport", [0, 1])
owncar = st.selectbox("OwnCar", [0, 1])
children = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "VP", "AVP", "Other"])
income = st.number_input("MonthlyIncome", min_value=0, value=50000)
pitch = st.slider("PitchSatisfactionScore", 1, 5, value=4)
productpitched = st.text_input("ProductPitched", value="Wellness Tourism Package")
followups = st.number_input("NumberOfFollowups", min_value=0, value=2)
duration = st.number_input("DurationOfPitch", min_value=1, value=10)

if st.button("Predict"):
    raw_input = pd.DataFrame([{
        "Age": age,
        "TypeofContact": typeofcontact,
        "CityTier": citytier,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": nopv,
        "PreferredPropertyStar": pps,
        "MaritalStatus": marital,
        "NumberOfTrips": notrips,
        "Passport": passport,
        "OwnCar": owncar,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": income,
        "PitchSatisfactionScore": pitch,
        "ProductPitched": productpitched,
        "NumberOfFollowups": followups,
        "DurationOfPitch": duration
    }])

    input_encoded = pd.get_dummies(raw_input, drop_first=True)
    input_aligned = input_encoded.reindex(columns=expected_cols, fill_value=0)

    pred = model.predict(input_aligned)[0]

    if pred == 1:
        st.success("Customer is LIKELY to purchase the Wellness Tourism Package.")
    else:
        st.error("Customer is NOT likely to purchase the package.")
