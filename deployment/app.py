
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_model.pkl"
FEATURE_PATH = BASE_DIR / "feature_columns.json"
DATA_PATH = BASE_DIR.parent / "data" / "tourism.csv"


# --------------------------------------------------
# Load model and feature information
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH, "r") as f:
    feature_columns = json.load(f)

reference_data = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# Streamlit App
# --------------------------------------------------

st.title("Visit with Us - Wellness Tourism Package Prediction")

st.write(
    "Enter customer details to predict whether the customer "
    "is likely to purchase the Wellness Tourism Package."
)


# --------------------------------------------------
# User Inputs
# --------------------------------------------------

age = st.number_input(
    "Age",
    min_value=int(reference_data["Age"].min()),
    max_value=int(reference_data["Age"].max()),
    value=int(reference_data["Age"].median())
)

type_of_contact = st.selectbox(
    "Type of Contact",
    sorted(reference_data["TypeofContact"].dropna().unique())
)

city_tier = st.selectbox(
    "City Tier",
    sorted(reference_data["CityTier"].dropna().unique())
)

occupation = st.selectbox(
    "Occupation",
    sorted(reference_data["Occupation"].dropna().unique())
)

gender = st.selectbox(
    "Gender",
    sorted(reference_data["Gender"].dropna().unique())
)

number_of_person_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=int(reference_data["NumberOfPersonVisiting"].min()),
    max_value=int(reference_data["NumberOfPersonVisiting"].max()),
    value=int(reference_data["NumberOfPersonVisiting"].median())
)

preferred_property_star = st.selectbox(
    "Preferred Property Star",
    sorted(reference_data["PreferredPropertyStar"].dropna().unique())
)

marital_status = st.selectbox(
    "Marital Status",
    sorted(reference_data["MaritalStatus"].dropna().unique())
)

number_of_trips = st.number_input(
    "Number of Trips",
    min_value=int(reference_data["NumberOfTrips"].min()),
    max_value=int(reference_data["NumberOfTrips"].max()),
    value=int(reference_data["NumberOfTrips"].median())
)

passport = st.selectbox(
    "Passport",
    sorted(reference_data["Passport"].dropna().unique())
)

own_car = st.selectbox(
    "Own Car",
    sorted(reference_data["OwnCar"].dropna().unique())
)

number_of_children = st.number_input(
    "Number of Children Visiting",
    min_value=int(reference_data["NumberOfChildrenVisiting"].min()),
    max_value=int(reference_data["NumberOfChildrenVisiting"].max()),
    value=int(reference_data["NumberOfChildrenVisiting"].median())
)

designation = st.selectbox(
    "Designation",
    sorted(reference_data["Designation"].dropna().unique())
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=float(reference_data["MonthlyIncome"].min()),
    max_value=float(reference_data["MonthlyIncome"].max()),
    value=float(reference_data["MonthlyIncome"].median())
)

pitch_satisfaction = st.selectbox(
    "Pitch Satisfaction Score",
    sorted(reference_data["PitchSatisfactionScore"].dropna().unique())
)

product_pitched = st.selectbox(
    "Product Pitched",
    sorted(reference_data["ProductPitched"].dropna().unique())
)

number_of_followups = st.number_input(
    "Number of Followups",
    min_value=int(reference_data["NumberOfFollowups"].min()),
    max_value=int(reference_data["NumberOfFollowups"].max()),
    value=int(reference_data["NumberOfFollowups"].median())
)

duration_of_pitch = st.number_input(
    "Duration of Pitch",
    min_value=int(reference_data["DurationOfPitch"].min()),
    max_value=int(reference_data["DurationOfPitch"].max()),
    value=int(reference_data["DurationOfPitch"].median())
)


# --------------------------------------------------
# Create input dataframe
# --------------------------------------------------

input_data = pd.DataFrame({
    "Age": [age],
    "TypeofContact": [type_of_contact],
    "CityTier": [city_tier],
    "Occupation": [occupation],
    "Gender": [gender],
    "NumberOfPersonVisiting": [number_of_person_visiting],
    "PreferredPropertyStar": [preferred_property_star],
    "MaritalStatus": [marital_status],
    "NumberOfTrips": [number_of_trips],
    "Passport": [passport],
    "OwnCar": [own_car],
    "NumberOfChildrenVisiting": [number_of_children],
    "Designation": [designation],
    "MonthlyIncome": [monthly_income],
    "PitchSatisfactionScore": [pitch_satisfaction],
    "ProductPitched": [product_pitched],
    "NumberOfFollowups": [number_of_followups],
    "DurationOfPitch": [duration_of_pitch]
})


# --------------------------------------------------
# Apply the same encoding used during training
# --------------------------------------------------

categorical_columns = input_data.select_dtypes(
    include=["object"]
).columns.tolist()

input_data = pd.get_dummies(
    input_data,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)


# Ensure exactly the same features used by the trained model

input_data = input_data.reindex(
    columns=feature_columns,
    fill_value=0
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict"):

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success(
            "Prediction: Customer is likely to purchase the Wellness Tourism Package."
        )
    else:
        st.info(
            "Prediction: Customer is unlikely to purchase the Wellness Tourism Package."
        )
