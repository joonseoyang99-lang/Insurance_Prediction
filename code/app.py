import streamlit as st
import numpy as np
import joblib
from pathlib import Path

DIR = Path(__file__).parent

model = joblib.load(DIR / "best_model.pkl")
feature_order = joblib.load(DIR / "feature_order.pkl")

def make_age_group(age: int) -> str:
    if age < 25:
        return "Youth"
    elif age < 45:
        return "Young Adults"
    elif age < 60:
        return "Middle Age"
    else:
        return "Senior"

def make_bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal"
    elif bmi < 30:
        return "overweight"
    else:
        return "obese"

def build_input(age, bmi, children, bloodpressure, gender, diabetic, smoker, region):
    raw = {
        "gender": gender,
        "bloodpressure": float(bloodpressure),
        "diabetic": diabetic,
        "children": float(children),
        "smoker": smoker,
        "region": region,
        "age_group": make_age_group(age),
        "bmi_category": make_bmi_category(bmi),
    }
    row = {}
    for feat in feature_order:
        if "[" in feat:
            col, val = feat.rstrip("]").split("[")
            row[feat] = 1.0 if str(raw[col]) == val else 0.0
        else:
            row[feat] = raw[feat]
    return np.array([list(row.values())], dtype=np.float32)

st.set_page_config(page_title="Insurance Claim Predictor", layout="centered")
st.title("Health Insurance Payment Prediction App")
st.write("Enter the details below to estimate your insurance payment")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        children = st.number_input("Number of Children", min_value=0, max_value=8, value=0)

    with col2:
        bloodpressure = st.number_input("Blood Pressure", min_value=60, max_value=200, value=120)
        gender = st.selectbox("Gender", options=["female", "male"])
        diabetic = st.selectbox("Diabetic", options=["No", "Yes"])
        smoker = st.selectbox("Smoker", options=["No", "Yes"])
        region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

    submitted = st.form_submit_button("Predict Payment")

if submitted:
    X = build_input(age, bmi, children, bloodpressure, gender, diabetic, smoker, region)
    prediction = float(model.predict(X)[0])
    st.success(f"**Estimated Insurance Payment Amount:** ${prediction:,.2f}")
