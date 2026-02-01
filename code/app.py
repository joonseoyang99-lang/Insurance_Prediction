import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("best_model.pkl")
model_spec = joblib.load("model_spec.pkl")
feature_order = joblib.load("feature_order.pkl")

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
    age_group = make_age_group(age)
    bmi_category = make_bmi_category(bmi)

    input_data = pd.DataFrame({
        "Id": [0],
        "claim": [0.0],
        "age": [age],
        "bmi": [bmi],
        "gender": [gender],
        "diabetic": [diabetic],
        "smoker": [smoker],
        "region": [region],
        "bloodpressure": [bloodpressure],
        "children": [children],
        "age_group": [age_group],
        "bmi_category": [bmi_category],
    })

    for c in ["gender", "region", "bmi_category"]:
        input_data[c] = input_data[c].str.lower()

    training_categories = {
        "gender": ["female", "male"],
        "diabetic": ["No", "Yes"],
        "smoker": ["No", "Yes"],
        "region": ["northeast", "northwest", "southeast", "southwest"],
        "age_group": ["Youth", "Young Adults", "Middle Age", "Senior"],
        "bmi_category": ["underweight", "normal", "overweight", "obese"]
    }

    for col, cats in training_categories.items():
        input_data[col] = pd.Categorical(input_data[col], categories=cats)

    X_input_df = model_spec.transform(input_data)
    X_input_df = X_input_df.reindex(columns=feature_order, fill_value=0)


    X_in = X_input_df.to_numpy(dtype=np.float32)
    prediction = float(model.predict(X_in)[0])
    st.success(f"**Estimated Insurance Payment Amount:** ${prediction:,.2f}")
