import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# Load the trained model and scaler
model = tf.keras.models.load_model('ann_model.h5')
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction")
st.write("""
This app predicts whether a customer is likely to churn based on their details.
""")

# User input form
with st.form("Churn Prediction Form"):
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=10, step=1)
    balance = st.number_input("Balance", min_value=0.0, step=0.01, format="%.2f")
    num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, step=1)
    has_credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=0.01, format="%.2f")
    submitted = st.form_submit_button("Predict")

# Processing input and making prediction
if submitted:
    # Encode the inputs
    geography_mapping = {"France": [1, 0, 0], "Spain": [0, 1, 0], "Germany": [0, 0, 1]}
    gender_mapping = {"Male": 1, "Female": 0}
    credit_card_mapping = {"Yes": 1, "No": 0}
    active_member_mapping = {"Yes": 1, "No": 0}

    geography_encoded = geography_mapping[geography]
    gender_encoded = gender_mapping[gender]
    has_credit_card_encoded = credit_card_mapping[has_credit_card]
    is_active_member_encoded = active_member_mapping[is_active_member]

    # Combine features
    features = geography_encoded + [
        credit_score,
        gender_encoded,
        age,
        tenure,
        balance,
        num_of_products,
        has_credit_card_encoded,
        is_active_member_encoded,
        estimated_salary,
    ]

    # Scale the features
    features_scaled = scaler.transform([features])

    # Make prediction
    prediction = model.predict(features_scaled)
    prediction = (prediction > 0.5).astype(int)

    # Display the result
    if prediction == 1:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")
