import numpy as np
import pickle
import tensorflow as tf

# Load model and scaler
model = tf.keras.models.load_model('ann_model.h5')
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Preprocessing function
def preprocess_input(features):
    geography_mapping = {"France": [1, 0, 0], "Spain": [0, 1, 0], "Germany": [0, 0, 1]}
    gender_mapping = {"Male": 1, "Female": 0}
    credit_card_mapping = {"Yes": 1, "No": 0}
    active_member_mapping = {"Yes": 1, "No": 0}

    geography_encoded = geography_mapping[features["geography"]]
    gender_encoded = gender_mapping[features["gender"]]
    has_credit_card_encoded = credit_card_mapping[features["has_credit_card"]]
    is_active_member_encoded = active_member_mapping[features["is_active_member"]]

    input_features = geography_encoded + [
        features["credit_score"],
        gender_encoded,
        features["age"],
        features["tenure"],
        features["balance"],
        features["num_of_products"],
        has_credit_card_encoded,
        is_active_member_encoded,
        features["estimated_salary"],
    ]

    return scaler.transform([input_features])

# Test inputs
churn_features = {
    "credit_score": 450,
    "geography": "Germany",
    "gender": "Male",
    "age": 50,
    "tenure": 1,
    "balance": 100000.0,
    "num_of_products": 1,
    "has_credit_card": "Yes",
    "is_active_member": "No",
    "estimated_salary": 40000.0
}

not_churn_features = {
    "credit_score": 800,
    "geography": "France",
    "gender": "Female",
    "age": 35,
    "tenure": 8,
    "balance": 75000.0,
    "num_of_products": 3,
    "has_credit_card": "Yes",
    "is_active_member": "Yes",
    "estimated_salary": 120000.0
}

# Preprocess and predict
churn_scaled = preprocess_input(churn_features)
not_churn_scaled = preprocess_input(not_churn_features)

churn_prediction = model.predict(churn_scaled) > 0.5
not_churn_prediction = model.predict(not_churn_scaled) > 0.5

print("Churn Prediction:", "Churn" if churn_prediction else "Not Churn")
print("Not Churn Prediction:", "Churn" if not_churn_prediction else "Not Churn")
