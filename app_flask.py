from flask import Flask, request, render_template
import numpy as np
import pickle
import tensorflow as tf

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model('ann_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        int(request.form['credit_score']),
        request.form['geography'],
        request.form['gender'],
        int(request.form['age']),
        int(request.form['tenure']),
        float(request.form['balance']),
        int(request.form['num_of_products']),
        request.form['has_credit_card'],
        request.form['is_active_member'],
        float(request.form['estimated_salary'])
    ]
    
    # Encoding categorical inputs
    geography_mapping = {"France": [1, 0, 0], "Spain": [0, 1, 0], "Germany": [0, 0, 1]}
    gender_mapping = {"Male": 1, "Female": 0}
    has_credit_card_mapping = {"Yes": 1, "No": 0}
    is_active_member_mapping = {"Yes": 1, "No": 0}

    encoded_features = (
        geography_mapping[features[1]] +
        [features[0], gender_mapping[features[2]], features[3], features[4], features[5], features[6],
         has_credit_card_mapping[features[7]], is_active_member_mapping[features[8]], features[9]]
    )

    # Scale features
    scaled_features = scaler.transform([encoded_features])
    
    # Make prediction
    prediction = model.predict(scaled_features)
    result = (prediction > 0.5).astype(int)
    
    # Set prediction text and class
    if result[0][0] == 1:
        prediction_text = "The customer is likely to churn."
        prediction_class = "churn"
    else:
        prediction_text = "The customer is not likely to churn."
        prediction_class = "not-churn"

    return render_template('index.html', prediction_text=prediction_text, prediction_class=prediction_class)

if __name__ == "__main__":
    app.run(debug=True)
