from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# Load the trained model
with open("xgboost_model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)  # ✅ FIXED

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON input from the Android app

    # Extract input features
    age = data.get("Age_at_Release")
    education = data.get("Education_Level")
    employment = data.get("Percent_Days_Employed")
    prior_arrests = data.get("Prior_Arrest_Episodes")
    drug_gun_charges = data.get("Prior_Arrest_Episodes_Drug_or_Gun_Charges")
    recidivism = data.get("Recidivism_Arrest_Last_3_Years")
    combined_offense = data.get("Combined_Offense")

    # Ensure valid input
    if None in [age, education, employment, prior_arrests, drug_gun_charges, recidivism, combined_offense]:
        return jsonify({"error": "Missing input values"}), 400

    # Convert input to numpy array for prediction
    input_features = np.array([age, education, employment, prior_arrests, drug_gun_charges, recidivism, 1, combined_offense]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_features)[0]

    return jsonify({"supervision_risk_score": float(prediction)})

# Run the Flask API
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=5000, debug=True)