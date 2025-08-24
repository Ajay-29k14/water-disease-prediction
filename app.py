from flask import Flask, request, jsonify
import joblib
import json
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("water_model.pkl")

# Load label-to-advice mapping
with open("label_to_advice.json", "r") as f:
    label_to_advice = json.load(f)

@app.route("/")
def home():
    return jsonify({"message": "Water Quality Disease Risk API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input from request
        data = request.get_json()

        # Extract features
        pH = float(data.get("pH", 7.0))
        turbidity = float(data.get("turbidity", 1.0))
        tds = float(data.get("tds", 300))

        # Make prediction
        features = np.array([[pH, turbidity, tds]])
        prediction = model.predict(features)[0]
        probs = model.predict_proba(features)[0]

        # Map probabilities to class names
        prob_dict = {cls: round(float(prob), 3) for cls, prob in zip(model.classes_, probs)}

        # Prepare response
        response = {
            "input": {"pH": pH, "turbidity": turbidity, "tds": tds},
            "predicted_risk": prediction,
            "advice": label_to_advice.get(prediction, "No advice available."),
            "confidence_scores": prob_dict
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
