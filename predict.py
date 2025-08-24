import joblib
import json
import argparse
import numpy as np

def predict_disease_risk(pH, turbidity, tds):
    """Predict disease risk from water quality parameters"""
    
    try:
        model = joblib.load('water_model.pkl')
        with open('label_to_advice.json', 'r') as f:
            label_to_advice = json.load(f)
    except FileNotFoundError:
        print("Error: Model files not found. Please run train_model.py first.")
        return
    
    # Prepare input features (no contam_prob anymore)
    features = np.array([[pH, turbidity, tds]])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    advice = label_to_advice.get(prediction, "No advice available.")
    
    return prediction, advice, probability

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict disease risk from water quality')
    parser.add_argument('--pH', type=float, required=True, help='pH value (5.0-9.5)')
    parser.add_argument('--turbidity', type=float, required=True, help='Turbidity in NTU (0-30)')
    parser.add_argument('--tds', type=float, required=True, help='TDS in mg/L (100-2000)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not (5.0 <= args.pH <= 9.5):
        print("Warning: pH value should be between 5.0 and 9.5")
    
    if not (0 <= args.turbidity <= 30):
        print("Warning: Turbidity should be between 0 and 30 NTU")
    
    if not (100 <= args.tds <= 2000):
        print("Warning: TDS should be between 100 and 2000 mg/L")
    
    prediction, advice, probabilities = predict_disease_risk(
        args.pH, args.turbidity, args.tds
    )
    
    if prediction:
        print(f"\nPredicted Disease Risk: {prediction}")
        print(f"Advice: {advice}")
        
        print(f"\nConfidence Scores:")
        model = joblib.load('water_model.pkl')
        for label, prob in zip(model.classes_, probabilities):
            print(f"  {label}: {prob:.3f}")
