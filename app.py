from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load your scikit-learn Random Forest model and encoders
model = joblib.load('dog_breeding_rf_model.pkl')
breed_encoder = joblib.load('breed_encoder.pkl')
temp_encoder = joblib.load('temp_encoder.pkl')

@app.route('/')
def home():
    return "Pet Breeding Compatibility API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        breed1 = data['breed1']
        breed2 = data['breed2']
        temp1 = data['temp1']
        temp2 = data['temp2']

        # Encode categorical features
        breed1_encoded = breed_encoder.transform([breed1])[0]
        breed2_encoded = breed_encoder.transform([breed2])[0]
        temp1_encoded = temp_encoder.transform([temp1])[0]
        temp2_encoded = temp_encoder.transform([temp2])[0]

        features = [
            breed1_encoded, data['age1'], data['weight1'],
            data['health1'], data['energy1'], temp1_encoded,
            breed2_encoded, data['age2'], data['weight2'],
            data['health2'], data['energy2'], temp2_encoded
        ]

        prediction = model.predict([features])[0]
        return jsonify({"compatibility_score": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
