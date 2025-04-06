from flask import Flask, request, jsonify
import joblib
import numpy as np
from pyngrok import ngrok
import threading
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load your scikit-learn Random Forest model
model = joblib.load('/content/dog_breeding_rf_model.pkl')
breed_encoder = joblib.load('/content/breed_encoder.pkl')
temp_encoder = joblib.load('/content/temp_encoder.pkl')

# Define an endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debugging
        #input_data = data.get("input", {}) 
        # Access directly from the root
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
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


# Function to run Flask app in the background
def run_flask():
    app.run(host='0.0.0.0', port=5017)

if __name__ == '__main__':
    # Terminate any existing ngrok tunnels
    ngrok.kill()

    # Start ngrok to expose the Flask app
    public_url = ngrok.connect(5017).public_url
    print(f"Public URL: {public_url}")

    # Start Flask in a background thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True  # Daemonize so it stops when the main program exits
    flask_thread.start()

    print("Flask app is running in the background. You can now run other cells.")