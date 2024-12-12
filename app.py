
from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        return render_template('index.html', predicted_text=f'Diabetes Prediction: {"Positive" if prediction[0] == 1 else "Negative"}')
    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', predicted_text="Error in prediction")

if __name__ == '__main__':
    app.run(debug=True)

