from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the model, scaler, SelectKBest, and feature names
with open('model/model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('model/kbest.pkl', 'rb') as file:
    kbest = pickle.load(file)

with open('model/feature_names.pkl', 'rb') as file:
    selected_feature_names = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html', features=selected_feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data fitur yang dipilih oleh SelectKBest
    input_data = request.form
    
    # Proses input
    input_features = []
    for feature in selected_feature_names:
        value = input_data.get(feature, 0)  # Jika tidak ada nilai, setel ke 0
        input_features.append(float(value))
    input_features = np.array(input_features).reshape(1, -1)
    
    # Pra-pemrosesan input menggunakan scaler yang sesuai
    input_features_scaled = scaler.transform(input_features)
    
    # Lakukan prediksi
    prediction = model.predict(input_features_scaled)
    
    # Tentukan hasil prediksi sebagai teks
    prediction_text = 'Legitimate' if prediction == 1 else 'Phishing'
    
    return render_template('index.html', features=selected_feature_names, prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
