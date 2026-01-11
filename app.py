from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)


with open('best_crop_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


MODEL_NEEDS_SCALING = ['Logistic Regression', 'SVM', 'KNN']
model_name = type(model).__name__


needs_scaling = any(name.replace(' ', '').lower() in model_name.lower() 
                   for name in ['logisticregression', 'svc', 'kneighbors'])

@app.route('/')
def home():
    return render_template('index.html', crops=label_encoder.classes_.tolist())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
       
        features = np.array([[nitrogen, phosphorus, potassium, temperature, 
                            humidity, ph, rainfall]])
        
        
        if needs_scaling:
            features = scaler.transform(features)
        
        
        prediction = model.predict(features)
        predicted_crop = label_encoder.inverse_transform(prediction)[0]
        
        
        probability = None
        if hasattr(model, 'predict_proba'):
            if needs_scaling:
                prob = model.predict_proba(features)
            else:
                prob = model.predict_proba(features)
            probability = round(float(np.max(prob) * 100), 2)
        
        return render_template('result.html', 
                             crop=predicted_crop,
                             probability=probability,
                             nitrogen=nitrogen,
                             phosphorus=phosphorus,
                             potassium=potassium,
                             temperature=temperature,
                             humidity=humidity,
                             ph=ph,
                             rainfall=rainfall)
    
    except Exception as e:
        return render_template('result.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        
        nitrogen = float(data['nitrogen'])
        phosphorus = float(data['phosphorus'])
        potassium = float(data['potassium'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        
       
        features = np.array([[nitrogen, phosphorus, potassium, temperature, 
                            humidity, ph, rainfall]])
        
        
        if needs_scaling:
            features = scaler.transform(features)
        
        
        prediction = model.predict(features)
        predicted_crop = label_encoder.inverse_transform(prediction)[0]
        
        
        probability = None
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(features)
            probability = round(float(np.max(prob) * 100), 2)
        
        return jsonify({
            'success': True,
            'predicted_crop': predicted_crop,
            'confidence': probability,
            'input_features': {
                'nitrogen': nitrogen,
                'phosphorus': phosphorus,
                'potassium': potassium,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
