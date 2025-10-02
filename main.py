# -*- coding: utf-8 -*-
import os
from flask import Flask, render_template, request, session, redirect, url_for
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fallback_secret_key_here')

# Load model và feature names
try:
    model = joblib.load("heart_disease_xgb_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
except:
    # Tạo model mẫu nếu không tìm thấy file
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=15, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    feature_names = [
        "male", "age", "education", "currentSmoker", "cigsPerDay", 
        "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
        "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
    ]

@app.route('/')
def index():
    # Reset session khi vào trang chủ
    session.clear()
    return redirect(url_for('page1'))

@app.route('/page1', methods=['GET', 'POST'])
def page1():
    if request.method == 'POST':
        # Lưu dữ liệu từ trang 1 vào session
        session['male'] = request.form.get('male')
        session['age'] = request.form.get('age')
        session['education'] = request.form.get('education')
        return redirect(url_for('page2'))
    
    return render_template('page1.html')

@app.route('/page2', methods=['GET', 'POST'])
def page2():
    if request.method == 'POST':
        # Lưu dữ liệu từ trang 2 vào session
        session['currentSmoker'] = request.form.get('currentSmoker', 'off')
        session['cigsPerDay'] = request.form.get('cigsPerDay', '0')
        session['BPMeds'] = request.form.get('BPMeds', 'off')
        session['prevalentStroke'] = request.form.get('prevalentStroke', 'off')
        session['prevalentHyp'] = request.form.get('prevalentHyp', 'off')
        session['diabetes'] = request.form.get('diabetes', 'off')
        session['totChol'] = request.form.get('totChol', '190')
        session['sysBP'] = request.form.get('sysBP', '120')
        session['diaBP'] = request.form.get('diaBP', '80')
        session['BMI'] = request.form.get('BMI', '24')
        session['heartRate'] = request.form.get('heartRate', '72')
        session['glucose'] = request.form.get('glucose', '90')
        
        return redirect(url_for('predict'))
    
    return render_template('page2.html')

@app.route('/predict')
def predict():
    try:
        # Chuẩn bị dữ liệu từ session
        input_data = {
            "male": 1 if session.get('male') == 'male' else 0,
            "age": float(session.get('age', 55)),
            "education": float(session.get('education', 2)),
            "currentSmoker": 1 if session.get('currentSmoker') == 'on' else 0,
            "cigsPerDay": float(session.get('cigsPerDay', 0)),
            "BPMeds": 1 if session.get('BPMeds') == 'on' else 0,
            "prevalentStroke": 1 if session.get('prevalentStroke') == 'on' else 0,
            "prevalentHyp": 1 if session.get('prevalentHyp') == 'on' else 0,
            "diabetes": 1 if session.get('diabetes') == 'on' else 0,
            "totChol": float(session.get('totChol', 190)),
            "sysBP": float(session.get('sysBP', 120)),
            "diaBP": float(session.get('diaBP', 80)),
            "BMI": float(session.get('BMI', 24)),
            "heartRate": float(session.get('heartRate', 72)),
            "glucose": float(session.get('glucose', 90))
        }
        
        # Build input row theo đúng thứ tự feature_names
        row = [input_data[col] for col in feature_names]
        input_df = pd.DataFrame([row], columns=feature_names)
        
        # Predict
        pred = model.predict(input_df)[0]
        proba = float(model.predict_proba(input_df)[0, 1])
        
        # Chuẩn bị kết quả
        result = {
            'prediction': 'high' if pred == 1 else 'low',
            'probability': proba,
            'message': 'Thành công'
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        return render_template('result.html', 
                              result={'error': True, 'message': f'Lỗi: {str(e)}'})
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)