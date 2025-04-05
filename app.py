# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open("xgboost_model_no_step.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'type_code': int(request.form['type_code']),
            'amount': float(request.form['amount']),
            'oldbalanceOrg': float(request.form['oldbalanceOrg']),
            'newbalanceOrig': float(request.form['newbalanceOrig']),
            'oldbalanceDest': float(request.form['oldbalanceDest']),
            'newbalanceDest': float(request.form['newbalanceDest'])
        }
        
        # Calculate derived features
        data['balance_diff_orig'] = data['oldbalanceOrg'] - data['newbalanceOrig']
        data['balance_diff_dest'] = data['oldbalanceDest'] - data['newbalanceDest']
        data['is_orig_balance_zero'] = int(data['oldbalanceOrg'] == 0)
        data['is_dest_balance_zero'] = int(data['newbalanceDest'] == 0)
        data['amount_to_orig_balance'] = data['amount'] / data['oldbalanceOrg'] if data['oldbalanceOrg'] > 0 else 0
        data['amount_to_dest_balance'] = data['amount'] / data['oldbalanceDest'] if data['oldbalanceDest'] > 0 else 0
        
        # Create DataFrame with the same structure as training data
        features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
                   'balance_diff_orig', 'balance_diff_dest', 'is_orig_balance_zero', 
                   'is_dest_balance_zero', 'amount_to_orig_balance', 'amount_to_dest_balance', 
                   'type_code']
        
        df = pd.DataFrame([data], columns=features)
        
        # Make prediction
        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])
        
        result = {
            'prediction': prediction,
            'probability': round(probability * 100, 2),
            'status': 'Fraudulent' if prediction == 1 else 'Legitimate'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/transaction-stats')
def transaction_stats():
    # This would typically come from your database
    # For demo purposes, we'll return mock data
    return jsonify({
        'total': 1000,
        'legitimate': 980,
        'fraudulent': 20,
        'fraud_by_type': {
            'TRANSFER': 10,
            'CASH_OUT': 8,
            'PAYMENT': 2
        },
        'recent_transactions': [
            {'time': '2025-04-04 10:15', 'amount': 423.45, 'type': 'TRANSFER', 'status': 'Legitimate'},
            {'time': '2025-04-04 09:23', 'amount': 5000.00, 'type': 'CASH_OUT', 'status': 'Fraudulent'},
            {'time': '2025-04-04 08:17', 'amount': 122.30, 'type': 'PAYMENT', 'status': 'Legitimate'}
        ]
    })

if __name__ == '__main__':
    app.run(debug=True)