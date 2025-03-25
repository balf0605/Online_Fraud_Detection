from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form values
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    prediction_label = encoder.inverse_transform(prediction)[0]

    return render_template('index.html', prediction_text=f'Prediction: {prediction_label}')

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form values
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    prediction_label = encoder.inverse_transform(prediction)[0]

    return render_template('index.html', prediction_text=f'Prediction: {prediction_label}')

if __name__ == "__main__":
    app.run(debug=True)
