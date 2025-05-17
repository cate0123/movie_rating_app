from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model/movie_rating_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])
    return render_template('index.html', prediction_text=f'Predicted Rating: {prediction[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
