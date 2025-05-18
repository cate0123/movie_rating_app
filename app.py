from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model
model = joblib.load('movie_rating_model()')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [data['feature1'], data['feature2'], ...]
    prediction = model.predict([features])[0]
    return jsonify({'predicted_rating': prediction})

if __name__ == '__main__':
    app.run(debug=True)
