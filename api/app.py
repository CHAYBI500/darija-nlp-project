from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("../models/sentiment_model.pkl")

@app.route("/")
def home():
    return {"message": "Darija Sentiment API running"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text")

    pred = model.predict([text])[0]

    return jsonify({
        "text": text,
        "sentiment": str(pred)
    })

if __name__ == "__main__":
    app.run(debug=True)