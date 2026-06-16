import sys
import joblib

model = joblib.load("models/sentiment_model.pkl")

def predict(text):
    return model.predict([text])[0]

if __name__ == "__main__":
    text = " ".join(sys.argv[1:])
    result = predict(text)

    print("🧠 Sentiment:", result)