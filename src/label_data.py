import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

df = pd.read_csv("data/processed/clean_data.csv")

analyzer = SentimentIntensityAnalyzer()

def get_label(text):
    score = analyzer.polarity_scores(str(text))["compound"]

    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["label"] = df["clean_text"].apply(get_label)

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/labeled_data.csv", index=False)

print("✅ Labels générés")
print(df["label"].value_counts())