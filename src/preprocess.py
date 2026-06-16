import pandas as pd
import re
import os
import yaml

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def clean_text(text):
    text = str(text).lower()

    # remove urls
    text = re.sub(r"http\S+", "", text)

    # keep Arabic + Latin + numbers
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def load_data():
    df = pd.read_csv(config["data"]["path"])
    df = df.dropna()

    df["clean_text"] = df[config["data"]["text_column"]].apply(clean_text)

    return df

if __name__ == "__main__":
    df = load_data()

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/clean_data.csv", index=False)

    print("✅ Preprocessing terminé")
    print(df.head())