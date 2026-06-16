import pandas as pd
import yaml
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import resample

# =========================
# CONFIG
# =========================
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/processed/labeled_data.csv")
df = df.dropna(subset=["clean_text", "label"])

print("\n📊 Distribution AVANT équilibrage:")
print(df["label"].value_counts())

# =========================
# BALANCING (OK FOR PROJECT BUT NOT PRODUCTION)
# =========================
dfs = []
max_size = df["label"].value_counts().max()

for label in df["label"].unique():
    df_label = df[df["label"] == label]

    df_label = resample(
        df_label,
        replace=True,
        n_samples=max_size,
        random_state=config["model"]["random_state"]
    )

    dfs.append(df_label)

df = pd.concat(dfs, ignore_index=True)
df = df.sample(frac=1, random_state=config["model"]["random_state"])

print("\n📊 Distribution APRÈS équilibrage:")
print(df["label"].value_counts())

# =========================
# SPLIT DATA
# =========================
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["model"]["test_size"],
    random_state=config["model"]["random_state"],
    stratify=y
)

# =========================
# MODEL
# =========================
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=config["tfidf"]["max_features"],
        ngram_range=eval(str(config["tfidf"]["ngram_range"]))
    )),
    ("clf", LinearSVC())
])

# =========================
# CROSS VALIDATION (PROPER POSITION)
# =========================
print("\n📊 CROSS VALIDATION (5-FOLD):")
scores = cross_val_score(model, X, y, cv=5)

print(scores)
print("Mean:", scores.mean())

# =========================
# TRAINING
# =========================
print("\n🚀 Training model...")
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

print("\n📊 ACCURACY:")
print(accuracy_score(y_test, y_pred))

print("\n📄 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

print("\n📉 CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sentiment_model.pkl")

print("\n✅ MODEL SAVED SUCCESSFULLY")