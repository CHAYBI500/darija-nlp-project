import matplotlib
matplotlib.use("Agg")

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import confusion_matrix

# =====================
# LOAD DATA + MODEL
# =====================
df = pd.read_csv("data/processed/labeled_data.csv")
model = joblib.load("models/sentiment_model.pkl")

# =====================
# FEATURES / LABELS
# =====================
X = df["clean_text"].astype(str).to_numpy()
y = df["label"].to_numpy()

# =====================
# SPLIT
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================
# 1. CONFUSION MATRIX
# =====================
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Matrice de Confusion")
plt.xlabel("Prédit")
plt.ylabel("Réel")

plt.savefig("reports/confusion_matrix.png", dpi=300)
plt.close()

# =====================
# 2. CROSS VALIDATION
# =====================
scores = cross_val_score(model, X, y, cv=3, n_jobs=-1)

plt.figure()
plt.plot(range(1, len(scores) + 1), scores, marker="o")
plt.title("Cross Validation (3-fold)")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0.9, 1.0)
plt.grid()

plt.savefig("reports/cross_validation.png", dpi=300)
plt.close()

# =====================
# 3. LEARNING CURVE
# =====================
lc = learning_curve(
    model,
    X,
    y,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0]
)

train_sizes = lc[0]
train_scores = lc[1]
test_scores = lc[2]

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, label="Train")
plt.plot(train_sizes, test_mean, label="Validation")
plt.title("Learning Curve")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.savefig("reports/learning_curve.png", dpi=300)
plt.close()

print("✅ Evaluation terminée - images sauvegardées dans /reports")