# Darija Sentiment Analysis

Projet de Deep Learning pour la classification automatique des sentiments en Darija marocaine.

## Objectif

Développer un système capable de détecter automatiquement le sentiment d'un texte rédigé en Darija :

- Positive
- Negative
- Neutral

## Dataset

Le dataset est constitué de commentaires YouTube en langue arabe dialectale marocaine (Darija).

Distribution initiale :

- Neutral : 129 082
- Positive : 1 445
- Negative : 324

Après équilibrage :

- Neutral : 129 082
- Positive : 129 082
- Negative : 129 082

Total : 387 246 échantillons

## Technologies utilisées

- Python
- Pandas
- Scikit-Learn
- TF-IDF
- Linear SVM (LinearSVC)
- Flask API
- Matplotlib
- Seaborn

## Résultats

### Cross Validation (5-Fold)

- Fold 1 : 99.79 %
- Fold 2 : 99.78 %
- Fold 3 : 99.79 %
- Fold 4 : 99.79 %
- Fold 5 : 99.74 %

Moyenne : 99.78 %

### Accuracy

99.77 %

### Matrice de confusion

```text
[[25816     0     0]
 [    4 25710   103]
 [    0    71 25746]]


Structure du projet
api/
config/
data/
models/
notebooks/
reports/
src/
Lancement

Installation :

pip install -r requirements.txt

Entraînement :

python src/train.py

Évaluation :

python src/evaluate.py

API Flask :

cd api
python app.py
Auteur

CHAYBI FATHI

ISGA Casablanca

2025–2026
