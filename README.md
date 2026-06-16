# 🇲🇦 Darija NLP Sentiment Analysis

Projet NLP pour analyser le sentiment en Darija marocaine.

## Pipeline
Text → Cleaning → TF-IDF → Logistic Regression → Sentiment

## Run

```bash
pip install -r requirements.txt
python src/preprocess.py
python src/train.py
python src/predict.py "hada projet mzyan"