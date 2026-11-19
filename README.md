# Spam Classifier

Detects whether an SMS / message is **spam** or **ham** (normal) using classic machine learning.

## Results (test set, 1,115 messages never seen in training)

| Model | Accuracy | Precision | Recall | F1 (spam) |
|---|---|---|---|---|
| Naive Bayes | 98.65% | 99.26% | 90.60% | 0.947 |
| Logistic Regression | 98.74% | 96.55% | 93.96% | 0.952 |
| **Linear SVM (kept)** | **99.01%** | **97.92%** | **94.63%** | **0.963** |

Training takes under a second on a laptop CPU.

## How it works
1. **Data** – SMS Spam Collection (UCI): 5,574 real messages, 13% spam. Downloaded automatically.
2. **Features** – TF-IDF on words and word pairs (bigrams): each message becomes a vector of "how important is each word here".
3. **Models** – 3 classifiers trained and compared; the best F1 on spam is saved.
4. **Evaluation** – 80/20 stratified split, precision / recall / F1, confusion matrix in `reports/`.

## Live demo (React + FastAPI)
Double-click **`start.bat`**, or run `python api.py`, then open http://127.0.0.1:8001

- `api.py` is the FastAPI server: a JSON API at `/api/...` that also serves the React page.
- `web/` holds the React app (Vite). After changing it, run `cd web && npm install && npm run build`.

## Run it
```powershell
cd "Desktop\AI Projects\1-spam-classifier"
.venv\Scripts\activate          # (first time: python -m venv .venv ; pip install -r requirements.txt)
python src/train.py             # trains + saves model and reports
python src/predict.py "You won a FREE prize, call now!"
python app.py                   # web page at http://127.0.0.1:7860
```

## Project layout
```
src/train.py      download data, train 3 models, pick best, save metrics + confusion matrix
src/predict.py    classify one message from the command line
app.py            Gradio web demo
reports/          metrics.json, confusion_matrix.png
models/           spam_model.joblib
```

## Ideas to go further
- Add French / Arabic spam messages and retrain.
- Compare with a fine-tuned DistilBERT (see project 2 for fine-tuning).
- Serve it as an API with FastAPI and a Docker image.
