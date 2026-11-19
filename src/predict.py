"""Classify a message from the command line.

Run:  python src/predict.py "WINNER!! Claim your free prize now"
"""
import sys
from pathlib import Path

import joblib

MODEL = Path(__file__).resolve().parent.parent / "models" / "spam_model.joblib"


def classify(text: str, model=None) -> dict:
    model = model or joblib.load(MODEL)
    clf = model.named_steps["clf"]
    is_spam = bool(model.predict([text])[0])
    if hasattr(clf, "predict_proba"):
        score = float(model.predict_proba([text])[0][1])
    else:  # LinearSVC: map decision distance to 0..1
        import math
        score = 1 / (1 + math.exp(-float(model.decision_function([text])[0])))
    return {"label": "SPAM" if is_spam else "HAM (not spam)", "spam_score": round(score, 3)}


if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) or input("Message: ")
    print(classify(msg))
