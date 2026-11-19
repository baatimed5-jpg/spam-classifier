"""Train and compare 3 classic spam classifiers on the SMS Spam Collection.

Run:  python src/train.py
Output: models/spam_model.joblib, reports/metrics.json, reports/confusion_matrix.png
"""
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "SMSSpamCollection"
URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def load_data() -> pd.DataFrame:
    if not DATA.exists():
        print("Downloading SMS Spam Collection (UCI)...")
        DATA.parent.mkdir(exist_ok=True)
        raw = urllib.request.urlopen(URL, timeout=60).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            DATA.write_bytes(z.read("SMSSpamCollection"))
    df = pd.read_csv(DATA, sep="\t", header=None, names=["label", "text"], quoting=3)
    df["y"] = (df["label"] == "spam").astype(int)
    return df


def make_models():
    tfidf = lambda: TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                    sublinear_tf=True)
    return {
        "naive_bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.1))]),
        "logistic_regression": Pipeline([("tfidf", tfidf()),
                                         ("clf", LogisticRegression(C=10, max_iter=2000,
                                                                    class_weight="balanced"))]),
        "linear_svm": Pipeline([("tfidf", tfidf()),
                                ("clf", LinearSVC(C=1.0, class_weight="balanced"))]),
    }


def main():
    df = load_data()
    print(f"{len(df)} messages | spam: {df.y.sum()} ({df.y.mean():.1%})")
    X_tr, X_te, y_tr, y_te = train_test_split(df.text, df.y, test_size=0.2,
                                              stratify=df.y, random_state=42)
    results, fitted = {}, {}
    for name, model in make_models().items():
        t = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_s = time.perf_counter() - t
        pred = model.predict(X_te)
        results[name] = {
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "precision": round(precision_score(y_te, pred), 4),
            "recall": round(recall_score(y_te, pred), 4),
            "f1": round(f1_score(y_te, pred), 4),
            "train_seconds": round(train_s, 2),
        }
        fitted[name] = model
        print(f"{name:20s} {results[name]}")

    best = max(results, key=lambda n: results[n]["f1"])
    print(f"\nBest model: {best}\n")
    print(classification_report(y_te, fitted[best].predict(X_te), target_names=["ham", "spam"]))

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(fitted[best], ROOT / "models" / "spam_model.joblib")
    (ROOT / "reports" / "metrics.json").write_text(
        json.dumps({"best": best, "results": results}, indent=2))
    ConfusionMatrixDisplay.from_estimator(fitted[best], X_te, y_te,
                                          display_labels=["ham", "spam"], cmap="Blues")
    plt.title(f"Confusion matrix - {best}")
    plt.savefig(ROOT / "reports" / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    print("Saved models/spam_model.joblib and reports/")


if __name__ == "__main__":
    main()
