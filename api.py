"""API + React demo server.  Run:  python api.py   ->  http://127.0.0.1:8001"""
import json
import sys
import time
from pathlib import Path

import joblib
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from predict import MODEL, classify  # noqa: E402

app = FastAPI(title="Spam Classifier API")
model = joblib.load(MODEL)


class Msg(BaseModel):
    text: str


@app.post("/api/predict")
def predict(m: Msg):
    t = time.perf_counter()
    r = classify(m.text, model)
    # top words that pushed the decision (linear model -> readable explanation)
    tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    vec = tfidf.transform([m.text])
    names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0] if hasattr(clf, "coef_") else None
    words = []
    if coefs is not None:
        for i in vec.nonzero()[1]:
            words.append({"word": names[i], "weight": round(float(vec[0, i] * coefs[i]), 3)})
        words.sort(key=lambda w: -abs(w["weight"]))
    return {"is_spam": r["label"] == "SPAM", "score": r["spam_score"],
            "ms": round((time.perf_counter() - t) * 1000, 2), "words": words[:6]}


@app.get("/api/stats")
def stats():
    return json.loads((ROOT / "reports" / "metrics.json").read_text())


app.mount("/", StaticFiles(directory=ROOT / "web" / "dist", html=True), name="web")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
