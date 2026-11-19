"""Small web page to test the spam classifier.  Run:  python app.py"""
import sys
from pathlib import Path

import gradio as gr
import joblib

sys.path.insert(0, str(Path(__file__).parent / "src"))
from predict import MODEL, classify  # noqa: E402

model = joblib.load(MODEL)


def run(text):
    r = classify(text, model)
    return r["label"], r["spam_score"]


demo = gr.Interface(
    fn=run,
    inputs=gr.Textbox(lines=4, label="Message"),
    outputs=[gr.Textbox(label="Result"), gr.Number(label="Spam score (0 = clean, 1 = spam)")],
    title="Spam Classifier",
    description="TF-IDF + linear model trained on 5,574 real SMS messages.",
    examples=[["Congratulations! You won a $1000 gift card. Click here to claim now!"],
              ["Hey, are we still meeting at 6 for the project?"]],
)

if __name__ == "__main__":
    demo.launch()
