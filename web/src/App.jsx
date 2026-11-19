import { useEffect, useState } from "react";

const EXAMPLES = [
  "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot! Txt the word CLAIM to 81010",
  "Hey, are we still meeting at 6 for the project? I'll bring the slides.",
  "Congratulations! Your mobile number has been selected for a free iPhone. Call 09061701461 now",
  "Can you send me the lecture notes tonight? Thanks!",
];

const PRETTY = { naive_bayes: "Naive Bayes", logistic_regression: "Logistic Regression", linear_svm: "Linear SVM" };

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    fetch("/api/stats").then((r) => r.json()).then(setStats).catch(() => {});
  }, []);

  async function check(msg = text) {
    if (!msg.trim()) return;
    setText(msg);
    setLoading(true);
    try {
      const r = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: msg }),
      }).then((x) => x.json());
      setResult({ ...r, text: msg });
      setHistory((h) => [{ ...r, text: msg }, ...h].slice(0, 6));
    } finally {
      setLoading(false);
    }
  }

  const pct = result ? Math.round(result.score * 100) : 0;

  return (
    <div className="page">
      <header className="top">
        <div className="brand">
          <span className="logo" aria-hidden>
            <svg viewBox="0 0 24 24" width="22" height="22"><path d="M12 2 4 5v6c0 5 3.4 9.6 8 11 4.6-1.4 8-6 8-11V5l-8-3Z" fill="currentColor" /></svg>
          </span>
          <div>
            <h1>Inbox Shield</h1>
            <p>Spam detection with classic machine learning</p>
          </div>
        </div>
        <span className="pill">TF-IDF + Linear SVM · runs locally</span>
      </header>

      <main className="grid">
        <section className="card composer">
          <label htmlFor="msg">Paste a message</label>
          <textarea
            id="msg"
            value={text}
            placeholder="e.g. You've won a free cruise! Reply YES to claim…"
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && (e.ctrlKey || e.metaKey) && check()}
          />
          <div className="row">
            <span className="hint">Ctrl + Enter to check</span>
            <button className="primary" onClick={() => check()} disabled={loading || !text.trim()}>
              {loading ? "Checking…" : "Check message"}
            </button>
          </div>
          <div className="examples">
            <span>Try:</span>
            {EXAMPLES.map((e, i) => (
              <button key={i} className="chip" onClick={() => check(e)}>
                {e.slice(0, 34)}…
              </button>
            ))}
          </div>
        </section>

        <section className={`card verdict ${result ? (result.is_spam ? "spam" : "ham") : "idle"}`}>
          {!result ? (
            <div className="empty">
              <div className="ring idle-ring" />
              <p>Your verdict appears here.</p>
            </div>
          ) : (
            <>
              <div className="gauge" style={{ "--p": pct }}>
                <div className="gauge-inner">
                  <strong>{pct}%</strong>
                  <span>spam score</span>
                </div>
              </div>
              <h2>{result.is_spam ? "Spam detected" : "Looks safe"}</h2>
              <p className="sub">
                {result.is_spam ? "This message matches patterns of unwanted promotions or scams." : "Nothing suspicious. This reads like a normal message."}
              </p>
              {result.words?.length > 0 && (
                <div className="why">
                  <h3>Why: the words that weighed most</h3>
                  <ul>
                    {result.words.map((w) => (
                      <li key={w.word} className={w.weight > 0 ? "up" : "down"}>
                        <code>{w.word}</code>
                        <span className="bar"><i style={{ width: `${Math.min(100, Math.abs(w.weight) * 120)}%` }} /></span>
                        <em>{w.weight > 0 ? "spammy" : "normal"}</em>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              <p className="meta">Answered in {result.ms} ms</p>
            </>
          )}
        </section>

        <section className="card">
          <h3 className="title">Model comparison <small>(1,115 unseen test messages)</small></h3>
          {stats ? (
            <table>
              <thead>
                <tr><th>Model</th><th>Accuracy</th><th>F1 (spam)</th></tr>
              </thead>
              <tbody>
                {Object.entries(stats.results).map(([k, v]) => (
                  <tr key={k} className={k === stats.best ? "best" : ""}>
                    <td>{PRETTY[k] || k}{k === stats.best && <span className="tag">in use</span>}</td>
                    <td>{(v.accuracy * 100).toFixed(1)}%</td>
                    <td>
                      <div className="f1"><span style={{ width: `${v.f1 * 100}%` }} />{v.f1.toFixed(3)}</div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted">Loading…</p>
          )}
        </section>

        <section className="card">
          <h3 className="title">Recent checks</h3>
          {history.length === 0 ? (
            <p className="muted">No messages checked yet.</p>
          ) : (
            <ul className="history">
              {history.map((h, i) => (
                <li key={i}>
                  <span className={`dot ${h.is_spam ? "spam" : "ham"}`} />
                  <span className="txt">{h.text}</span>
                  <b>{Math.round(h.score * 100)}%</b>
                </li>
              ))}
            </ul>
          )}
        </section>
      </main>

      <footer>Trained on the UCI SMS Spam Collection · 5,574 real messages · scikit-learn + FastAPI + React</footer>
    </div>
  );
}
