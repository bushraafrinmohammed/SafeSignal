"""
Alcon Device Safety Signals: NLP on FDA MAUDE adverse event reports

Part A, Post-market surveillance analytics:
  - Report volume over time by device category
  - Most frequent product problems
  - Complaint themes discovered with NMF topic modeling

Part B, Patient-harm triage classifier:
  - Target: report involves patient harm (Injury/Death = 1) vs device malfunction only (0)
  - Input: free-text event description only
  - TF-IDF (1-2 grams) + Logistic Regression / Linear SVM / Naive Bayes
  - Time-based split (newest 20% of reports = test); exact duplicate texts removed
  - Business metric: share of harm reports caught when reviewers read the top 30% riskiest

Usage:
  python src/train.py --data data/alcon_events.csv
"""
import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS, MODELS = ROOT / "results", ROOT / "results" / "figures", ROOT / "models"
N_TOPICS = 8
DOMAIN_STOP = {"alcon", "reported", "report", "patient", "pt", "device", "information",
               "received", "event", "mfr", "manufacturer", "date", "unknown", "stated",
               "approximately", "additional", "provided", "noted", "occurred", "per", "details", "impact", "requested", "customer", "health", "care", "professional", "healthcare", "non", "medical", "reports", "associated", "consumer", "product", "ophthalmic", "information", "attempt", "attempts", "obtain", "obtained", "available", "unable", "follow", "submitted"}
STOP = list(ENGLISH_STOP_WORDS | DOMAIN_STOP)


def clean(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"\(b\)\(\d\)", " ", t)       # FDA redaction tokens like (b)(6)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df["date"] = pd.to_datetime(df["date_received"], format="%Y%m%d", errors="coerce")
    df["text"] = df["event_text"].fillna("").map(clean)
    df = df[(df["text"].str.split().str.len() >= 8) & df["date"].notna()]
    df = df.drop_duplicates(subset="text")  # templated duplicates would leak across split
    cat = df["device_class_name"].fillna("").replace("", np.nan).fillna(df["generic_name"])
    df["category"] = cat.fillna("Unknown").str.title().str.slice(0, 40)
    return df.reset_index(drop=True)


# ---------------- Part A ----------------
def surveillance(df: pd.DataFrame) -> dict:
    top_cats = df["category"].value_counts().head(6).index
    monthly = (df[df["category"].isin(top_cats)]
               .groupby([pd.Grouper(key="date", freq="QS"), "category"]).size().unstack(fill_value=0))
    fig, ax = plt.subplots(figsize=(8, 4))
    monthly.plot(ax=ax, marker="o")
    ax.set_title("Adverse event reports per quarter, top device categories")
    ax.set_ylabel("Reports"); ax.set_xlabel(""); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "reports_over_time.png", dpi=150); plt.close(fig)

    probs = (df["product_problems"].dropna().str.split("; ").explode().str.strip()
             .replace("", np.nan).dropna().value_counts().head(10))
    fig, ax = plt.subplots(figsize=(7, 4))
    probs[::-1].plot.barh(ax=ax, color="#3b6fb6")
    ax.set_title("Most frequent product problems"); ax.set_xlabel("Reports")
    fig.tight_layout(); fig.savefig(FIGS / "product_problems.png", dpi=150); plt.close(fig)

    # Topic modeling
    vec = TfidfVectorizer(stop_words=STOP, max_df=0.6, min_df=5, ngram_range=(1, 2), max_features=8000)
    X = vec.fit_transform(df["text"])
    nmf = NMF(n_components=N_TOPICS, random_state=SEED, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    topics = {f"Topic {i+1}": ", ".join(terms[np.argsort(-c)[:8]]) for i, c in enumerate(nmf.components_)}
    df["topic"] = [f"Topic {i+1}" for i in W.argmax(1)]
    # One representative report per topic (highest topic weight) for the dashboard
    examples = {}
    for i in range(N_TOPICS):
        j = int(np.argmax(W[:, i]))
        examples[f"Topic {i+1}"] = str(df.iloc[j]["event_text"])[:400]
    topic_share = (df["topic"].value_counts(normalize=True) * 100).round(1).to_dict()
    harm_by_topic = (df.groupby("topic")["harm"].mean() * 100).round(1).to_dict()
    joblib.dump({"vec": vec, "nmf": nmf, "topics": topics}, MODELS / "topics.joblib")
    df[["date", "category", "event_type", "topic", "harm"]].to_csv(RESULTS / "report_index.csv", index=False)
    return {"categories": df["category"].value_counts().head(8).to_dict(),
            "top_problems": probs.to_dict(), "topics": topics,
            "topic_share_pct": topic_share, "harm_rate_by_topic_pct": harm_by_topic,
            "topic_examples": examples}


# ---------------- Part B ----------------
def capture_at(y, p, frac):
    n = max(1, int(len(p) * frac))
    return float(np.asarray(y)[np.argsort(-p)[:n]].sum() / np.asarray(y).sum())


def classifier(df: pd.DataFrame) -> dict:
    d = df.sort_values("date")
    cut = int(len(d) * 0.8)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    tfidf = lambda: TfidfVectorizer(stop_words=STOP, ngram_range=(1, 2), min_df=3,
                                    max_df=0.7, sublinear_tf=True, max_features=30000)
    cands = {
        "Logistic Regression": LogisticRegression(max_iter=3000, C=4, class_weight="balanced"),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3),
        "Naive Bayes": ComplementNB(alpha=0.5),
    }
    rows, fitted = [], {}
    for name, clf in cands.items():
        pipe = Pipeline([("tfidf", tfidf()), ("clf", clf)]).fit(tr["text"], tr["harm"])
        p = pipe.predict_proba(te["text"])[:, 1]
        pred = (p >= 0.5).astype(int)
        fitted[name] = (pipe, p)
        rows.append({"model": name, "recall": recall_score(te["harm"], pred),
                     "precision": precision_score(te["harm"], pred, zero_division=0),
                     "f1": f1_score(te["harm"], pred), "roc_auc": roc_auc_score(te["harm"], p),
                     "pr_auc": average_precision_score(te["harm"], p),
                     "capture_top30_pct": 100 * capture_at(te["harm"], p, 0.3)})
    res = pd.DataFrame(rows)
    best = res.sort_values("pr_auc", ascending=False).iloc[0]["model"]
    best_pipe, best_p = fitted[best]

    # Explainability: words pushing toward harm vs malfunction (from the LR model)
    lr = fitted["Logistic Regression"][0]
    coefs = lr.named_steps["clf"].coef_[0]
    vocab = np.array(lr.named_steps["tfidf"].get_feature_names_out())
    harm_words = vocab[np.argsort(-coefs)[:12]].tolist()
    malf_words = vocab[np.argsort(coefs)[:12]].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, idx, title, col in [(axes[0], np.argsort(-coefs)[:12], "Signals of patient harm", "#b33a3a"),
                                (axes[1], np.argsort(coefs)[:12], "Signals of malfunction only", "#3b6fb6")]:
        ax.barh(vocab[idx][::-1], np.abs(coefs[idx])[::-1], color=col); ax.set_title(title)
    fig.tight_layout(); fig.savefig(FIGS / "key_terms.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    fr = np.linspace(0.01, 1, 100)
    for name, (_, p) in fitted.items():
        ax.plot(fr * 100, [100 * capture_at(te["harm"], p, f) for f in fr], label=name)
    ax.plot(fr * 100, fr * 100, "--", c="gray", label="Random order")
    ax.set_xlabel("% of reports reviewed (highest risk first)"); ax.set_ylabel("% of harm reports found")
    ax.set_title("Triage efficiency (newest 20% of reports)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "triage_curve.png", dpi=150); plt.close(fig)

    joblib.dump({"pipeline": best_pipe, "model_name": best}, MODELS / "classifier.joblib")
    # Logistic Regression kept separately to explain which words drive a prediction
    joblib.dump({"vocab": vocab, "coefs": coefs, "tfidf": lr.named_steps["tfidf"]}, MODELS / "explainer.joblib")
    print(classification_report(te["harm"], (best_p >= 0.5).astype(int),
                                target_names=["Malfunction", "Patient harm"]))
    return {"train_reports": int(len(tr)), "test_reports": int(len(te)),
            "test_period": f"{te['date'].min():%Y-%m-%d} to {te['date'].max():%Y-%m-%d}",
            "test_harm_rate": float(te["harm"].mean()), "models": rows, "best_model": best,
            "harm_terms": harm_words, "malfunction_terms": malf_words}


def main(path: Path):
    for p in (RESULTS, FIGS, MODELS):
        p.mkdir(parents=True, exist_ok=True)
    df = load(path)
    df = df[df["event_type"].isin(["Injury", "Death", "Malfunction"])].copy()
    df["harm"] = df["event_type"].isin(["Injury", "Death"]).astype(int)
    print(f"{len(df):,} usable reports | harm rate {df['harm'].mean():.1%}")

    surv = surveillance(df)
    clf = classifier(df)
    (RESULTS / "metrics.json").write_text(json.dumps(
        {"n_reports": int(len(df)), "harm_rate": float(df["harm"].mean()),
         "date_range": f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}",
         "surveillance": surv, "classifier": clf}, indent=2, default=float))

    best = next(r for r in clf["models"] if r["model"] == clf["best_model"])
    md = [
        "# Results (auto-generated by `src/train.py`)\n",
        f"- Reports analyzed: **{len(df):,}** ({df['date'].min():%b %Y} – {df['date'].max():%b %Y}), "
        f"patient-harm share **{df['harm'].mean():.1%}**\n",
        "## Complaint themes (NMF topics)\n",
        "| Topic | Share | Harm rate | Top terms |", "|---|---|---|---|",
        *[f"| {t} | {surv['topic_share_pct'].get(t, 0)}% | {surv['harm_rate_by_topic_pct'].get(t, 0)}% | {w} |"
          for t, w in surv["topics"].items()],
        "\n## Patient-harm triage classifier\n",
        f"Train: {clf['train_reports']:,} older reports | Test: {clf['test_reports']:,} newest reports "
        f"({clf['test_period']})\n",
        pd.DataFrame(clf["models"]).to_markdown(index=False, floatfmt=".3f"),
        f"\n**{clf['best_model']}**: reviewing the riskiest 30% of reports first finds "
        f"**{best['capture_top30_pct']:.0f}%** of patient-harm reports.\n",
        f"- Harm signal terms: {', '.join(clf['harm_terms'][:8])}",
        f"- Malfunction signal terms: {', '.join(clf['malfunction_terms'][:8])}",
    ]
    (RESULTS / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "alcon_events.csv")
    main(ap.parse_args().data)
