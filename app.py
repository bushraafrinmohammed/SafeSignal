"""
SafeSignal: interactive NLP triage and safety-trend dashboard for FDA device adverse event reports.
Run:  python -m streamlit run app.py   (after src/download_data.py and src/train.py)
"""
import html
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from train import clean  # noqa: E402

TEAL, DEEP, HARM, MALF, INK = "#0E9F9A", "#0B6E6A", "#E5484D", "#3B82F6", "#1F2D3D"
SAMPLES = {
    "🩸 Post-op complication": "Following cataract surgery the patient developed corneal edema and blurred vision. "
                              "The intraocular lens was explanted and a secondary surgery was required.",
    "⚙️ Device malfunction": "During preparation the cartridge cracked when the plunger was advanced. The lens was "
                            "not implanted, a backup device was used and the procedure was completed without delay.",
    "👁️ Lens issue": "The surgeon reported the lens did not unfold properly in the eye. It was repositioned and the "
                    "patient later reported glare and halos requiring follow-up visits.",
}

st.set_page_config(page_title="SafeSignal", page_icon="🩺", layout="wide")


@st.cache_resource
def load_models():
    return (joblib.load(ROOT / "models" / "classifier.joblib"), joblib.load(ROOT / "models" / "topics.joblib"),
            joblib.load(ROOT / "models" / "explainer.joblib"))


@st.cache_data
def load_data():
    idx = pd.read_csv(RES / "report_index.csv", parse_dates=["date"])
    return json.loads((RES / "metrics.json").read_text()), idx


try:
    (clf, tp, ex), (m, idx) = load_models(), load_data()
except FileNotFoundError:
    st.error("Results not found. Run `python src/download_data.py` and `python src/train.py` first.")
    st.stop()

c, s = m["classifier"], m["surveillance"]
best = next(r for r in c["models"] if r["model"] == c["best_model"])

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stApp {{font-family: 'Plus Jakarta Sans', sans-serif;}}
.stApp {{
  background:
    radial-gradient(circle at 8% 0%, rgba(20,184,166,0.10) 0, transparent 35%),
    radial-gradient(circle at 100% 30%, rgba(14,159,154,0.08) 0, transparent 30%),
    radial-gradient(rgba(11,110,106,0.07) 1px, transparent 1px) 0 0/22px 22px,
    #F7FBFA;
}}
section[data-testid="stSidebar"] > div {{background: linear-gradient(180deg, #E6F6F4 0%, #F7FBFA 60%);}}
section[data-testid="stSidebar"] h3 {{color: {DEEP};}}

/* Animated hero */
@keyframes flow {{0% {{background-position: 0% 50%;}} 50% {{background-position: 100% 50%;}} 100% {{background-position: 0% 50%;}}}}
@keyframes float {{0%,100% {{transform: translateY(0) scale(1);}} 50% {{transform: translateY(-14px) scale(1.05);}}}}
@keyframes pulse {{0%,100% {{box-shadow: 0 0 0 0 rgba(204,251,241,0.7);}} 50% {{box-shadow: 0 0 0 8px rgba(204,251,241,0);}}}}
@keyframes fadeUp {{from {{opacity: 0; transform: translateY(14px);}} to {{opacity: 1; transform: translateY(0);}}}}
.hero {{position: relative; overflow: hidden; border-radius: 22px; padding: 34px 36px; margin-bottom: 22px; color: #fff;
        background: linear-gradient(120deg, #063F3C, {DEEP}, {TEAL}, #14B8A6, {DEEP});
        background-size: 300% 300%; animation: flow 14s ease infinite;
        box-shadow: 0 18px 40px -18px rgba(11,110,106,0.55);}}
.hero::before, .hero::after {{content: ""; position: absolute; border-radius: 50%; filter: blur(2px);
        background: radial-gradient(circle, rgba(255,255,255,0.28), rgba(255,255,255,0));}}
.hero::before {{width: 260px; height: 260px; right: -60px; top: -80px; animation: float 7s ease-in-out infinite;}}
.hero::after {{width: 160px; height: 160px; right: 180px; bottom: -70px; animation: float 9s ease-in-out infinite reverse;}}
.hero h1 {{margin: 0; font-size: 2.6rem; font-weight: 800; letter-spacing: -1px; color: #fff;}}
.hero .tag {{font-size: 1.1rem; opacity: 0.95; margin-top: 4px;}}
.hero .sub {{font-size: 0.85rem; opacity: 0.8; margin-top: 12px; display: flex; align-items: center; gap: 8px;}}
.live {{width: 9px; height: 9px; border-radius: 50%; background: #CCFBF1; display: inline-block; animation: pulse 2s infinite;}}
.chip {{display: inline-block; background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.3);
        border-radius: 999px; padding: 4px 13px; margin: 12px 6px 0 0; font-size: 0.8rem; backdrop-filter: blur(6px);
        transition: all .2s;}}
.chip:hover {{background: rgba(255,255,255,0.28); transform: translateY(-2px);}}
.bar {{width: 56px; height: 4px; background: #CCFBF1; border-radius: 2px; margin-bottom: 12px;}}

/* KPI cards */
.kpis {{display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px;}}
@media (max-width: 900px) {{ .kpis {{grid-template-columns: repeat(2, minmax(0, 1fr));}} }}
.kpi {{position: relative; background: rgba(255,255,255,0.85); backdrop-filter: blur(8px); border: 1px solid #D4EBE8;
        border-radius: 18px; padding: 16px 18px 14px; overflow: hidden; animation: fadeUp .6s ease both;
        transition: transform .2s, box-shadow .2s;}}
.kpi:hover {{transform: translateY(-4px); box-shadow: 0 14px 28px -14px rgba(11,110,106,0.45);}}
.kpi::before {{content: ""; position: absolute; inset: 0 0 auto 0; height: 4px; background: var(--accent);}}
.kpi .ico {{width: 36px; height: 36px; border-radius: 11px; display: grid; place-items: center; font-size: 1.15rem;
        background: var(--soft); margin-bottom: 8px;}}
.kpi .lbl {{font-size: 0.78rem; text-transform: uppercase; letter-spacing: .6px; color: #5B6B7B; font-weight: 600;}}
.kpi .val {{font-size: 1.9rem; font-weight: 800; color: {INK}; line-height: 1.15; margin-top: 2px;}}
.kpi .hint {{font-size: 0.78rem; color: #6B7C8C; margin-top: 2px;}}

/* Tabs as pills */
div[data-baseweb="tab-list"] {{gap: 8px; background: rgba(255,255,255,0.7); padding: 6px; border-radius: 14px;
        border: 1px solid #D4EBE8;}}
button[data-baseweb="tab"] {{border-radius: 10px !important; padding: 8px 16px !important; transition: all .2s;}}
button[data-baseweb="tab"]:hover {{background: #EEF7F6;}}
button[data-baseweb="tab"][aria-selected="true"] {{background: linear-gradient(120deg, {DEEP}, {TEAL}); color: #fff !important;}}
button[data-baseweb="tab"][aria-selected="true"] p {{color: #fff !important;}}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {{display: none;}}

/* Chart cards & widgets */
div[data-testid="stPlotlyChart"] {{background: #fff; border: 1px solid #E1F0EE; border-radius: 18px; padding: 8px;
        box-shadow: 0 6px 18px -12px rgba(11,110,106,0.35); animation: fadeUp .6s ease both;}}
div[data-testid="stImage"] img {{border-radius: 14px; border: 1px solid #E1F0EE;}}
.stButton > button {{border-radius: 12px; border: 1px solid #BFE3DE; background: #fff; font-weight: 600; transition: all .2s;}}
.stButton > button:hover {{background: linear-gradient(120deg, {DEEP}, {TEAL}); color: #fff; border-color: transparent;
        transform: translateY(-2px);}}
textarea {{border-radius: 14px !important;}}
div[data-testid="stMetric"] {{background: #fff; border: 1px solid #D4EBE8; border-radius: 14px; padding: 12px 14px;}}

.insight {{background: linear-gradient(120deg, #EEF7F6, #F7FBFA); border-left: 4px solid {TEAL}; border-radius: 12px;
        padding: 12px 16px; margin: 8px 0 14px 0; color: {INK}; animation: fadeUp .6s ease both;}}
.hl {{line-height: 2.1; font-size: 1.02rem; padding: 16px 18px; border: 1px solid #D4EBE8; border-radius: 14px; background: #fff;}}
.hl span.w {{border-radius: 6px; padding: 2px 4px; transition: all .15s;}}
.hl span.w:hover {{outline: 2px solid rgba(31,45,61,0.25);}}
.verdict {{border-radius: 14px; padding: 14px 18px; font-weight: 700; margin-top: 8px; animation: fadeUp .4s ease both;}}
.legend {{font-size: 0.82rem; color: #5B6B7B; margin-top: 6px;}}
.tchip {{display: inline-block; background: #EEF7F6; border: 1px solid #CDE7E3; color: {DEEP};
        border-radius: 999px; padding: 3px 11px; margin: 3px 4px 3px 0; font-size: 0.85rem; transition: all .15s;}}
.tchip:hover {{background: {TEAL}; color: #fff;}}
.quote {{background: #fff; border-left: 3px solid {TEAL}; padding: 12px 16px; border-radius: 10px;
        font-style: italic; color: #34495E; box-shadow: 0 4px 12px -8px rgba(11,110,106,0.4);}}
</style>""", unsafe_allow_html=True)

st.markdown(f"""<div class="hero"><div class="bar"></div><h1>🩺 SafeSignal</h1>
<div class="tag">NLP triage for medical device adverse event reports</div>
<div><span class="chip">🔌 openFDA API</span><span class="chip">🔤 TF-IDF</span><span class="chip">🧩 NMF topics</span>
<span class="chip">🔍 Explainable classifier</span></div>
<div class="sub"><span class="live"></span>{m['n_reports']:,} FDA MAUDE reports · Alcon devices · {m['date_range']}</div></div>""",
            unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)


def style(fig, h=380):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(color=INK), legend=dict(orientation="h", y=-0.2))
    return fig


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### ⚙️ Filters")
    cats = idx["category"].value_counts()
    chosen = st.multiselect("Device categories", list(cats.index), default=list(cats.index[:5]))
    dmin, dmax = idx["date"].min().date(), idx["date"].max().date()
    dr = st.slider("Report date range", dmin, dmax, (dmin, dmax), format="MMM YYYY")
    st.divider()
    st.markdown("### 📌 About")
    st.caption("Public FDA MAUDE data via openFDA. Reports are unverified submissions; counts are not "
               "incidence rates. The classifier is a prioritization aid for human reviewers.")

view = idx[idx["category"].isin(chosen or list(cats.index))
           & idx["date"].between(pd.Timestamp(dr[0]), pd.Timestamp(dr[1]))]

def kpi_cards(cards):
    html_cards = "".join(
        f'<div class="kpi" style="--accent:{c[4]};--soft:{c[5]};animation-delay:{i * 0.08:.2f}s">'
        f'<div class="ico">{c[0]}</div><div class="lbl">{c[1]}</div><div class="val">{c[2]}</div>'
        f'<div class="hint">{c[3]}</div></div>' for i, c in enumerate(cards))
    st.markdown(f'<div class="kpis">{html_cards}</div>', unsafe_allow_html=True)


kpi_cards([
    ("📄", "Reports", f"{len(view):,}", f"of {m['n_reports']:,} in selection", TEAL, "#E0F5F3"),
    ("🩸", "Patient-harm share", f"{view['harm'].mean():.0%}" if len(view) else "–", "Injury or death reports",
     HARM, "#FDE4E5"),
    ("🎯", "Harm recall", f"{best['recall']:.0%}", "Harm reports caught (test)", "#8B5CF6", "#EDE7FE"),
    ("📈", "PR-AUC", f"{best['pr_auc']:.2f}", f"{c['best_model']}", MALF, "#DCEBFE"),
    ("⚡", "Top-30% review", f"{best['capture_top30_pct']:.0f}%", "of harm reports found first", "#F59E0B", "#FEF3C7"),
])

tabs = st.tabs(["🩺 Triage a report", "📈 Trends", "🧩 Complaint themes", "🔤 Key signals", "📊 Model"])

# ---------------- Triage ----------------
with tabs[0]:
    st.markdown("**Try a sample or paste your own report**")
    cols = st.columns(len(SAMPLES))
    if "txt" not in st.session_state:
        st.session_state.txt = list(SAMPLES.values())[0]
    for col, (label, txt) in zip(cols, SAMPLES.items()):
        if col.button(label, width="stretch"):
            st.session_state.txt = txt
    text = st.text_area("Adverse event description", key="txt", height=130)

    if text.strip():
        t = clean(text)
        p = float(clf["pipeline"].predict_proba([t])[:, 1][0])
        w = tp["nmf"].transform(tp["vec"].transform([t]))[0]
        topic = f"Topic {int(np.argmax(w)) + 1}"

        a, b = st.columns([1, 2])
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=100 * p, number={"suffix": "%", "font": {"size": 42}},
            title={"text": "Patient-harm probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": INK},
                   "steps": [{"range": [0, 40], "color": "#DCEFFE"}, {"range": [40, 60], "color": "#FFF1D6"},
                             {"range": [60, 100], "color": "#FCDADB"}]}))
        a.plotly_chart(style(gauge, 260).update_layout(margin=dict(t=60, b=0)), width="stretch")
        if p >= 0.5:
            a.markdown(f'<div class="verdict" style="background:#FCDADB">🚨 Escalate to clinical safety review</div>',
                       unsafe_allow_html=True)
        else:
            a.markdown(f'<div class="verdict" style="background:#DCEFFE">📋 Standard malfunction queue</div>',
                       unsafe_allow_html=True)

        # Word-level explanation from the Logistic Regression coefficients
        coef = dict(zip(ex["vocab"], ex["coefs"]))
        scale = max(abs(ex["coefs"]).max(), 1e-9)
        parts = []
        for tok in text.split():
            key = clean(tok)
            v = coef.get(key, 0.0)
            if abs(v) < 0.15 * scale:
                parts.append(html.escape(tok))
            else:
                alpha = min(0.85, 0.2 + abs(v) / scale)
                rgb = "229,72,77" if v > 0 else "59,130,246"
                parts.append(f'<span class="w" style="background: rgba({rgb},{alpha:.2f})" '
                             f'title="{"harm" if v > 0 else "malfunction"} signal">{html.escape(tok)}</span>')
        b.markdown("**Why? Words the model reacted to**")
        b.markdown(f'<div class="hl">{" ".join(parts)}</div>', unsafe_allow_html=True)
        b.markdown('<div class="legend"><span style="background:rgba(229,72,77,.5);padding:1px 6px;'
                   'border-radius:4px">red</span> pushes toward patient harm · <span style="background:'
                   'rgba(59,130,246,.5);padding:1px 6px;border-radius:4px">blue</span> pushes toward malfunction '
                   'only · darker = stronger</div>', unsafe_allow_html=True)

        vec = ex["tfidf"].transform([t])
        contrib = pd.Series(vec.toarray()[0] * ex["coefs"], index=ex["vocab"])
        contrib = contrib[contrib != 0]
        if len(contrib):
            top = pd.concat([contrib.nlargest(6), contrib.nsmallest(6)]).drop_duplicates().sort_values()
            fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                                   marker_color=[HARM if v > 0 else MALF for v in top.values]))
            fig.update_xaxes(title="Contribution (← malfunction · harm →)")
            b.plotly_chart(style(fig, 300).update_layout(title="Top contributing terms"), width="stretch")

        hr = s["harm_rate_by_topic_pct"].get(topic, 0)
        insight(f"Closest complaint theme: <b>{topic}</b> ({s['topics'][topic]}). "
                f"Historically, {hr}% of reports in this theme involved patient harm.")

# ---------------- Trends ----------------
with tabs[1]:
    if view.empty:
        st.warning("No reports match the filters.")
    else:
        q = view.assign(quarter=view["date"].dt.to_period("Q").dt.start_time)
        a, b = st.columns([3, 2])
        vol = q.groupby(["quarter", "category"]).size().reset_index(name="reports")
        fig = px.area(vol, x="quarter", y="reports", color="category",
                      color_discrete_sequence=px.colors.qualitative.Safe)
        a.plotly_chart(style(fig, 400).update_layout(title="Reports per quarter by device category"),
                       width="stretch")
        et = view["event_type"].value_counts()
        donut = go.Figure(go.Pie(labels=et.index, values=et.values, hole=0.6,
                                 marker=dict(colors=[MALF, HARM, INK, "#94A3B8"])))
        b.plotly_chart(style(donut, 400).update_layout(title="Event types"), width="stretch")
        hs = q.groupby("quarter")["harm"].agg(["mean", "size"]).reset_index()
        fig = go.Figure()
        fig.add_bar(x=hs["quarter"], y=hs["size"], name="Reports", marker_color="#CDE7E3", yaxis="y2")
        fig.add_scatter(x=hs["quarter"], y=100 * hs["mean"], name="% patient harm", mode="lines+markers",
                        line=dict(color=HARM, width=3))
        fig.update_layout(yaxis=dict(title="% patient harm"),
                          yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Reports"))
        st.plotly_chart(style(fig, 360).update_layout(title="Patient-harm share over time"), width="stretch")
        peak = hs.loc[hs["size"].idxmax()]
        insight(f"Busiest quarter in this view: <b>{peak['quarter']:%b %Y}</b> with {int(peak['size']):,} reports "
                f"({100 * peak['mean']:.0f}% involving patient harm).")

# ---------------- Topics ----------------
with tabs[2]:
    tdf = pd.DataFrame({"topic": list(s["topics"]), "terms": list(s["topics"].values())})
    tdf["share"] = tdf["topic"].map(s["topic_share_pct"]).fillna(0)
    tdf["harm_rate"] = tdf["topic"].map(s["harm_rate_by_topic_pct"]).fillna(0)
    a, b = st.columns([3, 2])
    fig = px.scatter(tdf, x="share", y="harm_rate", size="share", text="topic", color="harm_rate",
                     color_continuous_scale=[MALF, "#C4B5FD", HARM], size_max=60, hover_data={"terms": True})
    fig.update_traces(textposition="top center")
    fig.update_xaxes(title="% of all reports")
    fig.update_yaxes(title="% involving patient harm")
    a.plotly_chart(style(fig, 460).update_layout(title="Themes: how common vs how serious",
                                                 coloraxis_showscale=False), width="stretch")
    pick = b.selectbox("Explore a theme", tdf["topic"])
    row = tdf.set_index("topic").loc[pick]
    b.metric("Share of reports", f"{row['share']}%")
    b.metric("Patient-harm rate", f"{row['harm_rate']}%")
    b.markdown("**Top terms**")
    b.markdown("".join(f'<span class="tchip">{html.escape(w.strip())}</span>' for w in row["terms"].split(",")),
               unsafe_allow_html=True)
    exm = s.get("topic_examples", {}).get(pick)
    if exm:
        b.markdown("**Most representative report**")
        b.markdown(f'<div class="quote">"{html.escape(exm)}…"</div>', unsafe_allow_html=True)

# ---------------- Key signals ----------------
with tabs[3]:
    co = pd.Series(ex["coefs"], index=ex["vocab"])
    n = st.slider("Terms to show", 5, 25, 12)
    a, b = st.columns(2)
    h = co.nlargest(n).sort_values()
    fig = px.bar(x=h.values, y=h.index, orientation="h", color_discrete_sequence=[HARM])
    fig.update_xaxes(title="Weight toward patient harm")
    fig.update_yaxes(title="")
    a.plotly_chart(style(fig, 30 * n + 120).update_layout(title="Signals of patient harm"), width="stretch")
    mf = co.nsmallest(n).abs().sort_values()
    fig = px.bar(x=mf.values, y=mf.index, orientation="h", color_discrete_sequence=[MALF])
    fig.update_xaxes(title="Weight toward malfunction only")
    fig.update_yaxes(title="")
    b.plotly_chart(style(fig, 30 * n + 120).update_layout(title="Signals of device malfunction"),
                   width="stretch")
    term = st.text_input("Look up any word or phrase", "edema")
    if term:
        v = co.get(clean(term))
        if v is None:
            st.caption("Not in the model's vocabulary (too rare or filtered out).")
        else:
            st.markdown(f"**{term}** → weight **{v:+.2f}** "
                        f"({'toward patient harm' if v > 0 else 'toward malfunction only'})")

# ---------------- Model ----------------
with tabs[4]:
    res = pd.DataFrame(c["models"])
    melt = res.melt("model", ["recall", "precision", "f1", "pr_auc"], var_name="metric", value_name="score")
    fig = px.bar(melt, x="metric", y="score", color="model", barmode="group", text_auto=".2f",
                 color_discrete_sequence=[TEAL, INK, "#94A3B8"])
    st.plotly_chart(style(fig, 380).update_layout(title=f"Test set: newest {c['test_reports']:,} reports "
                                                        f"({c['test_period']})"), width="stretch")
    a, b = st.columns(2)
    a.image(str(RES / "figures" / "triage_curve.png"), width="stretch")
    b.image(str(RES / "figures" / "product_problems.png"), width="stretch")
    insight(f"Trained on {c['train_reports']:,} older reports and tested on the newest {c['test_reports']:,}, "
            f"mimicking real deployment. Best model: <b>{c['best_model']}</b>.")
