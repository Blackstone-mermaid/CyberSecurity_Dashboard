"""
╔══════════════════════════════════════════════════════════╗
║  CYBERWATCH SOC DASHBOARD — UNSW-NB15 Dataset            ║
║  Built with Streamlit + Plotly + XGBoost                 ║
╚══════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberWatch SOC Dashboard",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Exo+2:wght@300;400;600;700&display=swap');
:root {
    --bg-void:#050a0f; --bg-panel:#090f18; --bg-card:#0d1620;
    --border:#112240; --border-glow:#1e3a5f;
    --cyan:#00e5ff; --cyan-dim:#00b8d9;
    --green:#00ff88; --orange:#ff6b2b;
    --red:#ff2d55; --yellow:#ffd60a; --purple:#b57bee;
    --text-main:#ccd6f6; --text-dim:#7a8ba8; --text-bright:#e6f1ff;
}
.stApp{background:var(--bg-void);}
html,body,.stApp{font-family:'Exo 2',sans-serif;color:var(--text-main);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1rem!important;padding-bottom:2rem!important;max-width:100%!important;}
[data-testid="stSidebar"]{background:var(--bg-panel)!important;border-right:1px solid var(--border-glow);}
[data-testid="stSidebar"] *{color:var(--text-main)!important;}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stRadio label{color:var(--cyan-dim)!important;font-size:.78rem!important;text-transform:uppercase;letter-spacing:.05em;}
[data-testid="stSidebar"] div[role="radiogroup"]{gap:.3rem;display:flex;flex-direction:column;}
[data-testid="stSidebar"] div[role="radiogroup"] label{
    display:flex!important;align-items:center;padding:.45rem .8rem!important;
    border-radius:6px!important;border:1px solid var(--border-glow)!important;
    background:var(--bg-card)!important;color:var(--text-dim)!important;
    font-family:Rajdhani,sans-serif!important;font-size:.85rem!important;font-weight:600!important;
    cursor:pointer;transition:all .2s;width:100%!important;box-sizing:border-box;}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover{border-color:var(--cyan)!important;color:var(--cyan)!important;background:rgba(0,229,255,.06)!important;}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){border-color:var(--cyan)!important;color:var(--cyan)!important;background:rgba(0,229,255,.10)!important;}
[data-testid="stSidebar"] div[role="radiogroup"] span[data-testid="stMarkdownContainer"]{display:none;}
[data-testid="stSidebar"] div[data-baseweb="radio"]>div:first-child{display:none!important;}
[data-testid="stSidebar"] div[data-baseweb="select"]>div,[data-testid="stSidebar"] div[data-baseweb="input"]>div{background:var(--bg-card)!important;border-color:var(--border-glow)!important;color:var(--text-main)!important;}
.kpi-card{background:linear-gradient(135deg,var(--bg-card) 0%,#0a1628 100%);border:1px solid var(--border-glow);border-radius:8px;padding:1.2rem 1.4rem;position:relative;overflow:hidden;transition:border-color .3s;}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);}
.kpi-card:hover{border-color:var(--cyan);}
.kpi-label{font-family:'Share Tech Mono',monospace;font-size:.68rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:.12em;margin-bottom:.4rem;}
.kpi-value{font-family:'Rajdhani',sans-serif;font-size:2.1rem;font-weight:700;color:var(--cyan);line-height:1.1;}
.kpi-sub{font-size:.75rem;color:var(--text-dim);margin-top:.3rem;}
.kpi-red{color:var(--red)!important;} .kpi-green{color:var(--green)!important;}
.kpi-yellow{color:var(--yellow)!important;} .kpi-orange{color:var(--orange)!important;}
.section-header{font-family:'Rajdhani',sans-serif;font-size:1.05rem;font-weight:600;color:var(--cyan-dim);text-transform:uppercase;letter-spacing:.1em;border-left:3px solid var(--cyan);padding-left:.7rem;margin:1.5rem 0 .8rem 0;}
.metric-box{background:var(--bg-card);border:1px solid var(--border-glow);border-radius:8px;padding:1rem;text-align:center;transition:border-color .3s;}
.metric-box:hover{border-color:var(--cyan);}
.metric-title{font-family:'Share Tech Mono',monospace;font-size:.65rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.4rem;}
.metric-val{font-family:'Rajdhani',sans-serif;font-size:1.7rem;font-weight:700;}
.pred-box-attack{background:rgba(255,45,85,.10);border:2px solid var(--red);border-radius:10px;padding:1.5rem;text-align:center;}
.pred-box-normal{background:rgba(0,255,136,.08);border:2px solid var(--green);border-radius:10px;padding:1.5rem;text-align:center;}
.pred-label{font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;letter-spacing:.1em;}
.pred-conf{font-family:'Share Tech Mono',monospace;font-size:.85rem;margin-top:.4rem;}
.stTabs [data-baseweb="tab-list"]{background:var(--bg-panel);border-bottom:1px solid var(--border-glow);gap:0;}
.stTabs [data-baseweb="tab"]{font-family:'Rajdhani',sans-serif!important;font-size:.85rem!important;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--text-dim)!important;padding:.5rem 1.2rem!important;background:transparent!important;border:none!important;}
.stTabs [aria-selected="true"]{color:var(--cyan)!important;border-bottom:2px solid var(--cyan)!important;}
.stTabs [data-baseweb="tab-panel"]{background:transparent!important;padding-top:1rem!important;}
[data-testid="stDataFrame"]{border:1px solid var(--border-glow);border-radius:6px;}
hr{border-color:var(--border-glow)!important;margin:1rem 0;}
details{border:1px solid var(--border-glow)!important;border-radius:6px!important;background:var(--bg-card)!important;}
summary{color:var(--cyan-dim)!important;font-family:'Rajdhani',sans-serif!important;font-weight:600!important;}
[data-testid="stExpander"]{border:1px solid var(--border-glow)!important;border-radius:6px!important;background:var(--bg-card)!important;margin-bottom:.5rem!important;}
[data-testid="stExpander"] summary{color:var(--cyan-dim)!important;font-family:'Rajdhani',sans-serif!important;font-weight:600!important;font-size:.9rem!important;letter-spacing:.04em!important;}
[data-testid="stExpander"] summary:hover{color:var(--cyan)!important;}
[data-testid="stExpander"] svg{fill:var(--cyan-dim)!important;}
[data-testid="stExpanderDetails"]{background:var(--bg-card)!important;padding:.5rem!important;}
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:var(--bg-void);}
::-webkit-scrollbar-thumb{background:var(--border-glow);border-radius:2px;}
section[data-testid="stSidebar"]{transform:none!important;width:260px!important;min-width:260px!important;visibility:visible!important;left:0!important;}
section[data-testid="stSidebar"]>div:first-child{min-width:260px!important;}
[data-testid="collapsedControl"],[data-testid="stSidebarCollapseButton"]{display:none!important;}
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY CONSTANTS ─────────────────────────────────────────────────────────
PLOT_BG    = "#090f18"
PAPER_BG   = "#090f18"
GRID_COLOR = "#112240"
TEXT_COLOR = "#ccd6f6"
CYAN       = "#00e5ff"
CYAN_DIM   = "#00b8d9"
GREEN      = "#00ff88"
ORANGE     = "#ff6b2b"
RED        = "#ff2d55"
YELLOW     = "#ffd60a"
PURPLE     = "#b57bee"
PROTO_PAL  = [CYAN, GREEN, ORANGE, YELLOW, PURPLE, RED,
              "#4cc9f0", "#f72585", "#7209b7", "#3a0ca3"]
ATTACK_PAL = {
    "Normal": "#00ff88", "Generic": "#ff2d55", "Exploits": "#ff6b2b",
    "Fuzzers": "#ffd60a", "Dos": "#ff4d6d", "Reconnaissance": "#00e5ff",
    "Analysis": "#b57bee", "Backdoor": "#ff6fd8",
    "Shellcode": "#ff9f1c", "Worms": "#caffbf",
}


def base_layout(title="", legend=None, xtitle="", ytitle="", xaxis=None, yaxis=None):
    """
    Builds a dark-theme Plotly layout dict.

    IMPORTANT: xtitle/ytitle are merged INTO the xaxis/yaxis dicts here so that
    callers never pass both `xaxis_title=` and `xaxis=dict(...)` — that would
    cause the 'multiple values for keyword argument' TypeError in Plotly.
    """
    base_x = dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
                  tickfont=dict(size=10), linecolor=GRID_COLOR)
    if xtitle:
        base_x["title"] = xtitle
    base_y = dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
                  tickfont=dict(size=10), linecolor=GRID_COLOR)
    if ytitle:
        base_y["title"] = ytitle
    # Merge caller overrides last so they take precedence
    merged_x = {**base_x, **(xaxis or {})}
    merged_y = {**base_y, **(yaxis or {})}
    base_leg = dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COLOR,
                    borderwidth=1, font=dict(size=10))
    merged_leg = {**base_leg, **(legend or {})}
    return dict(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family="Exo 2, sans-serif", color=TEXT_COLOR, size=11),
        title=dict(text=title,
                   font=dict(color=CYAN_DIM, size=13, family="Rajdhani, sans-serif"),
                   x=0, xanchor="left", pad=dict(l=4)),
        xaxis=merged_x,
        yaxis=merged_y,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=merged_leg,
        hoverlabel=dict(bgcolor="#0d1620", bordercolor=CYAN,
                        font=dict(color=TEXT_COLOR, size=11)),
    )


def fmt_num(n):
    if n >= 1_000_000: return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:     return f"{n / 1_000:.1f}K"
    return str(int(n))


def kpi(label, value, sub="", color="cyan"):
    cls = {"red": "kpi-red", "green": "kpi-green",
           "yellow": "kpi-yellow", "orange": "kpi-orange"}.get(color, "")
    return (f"<div class='kpi-card'><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value {cls}'>{value}</div>"
            f"<div class='kpi-sub'>{sub}</div></div>")


def mbox(label, val, color=CYAN):
    return (f"<div class='metric-box'><div class='metric-title'>{label}</div>"
            f"<div class='metric-val' style='color:{color};'>{val}</div></div>")


# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────
def engineer(d: pd.DataFrame) -> pd.DataFrame:
    """Add 10 engineered features. Works on both train/test DataFrames and
    on a single-row DataFrame built from live-prediction inputs."""
    d = d.copy()
    d["byte_ratio"]    = d["sbytes"]  / (d["dbytes"]  + 1)
    d["pkt_ratio"]     = d["spkts"]   / (d["dpkts"]   + 1)
    d["load_ratio"]    = d["sload"]   / (d["dload"]   + 1)
    d["total_bytes"]   = d["sbytes"]  + d["dbytes"]
    d["total_pkts"]    = d["spkts"]   + d["dpkts"]
    d["mean_pkt_diff"] = d["smean"]   - d["dmean"]
    d["ttl_diff"]      = (d["sttl"]   - d["dttl"]).abs()
    d["bytes_per_pkt"] = d["total_bytes"] / (d["total_pkts"] + 1)
    d["jit_sum"]       = d["sjit"]    + d["djit"]
    d["win_diff"]      = (d["swin"]   - d["dwin"]).abs()
    return d


# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_train():
    df = pd.read_csv("UNSW_NB15_training-set.csv")
    df["attack_cat"]   = df["attack_cat"].str.strip().str.title().fillna("Normal")
    df["is_attack"]    = df["label"].astype(int)
    df["traffic_type"] = df["label"].map({0: "Normal", 1: "Attack"})
    rng = np.random.default_rng(42)
    n   = len(df)
    df["sim_hour"] = rng.integers(0, 24, n)
    df["sim_day"]  = rng.integers(1,  8, n)
    return df


# ─── ML TRAINING ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models():
    import time
    from collections import Counter
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score, roc_curve,
                                 confusion_matrix, classification_report)
    from xgboost import XGBClassifier

    # ── Load ──────────────────────────────────────────────────────────────────
    train_df = pd.read_csv("UNSW_NB15_training-set.csv")
    test_df  = pd.read_csv("UNSW_NB15_testing-set.csv")
    for d in [train_df, test_df]:
        d["attack_cat"] = d["attack_cat"].str.strip().str.title().fillna("Normal")

    train_df = engineer(train_df)
    test_df  = engineer(test_df)

    DROP = ["id", "attack_cat", "label"]
    CAT  = ["proto", "service", "state"]
    FEAT = [c for c in train_df.columns if c not in DROP]

    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_df[CAT] = oe.fit_transform(train_df[CAT].astype(str))
    test_df[CAT]  = oe.transform(test_df[CAT].astype(str))

    X_tr  = train_df[FEAT].values
    X_te  = test_df[FEAT].values
    yb_tr = train_df["label"].values
    yb_te = test_df["label"].values

    le = LabelEncoder()
    le.fit(train_df["attack_cat"])
    ym_tr = le.transform(train_df["attack_cat"])
    ym_te = le.transform(test_df["attack_cat"])

    # ── MODEL 1: Binary ───────────────────────────────────────────────────────
    t0 = time.time()
    xgb_bin = XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
        eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)
    xgb_bin.fit(X_tr, yb_tr)
    bin_time = round(time.time() - t0, 1)

    yb_pred = xgb_bin.predict(X_te)
    yb_prob = xgb_bin.predict_proba(X_te)[:, 1]
    fpr_a, tpr_a, _ = roc_curve(yb_te, yb_prob)
    idx = np.linspace(0, len(fpr_a) - 1, min(500, len(fpr_a))).astype(int)

    bin_metrics = {
        "accuracy":   round(accuracy_score(yb_te,  yb_pred) * 100, 2),
        "precision":  round(precision_score(yb_te, yb_pred) * 100, 2),
        "recall":     round(recall_score(yb_te,    yb_pred) * 100, 2),
        "f1":         round(f1_score(yb_te,        yb_pred) * 100, 2),
        "auc":        round(roc_auc_score(yb_te,   yb_prob) * 100, 2),
        "train_time": bin_time,
        "cm":         confusion_matrix(yb_te, yb_pred).tolist(),
        "fpr":        fpr_a[idx].tolist(),
        "tpr":        tpr_a[idx].tolist(),
    }

    # ── MODEL 2: Multi-class with sqrt-balanced weights ───────────────────────
    # sqrt(max_count / class_count) per sample — improves rare-class F1
    # without over-weighting the tiny minority classes to extreme values
    counts = Counter(ym_tr.tolist())
    max_c  = max(counts.values())
    sw     = np.array([np.sqrt(max_c / counts[c]) for c in ym_tr])

    t0 = time.time()
    xgb_multi = XGBClassifier(
        n_estimators=600, max_depth=10, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, colsample_bylevel=0.85,
        min_child_weight=1, gamma=0.05,
        reg_alpha=0.1, reg_lambda=0.5,
        eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0)
    xgb_multi.fit(X_tr, ym_tr, sample_weight=sw)
    multi_time = round(time.time() - t0, 1)

    ym_pred = xgb_multi.predict(X_te)
    cr = classification_report(ym_te, ym_pred,
                               target_names=le.classes_, output_dict=True)

    multi_metrics = {
        "accuracy":    round(accuracy_score(ym_te,  ym_pred) * 100, 2),
        "f1_weighted": round(f1_score(ym_te, ym_pred, average="weighted") * 100, 2),
        "f1_macro":    round(f1_score(ym_te, ym_pred, average="macro")    * 100, 2),
        "train_time":  multi_time,
        "cm":          confusion_matrix(ym_te, ym_pred).tolist(),
        "class_report": cr,
    }

    return {
        "xgb_bin":       xgb_bin,
        "xgb_multi":     xgb_multi,
        "oe":            oe,
        "le":            le,
        "feat_cols":     FEAT,
        "cat_cols":      CAT,
        "bin_metrics":   bin_metrics,
        "multi_metrics": multi_metrics,
        "fi_bin":        dict(zip(FEAT, xgb_bin.feature_importances_.tolist())),
        "fi_multi":      dict(zip(FEAT, xgb_multi.feature_importances_.tolist())),
        "classes":       list(le.classes_),
    }


# ─── LOAD DATA & SIDEBAR ──────────────────────────────────────────────────────
df_raw = load_train()

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 1.5rem 0;'>
      <div style='font-family:Share Tech Mono,monospace;font-size:1.3rem;color:#00e5ff;letter-spacing:.15em;'>CYBERWATCH</div>
      <div style='font-size:.65rem;color:#7a8ba8;letter-spacing:.2em;text-transform:uppercase;margin-top:2px;'>SOC Intelligence Dashboard</div>
      <div style='height:1px;background:linear-gradient(90deg,transparent,#1e3a5f,transparent);margin:1rem 0 0 0;'></div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("NAVIGATION", [
        "⬡  Overview", "⬡  Traffic Analysis", "⬡  Attack Analysis",
        "⬡  ML Prediction", "⬡  Dataset Info"])

    st.markdown("---")
    st.markdown("<div style='font-size:.7rem;color:#7a8ba8;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem;'>FILTERS</div>", unsafe_allow_html=True)
    sel_proto   = st.selectbox("Protocol",        ["All"] + sorted(df_raw["proto"].dropna().unique().tolist()))
    sel_attack  = st.selectbox("Attack Category", ["All"] + sorted(df_raw["attack_cat"].dropna().unique().tolist()))
    sel_service = st.selectbox("Service",         ["All"] + sorted(df_raw["service"].dropna().unique().tolist()))
    sel_state   = st.selectbox("State",           ["All"] + sorted(df_raw["state"].dropna().unique().tolist()))
    st.markdown("---")
    st.markdown(f"<div style='font-size:.68rem;color:#7a8ba8;font-family:Share Tech Mono,monospace;'>RECORDS LOADED<br><span style='color:#00e5ff;font-size:1.1rem;'>{len(df_raw):,}</span></div>", unsafe_allow_html=True)

# ─── APPLY FILTERS ────────────────────────────────────────────────────────────
df = df_raw.copy()
if sel_proto   != "All": df = df[df["proto"]      == sel_proto]
if sel_attack  != "All": df = df[df["attack_cat"] == sel_attack]
if sel_service != "All": df = df[df["service"]    == sel_service]
if sel_state   != "All": df = df[df["state"]      == sel_state]


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("<div style='font-family:Rajdhani,sans-serif;font-size:1.6rem;font-weight:700;color:#e6f1ff;letter-spacing:.05em;margin-bottom:1rem;'>Security Operations Overview</div>", unsafe_allow_html=True)

    total       = len(df)
    attacks     = int(df["is_attack"].sum())
    normals     = total - attacks
    pct         = attacks / total * 100 if total else 0
    ratio       = f"{pct:.1f}%"
    top_att     = (df[df["is_attack"] == 1]["attack_cat"].value_counts().idxmax()
                   if attacks else "—")
    total_bytes = int(df["sbytes"].sum() + df["dbytes"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi("Total Records",   fmt_num(total),   "network events captured"),                   unsafe_allow_html=True)
    c2.markdown(kpi("Total Attacks",   fmt_num(attacks), f"{ratio} of all traffic",   "red"),          unsafe_allow_html=True)
    c3.markdown(kpi("Normal Traffic",  fmt_num(normals), f"{100 - pct:.1f}% benign",  "green"),        unsafe_allow_html=True)
    c4.markdown(kpi("Top Attack",      top_att[:12],     "dominant threat vector",    "orange"),        unsafe_allow_html=True)
    c5.markdown(kpi("Data Transferred",fmt_num(total_bytes) + "B", "total src+dst bytes"),             unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns([1, 1.6])

    with col_left:
        st.markdown("<div class='section-header'>Traffic Composition</div>", unsafe_allow_html=True)
        pie_data = df["traffic_type"].value_counts().reset_index()
        pie_data.columns = ["Type", "Count"]
        fig_pie = go.Figure(go.Pie(
            labels=pie_data["Type"], values=pie_data["Count"], hole=0.62,
            marker=dict(colors=[GREEN, RED], line=dict(color="#050a0f", width=2)),
            textinfo="percent", textfont=dict(size=12, color=TEXT_COLOR),
            hovertemplate="<b>%{label}</b><br>%{value:,} records<br>%{percent}<extra></extra>",
        ))
        fig_pie.add_annotation(
            text=f"<b>{total:,}</b><br><span style='font-size:10px'>TOTAL</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_COLOR, size=14))
        fig_pie.update_layout(
            **base_layout(legend=dict(orientation="h", yanchor="bottom",
                                      y=-0.15, xanchor="center", x=0.5)),
            showlegend=True, height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-header'>Attack Category Distribution</div>", unsafe_allow_html=True)
        cat_c = df["attack_cat"].value_counts().reset_index()
        cat_c.columns = ["Category", "Count"]
        cat_c = cat_c.sort_values("Count", ascending=True)
        fig_bar = go.Figure(go.Bar(
            x=cat_c["Count"], y=cat_c["Category"], orientation="h",
            marker=dict(color=[ATTACK_PAL.get(c, CYAN) for c in cat_c["Category"]], opacity=0.85),
            text=cat_c["Count"].apply(fmt_num), textposition="outside",
            textfont=dict(size=10, color=TEXT_COLOR),
            hovertemplate="<b>%{y}</b>: %{x:,}<extra></extra>",
        ))
        fig_bar.update_layout(**base_layout(xtitle="Count"), height=300, bargap=0.25)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<div class='section-header'>Simulated Traffic Trend (by Hour of Day)</div>", unsafe_allow_html=True)
    trend = df.groupby(["sim_hour", "traffic_type"]).size().reset_index(name="count")
    fig_trend = px.line(trend, x="sim_hour", y="count", color="traffic_type",
                        color_discrete_map={"Normal": GREEN, "Attack": RED}, markers=True)
    fig_trend.update_traces(line=dict(width=2))
    fig_trend.update_layout(
        **base_layout(legend=dict(title=""), xtitle="Hour of Day", ytitle="Event Count",
                      xaxis=dict(tickmode="linear", dtick=2)),
        height=250)
    st.plotly_chart(fig_trend, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRAFFIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Traffic" in page:
    st.markdown("<div style='font-family:Rajdhani,sans-serif;font-size:1.6rem;font-weight:700;color:#e6f1ff;letter-spacing:.05em;margin-bottom:1rem;'>Network Traffic Analysis</div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["  Protocol & Service", "  Bytes & Packets", "  Connection Behaviour"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='section-header'>Protocol Distribution</div>", unsafe_allow_html=True)
            proto_c = df["proto"].value_counts().head(12).reset_index()
            proto_c.columns = ["Protocol", "Count"]
            fig_p = go.Figure(go.Bar(
                x=proto_c["Protocol"], y=proto_c["Count"],
                marker=dict(color=PROTO_PAL[:len(proto_c)], opacity=0.85),
                text=proto_c["Count"].apply(fmt_num), textposition="outside",
                textfont=dict(size=10, color=TEXT_COLOR),
                hovertemplate="<b>%{x}</b>: %{y:,}<extra></extra>",
            ))
            fig_p.update_layout(**base_layout(ytitle="Count", xtitle="Protocol"),
                                height=300, bargap=0.3)
            st.plotly_chart(fig_p, use_container_width=True)

        with c2:
            st.markdown("<div class='section-header'>Service Distribution</div>", unsafe_allow_html=True)
            svc_c = df["service"].value_counts().head(10).reset_index()
            svc_c.columns = ["Service", "Count"]
            fig_s = px.pie(svc_c, values="Count", names="Service",
                           color_discrete_sequence=PROTO_PAL, hole=0.5)
            fig_s.update_traces(textposition="inside", textinfo="percent+label",
                                hovertemplate="<b>%{label}</b>: %{value:,}<extra></extra>")
            fig_s.update_layout(**base_layout(), height=300, showlegend=False)
            st.plotly_chart(fig_s, use_container_width=True)

        st.markdown("<div class='section-header'>Connection State Distribution</div>", unsafe_allow_html=True)
        state_c = df["state"].value_counts().reset_index()
        state_c.columns = ["State", "Count"]
        fig_st = go.Figure(go.Bar(
            x=state_c["State"], y=state_c["Count"],
            marker=dict(color=CYAN, opacity=0.75),
            text=state_c["Count"].apply(fmt_num), textposition="outside",
            hovertemplate="<b>%{x}</b>: %{y:,}<extra></extra>",
        ))
        fig_st.update_layout(**base_layout(ytitle="Count", xtitle="State"),
                             height=260, bargap=0.35)
        st.plotly_chart(fig_st, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='section-header'>Src vs Dst Bytes (Top Protocols)</div>", unsafe_allow_html=True)
            bd = df.groupby("proto")[["sbytes", "dbytes"]].sum().reset_index()
            bd = bd.sort_values("sbytes", ascending=False).head(10)
            fig_bd = go.Figure()
            fig_bd.add_trace(go.Bar(name="Src Bytes", x=bd["proto"], y=bd["sbytes"], marker_color=CYAN,   opacity=0.8))
            fig_bd.add_trace(go.Bar(name="Dst Bytes", x=bd["proto"], y=bd["dbytes"], marker_color=ORANGE, opacity=0.8))
            fig_bd.update_layout(**base_layout(xtitle="Protocol", ytitle="Bytes"),
                                 barmode="group", height=300)
            st.plotly_chart(fig_bd, use_container_width=True)

        with c2:
            st.markdown("<div class='section-header'>Packet Count Distribution (log scale)</div>", unsafe_allow_html=True)
            pkt_s = df[df["spkts"] < df["spkts"].quantile(0.98)]
            fig_pk = go.Figure()
            fig_pk.add_trace(go.Histogram(x=pkt_s["spkts"], name="Src Pkts", marker_color=CYAN,  opacity=0.65, nbinsx=40))
            fig_pk.add_trace(go.Histogram(x=pkt_s["dpkts"], name="Dst Pkts", marker_color=GREEN, opacity=0.65, nbinsx=40))
            fig_pk.update_layout(
                **base_layout(xtitle="Packet Count", ytitle="Frequency",
                              yaxis=dict(type="log")),
                barmode="overlay", height=300)
            st.plotly_chart(fig_pk, use_container_width=True)

        st.markdown("<div class='section-header'>Duration vs Source Bytes (scatter sample)</div>", unsafe_allow_html=True)
        sample = df.sample(min(3000, len(df)), random_state=42)
        fig_sc = px.scatter(sample, x="dur", y="sbytes", color="traffic_type",
                            color_discrete_map={"Normal": GREEN, "Attack": RED},
                            opacity=0.45, log_x=True, log_y=True,
                            hover_data=["proto", "service", "attack_cat"])
        fig_sc.update_traces(marker=dict(size=4))
        fig_sc.update_layout(
            **base_layout(legend=dict(title=""), xtitle="Duration (log)", ytitle="Src Bytes (log)"),
            height=300)
        st.plotly_chart(fig_sc, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='section-header'>TTL Distribution — Src vs Dst</div>", unsafe_allow_html=True)
            fig_ttl = go.Figure()
            fig_ttl.add_trace(go.Histogram(x=df["sttl"], name="Src TTL", marker_color=CYAN,   opacity=0.7, nbinsx=30))
            fig_ttl.add_trace(go.Histogram(x=df["dttl"], name="Dst TTL", marker_color=ORANGE, opacity=0.7, nbinsx=30))
            fig_ttl.update_layout(**base_layout(xtitle="TTL Value", ytitle="Frequency"),
                                  barmode="overlay", height=280)
            st.plotly_chart(fig_ttl, use_container_width=True)

        with c2:
            st.markdown("<div class='section-header'>Packet Rate by Traffic Type</div>", unsafe_allow_html=True)
            rate_cap = df[df["rate"] < df["rate"].quantile(0.97)]
            fig_rt = go.Figure()
            for tt, col in [("Normal", GREEN), ("Attack", RED)]:
                fig_rt.add_trace(go.Histogram(
                    x=rate_cap[rate_cap["traffic_type"] == tt]["rate"],
                    name=tt, marker_color=col, opacity=0.65, nbinsx=40))
            fig_rt.update_layout(**base_layout(xtitle="Packets/s", ytitle="Frequency"),
                                 barmode="overlay", height=280)
            st.plotly_chart(fig_rt, use_container_width=True)

        st.markdown("<div class='section-header'>Mean Packet Size by Protocol</div>", unsafe_allow_html=True)
        mp = df.groupby("proto")[["smean", "dmean"]].mean().reset_index()
        mp = mp.sort_values("smean", ascending=False).head(12)
        fig_mp = go.Figure()
        fig_mp.add_trace(go.Bar(name="Src Mean", x=mp["proto"], y=mp["smean"], marker_color=CYAN,   opacity=0.8))
        fig_mp.add_trace(go.Bar(name="Dst Mean", x=mp["proto"], y=mp["dmean"], marker_color=PURPLE, opacity=0.8))
        fig_mp.update_layout(**base_layout(xtitle="Protocol", ytitle="Avg Packet Size (bytes)"),
                             barmode="group", height=270)
        st.plotly_chart(fig_mp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ATTACK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Attack" in page:
    st.markdown("<div style='font-family:Rajdhani,sans-serif;font-size:1.6rem;font-weight:700;color:#e6f1ff;letter-spacing:.05em;margin-bottom:1rem;'>Attack Analysis & Threat Intelligence</div>", unsafe_allow_html=True)
    atk = df[df["is_attack"] == 1]

    if len(atk) == 0:
        st.markdown("<div style='background:rgba(255,214,10,.07);border:1px solid #ffd60a;border-radius:6px;padding:.6rem 1rem;font-size:.82rem;color:#ffd60a;font-family:Share Tech Mono,monospace;'>⚠ No attacks found with current filters.</div>", unsafe_allow_html=True)
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(kpi("Attack Events", fmt_num(len(atk)),                         "in filtered dataset", "red"),    unsafe_allow_html=True)
        k2.markdown(kpi("Top Attack",    atk["attack_cat"].value_counts().idxmax()[:14], "most frequent",  "orange"), unsafe_allow_html=True)
        k3.markdown(kpi("Top Protocol",  atk["proto"].value_counts().idxmax(),       "most used",          "yellow"), unsafe_allow_html=True)
        k4.markdown(kpi("Avg Src Bytes", fmt_num(int(atk["sbytes"].mean())),         "per attack conn"),              unsafe_allow_html=True)
        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='section-header'>Attack Category Breakdown</div>", unsafe_allow_html=True)
            ac = atk["attack_cat"].value_counts().reset_index()
            ac.columns = ["Category", "Count"]
            ac = ac.sort_values("Count", ascending=True)
            fig_ac = go.Figure(go.Bar(
                x=ac["Count"], y=ac["Category"], orientation="h",
                marker=dict(color=[ATTACK_PAL.get(c, CYAN) for c in ac["Category"]], opacity=0.85),
                text=ac["Count"].apply(fmt_num), textposition="outside",
                textfont=dict(size=10, color=TEXT_COLOR),
                hovertemplate="<b>%{y}</b>: %{x:,}<extra></extra>",
            ))
            fig_ac.update_layout(**base_layout(xtitle="Count"), height=320, bargap=0.25)
            st.plotly_chart(fig_ac, use_container_width=True)

        with c2:
            st.markdown("<div class='section-header'>Attack Protocol Mix</div>", unsafe_allow_html=True)
            ap = atk["proto"].value_counts().head(8).reset_index()
            ap.columns = ["Protocol", "Count"]
            fig_ap = px.pie(ap, values="Count", names="Protocol",
                            color_discrete_sequence=PROTO_PAL, hole=0.55)
            fig_ap.update_traces(textinfo="percent+label",
                                 hovertemplate="<b>%{label}</b>: %{value:,}<extra></extra>")
            fig_ap.update_layout(**base_layout(), height=320, showlegend=False)
            st.plotly_chart(fig_ap, use_container_width=True)

        st.markdown("<div class='section-header'>Attack Category × Protocol Heatmap</div>", unsafe_allow_html=True)
        heat = atk.groupby(["attack_cat", "proto"]).size().reset_index(name="count")
        top_p = atk["proto"].value_counts().head(8).index.tolist()
        heat = heat[heat["proto"].isin(top_p)]
        hp = heat.pivot(index="attack_cat", columns="proto", values="count").fillna(0)
        fig_h = go.Figure(go.Heatmap(
            z=hp.values, x=hp.columns.tolist(), y=hp.index.tolist(),
            colorscale=[[0,"#090f18"],[0.3,"#0d2137"],[0.6,"#1e4976"],[0.85,"#00b8d9"],[1,"#00e5ff"]],
            text=hp.values.astype(int), texttemplate="%{text:,}",
            textfont=dict(size=9, color=TEXT_COLOR),
            hovertemplate="<b>%{y} × %{x}</b>: %{z:,}<extra></extra>",
        ))
        fig_h.update_layout(**base_layout(xtitle="Protocol", ytitle="Attack Category"), height=300)
        st.plotly_chart(fig_h, use_container_width=True)

        st.markdown("<div class='section-header'>Top Targeted Services</div>", unsafe_allow_html=True)
        sv = atk["service"].value_counts().head(10).reset_index()
        sv.columns = ["Service", "Count"]
        fig_sv = go.Figure(go.Bar(
            x=sv["Service"], y=sv["Count"],
            marker=dict(color=RED, opacity=0.8),
            text=sv["Count"].apply(fmt_num), textposition="outside",
            textfont=dict(size=10, color=TEXT_COLOR),
            hovertemplate="<b>%{x}</b>: %{y:,}<extra></extra>",
        ))
        fig_sv.update_layout(**base_layout(ytitle="Attack Count", xtitle="Service"),
                             height=260, bargap=0.3)
        st.plotly_chart(fig_sv, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ML PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif "ML Prediction" in page:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:1rem;margin-bottom:.5rem;'>
      <div style='font-family:Rajdhani,sans-serif;font-size:1.6rem;font-weight:700;color:#e6f1ff;letter-spacing:.05em;'>XGBoost Intrusion Detection Engine</div>
      <div style='font-family:Share Tech Mono,monospace;font-size:.7rem;color:#ffd60a;background:rgba(255,214,10,.08);border:1px solid rgba(255,214,10,.3);padding:.2rem .6rem;border-radius:3px;'>⚡ XGBoost</div>
    </div>
    <div style='font-size:.82rem;color:#7a8ba8;margin-bottom:1.5rem;'>
      Trained on UNSW-NB15 training set · Evaluated on 175,341-record testing set ·
      52 features (42 raw + 10 engineered) · sqrt-balanced class weights for improved minority-class detection.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔧 Training XGBoost models on UNSW-NB15… (~30–90 s on first load, cached afterwards)"):
        ml = train_models()

    tab_live, tab_perf, tab_feat = st.tabs(
        ["  Live Prediction", "  Model Performance", "  Feature Intelligence"])

    # ── Tab 1: Performance ────────────────────────────────────────────────────
    with tab_perf:
        st.markdown("<div class='section-header'>Binary Detection — Normal vs Attack</div>", unsafe_allow_html=True)
        bm = ml["bin_metrics"]
        bc1, bc2, bc3, bc4, bc5 = st.columns(5)
        for col, lbl, val, clr in [
            (bc1, "Accuracy",  f"{bm['accuracy']}%",  GREEN),
            (bc2, "Precision", f"{bm['precision']}%", CYAN),
            (bc3, "Recall",    f"{bm['recall']}%",    CYAN),
            (bc4, "F1 Score",  f"{bm['f1']}%",        ORANGE),
            (bc5, "ROC-AUC",   f"{bm['auc']}%",       YELLOW),
        ]:
            col.markdown(mbox(lbl, val, clr), unsafe_allow_html=True)

        st.markdown(f"<div style='font-size:.75rem;color:#7a8ba8;font-family:Share Tech Mono,monospace;margin:.5rem 0 1.5rem 0;'>n_estimators=400 · max_depth=7 · lr=0.08 · train time: {bm['train_time']}s</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='section-header'>Confusion Matrix — Binary</div>", unsafe_allow_html=True)
            cm_b = np.array(bm["cm"])
            fig_cmb = go.Figure(go.Heatmap(
                z=cm_b, x=["Normal", "Attack"], y=["Normal", "Attack"],
                colorscale=[[0,"#090f18"],[0.4,"#0d2137"],[0.7,"#1e4976"],[1,"#00e5ff"]],
                text=cm_b, texttemplate="%{text:,}", textfont=dict(size=14, color=TEXT_COLOR),
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>",
                showscale=False,
            ))
            fig_cmb.update_layout(**base_layout(xtitle="Predicted", ytitle="Actual"), height=280)
            st.plotly_chart(fig_cmb, use_container_width=True)

        with c2:
            st.markdown("<div class='section-header'>ROC Curve — Binary Detection</div>", unsafe_allow_html=True)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=bm["fpr"], y=bm["tpr"], mode="lines",
                name=f"XGBoost (AUC={bm['auc']}%)",
                line=dict(color=CYAN, width=2.5),
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color="#7a8ba8", width=1, dash="dash"),
                name="Random", showlegend=True))
            fig_roc.update_layout(
                **base_layout(legend=dict(title=""),
                              xtitle="False Positive Rate", ytitle="True Positive Rate"),
                height=280)
            st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("---")
        st.markdown("<div class='section-header'>Multi-Class Attack Classification — 10 Categories</div>", unsafe_allow_html=True)
        mm = ml["multi_metrics"]
        mc1, mc2, mc3, mc4 = st.columns(4)
        for col, lbl, val, clr in [
            (mc1, "Accuracy",    f"{mm['accuracy']}%",    GREEN),
            (mc2, "F1 Weighted", f"{mm['f1_weighted']}%", CYAN),
            (mc3, "F1 Macro",    f"{mm['f1_macro']}%",    ORANGE),
            (mc4, "Train Time",  f"{mm['train_time']}s",  YELLOW),
        ]:
            col.markdown(mbox(lbl, val, clr), unsafe_allow_html=True)

        st.markdown(f"<div style='font-size:.75rem;color:#7a8ba8;font-family:Share Tech Mono,monospace;margin:.5rem 0 1.5rem 0;'>n_estimators=600 · max_depth=10 · lr=0.05 · sqrt-balanced weights · colsample_bylevel=0.85</div>", unsafe_allow_html=True)

        c1, c2 = st.columns([1.15, 0.85])
        with c1:
            st.markdown("<div class='section-header'>Confusion Matrix — Multi-Class (row-normalised recall)</div>", unsafe_allow_html=True)
            cm_m    = np.array(mm["cm"])
            classes = ml["classes"]
            cm_norm = cm_m.astype(float) / (cm_m.sum(axis=1, keepdims=True) + 1e-9)
            fig_cmm = go.Figure(go.Heatmap(
                z=cm_norm, x=classes, y=classes,
                colorscale=[[0,"#090f18"],[0.3,"#0d2137"],[0.6,"#1e4976"],[0.85,"#00b8d9"],[1,"#00e5ff"]],
                text=cm_m, texttemplate="%{text:,}",
                textfont=dict(size=9, color=TEXT_COLOR),
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>",
                showscale=True,
                colorbar=dict(title="Recall", tickformat=".0%",
                              tickfont=dict(color=TEXT_COLOR, size=9)),
            ))
            # Pass tick overrides inside xaxis/yaxis — NOT as separate xaxis_title= kwargs
            fig_cmm.update_layout(
                **base_layout(
                    xtitle="Predicted", ytitle="Actual",
                    xaxis=dict(tickangle=-30, tickfont=dict(size=9)),
                    yaxis=dict(tickfont=dict(size=9))
                ),
                height=400)
            st.plotly_chart(fig_cmm, use_container_width=True)

        with c2:
            st.markdown("<div class='section-header'>Per-Class F1 Scores</div>", unsafe_allow_html=True)
            cr = mm["class_report"]
            f1_df = pd.DataFrame([
                {"Class": k, "F1": round(v["f1-score"] * 100, 1)}
                for k, v in cr.items()
                if k not in ("accuracy", "macro avg", "weighted avg")
            ]).sort_values("F1", ascending=True)
            fig_f1 = go.Figure(go.Bar(
                x=f1_df["F1"], y=f1_df["Class"], orientation="h",
                marker=dict(color=[ATTACK_PAL.get(c, CYAN) for c in f1_df["Class"]], opacity=0.85),
                text=[f"{v}%" for v in f1_df["F1"]], textposition="outside",
                textfont=dict(size=10, color=TEXT_COLOR),
                hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
            ))
            fig_f1.update_layout(**base_layout(xtitle="F1 Score (%)"),
                                 height=400, xaxis_range=[0, 115], bargap=0.25)
            st.plotly_chart(fig_f1, use_container_width=True)

        # Detailed per-class table
        st.markdown("<div class='section-header'>Detailed Per-Class Metrics</div>", unsafe_allow_html=True)
        rows_html = ""
        for cls in ml["classes"]:
            if cls in cr and cls not in ("accuracy", "macro avg", "weighted avg"):
                v   = cr[cls]
                clr = ATTACK_PAL.get(cls, CYAN)
                rows_html += (
                    f"<tr>"
                    f"<td style='padding:.4rem .8rem;font-family:Share Tech Mono,monospace;color:{clr};font-size:.8rem;'>{cls}</td>"
                    f"<td style='padding:.4rem .8rem;text-align:center;color:{TEXT_COLOR};'>{v['precision']*100:.1f}%</td>"
                    f"<td style='padding:.4rem .8rem;text-align:center;color:{TEXT_COLOR};'>{v['recall']*100:.1f}%</td>"
                    f"<td style='padding:.4rem .8rem;text-align:center;color:{TEXT_COLOR};'>{v['f1-score']*100:.1f}%</td>"
                    f"<td style='padding:.4rem .8rem;text-align:center;color:{TEXT_COLOR};'>{int(v['support']):,}</td>"
                    f"</tr>")
        st.markdown(f"""
        <table style='width:100%;border-collapse:collapse;'>
          <thead><tr style='border-bottom:1px solid #1e3a5f;'>
            <th style='text-align:left;padding:.4rem .8rem;color:#7a8ba8;font-size:.7rem;text-transform:uppercase;'>Class</th>
            <th style='text-align:center;padding:.4rem .8rem;color:#7a8ba8;font-size:.7rem;text-transform:uppercase;'>Precision</th>
            <th style='text-align:center;padding:.4rem .8rem;color:#7a8ba8;font-size:.7rem;text-transform:uppercase;'>Recall</th>
            <th style='text-align:center;padding:.4rem .8rem;color:#7a8ba8;font-size:.7rem;text-transform:uppercase;'>F1</th>
            <th style='text-align:center;padding:.4rem .8rem;color:#7a8ba8;font-size:.7rem;text-transform:uppercase;'>Support</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

    # ── Tab 2: Feature Intelligence ───────────────────────────────────────────
    with tab_feat:
        st.markdown("<div class='section-header'>Top 20 Features — Binary Detection Model</div>", unsafe_allow_html=True)
        fi_b  = pd.Series(ml["fi_bin"]).sort_values(ascending=False)
        fi_top = fi_b.head(20).sort_values(ascending=True)
        max_v  = fi_top.values.max()
        fig_fi = go.Figure(go.Bar(
            x=fi_top.values * 100, y=fi_top.index, orientation="h",
            marker=dict(
                color=[GREEN if v > max_v * .5 else CYAN if v > max_v * .2 else CYAN_DIM
                       for v in fi_top.values],
                opacity=0.85),
            text=[f"{v*100:.2f}%" for v in fi_top.values], textposition="outside",
            textfont=dict(size=10, color=TEXT_COLOR),
            hovertemplate="<b>%{y}</b>: %{x:.3f}%<extra></extra>",
        ))
        fig_fi.update_layout(**base_layout(xtitle="Importance Score (%)"), height=500, bargap=0.2)
        st.plotly_chart(fig_fi, use_container_width=True)

        c1, c2 = st.columns(2)
        fi_b10 = pd.Series(ml["fi_bin"]).sort_values(ascending=False).head(10).sort_values(ascending=True)
        fi_m10 = pd.Series(ml["fi_multi"]).sort_values(ascending=False).head(10).sort_values(ascending=True)

        with c1:
            st.markdown("<div class='section-header'>Binary Model — Top 10</div>", unsafe_allow_html=True)
            fig_fb = go.Figure(go.Bar(
                x=fi_b10.values * 100, y=fi_b10.index, orientation="h",
                marker=dict(color=CYAN, opacity=0.8),
                text=[f"{v*100:.2f}%" for v in fi_b10.values],
                textposition="outside", textfont=dict(size=9, color=TEXT_COLOR)))
            fig_fb.update_layout(**base_layout(xtitle="Importance (%)"), height=320, bargap=0.25)
            st.plotly_chart(fig_fb, use_container_width=True)

        with c2:
            st.markdown("<div class='section-header'>Multi-Class Model — Top 10</div>", unsafe_allow_html=True)
            fig_fm = go.Figure(go.Bar(
                x=fi_m10.values * 100, y=fi_m10.index, orientation="h",
                marker=dict(color=ORANGE, opacity=0.8),
                text=[f"{v*100:.2f}%" for v in fi_m10.values],
                textposition="outside", textfont=dict(size=9, color=TEXT_COLOR)))
            fig_fm.update_layout(**base_layout(xtitle="Importance (%)"), height=320, bargap=0.25)
            st.plotly_chart(fig_fm, use_container_width=True)

        st.markdown("<div class='section-header'>Key Feature Insights</div>", unsafe_allow_html=True)
        for feat, clr, name, desc in [
            ("sttl",            CYAN,   "Source TTL",               "Dominant feature. Attackers use non-standard TTL values to disguise traffic origin."),
            ("ct_state_ttl",    GREEN,  "CT State + TTL Count",     "Connections with same state/TTL pair. High count signals scanning or botnet patterns."),
            ("byte_ratio",      ORANGE, "Byte Ratio (engineered)",  "sbytes/(dbytes+1). Asymmetric byte flows reveal DoS floods and exfiltration."),
            ("swin",            YELLOW, "Source TCP Window",        "OS fingerprint via window size. Exploit kits craft specific values during setup."),
            ("ct_dst_sport_ltm",PURPLE, "CT Dst+Sport Count (LTM)", "Same dst IP + src port in last 100 conns. Strong indicator for port scans and DoS."),
        ]:
            st.markdown(f"""
            <div style='background:var(--bg-card);border:1px solid var(--border-glow);border-left:3px solid {clr};border-radius:6px;padding:.8rem 1rem;margin-bottom:.6rem;'>
              <span style='font-family:Share Tech Mono,monospace;color:{clr};font-size:.8rem;'>{feat}</span>
              <span style='font-family:Rajdhani,sans-serif;font-weight:600;color:var(--text-bright);margin-left:.8rem;font-size:.95rem;'>{name}</span>
              <div style='font-size:.8rem;color:var(--text-dim);margin-top:.3rem;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    # ── Tab 3: Live Prediction ────────────────────────────────────────────────
    with tab_live:
        st.markdown("""
        <div style='background:var(--bg-card);border:1px solid var(--border-glow);border-radius:8px;padding:1rem 1.2rem;margin-bottom:1.2rem;'>
          <div style='font-family:Rajdhani,sans-serif;font-weight:700;color:var(--cyan);font-size:1rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem;'>Live Network Traffic Analyzer</div>
          <div style='font-size:.82rem;color:var(--text-dim);line-height:1.6;'>
            Fill in the fields you can read from a network log, Wireshark capture, or firewall alert.
            All other technical features are automatically estimated from your inputs.
          </div>
        </div>""", unsafe_allow_html=True)

        col_in, col_res = st.columns([1.1, 1])

        with col_in:
            # ── Section 1: Connection Identity ───────────────────────────────
            st.markdown("<div class='section-header'>Connection Identity</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:.75rem;color:#7a8ba8;margin-bottom:.6rem;'>Readable from any firewall log, Wireshark, or SIEM alert</div>", unsafe_allow_html=True)
            r1c1, r1c2, r1c3 = st.columns(3)
            inp_proto   = r1c1.selectbox("Protocol",        ["tcp","udp","icmp","arp","ospf","igmp","other"],
                                         help="Transport layer protocol of the connection")
            inp_service = r1c2.selectbox("Application Service", ["-","http","https / ssl","dns","smtp","ftp","ftp-data","ssh","irc","pop3","radius","snmp","dhcp"],
                                         help="Application layer service detected")
            inp_state   = r1c3.selectbox("Connection State", ["FIN","CON","INT","REQ","RST","ECO","no"],
                                         help="FIN=closed normally, CON=connected, INT=in progress, REQ=request, RST=reset")

            st.markdown("<div style='height:.5rem;'></div>", unsafe_allow_html=True)

            # ── Section 2: Packet & Byte Counts ──────────────────────────────
            st.markdown("<div class='section-header'>Packet and Byte Counts</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:.75rem;color:#7a8ba8;margin-bottom:.6rem;'>Visible in Wireshark, NetFlow, or any packet capture tool</div>", unsafe_allow_html=True)
            r2c1, r2c2 = st.columns(2)
            inp_dur    = r2c1.number_input("Duration (seconds)",       min_value=0.0,  value=0.5,   step=0.1,   format="%.2f",
                                           help="How long the connection lasted")
            inp_sbytes = r2c1.number_input("Bytes Sent (source)",      min_value=0,    value=1024,  step=64,
                                           help="Total bytes sent from source to destination")
            inp_dbytes = r2c1.number_input("Bytes Received (dest)",    min_value=0,    value=512,   step=64,
                                           help="Total bytes sent from destination back to source")
            inp_spkts  = r2c2.number_input("Packets Sent (source)",    min_value=0,    value=8,     step=1,
                                           help="Number of packets sent by source")
            inp_dpkts  = r2c2.number_input("Packets Received (dest)",  min_value=0,    value=6,     step=1,
                                           help="Number of packets sent by destination")

            st.markdown("<div style='height:.5rem;'></div>", unsafe_allow_html=True)

            # ── Section 3: Network-Level Fields ──────────────────────────────
            st.markdown("<div class='section-header'>Network-Level Fields</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:.75rem;color:#7a8ba8;margin-bottom:.6rem;'>Readable from packet headers in Wireshark (IPv4 TTL field, TCP Window field)</div>", unsafe_allow_html=True)
            r3c1, r3c2 = st.columns(2)
            inp_sttl = r3c1.number_input("Source TTL",       min_value=0, max_value=255, value=64,
                                          help="IP Time-To-Live from source packets. Linux=64, Windows=128, Cisco=255")
            inp_dttl = r3c2.number_input("Destination TTL",  min_value=0, max_value=255, value=252,
                                          help="IP Time-To-Live from destination response packets")
            inp_swin = r3c1.number_input("Source TCP Window",     min_value=0, max_value=65535, value=8192,
                                          help="TCP receive window size advertised by source (seen in SYN packet)")
            inp_dwin = r3c2.number_input("Destination TCP Window",min_value=0, max_value=65535, value=8192,
                                          help="TCP receive window size advertised by destination (seen in SYN-ACK)")

            st.markdown("<div style='height:.5rem;'></div>", unsafe_allow_html=True)

            # ── Note about auto-computed fields ──────────────────────────────
            st.markdown("""
            <div style='background:rgba(0,229,255,.04);border:1px solid rgba(0,229,255,.15);
                        border-radius:6px;padding:.7rem 1rem;margin-bottom:.8rem;'>
              <div style='font-size:.75rem;color:#7a8ba8;line-height:1.6;'>
                <span style='color:#00b8d9;font-weight:600;'>AUTO-COMPUTED:</span>
                Rate, load, jitter, RTT, inter-packet times, mean packet sizes, TCP sequence numbers,
                and all connection-table counters are automatically estimated from your inputs above.
                You only need to fill what you can actually read from a network tool.
              </div>
            </div>""", unsafe_allow_html=True)

            predict_btn = st.button("  RUN PREDICTION", use_container_width=True, type="primary")

        with col_res:
            st.markdown("<div class='section-header'>Prediction Results</div>", unsafe_allow_html=True)

            if predict_btn:
                oe_m      = ml["oe"]
                feat_cols = ml["feat_cols"]

                # Map the service dropdown label back to dataset value
                svc_map = {"https / ssl": "ssl"}
                inp_service_raw = svc_map.get(inp_service, inp_service)

                # Encode categorical inputs
                cat_enc = oe_m.transform([[inp_proto, inp_service_raw, inp_state]])[0]

                # ── Auto-compute all remaining features from the user inputs ─
                # Rate: total packets / duration
                safe_dur  = max(inp_dur, 0.001)
                auto_rate = (inp_spkts + inp_dpkts) / safe_dur

                # Load: bytes * 8 / duration  (bits per second)
                auto_sload = (inp_sbytes * 8) / safe_dur
                auto_dload = (inp_dbytes * 8) / safe_dur

                # Mean packet size
                auto_smean = inp_sbytes / max(inp_spkts, 1)
                auto_dmean = inp_dbytes / max(inp_dpkts, 1)

                # Inter-packet time (ms): duration / packets * 1000
                auto_sinpkt = (safe_dur / max(inp_spkts, 1)) * 1000
                auto_dinpkt = (safe_dur / max(inp_dpkts, 1)) * 1000

                # Jitter: rough estimate — 5% of inter-packet time
                auto_sjit = auto_sinpkt * 0.05
                auto_djit = auto_dinpkt * 0.05

                # TCP timing: simple estimates
                auto_tcprtt = min(safe_dur * 0.1, 0.5)
                auto_synack = auto_tcprtt * 0.5
                auto_ackdat = auto_tcprtt * 0.5

                # Loss: assume 0 for normal input
                auto_sloss = 0
                auto_dloss = 0

                # TCP base sequence: use 0 as default
                auto_stcpb = 0
                auto_dtcpb = 0

                # Connection-table counts: reasonable defaults based on service
                # (if service is known busy protocol, bump counts slightly)
                busy_services = {"http", "dns", "ssl", "smtp"}
                ct_base = 3 if inp_service_raw in busy_services else 1
                auto_ct_srv_src      = ct_base
                auto_ct_state_ttl    = ct_base
                auto_ct_dst_ltm      = ct_base
                auto_ct_src_dport    = 1
                auto_ct_dst_sport    = 1
                auto_ct_dst_src_ltm  = ct_base
                auto_ct_src_ltm      = ct_base
                auto_ct_srv_dst      = ct_base

                # HTTP / FTP application fields
                auto_trans_depth      = 1 if inp_service_raw in {"http","ssl"} else 0
                auto_resp_body_len    = inp_dbytes if inp_service_raw in {"http","ssl"} else 0
                auto_is_ftp_login     = 1 if inp_service_raw in {"ftp","ftp-data"} else 0
                auto_ct_ftp_cmd       = 1 if inp_service_raw in {"ftp","ftp-data"} else 0
                auto_ct_flw_http_mthd = 1 if inp_service_raw in {"http","ssl"} else 0
                auto_is_sm_ips_ports  = 0

                # Build full row DataFrame
                row_dict = {
                    "dur":    inp_dur,
                    "proto":  cat_enc[0], "service": cat_enc[1], "state": cat_enc[2],
                    "spkts":  inp_spkts,  "dpkts":   inp_dpkts,
                    "sbytes": inp_sbytes, "dbytes":  inp_dbytes,
                    "rate":   auto_rate,
                    "sttl":   inp_sttl,   "dttl":    inp_dttl,
                    "sload":  auto_sload, "dload":   auto_dload,
                    "sloss":  auto_sloss, "dloss":   auto_dloss,
                    "sinpkt": auto_sinpkt,"dinpkt":  auto_dinpkt,
                    "sjit":   auto_sjit,  "djit":    auto_djit,
                    "swin":   inp_swin,   "stcpb":   auto_stcpb,
                    "dtcpb":  auto_dtcpb, "dwin":    inp_dwin,
                    "tcprtt": auto_tcprtt,"synack":  auto_synack,"ackdat": auto_ackdat,
                    "smean":  auto_smean, "dmean":   auto_dmean,
                    "trans_depth":       auto_trans_depth,
                    "response_body_len": auto_resp_body_len,
                    "ct_srv_src":        auto_ct_srv_src,
                    "ct_state_ttl":      auto_ct_state_ttl,
                    "ct_dst_ltm":        auto_ct_dst_ltm,
                    "ct_src_dport_ltm":  auto_ct_src_dport,
                    "ct_dst_sport_ltm":  auto_ct_dst_sport,
                    "ct_dst_src_ltm":    auto_ct_dst_src_ltm,
                    "is_ftp_login":      auto_is_ftp_login,
                    "ct_ftp_cmd":        auto_ct_ftp_cmd,
                    "ct_flw_http_mthd":  auto_ct_flw_http_mthd,
                    "ct_src_ltm":        auto_ct_src_ltm,
                    "ct_srv_dst":        auto_ct_srv_dst,
                    "is_sm_ips_ports":   auto_is_sm_ips_ports,
                }
                row_df = pd.DataFrame([row_dict])
                row_df = engineer(row_df)
                X_in   = row_df[feat_cols].values

                # Predictions
                bin_pred  = int(ml["xgb_bin"].predict(X_in)[0])
                bin_proba = ml["xgb_bin"].predict_proba(X_in)[0]
                bin_conf  = bin_proba[bin_pred] * 100

                mp_idx   = int(ml["xgb_multi"].predict(X_in)[0])
                mp_proba = ml["xgb_multi"].predict_proba(X_in)[0]
                mp_conf  = mp_proba[mp_idx] * 100
                att_type = ml["classes"][mp_idx]

                # Binary result
                if bin_pred == 1:
                    st.markdown(f"""
                    <div class='pred-box-attack'>
                      <div style='font-size:.7rem;font-family:Share Tech Mono,monospace;color:#ff2d55;letter-spacing:.15em;margin-bottom:.3rem;'>BINARY DETECTION</div>
                      <div class='pred-label' style='color:#ff2d55;'>⚠ ATTACK DETECTED</div>
                      <div class='pred-conf' style='color:#ff2d55;'>Confidence: {bin_conf:.1f}% &nbsp;|&nbsp; P(Attack)={bin_proba[1]*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='pred-box-normal'>
                      <div style='font-size:.7rem;font-family:Share Tech Mono,monospace;color:#00ff88;letter-spacing:.15em;margin-bottom:.3rem;'>BINARY DETECTION</div>
                      <div class='pred-label' style='color:#00ff88;'>✓ NORMAL TRAFFIC</div>
                      <div class='pred-conf' style='color:#00ff88;'>Confidence: {bin_conf:.1f}% &nbsp;|&nbsp; P(Normal)={bin_proba[0]*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Attack category result
                att_clr = ATTACK_PAL.get(att_type, CYAN)
                st.markdown(f"""
                <div style='background:rgba(0,0,0,.3);border:2px solid {att_clr};border-radius:10px;padding:1.5rem;text-align:center;'>
                  <div style='font-size:.7rem;font-family:Share Tech Mono,monospace;color:{att_clr};letter-spacing:.15em;margin-bottom:.3rem;'>ATTACK CLASSIFICATION</div>
                  <div style='font-family:Rajdhani,sans-serif;font-size:2rem;font-weight:700;color:{att_clr};letter-spacing:.1em;'>{att_type.upper()}</div>
                  <div style='font-family:Share Tech Mono,monospace;font-size:.82rem;color:{att_clr};margin-top:.4rem;'>Confidence: {mp_conf:.1f}%</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Probability distribution
                st.markdown("<div class='section-header'>Class Probability Distribution</div>", unsafe_allow_html=True)
                prob_df = pd.DataFrame({"Class": ml["classes"], "Prob": mp_proba * 100})
                prob_df = prob_df.sort_values("Prob", ascending=True)
                fig_prob = go.Figure(go.Bar(
                    x=prob_df["Prob"], y=prob_df["Class"], orientation="h",
                    marker=dict(color=[ATTACK_PAL.get(c, CYAN) for c in prob_df["Class"]], opacity=0.85),
                    text=[f"{v:.1f}%" for v in prob_df["Prob"]], textposition="outside",
                    textfont=dict(size=9, color=TEXT_COLOR),
                    hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
                ))
                fig_prob.update_layout(**base_layout(xtitle="Probability (%)"),
                                      height=340, xaxis_range=[0, 115], bargap=0.2)
                st.plotly_chart(fig_prob, use_container_width=True)

            else:
                st.markdown("""
                <div style='background:var(--bg-card);border:1px dashed var(--border-glow);border-radius:8px;padding:3rem 2rem;text-align:center;margin-top:1rem;'>
                  <div style='font-size:2rem;margin-bottom:1rem;'>⚡</div>
                  <div style='font-family:Rajdhani,sans-serif;font-size:1.1rem;color:var(--cyan-dim);font-weight:600;text-transform:uppercase;letter-spacing:.08em;'>Awaiting Input</div>
                  <div style='font-size:.8rem;color:var(--text-dim);margin-top:.5rem;'>Fill in the connection details on the left<br>then click RUN PREDICTION</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br><div class='section-header'>Quick Test Scenarios</div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:.78rem;color:#7a8ba8;margin-bottom:.8rem;'>Try these typical input combinations to test the model:</div>", unsafe_allow_html=True)
                for title, details in [
                    ("Normal Web Browse",
                     "Protocol: tcp  |  Service: http  |  State: FIN\nSrc Bytes: 1200  |  Dst Bytes: 8500  |  Src Pkts: 9  |  Dst Pkts: 12\nSrc TTL: 64  |  Dst TTL: 128  |  Duration: 0.8s"),
                    ("DoS Flood Attack",
                     "Protocol: udp  |  Service: –  |  State: CON\nSrc Bytes: 65000  |  Dst Bytes: 0  |  Src Pkts: 500  |  Dst Pkts: 0\nSrc TTL: 30  |  Dst TTL: 0  |  Duration: 0.1s"),
                    ("SSH Brute Force",
                     "Protocol: tcp  |  Service: ssh  |  State: REQ\nSrc Bytes: 200  |  Dst Bytes: 80  |  Src Pkts: 4  |  Dst Pkts: 3\nSrc TTL: 64  |  Dst TTL: 64  |  Duration: 0.05s"),
                    ("Port Scan (Recon)",
                     "Protocol: tcp  |  Service: –  |  State: RST\nSrc Bytes: 40  |  Dst Bytes: 0  |  Src Pkts: 1  |  Dst Pkts: 0\nSrc TTL: 64  |  Dst TTL: 0  |  Duration: 0.001s"),
                ]:
                    st.markdown(f"""
                    <div style='background:var(--bg-card);border:1px solid var(--border-glow);border-radius:6px;padding:.8rem 1rem;margin-bottom:.5rem;'>
                      <div style='font-family:Rajdhani,sans-serif;font-weight:700;color:var(--text-bright);font-size:.95rem;margin-bottom:.3rem;'>{title}</div>
                      <div style='font-size:.73rem;color:var(--text-dim);font-family:Share Tech Mono,monospace;white-space:pre-line;line-height:1.7;'>{details}</div>
                    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DATASET INFO
# ═══════════════════════════════════════════════════════════════════════════════
elif "Dataset" in page:
    st.markdown("<div style='font-family:Rajdhani,sans-serif;font-size:1.6rem;font-weight:700;color:#e6f1ff;letter-spacing:.05em;margin-bottom:1rem;'>Dataset Documentation</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='kpi-card' style='margin-bottom:1.5rem;'>
      <div style='font-family:Rajdhani,sans-serif;font-size:1.1rem;font-weight:700;color:#00e5ff;margin-bottom:.6rem;'>UNSW-NB15 Network Intrusion Dataset</div>
      <div style='font-size:.88rem;color:#ccd6f6;line-height:1.7;'>
        Created by the Australian Centre for Cyber Security (ACCS) at UNSW Canberra using IXIA PerfectStorm.
        Mixes real modern normal traffic with synthetic attack behaviours across 9 attack families.
        Widely used for NIDS benchmarking.
      </div>
      <div style='margin-top:.8rem;display:flex;gap:1rem;flex-wrap:wrap;'>
        <span style='font-family:Share Tech Mono,monospace;font-size:.72rem;background:rgba(0,229,255,.08);border:1px solid rgba(0,229,255,.25);padding:.2rem .6rem;border-radius:3px;color:#00e5ff;'>SOURCE: UNSW ACCS</span>
        <span style='font-family:Share Tech Mono,monospace;font-size:.72rem;background:rgba(0,255,136,.08);border:1px solid rgba(0,255,136,.25);padding:.2rem .6rem;border-radius:3px;color:#00ff88;'>TRAINING: 82,332 RECORDS</span>
        <span style='font-family:Share Tech Mono,monospace;font-size:.72rem;background:rgba(255,107,43,.08);border:1px solid rgba(255,107,43,.25);padding:.2rem .6rem;border-radius:3px;color:#ff6b2b;'>TESTING: 175,341 RECORDS</span>
        <span style='font-family:Share Tech Mono,monospace;font-size:.72rem;background:rgba(181,123,238,.08);border:1px solid rgba(181,123,238,.25);padding:.2rem .6rem;border-radius:3px;color:#b57bee;'>52 FEATURES (42 raw + 10 engineered)</span>
      </div>
    </div>""", unsafe_allow_html=True)

    FEATURES = {
        "🎯 Labels": {
            "label":      "Binary — 0 = Normal, 1 = Attack",
            "attack_cat": "Attack type: Normal / Generic / Exploits / Fuzzers / DoS / Reconnaissance / Analysis / Backdoor / Shellcode / Worms",
        },
        "🔧 Engineered Features": {
            "byte_ratio":    "sbytes / (dbytes+1) — asymmetric flow indicator",
            "pkt_ratio":     "spkts / (dpkts+1) — packet direction ratio",
            "load_ratio":    "sload / (dload+1) — bandwidth direction ratio",
            "total_bytes":   "sbytes + dbytes",
            "total_pkts":    "spkts + dpkts",
            "mean_pkt_diff": "smean − dmean — packet size asymmetry",
            "ttl_diff":      "|sttl − dttl| — TTL hop difference",
            "bytes_per_pkt": "total_bytes / (total_pkts+1)",
            "jit_sum":       "sjit + djit — combined jitter",
            "win_diff":      "|swin − dwin| — TCP window asymmetry",
        },
        "🌐 Network Identifiers": {
            "proto":   "Transaction protocol (tcp, udp, arp…)",
            "service": "Application layer service (http, dns, smtp…)",
            "state":   "Connection state (FIN / INT / CON / REQ…)",
        },
        "📦 Traffic Volume": {
            "dur":    "Connection duration (seconds)",
            "sbytes": "Source → destination bytes",
            "dbytes": "Destination → source bytes",
            "spkts":  "Source packet count",
            "dpkts":  "Destination packet count",
            "rate":   "Total packets / second",
            "sload":  "Source bits / second",
            "dload":  "Destination bits / second",
            "smean":  "Mean source packet size (bytes)",
            "dmean":  "Mean destination packet size (bytes)",
        },
        "⏱ Timing": {
            "sttl":   "Source TTL",
            "dttl":   "Destination TTL",
            "sinpkt": "Source inter-packet arrival time (ms)",
            "dinpkt": "Destination inter-packet arrival time (ms)",
            "sjit":   "Source jitter (ms)",
            "djit":   "Destination jitter (ms)",
            "tcprtt": "TCP connection round-trip time",
            "synack": "SYN → SYN-ACK delay",
            "ackdat": "SYN-ACK → ACK delay",
        },
        "🔌 TCP & Loss": {
            "swin":  "Source TCP window advertisement",
            "dwin":  "Destination TCP window advertisement",
            "stcpb": "Source TCP base sequence number",
            "dtcpb": "Destination TCP base sequence number",
            "sloss": "Source retransmitted / dropped packets",
            "dloss": "Destination retransmitted / dropped packets",
        },
        "🔗 Connection-Table Counts (last 100 connections)": {
            "ct_state_ttl":      "Same state + TTL pair",
            "ct_srv_src":        "Same service + source IP",
            "ct_srv_dst":        "Same service + destination IP",
            "ct_dst_ltm":        "Same destination IP",
            "ct_src_ltm":        "Same source IP",
            "ct_src_dport_ltm":  "Same source IP + destination port",
            "ct_dst_sport_ltm":  "Same destination IP + source port",
            "ct_dst_src_ltm":    "Same source-destination IP pair",
        },
        "🖥 Application": {
            "trans_depth":       "HTTP pipeline depth",
            "response_body_len": "HTTP response body size",
            "is_ftp_login":      "FTP authenticated successfully (bool)",
            "ct_ftp_cmd":        "FTP commands in flow",
            "ct_flw_http_mthd":  "HTTP method count (last 100)",
            "is_sm_ips_ports":   "Source == destination IP and ports (bool)",
        },
    }

    for section, fields in FEATURES.items():
        with st.expander(section, expanded=(section == "🎯 Labels")):
            rows = ""
            for col_name, desc in fields.items():
                rows += (f"<tr>"
                         f"<td style='font-family:Share Tech Mono,monospace;color:{CYAN};padding:.35rem .8rem;font-size:.8rem;white-space:nowrap;'>{col_name}</td>"
                         f"<td style='color:{TEXT_COLOR};padding:.35rem .8rem;font-size:.83rem;'>{desc}</td>"
                         f"</tr>")
            st.markdown(f"""
            <table style='width:100%;border-collapse:collapse;'>
              <thead><tr style='border-bottom:1px solid #112240;'>
                <th style='text-align:left;color:#7a8ba8;font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;padding:.3rem .8rem;'>Column</th>
                <th style='text-align:left;color:#7a8ba8;font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;padding:.3rem .8rem;'>Description</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:Share Tech Mono,monospace;font-size:.75rem;color:#7a8ba8;line-height:1.8;'>
      CITATION: Moustafa, N. &amp; Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion
      detection systems. Military Communications and Information Systems Conference (MilCIS), IEEE.<br>
      URL: https://research.unsw.edu.au/projects/unsw-nb15-dataset
    </div>""", unsafe_allow_html=True)
