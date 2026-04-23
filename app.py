"""
╔══════════════════════════════════════════════════════════╗
║   CYBERWATCH SOC DASHBOARD — UNSW-NB15 Dataset           ║
║   Built with Streamlit + Plotly                          ║
╚══════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberWatch SOC Dashboard",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Exo+2:wght@300;400;600;700&display=swap');

  /* Root palette */
  :root {
    --bg-void:    #050a0f;
    --bg-panel:   #090f18;
    --bg-card:    #0d1620;
    --border:     #112240;
    --border-glow:#1e3a5f;
    --cyan:       #00e5ff;
    --cyan-dim:   #00b8d9;
    --green:      #00ff88;
    --orange:     #ff6b2b;
    --red:        #ff2d55;
    --yellow:     #ffd60a;
    --text-main:  #ccd6f6;
    --text-dim:   #7a8ba8;
    --text-bright:#e6f1ff;
  }

  /* Base */
  .stApp { background: var(--bg-void); }
  html, body, .stApp { font-family: 'Exo 2', sans-serif; color: var(--text-main); }

  /* Kill Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; max-width: 100% !important; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border-glow);
  }
  [data-testid="stSidebar"] * { color: var(--text-main) !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stMultiSelect label,
  [data-testid="stSidebar"] .stRadio label { color: var(--cyan-dim) !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; }

  /* ── Style radio nav options as pill buttons ── */
  [data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.3rem; display: flex; flex-direction: column; }
  [data-testid="stSidebar"] div[role="radiogroup"] label {
    display: flex !important; align-items: center;
    padding: 0.45rem 0.8rem !important;
    border-radius: 6px !important;
    border: 1px solid var(--border-glow) !important;
    background: var(--bg-card) !important;
    color: var(--text-dim) !important;
    font-family: Rajdhani, sans-serif !important;
    font-size: 0.85rem !important; font-weight: 600 !important;
    text-transform: none !important; letter-spacing: 0.03em !important;
    cursor: pointer; transition: all 0.2s;
    width: 100% !important; box-sizing: border-box;
  }
  [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
    background: rgba(0,229,255,0.06) !important;
  }
  [data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked),
  [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
    background: rgba(0,229,255,0.10) !important;
  }
  /* Hide the default radio dot */
  [data-testid="stSidebar"] div[role="radiogroup"] span[data-testid="stMarkdownContainer"] { display:none; }
  [data-testid="stSidebar"] div[role="radiogroup"] div[data-baseweb="radio"] > div:first-child { display:none !important; }

  /* Inputs */
  [data-testid="stSidebar"] div[data-baseweb="select"] > div,
  [data-testid="stSidebar"] div[data-baseweb="input"] > div {
    background: var(--bg-card) !important;
    border-color: var(--border-glow) !important;
    color: var(--text-main) !important;
  }

  /* KPI Card */
  .kpi-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, #0a1628 100%);
    border: 1px solid var(--border-glow);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
  }
  .kpi-card::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  }
  .kpi-card:hover { border-color: var(--cyan); }
  .kpi-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 0.4rem;
  }
  .kpi-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.1rem; font-weight: 700;
    color: var(--cyan); line-height: 1.1;
  }
  .kpi-sub {
    font-size: 0.75rem; color: var(--text-dim);
    margin-top: 0.3rem;
  }
  .kpi-badge-red   { color: var(--red); }
  .kpi-badge-green { color: var(--green); }
  .kpi-badge-yellow{ color: var(--yellow); }
  .kpi-badge-orange{ color: var(--orange); }

  /* Section headers */
  .section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.05rem; font-weight: 600;
    color: var(--cyan-dim);
    text-transform: uppercase; letter-spacing: 0.1em;
    border-left: 3px solid var(--cyan);
    padding-left: 0.7rem;
    margin: 1.5rem 0 0.8rem 0;
  }

  /* Chart containers */
  .chart-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
  }
  .chart-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
  }

  /* Page nav */
  .nav-pill {
    display:inline-block; padding: 0.35rem 1rem;
    background: var(--bg-card); border: 1px solid var(--border-glow);
    border-radius: 20px; font-size:0.82rem; color: var(--text-dim);
    cursor:pointer; margin-right:0.5rem;
    font-family: 'Exo 2', sans-serif;
  }

  /* Dividers */
  hr { border-color: var(--border-glow) !important; margin: 1rem 0; }

  /* Alert boxes */
  .alert-danger {
    background: rgba(255,45,85,0.08);
    border: 1px solid var(--red);
    border-radius:6px; padding:0.6rem 1rem;
    font-size:0.82rem; color:var(--red);
    font-family:'Share Tech Mono', monospace;
  }
  .alert-warn {
    background: rgba(255,214,10,0.07);
    border: 1px solid var(--yellow);
    border-radius:6px; padding:0.6rem 1rem;
    font-size:0.82rem; color:var(--yellow);
    font-family:'Share Tech Mono', monospace;
  }

  /* Tabs override */
  .stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border-glow);
    gap: 0;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.85rem !important; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-dim) !important;
    padding: 0.5rem 1.2rem !important;
    background: transparent !important;
    border: none !important;
  }
  .stTabs [aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan) !important;
  }
  .stTabs [data-baseweb="tab-panel"] { background: transparent !important; padding-top: 1rem !important; }

  /* DataFrame */
  [data-testid="stDataFrame"] { border: 1px solid var(--border-glow); border-radius: 6px; }

  /* Metric delta */
  [data-testid="stMetricDelta"] { font-family: 'Share Tech Mono', monospace !important; }

  /* Expander */
  details { border: 1px solid var(--border-glow) !important; border-radius: 6px !important; background: var(--bg-card) !important; }
  summary { color: var(--cyan-dim) !important; font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width:5px; }
  ::-webkit-scrollbar-track { background: var(--bg-void); }
  ::-webkit-scrollbar-thumb { background: var(--border-glow); border-radius:2px; }

  /* ── Hide ONLY the collapse/expand (<<<) button, keep sidebar open ── */
  [data-testid="collapsedControl"]                             { display: none !important; }
  [data-testid="stSidebarCollapseButton"]                      { display: none !important; }
  [data-testid="stSidebarNavCollapseIcon"]                     { display: none !important; }
  button[data-testid="stSidebarCollapseButton"]                { display: none !important; }
  /* Target the collapse button specifically inside sidebar header */
  section[data-testid="stSidebar"] > div > div > button        { display: none !important; }
  /* Emotion cache classes used by recent Streamlit for the collapse btn */
  .st-emotion-cache-zq5wmm, .st-emotion-cache-1q1n0ol,
  .st-emotion-cache-79elbk, .st-emotion-cache-6qob1r           { display: none !important; }
  /* Lock sidebar always open and visible */
  section[data-testid="stSidebar"] {
    transform: none !important;
    width: 260px !important;
    min-width: 260px !important;
    visibility: visible !important;
    left: 0 !important;
  }
  section[data-testid="stSidebar"] > div:first-child { min-width: 260px !important; }
  /* Make sure main content doesn't cover sidebar */
  .main .block-container { margin-left: 0 !important; }

</style>
""", unsafe_allow_html=True)

# ─── KEEP SIDEBAR ALWAYS EXPANDED (JS) ──────────────────────────────────────
st.markdown("""
<script>
(function keepSidebarOpen() {
  function expand() {
    var collapsed = document.querySelector('[data-testid="collapsedControl"] button');
    if (collapsed) collapsed.click();
    document.querySelectorAll(
      '[data-testid="stSidebarCollapseButton"],[data-testid="collapsedControl"]'
    ).forEach(function(el){ el.style.setProperty('display','none','important'); });
  }
  expand();
  new MutationObserver(expand).observe(document.body, {childList:true, subtree:true});
})();
</script>
""", unsafe_allow_html=True)

# ─── PLOTLY THEME ─────────────────────────────────────────────────────────────
PLOT_BG      = "#090f18"
PAPER_BG     = "#090f18"
GRID_COLOR   = "#112240"
TEXT_COLOR   = "#ccd6f6"
CYAN         = "#00e5ff"
CYAN_DIM     = "#00b8d9"
GREEN        = "#00ff88"
ORANGE       = "#ff6b2b"
RED          = "#ff2d55"
YELLOW       = "#ffd60a"
PURPLE       = "#b57bee"

ATTACK_PALETTE = {
    "Normal":         GREEN,
    "Generic":        RED,
    "Exploits":       ORANGE,
    "Fuzzers":        YELLOW,
    "DoS":            "#ff4d6d",
    "Reconnaissance": CYAN,
    "Analysis":       PURPLE,
    "Backdoor":       "#ff6fd8",
    "Shellcode":      "#ff9f1c",
    "Worms":          "#caffbf",
}

PROTO_PALETTE = [CYAN, GREEN, ORANGE, YELLOW, PURPLE, RED, "#4cc9f0", "#f72585", "#7209b7", "#3a0ca3"]

def base_layout(title="", legend=None, xaxis=None, yaxis=None):
    def _m(base, override):
        return base if override is None else {**base, **override}
    return dict(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(family="Exo 2, sans-serif", color=TEXT_COLOR, size=11),
        title=dict(text=title, font=dict(color=CYAN_DIM, size=13, family="Rajdhani, sans-serif"),
                   x=0, xanchor="left", pad=dict(l=4)),
        xaxis=_m(dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
                      tickfont=dict(size=10), linecolor=GRID_COLOR), xaxis),
        yaxis=_m(dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
                      tickfont=dict(size=10), linecolor=GRID_COLOR), yaxis),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=_m(dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COLOR,
                       borderwidth=1, font=dict(size=10)), legend),
        hoverlabel=dict(bgcolor="#0d1620", bordercolor=CYAN, font=dict(color=TEXT_COLOR, size=11)),
    )

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("UNSW_NB15_training-set.csv")
    # Standardize attack_cat
    df["attack_cat"] = df["attack_cat"].str.strip().str.title()
    df["attack_cat"] = df["attack_cat"].fillna("Normal")
    df["is_attack"] = df["label"].astype(int)
    df["traffic_type"] = df["label"].map({0: "Normal", 1: "Attack"})
    # Simulated time index for trend (dataset has no timestamp)
    np.random.seed(42)
    n = len(df)
    df["sim_hour"] = np.random.randint(0, 24, n)
    df["sim_day"]  = np.random.randint(1,  8, n)
    df["sim_ts"]   = pd.to_datetime("2024-01-01") + pd.to_timedelta(df["sim_day"]*24 + df["sim_hour"], unit="h")
    return df

df_raw = load_data()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
      <div style='font-family:Share Tech Mono,monospace; font-size:1.3rem;
                  color:#00e5ff; letter-spacing:0.15em;'>CYBERWATCH</div>
      <div style='font-size:0.65rem; color:#7a8ba8; letter-spacing:0.2em;
                  text-transform:uppercase; margin-top:2px;'>SOC Intelligence Dashboard</div>
      <div style='height:1px; background:linear-gradient(90deg,transparent,#1e3a5f,transparent); margin:1rem 0 0 0;'></div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        ["  Overview", "  Traffic Analysis", "  Attack Analysis", "  Dataset Info"],
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown("<div style='font-size:0.7rem;color:#7a8ba8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;'>FILTERS</div>", unsafe_allow_html=True)

    proto_opts = ["All"] + sorted(df_raw["proto"].dropna().unique().tolist())
    sel_proto = st.selectbox("Protocol", proto_opts)

    attack_opts = ["All"] + sorted(df_raw["attack_cat"].dropna().unique().tolist())
    sel_attack = st.selectbox("Attack Category", attack_opts)

    service_opts = ["All"] + sorted(df_raw["service"].dropna().unique().tolist())
    sel_service = st.selectbox("Service", service_opts)

    state_opts = ["All"] + sorted(df_raw["state"].dropna().unique().tolist())
    sel_state = st.selectbox("State", state_opts)

    st.markdown("---")
    st.markdown(f"<div style='font-size:0.68rem;color:#7a8ba8;font-family:Share Tech Mono,monospace;'>RECORDS LOADED<br><span style='color:#00e5ff;font-size:1.1rem;'>{len(df_raw):,}</span></div>", unsafe_allow_html=True)

# ─── FILTER DATA ──────────────────────────────────────────────────────────────
df = df_raw.copy()
if sel_proto   != "All": df = df[df["proto"]      == sel_proto]
if sel_attack  != "All": df = df[df["attack_cat"] == sel_attack]
if sel_service != "All": df = df[df["service"]    == sel_service]
if sel_state   != "All": df = df[df["state"]      == sel_state]

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def fmt_num(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}K"
    return str(n)

def kpi(label, value, sub="", color="cyan"):
    badge_cls = {"red":"kpi-badge-red","green":"kpi-badge-green",
                 "yellow":"kpi-badge-yellow","orange":"kpi-badge-orange","cyan":""}.get(color,"")
    return f"""
    <div class='kpi-card'>
      <div class='kpi-label'>{label}</div>
      <div class='kpi-value {badge_cls}'>{value}</div>
      <div class='kpi-sub'>{sub}</div>
    </div>"""


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("""
    <div style='display:flex; align-items:center; gap:1rem; margin-bottom:1rem;'>
      <div style='font-family:Rajdhani,sans-serif; font-size:1.6rem; font-weight:700;
                  color:#e6f1ff; letter-spacing:0.05em;'>Security Operations Overview</div>
      <div style='font-family:Share Tech Mono,monospace; font-size:0.7rem; color:#00ff88;
                  background:rgba(0,255,136,0.08); border:1px solid rgba(0,255,136,0.3);
                  padding:0.2rem 0.6rem; border-radius:3px;'>● LIVE FEED</div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    total     = len(df)
    attacks   = df["is_attack"].sum()
    normals   = total - attacks
    ratio     = f"{attacks/total*100:.1f}%" if total else "—"
    top_att   = df[df["is_attack"]==1]["attack_cat"].value_counts().idxmax() if attacks else "—"
    total_bytes = df["sbytes"].sum() + df["dbytes"].sum()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(kpi("Total Records",    fmt_num(total),      "network events captured"),              unsafe_allow_html=True)
    c2.markdown(kpi("Total Attacks",    fmt_num(attacks),    f"{ratio} of all traffic", "red"),       unsafe_allow_html=True)
    c3.markdown(kpi("Normal Traffic",   fmt_num(normals),    f"{100-float(ratio[:-1]):.1f}% benign", "green"), unsafe_allow_html=True)
    c4.markdown(kpi("Top Attack",       top_att[:12] if top_att != "—" else "—", "dominant threat vector", "orange"), unsafe_allow_html=True)
    c5.markdown(kpi("Data Transferred", fmt_num(total_bytes)+"B", "total src+dst bytes"),             unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns([1, 1.6])

    # Pie chart
    with col_left:
        st.markdown("<div class='section-header'>Traffic Composition</div>", unsafe_allow_html=True)
        pie_data = df["traffic_type"].value_counts().reset_index()
        pie_data.columns = ["Type", "Count"]
        pie_data["_o"] = pie_data["Type"].map({"Normal": 0, "Attack": 1})
        pie_data = pie_data.sort_values("_o").drop(columns="_o").reset_index(drop=True)
        fig_pie = go.Figure(go.Pie(
            labels=pie_data["Type"], values=pie_data["Count"],
            hole=0.62,
            marker=dict(colors=[GREEN, RED], line=dict(color="#050a0f", width=2)),
            textinfo="percent", textfont=dict(size=12, color=TEXT_COLOR),
            hovertemplate="<b>%{label}</b><br>%{value:,} records<br>%{percent}<extra></extra>",
        ))
        fig_pie.add_annotation(text=f"<b>{total:,}</b><br><span style='font-size:10px'>TOTAL</span>",
                                x=0.5, y=0.5, showarrow=False,
                                font=dict(color=TEXT_COLOR, size=14))
        fig_pie.update_layout(**base_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)),
                               showlegend=True, height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Attack category bar
    with col_right:
        st.markdown("<div class='section-header'>Attack Category Distribution</div>", unsafe_allow_html=True)
        cat_counts = df["attack_cat"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        cat_counts = cat_counts.sort_values("Count", ascending=True)
        colors_bar = [ATTACK_PALETTE.get(c, CYAN) for c in cat_counts["Category"]]
        fig_bar = go.Figure(go.Bar(
            x=cat_counts["Count"], y=cat_counts["Category"],
            orientation="h",
            marker=dict(color=colors_bar, opacity=0.85,
                        line=dict(color=[c for c in colors_bar], width=0.5)),
            text=cat_counts["Count"].apply(fmt_num),
            textposition="outside", textfont=dict(size=10, color=TEXT_COLOR),
            hovertemplate="<b>%{y}</b>: %{x:,}<extra></extra>",
        ))
        fig_bar.update_layout(**base_layout(), height=300,
                               xaxis_title="", yaxis_title="",
                               bargap=0.25)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Traffic trend (simulated)
    st.markdown("<div class='section-header'>Simulated Traffic Trend (by Hour of Day)</div>", unsafe_allow_html=True)
    trend = df.groupby(["sim_hour","traffic_type"]).size().reset_index(name="count")
    fig_trend = px.line(trend, x="sim_hour", y="count", color="traffic_type",
                        color_discrete_map={"Normal": GREEN, "Attack": RED},
                        markers=True)
    fig_trend.update_traces(line=dict(width=2))
    fig_trend.update_layout(**base_layout(legend=dict(title=""),
                                      xaxis=dict(tickmode="linear", dtick=2)),
                             height=250,
                             xaxis_title="Hour of Day", yaxis_title="Event Count")
    fig_trend.add_annotation(text="⚠ Timestamps simulated — dataset has no native timestamps",
                              x=0, y=-0.22, xref="paper", yref="paper",
                              showarrow=False, font=dict(size=9, color="#7a8ba8"))
    st.plotly_chart(fig_trend, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRAFFIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Traffic" in page:
    st.markdown("<div style='font-family:Rajdhani,sans-serif; font-size:1.6rem; font-weight:700; color:#e6f1ff; letter-spacing:0.05em; margin-bottom:1rem;'>Network Traffic Analysis</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([" Protocol & Service", " Bytes & Packets", " Connection Behaviour"])

    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='section-header'>Protocol Distribution</div>", unsafe_allow_html=True)
            proto_c = df["proto"].value_counts().head(12).reset_index()
            proto_c.columns = ["Protocol", "Count"]
            fig_proto = go.Figure(go.Bar(
                x=proto_c["Protocol"], y=proto_c["Count"],
                marker=dict(color=PROTO_PALETTE[:len(proto_c)], opacity=0.85,
                            line=dict(color=PROTO_PALETTE[:len(proto_c)], width=0.5)),
                text=proto_c["Count"].apply(fmt_num), textposition="outside",
                textfont=dict(size=10, color=TEXT_COLOR),
                hovertemplate="<b>%{x}</b>: %{y:,}<extra></extra>",
            ))
            fig_proto.update_layout(**base_layout(), height=300, bargap=0.3,
                                    yaxis_title="Count", xaxis_title="Protocol")
            st.plotly_chart(fig_proto, use_container_width=True)

        with c2:
            st.markdown("<div class='section-header'>Service Distribution</div>", unsafe_allow_html=True)
            svc_c = df["service"].value_counts().head(10).reset_index()
            svc_c.columns = ["Service", "Count"]
            fig_svc = px.pie(svc_c, values="Count", names="Service",
                             color_discrete_sequence=[CYAN,GREEN,ORANGE,YELLOW,PURPLE,RED,
                                                      "#4cc9f0","#f72585","#7209b7","#3a0ca3"],
                             hole=0.5)
            fig_svc.update_traces(textposition="inside", textinfo="percent+label",
                                  hovertemplate="<b>%{label}</b>: %{value:,}<extra></extra>")
            fig_svc.update_layout(**base_layout(), height=300, showlegend=False)
            st.plotly_chart(fig_svc, use_container_width=True)

        st.markdown("<div class='section-header'>State Distribution</div>", unsafe_allow_html=True)
        state_c = df["state"].value_counts().reset_index()
        state_c.columns = ["State","Count"]
        fig_state = go.Figure(go.Bar(
            x=state_c["State"], y=state_c["Count"],
            marker=dict(color=CYAN, opacity=0.75,
                        line=dict(color=CYAN_DIM, width=0.5)),
            text=state_c["Count"].apply(fmt_num), textposition="outside",
            hovertemplate="<b>%{x}</b>: %{y:,}<extra></extra>",
        ))
        fig_state.update_layout(**base_layout(), height=260, bargap=0.35,
                                 yaxis_title="Count", xaxis_title="Connection State")
        st.plotly_chart(fig_state, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='section-header'>Source vs Destination Bytes (by Proto)</div>", unsafe_allow_html=True)
            bd = df.groupby("proto")[["sbytes","dbytes"]].sum().reset_index()
            bd = bd.sort_values("sbytes", ascending=False).head(10)
            fig_bd = go.Figure()
            fig_bd.add_trace(go.Bar(name="Source Bytes", x=bd["proto"], y=bd["sbytes"],
                                    marker_color=CYAN, opacity=0.8))
            fig_bd.add_trace(go.Bar(name="Dest Bytes",   x=bd["proto"], y=bd["dbytes"],
                                    marker_color=ORANGE, opacity=0.8))
            fig_bd.update_layout(**base_layout(), barmode="group", height=300,
                                  xaxis_title="Protocol", yaxis_title="Bytes")
            st.plotly_chart(fig_bd, use_container_width=True)

        with c2:
            st.markdown("<div class='section-header'>Packet Count Distribution (log scale)</div>", unsafe_allow_html=True)
            pkt_sample = df[df["spkts"] < df["spkts"].quantile(0.98)]
            fig_pkt = go.Figure()
            fig_pkt.add_trace(go.Histogram(x=pkt_sample["spkts"], name="Src Pkts",
                                            marker_color=CYAN, opacity=0.65, nbinsx=40))
            fig_pkt.add_trace(go.Histogram(x=pkt_sample["dpkts"], name="Dst Pkts",
                                            marker_color=GREEN, opacity=0.65, nbinsx=40))
            fig_pkt.update_layout(**base_layout(), barmode="overlay", height=300,
                                   xaxis_title="Packet Count", yaxis_title="Frequency",
                                   yaxis_type="log")
            st.plotly_chart(fig_pkt, use_container_width=True)

        st.markdown("<div class='section-header'>Duration vs Source Bytes (scatter sample)</div>", unsafe_allow_html=True)
        sample = df.sample(min(3000, len(df)), random_state=42)
        fig_sc = px.scatter(sample, x="dur", y="sbytes",
                            color="traffic_type",
                            color_discrete_map={"Normal": GREEN, "Attack": RED},
                            opacity=0.45, log_x=True, log_y=True,
                            hover_data=["proto","service","attack_cat"])
        fig_sc.update_traces(marker=dict(size=4))
        fig_sc.update_layout(**base_layout(legend=dict(title="")), height=300,
                              xaxis_title="Duration (log)", yaxis_title="Source Bytes (log)")
        st.plotly_chart(fig_sc, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='section-header'>TTL Distribution — Source vs Destination</div>", unsafe_allow_html=True)
            fig_ttl = go.Figure()
            fig_ttl.add_trace(go.Histogram(
                x=df["sttl"], name="Src TTL",
                marker_color=CYAN, opacity=0.7, nbinsx=30,
                hovertemplate="TTL %{x}: %{y:,}<extra></extra>"))
            fig_ttl.add_trace(go.Histogram(
                x=df["dttl"], name="Dst TTL",
                marker_color=ORANGE, opacity=0.7, nbinsx=30,
                hovertemplate="TTL %{x}: %{y:,}<extra></extra>"))
            fig_ttl.update_layout(**base_layout(legend=dict(title="")),
                                   barmode="overlay", height=280,
                                   xaxis_title="TTL Value", yaxis_title="Frequency")
            st.plotly_chart(fig_ttl, use_container_width=True)

        with c2:
            st.markdown("<div class='section-header'>Packet Rate Distribution (by Traffic Type)</div>", unsafe_allow_html=True)
            rate_cap = df[df["rate"] < df["rate"].quantile(0.97)]
            fig_rate = go.Figure()
            for ttype, col in [("Normal", GREEN), ("Attack", RED)]:
                subset = rate_cap[rate_cap["traffic_type"] == ttype]["rate"]
                fig_rate.add_trace(go.Histogram(
                    x=subset, name=ttype,
                    marker_color=col, opacity=0.65, nbinsx=40,
                    hovertemplate=f"{ttype} — rate %{{x:.1f}}: %{{y:,}}<extra></extra>"))
            fig_rate.update_layout(**base_layout(legend=dict(title="")),
                                    barmode="overlay", height=280,
                                    xaxis_title="Packets / Second", yaxis_title="Frequency")
            st.plotly_chart(fig_rate, use_container_width=True)

        st.markdown("<div class='section-header'>Inter-Packet Arrival Time — Src vs Dst (log scale)</div>", unsafe_allow_html=True)
        c1b, c2b = st.columns(2)
        with c1b:
            sinpkt_cap = df[df["sinpkt"] > 0]["sinpkt"]
            sinpkt_cap = sinpkt_cap[sinpkt_cap < sinpkt_cap.quantile(0.97)]
            fig_sinpkt = go.Figure()
            for ttype, col in [("Normal", GREEN), ("Attack", RED)]:
                vals = df[(df["traffic_type"] == ttype) & (df["sinpkt"] > 0)]["sinpkt"]
                vals = vals[vals < vals.quantile(0.97)]
                fig_sinpkt.add_trace(go.Histogram(
                    x=vals, name=ttype, marker_color=col, opacity=0.65, nbinsx=40))
            fig_sinpkt.update_layout(**base_layout(legend=dict(title="")),
                                      barmode="overlay", height=260,
                                      xaxis_title="Src Inter-Packet Time (ms)",
                                      yaxis_title="Count", yaxis_type="log")
            st.plotly_chart(fig_sinpkt, use_container_width=True)

        with c2b:
            fig_dinpkt = go.Figure()
            for ttype, col in [("Normal", GREEN), ("Attack", RED)]:
                vals = df[(df["traffic_type"] == ttype) & (df["dinpkt"] > 0)]["dinpkt"]
                vals = vals[vals < vals.quantile(0.97)]
                fig_dinpkt.add_trace(go.Histogram(
                    x=vals, name=ttype, marker_color=col, opacity=0.65, nbinsx=40))
            fig_dinpkt.update_layout(**base_layout(legend=dict(title="")),
                                      barmode="overlay", height=260,
                                      xaxis_title="Dst Inter-Packet Time (ms)",
                                      yaxis_title="Count", yaxis_type="log")
            st.plotly_chart(fig_dinpkt, use_container_width=True)

        st.markdown("<div class='section-header'>Mean Packet Size — Source vs Destination (by Protocol)</div>", unsafe_allow_html=True)
        mean_pkt = df.groupby("proto")[["smean","dmean"]].mean().reset_index()
        mean_pkt = mean_pkt.sort_values("smean", ascending=False).head(12)
        fig_mean = go.Figure()
        fig_mean.add_trace(go.Bar(name="Src Mean Pkt Size", x=mean_pkt["proto"], y=mean_pkt["smean"],
                                   marker_color=CYAN, opacity=0.8))
        fig_mean.add_trace(go.Bar(name="Dst Mean Pkt Size", x=mean_pkt["proto"], y=mean_pkt["dmean"],
                                   marker_color=PURPLE, opacity=0.8))
        fig_mean.update_layout(**base_layout(), barmode="group", height=270,
                                xaxis_title="Protocol", yaxis_title="Avg Packet Size (bytes)")
        st.plotly_chart(fig_mean, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ATTACK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Attack" in page:
    st.markdown("<div style='font-family:Rajdhani,sans-serif; font-size:1.6rem; font-weight:700; color:#e6f1ff; letter-spacing:0.05em; margin-bottom:1rem;'>Attack Analysis & Threat Intelligence</div>", unsafe_allow_html=True)

    attacks_df = df[df["is_attack"] == 1]
    if len(attacks_df) == 0:
        st.markdown("<div class='alert-warn'>⚠ No attacks found with current filters.</div>", unsafe_allow_html=True)
    else:
        total_att  = len(attacks_df)
        top_cat    = attacks_df["attack_cat"].value_counts().idxmax()
        top_proto  = attacks_df["proto"].value_counts().idxmax()
        avg_bytes  = int(attacks_df["sbytes"].mean())
        k1,k2,k3,k4 = st.columns(4)
        k1.markdown(kpi("Attack Events",  fmt_num(total_att), "in filtered dataset", "red"),    unsafe_allow_html=True)
        k2.markdown(kpi("Top Attack",     top_cat[:14],       "most frequent category","orange"),unsafe_allow_html=True)
        k3.markdown(kpi("Top Protocol",   top_proto,          "most used by attackers","yellow"),unsafe_allow_html=True)
        k4.markdown(kpi("Avg Src Bytes",  fmt_num(avg_bytes), "per attack connection"),         unsafe_allow_html=True)
        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='section-header'>Attack Category Breakdown</div>", unsafe_allow_html=True)
            acat = attacks_df["attack_cat"].value_counts().reset_index()
            acat.columns = ["Category","Count"]
            acat = acat.sort_values("Count", ascending=True)
            clrs = [ATTACK_PALETTE.get(c, CYAN) for c in acat["Category"]]
            fig_acat = go.Figure(go.Bar(
                x=acat["Count"], y=acat["Category"], orientation="h",
                marker=dict(color=clrs, opacity=0.85),
                text=acat["Count"].apply(fmt_num), textposition="outside",
                textfont=dict(color=TEXT_COLOR, size=10),
                hovertemplate="<b>%{y}</b>: %{x:,}<extra></extra>",
            ))
            fig_acat.update_layout(**base_layout(), height=320,
                                    xaxis_title="Count", yaxis_title="", bargap=0.25)
            st.plotly_chart(fig_acat, use_container_width=True)

        with c2:
            st.markdown("<div class='section-header'>Attack Protocol Mix</div>", unsafe_allow_html=True)
            apr = attacks_df["proto"].value_counts().head(8).reset_index()
            apr.columns = ["Protocol","Count"]
            fig_apr = px.pie(apr, values="Count", names="Protocol",
                             color_discrete_sequence=PROTO_PALETTE, hole=0.55)
            fig_apr.update_traces(textinfo="percent+label",
                                   hovertemplate="<b>%{label}</b>: %{value:,}<extra></extra>")
            fig_apr.update_layout(**base_layout(legend=dict(orientation="v", x=1.0)),
                                   height=320, showlegend=True)
            st.plotly_chart(fig_apr, use_container_width=True)

        st.markdown("<div class='section-header'>Attack Category × Protocol Heatmap</div>", unsafe_allow_html=True)
        heat = attacks_df.groupby(["attack_cat","proto"]).size().reset_index(name="count")
        top_protos = attacks_df["proto"].value_counts().head(8).index.tolist()
        heat = heat[heat["proto"].isin(top_protos)]
        heat_pivot = heat.pivot(index="attack_cat", columns="proto", values="count").fillna(0)
        fig_heat = go.Figure(go.Heatmap(
            z=heat_pivot.values, x=heat_pivot.columns.tolist(), y=heat_pivot.index.tolist(),
            colorscale=[[0,"#090f18"],[0.3,"#0d2137"],[0.6,"#1e4976"],[0.85,"#00b8d9"],[1,"#00e5ff"]],
            text=heat_pivot.values.astype(int),
            texttemplate="%{text:,}", textfont=dict(size=9, color=TEXT_COLOR),
            hovertemplate="<b>%{y} × %{x}</b>: %{z:,}<extra></extra>",
        ))
        fig_heat.update_layout(**base_layout(), height=300,
                                xaxis_title="Protocol", yaxis_title="Attack Category")
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("<div class='section-header'>Attack Volume by Simulated Hour</div>", unsafe_allow_html=True)
        trend = attacks_df.groupby(["sim_hour","attack_cat"]).size().reset_index(name="count")
        top5 = attacks_df["attack_cat"].value_counts().head(5).index.tolist()
        trend5 = trend[trend["attack_cat"].isin(top5)]
        fig_t = px.line(trend5, x="sim_hour", y="count", color="attack_cat",
                        color_discrete_sequence=[RED,ORANGE,YELLOW,CYAN,PURPLE],
                        markers=True)
        fig_t.update_traces(line=dict(width=2))
        fig_t.update_layout(**base_layout(legend=dict(title=""),
                                  xaxis=dict(dtick=2)),
                             height=260,
                             xaxis_title="Hour", yaxis_title="Events")
        st.plotly_chart(fig_t, use_container_width=True)

        st.markdown("<div class='section-header'>Top Targeted Services</div>", unsafe_allow_html=True)
        svc_att = attacks_df["service"].value_counts().head(10).reset_index()
        svc_att.columns = ["Service","Count"]
        fig_svc_att = go.Figure(go.Bar(
            x=svc_att["Service"], y=svc_att["Count"],
            marker=dict(color=RED, opacity=0.8),
            text=svc_att["Count"].apply(fmt_num), textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=10),
            hovertemplate="<b>%{x}</b>: %{y:,}<extra></extra>",
        ))
        fig_svc_att.update_layout(**base_layout(), height=260, bargap=0.3,
                                    yaxis_title="Attack Count", xaxis_title="Service")
        st.plotly_chart(fig_svc_att, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DATASET INFO
# ═══════════════════════════════════════════════════════════════════════════════
elif "Dataset" in page:
    st.markdown("<div style='font-family:Rajdhani,sans-serif; font-size:1.6rem; font-weight:700; color:#e6f1ff; letter-spacing:0.05em; margin-bottom:1rem;'>Dataset Documentation</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='kpi-card' style='margin-bottom:1.5rem;'>
      <div style='font-family:Rajdhani,sans-serif;font-size:1.1rem;font-weight:700;color:#00e5ff;margin-bottom:0.6rem;'>
        UNSW-NB15 Network Intrusion Dataset
      </div>
      <div style='font-size:0.88rem;color:#ccd6f6;line-height:1.7;'>
        The UNSW-NB15 dataset was created by the Australian Centre for Cyber Security (ACCS) 
        at UNSW Canberra. It contains 2,540,044 records of raw network packets generated by 
        the IXIA PerfectStorm tool in a cyber range infrastructure. The dataset mixes real 
        modern normal activities and synthetic contemporary attack behaviours across 9 attack 
        families. It is widely used for benchmarking network intrusion detection systems (NIDS).
      </div>
      <div style='margin-top:0.8rem; display:flex; gap:1rem; flex-wrap:wrap;'>
        <span style='font-family:Share Tech Mono,monospace;font-size:0.72rem;background:rgba(0,229,255,0.08);
               border:1px solid rgba(0,229,255,0.25);padding:0.2rem 0.6rem;border-radius:3px;color:#00e5ff;'>
          SOURCE: UNSW Canberra ACCS
        </span>
        <span style='font-family:Share Tech Mono,monospace;font-size:0.72rem;background:rgba(0,255,136,0.08);
               border:1px solid rgba(0,255,136,0.25);padding:0.2rem 0.6rem;border-radius:3px;color:#00ff88;'>
          DATASET: 82,332 RECORDS
        </span>
        <span style='font-family:Share Tech Mono,monospace;font-size:0.72rem;background:rgba(255,107,43,0.08);
               border:1px solid rgba(255,107,43,0.25);padding:0.2rem 0.6rem;border-radius:3px;color:#ff6b2b;'>
          FEATURES: 45 COLUMNS
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    FEATURES = {
        " Identifiers": {
            "id":    "Unique row identifier",
        },
        " Network Identifiers (full dataset)": {
            "srcip":  "Source IP address",
            "sport":  "Source port number",
            "dstip":  "Destination IP address",
            "dsport": "Destination port number",
        },
        " Traffic Basics": {
            "proto":   "Transaction protocol (tcp, udp, arp, etc.)",
            "service": "Application layer service (-: not determined, dns, http, smtp, etc.)",
            "state":   "Connection state (FIN, INT, CON, REQ, ACC, etc.)",
            "dur":     "Total duration of the connection record (seconds)",
        },
        " Byte & Packet Counts": {
            "sbytes":  "Source to destination bytes",
            "dbytes":  "Destination to source bytes",
            "spkts":   "Source to destination packet count",
            "dpkts":   "Destination to source packet count",
            "rate":    "Total packets per second (bidirectional)",
            "sload":   "Source bits per second",
            "dload":   "Destination bits per second",
            "smean":   "Mean of the source packet size",
            "dmean":   "Mean of the destination packet size",
        },
        " Timing Features": {
            "sttl":    "Source to destination time-to-live value",
            "dttl":    "Destination to source TTL value",
            "sinpkt":  "Source inter-packet arrival time (ms)",
            "dinpkt":  "Destination inter-packet arrival time (ms)",
            "sjit":    "Source jitter (ms)",
            "djit":    "Destination jitter (ms)",
            "tcprtt":  "TCP connection setup round-trip time (synack + ackdat)",
            "synack":  "Time between SYN and SYN-ACK packets",
            "ackdat":  "Time between SYN-ACK and ACK packets",
        },
        " TCP Window & Sequence": {
            "swin":    "Source TCP window advertisement size",
            "dwin":    "Destination TCP window advertisement size",
            "stcpb":   "Source TCP base sequence number",
            "dtcpb":   "Destination TCP base sequence number",
            "sloss":   "Source packets retransmitted or dropped",
            "dloss":   "Destination packets retransmitted or dropped",
        },
        " HTTP / Application Layer": {
            "trans_depth":       "Number of pipelined HTTP request/response transactions",
            "response_body_len": "Actual uncompressed content size of HTTP response body",
        },
        " Connection-Table Features": {
            "ct_state_ttl":     "No. connections with same src-dst TTL and state (last 100)",
            "ct_flw_http_mthd": "No. flows using same HTTP method (last 100, if applicable)",
            "ct_ftp_cmd":       "No. FTP session commands in the flow (if FTP)",
            "ct_srv_src":       "No. connections with same service and source IP (last 100)",
            "ct_srv_dst":       "No. connections with same service and destination IP (last 100)",
            "ct_dst_ltm":       "No. connections to same destination IP (last 100)",
            "ct_src_ltm":       "No. connections from same source IP (last 100)",
            "ct_src_dport_ltm": "No. connections from same source IP and dest port (last 100)",
            "ct_dst_sport_ltm": "No. connections to same destination IP and src port (last 100)",
            "ct_dst_src_ltm":   "No. connections with same src-dst IP pair (last 100)",
        },
        " Boolean Flags": {
            "is_ftp_login":     "1 if FTP session authenticated with valid user/password",
            "is_sm_ips_ports":  "1 if source == destination IP and ports are equal",
        },
        " Labels": {
            "attack_cat": "Attack category name: Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms",
            "label":      "Binary label — 0 = Normal traffic, 1 = Attack traffic",
        },
    }

    for section, fields in FEATURES.items():
        with st.expander(section, expanded=(section == "🎯 Labels")):
            rows = ""
            for col, desc in fields.items():
                rows += f"<tr><td style='font-family:Share Tech Mono,monospace;color:{CYAN};padding:0.35rem 0.8rem;font-size:0.8rem;white-space:nowrap;'>{col}</td><td style='color:{TEXT_COLOR};padding:0.35rem 0.8rem;font-size:0.83rem;'>{desc}</td></tr>"
            st.markdown(f"""
            <table style='width:100%;border-collapse:collapse;'>
              <thead>
                <tr style='border-bottom:1px solid #112240;'>
                  <th style='text-align:left;color:#7a8ba8;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;padding:0.3rem 0.8rem;'>Column</th>
                  <th style='text-align:left;color:#7a8ba8;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;padding:0.3rem 0.8rem;'>Description</th>
                </tr>
              </thead>
              <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#7a8ba8;line-height:1.8;'>
    CITATION: Moustafa, N. & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network 
    intrusion detection systems. Military Communications and Information Systems Conference 
    (MilCIS), Canberra, Australia. IEEE.<br>
    DATASET URL: https://research.unsw.edu.au/projects/unsw-nb15-dataset
    </div>
    """, unsafe_allow_html=True)