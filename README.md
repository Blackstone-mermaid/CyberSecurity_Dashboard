# CyberShield-Intelligence

A Security Operations Center (SOC) style dashboard built with Streamlit for exploring, visualizing, and performing live machine learning inference on network traffic data from the UNSW-NB15 dataset. The application combines interactive threat analysis across four analytical pages with a fifth page that trains an XGBoost classifier on the training set and runs live attack prediction against the testing set.

---

## Live Demo

The application is deployed and accessible on Streamlit Community Cloud:

https://cybersecurityxdashboard.streamlit.app/

> Streamlit Community Cloud apps sleep after a period of inactivity. If you see a sleeping screen, click the wake button — the app typically loads within 30 seconds.

---

## About the Project

This project demonstrates how a well-known cybersecurity benchmark dataset can be made accessible through an interactive, SOC-style interface. Rather than working with the data in a static notebook, the entire analysis is surfaced through a dark-themed, multi-page dashboard with persistent sidebar filters, custom CSS theming, and a complete XGBoost machine learning pipeline with live inference.

The underlying data comes from the UNSW-NB15 dataset, created by the Australian Centre for Cyber Security (ACCS) at UNSW Canberra. It contains network connection records labeled across nine attack categories alongside normal traffic, and is widely used for benchmarking Network Intrusion Detection Systems (NIDS). This project uses both the official training and testing splits.

---

## Features

**Overview Page**

Five KPI cards summarize total records, total attacks, normal traffic volume, the dominant attack category, and total bytes transferred for the currently filtered view. Below the KPIs, a traffic composition donut chart and an attack category horizontal bar chart give an immediate picture of the dataset's distribution. A simulated hourly trend line (the dataset carries no native timestamps, so hours are randomly assigned at load time with a fixed seed — this is clearly annotated on the chart) shows how attack and normal traffic volumes vary across the day.

**Traffic Analysis Page**

Organized into three tabs. The Protocol and Service tab covers protocol distribution across the top 12 protocols, service distribution as a donut chart, and connection state breakdown. The Bytes and Packets tab covers source versus destination byte volumes by protocol, overlapping packet count histograms on a log scale, and a scatter plot of connection duration versus source bytes colored by traffic type. The Connection Behaviour tab covers TTL distribution for source and destination, packet rate histograms split by traffic type, inter-packet arrival time distributions for both directions on a log scale, and mean packet size by protocol.

**Attack Analysis Page**

Focused entirely on the attack subset of the filtered data. Displays four attack-specific KPIs (event count, top category, top protocol, average source bytes), a horizontal bar chart of attack categories with per-category color coding, a protocol mix donut chart for attacks only, an attack category-by-protocol heatmap, an hourly volume trend for the top five attack types, and a targeted-services bar chart.

**Dataset Info Page**

A reference page documenting all 45 columns of the UNSW-NB15 dataset, organized into logical groups — identifiers, network identifiers, traffic basics, byte and packet counts, timing features, TCP window and sequence features, HTTP application layer fields, connection-table features, boolean flags, and labels. Each field is explained in plain language. The page also includes the full academic citation.

**ML Prediction Page (XGBoost)**

The fifth page trains an XGBoost classifier using `UNSW_NB15_training-set.csv` and evaluates it against `UNSW_NB15_testing-set.csv` for live inference. The model is trained on the relevant numerical and categorical features from the dataset, and predictions are surfaced through the dashboard interface. This page makes scikit-learn and XGBoost active parts of the application rather than placeholder dependencies.

**Sidebar Filters**

All pages respond to four persistent filters in the sidebar: Protocol, Attack Category, Service, and Connection State. Any combination narrows the data shown across all charts and KPIs on the current page in real time.

---

## Dataset

**UNSW-NB15** — Network Intrusion Dataset  
Source: Australian Centre for Cyber Security, UNSW Canberra  
Training set: `UNSW_NB15_training-set.csv` — used for dashboard visualization and XGBoost model training  
Testing set: `UNSW_NB15_testing-set.csv` — used for live ML inference on the prediction page  
Total columns: 45  
Attack families: Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms  

Citation:
> Moustafa, N. & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems. Military Communications and Information Systems Conference (MilCIS), Canberra, Australia. IEEE.

Dataset homepage: https://research.unsw.edu.au/projects/unsw-nb15-dataset

---

## Tech Stack

| Component | Library / Version |
|---|---|
| Web framework | Streamlit >= 1.32.0 |
| Data manipulation | Pandas >= 2.0.0, NumPy >= 1.24.0 |
| Visualization | Plotly >= 5.18.0 |
| ML — gradient boosting | XGBoost >= 2.0.0 |
| ML — preprocessing and metrics | scikit-learn >= 1.3.0 |
| Fonts | Share Tech Mono, Rajdhani, Exo 2 (Google Fonts, loaded via CSS) |

---

## Project Structure

```
CyberSecurity_Dashboard/
├── app.py                          # Main application — all five pages, CSS, JS, ML pipeline
├── UNSW_NB15_training-set.csv      # Training data — visualization + XGBoost model training
├── UNSW_NB15_testing-set.csv       # Testing data — live inference on ML prediction page
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Local Setup

### Prerequisites

- Python 3.9 or higher
- pip

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/Blackstone-mermaid/CyberShield-Intelligence.git
cd CyberShield-Intelligence
```

**2. (Optional) Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the application**

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## Design Notes

The interface uses a custom dark color palette with CSS variables injected via `st.markdown`. Key implementation details:

- The sidebar collapse button is hidden and the sidebar is locked open using both CSS overrides and a MutationObserver JavaScript snippet, so navigation always stays visible regardless of screen size changes.
- All Plotly charts share a `base_layout` helper function that applies a consistent background, grid color, font family, and hover style across the entire app.
- The attack category color palette is semantically fixed (Generic = red, Reconnaissance = cyan, DoS = pink-red, etc.) so colors carry the same meaning across all charts on every page.
- Because the UNSW-NB15 training set has no timestamp column, a `sim_hour` field (0–23) is generated with a fixed random seed at load time. All charts that use this field are annotated with a visible warning to avoid misleading interpretation.
- Data loading uses `@st.cache_data` so the CSV is read and preprocessed only once per session, keeping the app responsive when switching pages or changing filters.

---

## Known Limitations

- Timestamps are simulated. Any time-based trend chart reflects randomly assigned hours, not actual capture times from the dataset.
- The dashboard loads the training split for visualization. The testing split is used exclusively on the ML prediction page.
- Streamlit Community Cloud instances sleep after inactivity, so the first load after a dormant period takes additional time.

---

## Acknowledgments

- UNSW-NB15 dataset created by the Australian Centre for Cyber Security (ACCS) at UNSW Canberra.
- Built using Streamlit's open-source framework, Plotly's interactive charting library, and the XGBoost gradient boosting library.
