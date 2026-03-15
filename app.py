import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
from src.config import (
    CITY_MAP, FEATURE_ALL, CSV_PATH, 
    UI_COLORS, APP_INFO, UNCERTAINTY_BANDS
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title=APP_INFO["title"],
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Sora', sans-serif !important; }}
.stApp {{ background-color: {UI_COLORS['background']}; color: {UI_COLORS['text_main']}; }}

/* Sidebar */
[data-testid="stSidebar"] {{ 
    background-color: {UI_COLORS['card']} !important; 
    border-right: 1px solid {UI_COLORS['border']} !important; 
}}
[data-testid="stSidebar"] label {{ color: {UI_COLORS['text_muted']} !important; font-size: 11px !important; font-weight: 600 !important; }}

/* Buttons */
.stButton > button {{ 
    background: {UI_COLORS['primary']} !important; 
    color: #FFFFFF !important; 
    border-radius: 8px !important; 
    border:none; 
    font-weight: 600;
    transition: 0.2s ease;
}}
.stButton > button:hover {{ 
    background: {UI_COLORS['primary_hover']} !important; 
    transform: translateY(-1px); 
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
}}

/* Cards & Metrics */
[data-testid="metric-container"] {{ 
    background: {UI_COLORS['card']} !important; 
    border: 1px solid {UI_COLORS['border']} !important; 
    border-radius: 12px; 
    padding: 20px; 
}}
[data-testid="stMetricValue"] {{ font-family: 'DM Mono', monospace !important; color: {UI_COLORS['text_main']}; }}

/* Info & Alerts */
[data-testid="stInfo"] {{ 
    background: {UI_COLORS['accent_bg']} !important; 
    border-left: 4px solid {UI_COLORS['primary']} !important; 
    color: {UI_COLORS['primary']} !important; 
}}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_training_data():
    try:
        df = pd.read_csv(CSV_PATH)
        df = df.dropna(subset=["cost_per_km_2023_musd"])
        df["tunnel_pct"] = df["tunnel_pct"].apply(lambda x: x / 100 if x > 1 else x)
        return df
    except FileNotFoundError: return None

@st.cache_resource
def load_assets():
    try:
        model  = joblib.load("rail_cost_model.pkl")
        memory = joblib.load("memory_package.pkl")
        return model, memory, None
    except Exception as e: return None, None, str(e)

def find_similars(df, country, city, length_km, tunnel_pct, n=8):
    d = df.copy()
    d["loc_match"]     = ((d["country"] == country) & (d["city"] == city)).astype(float)
    d["country_match"] = (d["country"] == country).astype(float)
    d["len_diff"]      = np.abs(np.log(d["length_km"] + 0.1) - np.log(length_km + 0.1))
    d["tun_diff"]      = np.abs(d["tunnel_pct"] - tunnel_pct)
    d["score"] = (d["loc_match"] * 3.0 + d["country_match"] * 1.5 + 
                  1 / (1 + d["len_diff"]) + 1 / (1 + d["tun_diff"] * 3))
    
    cols = ["score", "country", "city", "line", "start_year", "end_year", 
            "length_km", "tunnel_pct", "num_stations", "cost_per_km_2023_musd"]
    
    return d.nlargest(n, "score")[cols].reset_index(drop=True)

model, memory, load_error = load_assets()

def predict(country, city, length_km, tunnel_pct, num_stations,
            start_year, end_year, is_regional):
    gm   = memory["global_mean"]
    tm   = memory["train_medians"]
    mid  = (start_year + end_year) / 2.0
    st_n = num_stations if num_stations > 0 else tm["num_stations"]
    
    row  = {
        "country_te":      memory["country_te_map"].get(country, gm),
        "country_freq":    memory["country_freq_map"].get(country, 1),
        "city_te":         memory["city_te_map"].get(city, memory["country_te_map"].get(country, gm)),
        "city_freq":       memory["city_freq_map"].get(city, 1),
        "tunnel_pct":      tunnel_pct,
        "station_density": st_n / (length_km + 0.1),
        "log_length":      np.log(max(length_km, 0.1)),
        "is_regional_rail": 1.0 if is_regional else 0.0,
        "mid_year":        mid,
    }
    X = pd.DataFrame([row])[FEATURE_ALL]
    return np.exp(model.predict(X)[0])

def band(pred): 
    return pred * UNCERTAINTY_BANDS["lower"], pred * UNCERTAINTY_BANDS["upper"]

with st.sidebar:
    st.markdown("### Project Configuration")
    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)

    clist = sorted(CITY_MAP.keys()) + ["Other"]
    default_idx = clist.index("TR") if "TR" in clist else 0
    country_choice = st.selectbox("Country", clist, index=default_idx)
    
    if country_choice == "Other":
        country = st.text_input("New Country Name", "Unknown")
        city = st.text_input("New City Name", "New City")
        st.info("Using global baseline for unknown locations.")
    else:
        country = country_choice
        cities = CITY_MAP.get(country, []) + ["Other"]
        city_choice = st.selectbox("City", cities)
        
        if city_choice == "Other":
            city = st.text_input("New City Name", "New City")
            st.info(f"Using {country} national baseline for this city.")
        else:
            city = city_choice

    st.divider()
    length_km    = st.number_input("Line length (km)", 0.5, 250.0, 15.0, 0.5)
    tunnel_pct   = st.slider("Tunnel share (%)", 0, 100, 80, 5) / 100.0
    num_stations = st.number_input("Stations", 0, 150, 12, 1)

    st.divider()
    c1, c2 = st.columns(2)
    with c1: start_year = st.number_input("Start", 1950, 2040, 2022)
    with c2: end_year   = st.number_input("End",   1950, 2040, 2028)
    is_regional = st.checkbox("Regional Rail project")

    st.divider()
    actual_cost = st.number_input("Actual cost/km (M$)", 0.0, 10000.0, 0.0, 10.0)
    
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    run = st.button("Generate Prediction →")

st.markdown(f"""
<div style="padding-bottom:1.5rem; margin-bottom:2rem;">
  <span style="font-family:'DM Mono',monospace; font-size:11px; color:{UI_COLORS['primary']}; 
               background:{UI_COLORS['accent_bg']}; padding:4px 10px; border-radius:6px; letter-spacing:0.1em; font-weight:600;">
    {APP_INFO['engine_name']} · {APP_INFO['dataset_version']} · {APP_INFO['accuracy_metrics']}
  </span>
  <h1 style="font-size:36px; margin:12px 0 8px;">{APP_INFO['title']}</h1>
  <p style="color:{UI_COLORS['text_muted']}; font-size:15px; margin:0; max-width:850px;">
    Advanced machine learning framework for predicting transit infrastructure costs. 
    Global benchmark data normalized to 2023 USD.
  </p>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"**Asset Load Error:** {load_error}")
    st.stop()

if run and end_year >= start_year:
    pred = predict(country, city, length_km, tunnel_pct, num_stations, start_year, end_year, is_regional)
    lo, hi = band(pred)
    st.session_state["result"] = {
        "pred": pred, "lo": lo, "hi": hi, "country": country, "city": city,
        "length_km": length_km, "tunnel_pct": tunnel_pct, "num_stations": num_stations,
        "start_year": start_year, "end_year": end_year, "actual": actual_cost, "is_regional": is_regional
    }

r = st.session_state.get("result")

if not r:
    st.markdown(f"""
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:45vh; text-align:center;">
      <div style="font-size:72px; margin-bottom:20px; opacity:0.8;">🚉</div>
      <div style="font-size:16px; color:{UI_COLORS['text_muted']};">Select project parameters and click Estimate to begin analysis.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

pred, lo, hi = r["pred"], r["lo"], r["hi"]
total_b = pred * r["length_km"] / 1000.0

k1, k2, k3 = st.columns(3)
with k1: 
    st.metric(label="Predicted Cost / KM", value=f"${pred:,.0f} M")
with k2: 
    st.markdown(f"""
    <div style="padding-top: 0.2rem;">
        <div style="color:{UI_COLORS['text_muted']}; font-size:14px; margin-bottom:4px;">
            Probability Band
        </div>
        <div style="font-family:'DM Mono', monospace; font-size:35px; color:{UI_COLORS['text_main']}; line-height:1.2;">
            ${lo:,.0f} - ${hi:,.0f} M
        </div>
    </div>
    """, unsafe_allow_html=True)
with k3: 
    st.metric(label="Total Project Budget", value=f"${total_b:.2f} B")

if r["actual"] > 0:
    err_pct = (pred - r["actual"]) / r["actual"] * 100
    ec = "#10B981" if abs(err_pct) <= 30 else "#EF4444"
    st.markdown(f"""
    <div style="background:{UI_COLORS['card']}; border:1px solid {UI_COLORS['border']}; border-radius:16px; padding:24px; margin:1.5rem 0;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
        <span style="font-size:13px; font-weight:600; color:{UI_COLORS['text_main']};">Benchmark Verification</span>
        <span style="font-family:'DM Mono',monospace; font-size:13px; color:{ec}; font-weight:600;">Error: {err_pct:+.1f}%</span>
      </div>
      <div style="margin-bottom:15px;">
        <div style="display:flex; justify-content:space-between; font-size:12px; color:{UI_COLORS['text_muted']}; margin-bottom:6px;"><span>Model Prediction</span><span style="font-weight:600;">${pred:,.0f}M</span></div>
        <div style="background:#F3F4F6; height:8px; border-radius:4px;"><div style="background:{UI_COLORS['primary']}; width:{min(100, (pred/max(pred,r['actual']))*100)}%; height:100%; border-radius:4px;"></div></div>
      </div>
      <div>
        <div style="display:flex; justify-content:space-between; font-size:12px; color:{UI_COLORS['text_muted']}; margin-bottom:6px;"><span>Realized Project Cost</span><span style="font-weight:600;">${r['actual']:,.0f}M</span></div>
        <div style="background:#F3F4F6; height:8px; border-radius:4px;"><div style="background:{ec}; width:{min(100, (r['actual']/max(pred,r['actual']))*100)}%; height:100%; border-radius:4px;"></div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

left, right = st.columns([3, 2])
with left:
    st.markdown(f"<h4 style='font-size:13px; color:{UI_COLORS['text_muted']}; text-transform:uppercase; margin-bottom:15px;'>Market Context & Local Signals</h4>", unsafe_allow_html=True)
    gm = memory["global_mean"]
    city_te = memory["city_te_map"].get(r["city"], memory["country_te_map"].get(r["country"], gm))
    ctry_te = memory["country_te_map"].get(r["country"], gm)
    
    ctx = {
        "City Baseline Cost": f"${np.exp(city_te):,.0f} M$/km",
        "Country Baseline Cost": f"${np.exp(ctry_te):,.0f} M$/km",
        "Market Sample Density": f"{memory['city_freq_map'].get(r['city'], 0)} City Projects",
        "Inflation Window": f"Mid-Year {r['start_year'] + (r['end_year'] - r['start_year'])/2:.1f}"
    }
    st.table(pd.DataFrame(ctx.items(), columns=["Signal", "Status"]))

with right:
    st.markdown(f"<h4 style='font-size:13px; color:{UI_COLORS['text_muted']}; text-transform:uppercase; margin-bottom:15px;'>Project Summary</h4>", unsafe_allow_html=True)
    summ = {
        "Location": f"{r['city']}, {r['country']}",
        "Topology": f"{r['length_km']:.1f} km / {r['tunnel_pct']*100:.0f}% Tunnel",
        "Stations": f"{r['num_stations']} Units",
        "Duration": f"{r['start_year']} – {r['end_year']}",
        "Class": "Regional Rail" if r["is_regional"] else "Urban Transit"
    }
    st.table(pd.DataFrame(summ.items(), columns=["Field", "Value"]))

st.divider()
st.markdown(f"<h4 style='font-size:13px; color:{UI_COLORS['text_muted']}; text-transform:uppercase; margin-bottom:15px;'>Global Comparative Benchmarks</h4>", unsafe_allow_html=True)

train_df = load_training_data()
if train_df is not None:
    sim = find_similars(train_df, r["country"], r["city"], r["length_km"], r["tunnel_pct"])
    disp = sim.copy()

    disp["length_km"] = disp["length_km"].apply(lambda x: f"{x:.1f}")
    disp["Match %"] = ((disp["score"] / 6.5) * 100).clip(0, 100).round(0).astype(int).astype(str) + "%"
    disp["tunnel_pct"] = (disp["tunnel_pct"] * 100).round(0).astype(int).astype(str) + "%"
    disp["cost_per_km_2023_musd"] = disp["cost_per_km_2023_musd"].round(0).astype(int)
    
    display_cols = {
        "Match %": "Match", "country": "Country", "city": "City", 
        "line": "Line Name", "length_km": "Length (km)", 
        "tunnel_pct": "Tunnel", "cost_per_km_2023_musd": "Cost (M$/km)"
    }
    disp_final = disp[list(display_cols.keys())].rename(columns=display_cols)
    
    def highlight_match(row):
        if row["Country"] == r["country"] and row["City"] == r["city"]:
            return [f"background-color: {UI_COLORS['accent_bg']}; color: {UI_COLORS['primary']}; font-weight: 600"] * len(row)
        return [""] * len(row)

    st.dataframe(disp_final.style.apply(highlight_match, axis=1), use_container_width=True, hide_index=True)
else:
    st.info("Historical CSV not found for similarity analysis.")


st.divider()
st.markdown(f"<h4 style='font-size:13px; color:{UI_COLORS['text_muted']}; text-transform:uppercase; margin-bottom:15px;'>Interactive Cost Drivers</h4>", unsafe_allow_html=True)

gm, tm = memory["global_mean"], memory["train_medians"]
row_for_shap = {
    "country_te": memory["country_te_map"].get(r["country"], gm),
    "country_freq": memory["country_freq_map"].get(r["country"], 1),
    "city_te": memory["city_te_map"].get(r["city"], memory["country_te_map"].get(r["country"], gm)),
    "city_freq": memory["city_freq_map"].get(r["city"], 1),
    "tunnel_pct": r["tunnel_pct"], 
    "station_density": (r["num_stations"] if r["num_stations"] > 0 else tm["num_stations"]) / (r["length_km"] + 0.1), 
    "log_length": np.log(max(r["length_km"], 0.1)), 
    "is_regional_rail": 1.0 if r["is_regional"] else 0.0, 
    "mid_year": (r["start_year"] + r["end_year"]) / 2.0, 
}
X_inf = pd.DataFrame([row_for_shap])[FEATURE_ALL]

nice_names = {
    "country_te": "National Premium", "country_freq": "National Market Size",
    "city_te": "Local Premium", "city_freq": "Local Market Size",
    "tunnel_pct": "Tunnel Proportion", "station_density": "Station Density",
    "log_length": "Line Length (log)", "is_regional_rail": "Is Regional Rail?",
    "mid_year": "Inflation (Mid-Year)"
}
X_inf_display = X_inf.rename(columns=nice_names)

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_inf)
    sv = shap_values[0].values
    total_impact = np.sum(np.abs(sv))
    
    impact_df = pd.DataFrame({
        "Factor": X_inf_display.columns,
        "Raw Value": sv,
        "Impact Weight": np.abs(sv),
        "Signed Percent": (sv / total_impact) * 100 
    })
    
    impact_df = impact_df[impact_df["Impact Weight"] / total_impact > 0.01]
    
    impact_df["Direction"] = impact_df["Raw Value"].apply(lambda x: "Increases Cost 🔺" if x > 0 else "Decreases Cost 🔻")
    
    bar_df = impact_df.sort_values(by="Signed Percent", ascending=True)
    
    bar_df["Text Value"] = bar_df["Signed Percent"].apply(lambda x: f"{x:+.1f}%")
    
    fig_bar = px.bar(
        bar_df, x="Signed Percent", y="Factor", orientation='h', color="Direction",
        color_discrete_map={"Increases Cost 🔺": "#EF4444", "Decreases Cost 🔻": "#3B82F6"},
        text="Text Value"
    )
    
    fig_bar.update_traces(
        textposition='outside',
        marker=dict(line=dict(color=UI_COLORS['card'], width=1)),
        insidetextfont=dict(color=UI_COLORS['text_main'])
    )
    
    fig_bar.update_layout(
        showlegend=False,
        xaxis_title="Relative Impact on Final Cost (%)", 
        yaxis_title="",
        margin=dict(t=50, b=50, l=10, r=60), 
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        title=dict(text="Main Cost Drivers (Percentage Impact)", font=dict(size=14, color=UI_COLORS['text_main']), x=0.5)
    )
    
    fig_bar.update_yaxes(tickmode='linear', tickfont=dict(size=11, color=UI_COLORS['text_main']))
    fig_bar.update_xaxes(showgrid=True, gridcolor=UI_COLORS['border'], zeroline=True, zerolinecolor=UI_COLORS['text_muted'])
    st.plotly_chart(fig_bar, use_container_width=True)
    
except Exception as e:
    st.warning(f"Could not generate interactive explanation: {e}")

st.divider()
st.caption(APP_INFO["footer"])