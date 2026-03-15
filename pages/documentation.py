import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="How It Works — Rail Cost Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_resource
def load_assets():
    model   = joblib.load("rail_cost_model.pkl")
    memory  = joblib.load("memory_package.pkl")
    features = joblib.load("feature_names.pkl")
    return model, memory, features

model, memory, features = load_assets()

fi_dict = dict(zip(features, model.feature_importances_))

nice_names = {
    "country_te":       "National Cost Premium",
    "country_freq":     "National Market Size",
    "city_te":          "Local Cost Premium",
    "city_freq":        "Local Market Size",
    "tunnel_pct":       "Tunnel Proportion",
    "station_density":  "Station Density",
    "log_length":       "Line Length (log)",
    "is_regional_rail": "Is Regional Rail?",
    "mid_year":         "Inflation Window (Mid-Year)",
}

feature_descriptions = {
    "city_te":          "Bayesian-smoothed average historical cost for this city. The single most predictive signal — local geology, labor, and procurement practices are all embedded here.",
    "tunnel_pct":       "Share of the line built underground. Underground construction typically costs 3–8× more per km than at-grade or elevated, making this a dominant cost lever.",
    "station_density":  "Stations per km of line. Dense station spacing (e.g. urban metros) drives up civil works, fit-out, and system complexity per km.",
    "mid_year":         "Midpoint year of construction. Captures decade-level inflation trends in global construction materials and labor.",
    "log_length":       "Log-transformed total length. Longer lines gain economies of scale but also face greater geological and logistical variance.",
    "city_freq":        "How many prior projects exist for this city in the training set. More data → Bayesian encoder trusts local estimate more than global prior.",
    "country_te":       "Country-level cost baseline after Bayesian smoothing. Captures national regulatory, labor, and procurement environment.",
    "country_freq":     "Number of projects from this country in the training dataset. Determines how much the encoder relies on national vs. global prior.",
    "is_regional_rail": "Binary flag distinguishing commuter/regional rail (longer station spacing, lower civil cost) from urban metro.",
}

params = model.get_params()

country_costs = {
    k: np.exp(v)
    for k, v in memory["country_te_map"].items()
}
country_n = memory["country_freq_map"]

city_costs = {
    k: np.exp(v)
    for k, v in memory["city_te_map"].items()
}
city_n = memory["city_freq_map"]

global_baseline = np.exp(memory["global_mean"])
median_stations = memory["train_medians"]["num_stations"]
median_mid_year = memory["train_medians"]["mid_year"]

cv_folds = [
    {"fold": "Fold 1", "r2": 0.4927, "mae": 0.2926},
    {"fold": "Fold 2", "r2": 0.5059, "mae": 0.2795},
    {"fold": "Fold 3", "r2": 0.5062, "mae": 0.3082},
    {"fold": "Fold 4", "r2": 0.5385, "mae": 0.3048},
    {"fold": "Fold 5", "r2": 0.4962, "mae": 0.2846},
]
mean_r2   = np.mean([f["r2"]  for f in cv_folds])
mean_mae  = np.mean([f["mae"] for f in cv_folds])

country_perf = [
    {"country": "CN", "n": 440, "r2": 0.502, "mdape": 12.8,  "bias": 2.0},
    {"country": "JP", "n": 55,  "r2": 0.586, "mdape": 23.6,  "bias": -8.3},
    {"country": "RU", "n": 54,  "r2": 0.115, "mdape": 21.4,  "bias": -3.5},
    {"country": "IN", "n": 38,  "r2": -0.035,"mdape": 23.7,  "bias": -1.6},
    {"country": "IT", "n": 34,  "r2": 0.463, "mdape": 33.8,  "bias": 10.0},
    {"country": "TR", "n": 29,  "r2": -0.160,"mdape": 31.7,  "bias": 9.5},
    {"country": "CA", "n": 27,  "r2": 0.312, "mdape": 51.1,  "bias": -10.0},
    {"country": "DE", "n": 23,  "r2": 0.203, "mdape": 42.5,  "bias": -9.0},
    {"country": "US", "n": 21,  "r2": 0.497, "mdape": 30.0,  "bias": -17.1},
    {"country": "FR", "n": 17,  "r2": 0.215, "mdape": 31.6,  "bias": 22.3},
    {"country": "TW", "n": 17,  "r2": -0.560,"mdape": 43.2,  "bias": -20.4},
    {"country": "KR", "n": 16,  "r2": -0.223,"mdape": 16.2,  "bias": 10.6},
    {"country": "UK", "n": 16,  "r2": 0.143, "mdape": 49.5,  "bias": 13.0},
    {"country": "ES", "n": 16,  "r2": -0.443,"mdape": 28.8,  "bias": 11.6},
    {"country": "HK", "n": 14,  "r2": 0.330, "mdape": 40.3,  "bias": -24.5},
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Sora', sans-serif !important;
    background-color: #FAFAFA;
    color: #111111;
}
.section-divider {
    border: none;
    border-top: 1px solid #E5E5E5;
    margin: 2.5rem 0;
}
.tag {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: #EEF2FF;
    color: #4F46E5;
    padding: 3px 10px;
    border-radius: 5px;
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 12px;
    padding: 22px 24px;
}
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    color: #888;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'DM Mono', monospace;
    font-size: 28px;
    font-weight: 500;
    color: #111;
    line-height: 1.2;
}
.kpi-sub {
    font-size: 11px;
    color: #AAA;
    margin-top: 4px;
}
.feature-card {
    background: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-left: 3px solid #4F46E5;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.feature-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.feature-name {
    font-size: 14px;
    font-weight: 600;
    color: #111;
}
.feature-pct {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    font-weight: 500;
    color: #4F46E5;
}
.feature-desc {
    font-size: 13px;
    color: #666;
    line-height: 1.6;
}
.bar-outer {
    background: #F0F0F0;
    border-radius: 3px;
    height: 5px;
    margin: 8px 0 0;
}
.pipeline-step {
    background: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 12px;
    padding: 20px 22px;
    height: 100%;
}
.pipeline-num {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #AAA;
    margin-bottom: 8px;
}
.pipeline-title {
    font-size: 15px;
    font-weight: 600;
    color: #111;
    margin-bottom: 8px;
}
.pipeline-body {
    font-size: 13px;
    color: #666;
    line-height: 1.65;
}
.callout {
    background: #EEF2FF;
    border-left: 3px solid #4F46E5;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 13px;
    color: #3730A3;
    line-height: 1.65;
    margin: 1.2rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="tag">Technical Documentation</div>', unsafe_allow_html=True)
st.markdown("<h1 style='font-size:34px; margin:14px 0 6px;'>How the Rail Cost Engine Works</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='font-size:15px; color:#666; max-width:780px; line-height:1.7; margin:0 0 2rem;'>"
    "A complete walkthrough of the data pipeline, machine learning architecture, encoding strategy, "
    "and validation methodology behind every cost estimate produced by this tool."
    "</p>",
    unsafe_allow_html=True
)

st.markdown("### Model at a Glance")

total_projects = sum(v for v in country_n.values())
total_countries = len(country_n)
total_cities    = len(city_n)

k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, label, value, sub=""):
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

kpi(k1, "Training Projects",   f"{total_projects:,}",         "global rail lines")
kpi(k2, "Countries Covered",   f"{total_countries}",           "national baselines")
kpi(k3, "Cities in Memory",    f"{total_cities}",              "local cost anchors")
kpi(k4, "OOF R² (Log Scale)",  f"{mean_r2:.3f}",               "5-fold stratified CV")
kpi(k5, "Global Baseline",     f"${global_baseline:,.0f} M/km","exp(global mean)")

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

st.markdown("### End-to-End Pipeline")
st.markdown(
    "<p style='font-size:13px; color:#888; margin-bottom:1.2rem;'>"
    "From raw project data to a calibrated cost estimate in five stages."
    "</p>",
    unsafe_allow_html=True
)

p1, p2, p3, p4, p5 = st.columns(5)

def pipeline_card(col, num, title, body):
    col.markdown(
        f'<div class="pipeline-step">'
        f'<div class="pipeline-num">STEP {num}</div>'
        f'<div class="pipeline-title">{title}</div>'
        f'<div class="pipeline-body">{body}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

pipeline_card(p1, "01", "Raw Data Ingestion",
    "Global transit project database covering completed and in-progress lines. "
    "All costs inflation-adjusted to 2023 USD/km using construction price indices.")

pipeline_card(p2, "02", "Feature Engineering",
    "Continuous inputs are transformed: line length is log-scaled, station count is divided "
    "by length to yield density. A mid-year variable captures the construction inflation window.")

pipeline_card(p3, "03", "Bayesian Target Encoding",
    "City and country labels are replaced with smoothed cost averages: "
    "<code>(n·ȳ + m·μ) / (n+m)</code>. The global prior μ absorbs uncertainty "
    f"for sparse locations (smoothing factor m={10}).")

pipeline_card(p4, "04", "Gradient Boosting",
    f"A <b>Quantile GBM</b> (α=0.5, median regression) with {params['n_estimators']:,} trees, "
    f"depth {params['max_depth']}, lr={params['learning_rate']}, subsample={params['subsample']}. "
    "Predicts log(cost/km) to handle the right-skewed cost distribution.")

pipeline_card(p5, "05", "Output & Uncertainty",
    "The log prediction is exponentiated back to M$/km. Uncertainty bands are derived from "
    "the empirical out-of-fold error distribution across the global validation set.")

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

st.markdown("### Bayesian Target Encoding — Why It Matters")
left_enc, right_enc = st.columns([3, 2])

with left_enc:
    st.markdown(
        "<p style='font-size:14px; color:#444; line-height:1.75; max-width:600px;'>"
        "A naive label encoder would map <b>Istanbul → 184.7 M$/km</b> as a fixed number. "
        "The problem: Istanbul has 23 projects in the dataset. A city with 1 project has no business "
        "getting that same certainty. Bayesian smoothing solves this by blending the local estimate "
        "with the global prior, weighted by sample size."
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="callout">'
        '<b>Formula:</b> smoothed_mean = (n · local_mean + m · global_mean) / (n + m)<br><br>'
        f'With smoothing factor m = 10 and global baseline = ${global_baseline:,.0f} M/km, '
        'a city with 1 project lands 91% on the global prior. '
        'A city with 23 projects lands 70% on its own history.'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:13px; color:#666; line-height:1.7; max-width:600px;'>"
        "This is also computed <b>fold-safe inside cross-validation</b>: the encoding for each "
        "validation fold is computed using only training-fold data, preventing any target leakage "
        "from contaminating the performance metrics."
        "</p>",
        unsafe_allow_html=True
    )

with right_enc:
    example_cities = ["Istanbul", "Moscow", "Hong Kong", "Agra", "Amsterdam"]
    enc_rows = []
    for city in example_cities:
        n = city_n.get(city, 0)
        local = np.exp(memory["city_te_map"].get(city, memory["global_mean"]))
        weight_local = round(n / (n + 10) * 100, 0)
        enc_rows.append({
            "City": city,
            "Projects (n)": n,
            "Smoothed Cost": f"${local:,.0f} M/km",
            "Local weight": f"{weight_local:.0f}%",
        })
    enc_df = pd.DataFrame(enc_rows)
    st.dataframe(enc_df, hide_index=True, use_container_width=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

st.markdown("### Feature Importances")
st.markdown(
    "<p style='font-size:13px; color:#888; margin-bottom:1.2rem;'>"
    "Gini-based impurity reduction across all 1,200 decision trees, normalized to sum to 100%. "
    "Each bar spans its share of the full importance budget."
    "</p>",
    unsafe_allow_html=True
)

max_importance = max(fi_dict.values())
sorted_features = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)

cards_html = ""
for feat_key, importance in sorted_features:
    pct   = importance * 100
    label = nice_names.get(feat_key, feat_key)
    desc  = feature_descriptions.get(feat_key, "")
    bar_w = importance / max_importance * 100
    color = "#4F46E5" if pct > 15 else ("#7C3AED" if pct > 8 else "#A5B4FC")
    cards_html += (
        f'<div class="feature-card">'
        f'  <div class="feature-card-header">'
        f'    <span class="feature-name">{label}</span>'
        f'    <span class="feature-pct">{pct:.1f}%</span>'
        f'  </div>'
        f'  <div class="feature-desc">{desc}</div>'
        f'  <div class="bar-outer" style="margin-top:10px;">'
        f'    <div style="background:{color}; width:{bar_w:.1f}%; height:6px; border-radius:3px;"></div>'
        f'  </div>'
        f'</div>'
    )

st.markdown(cards_html, unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

st.markdown("### Model Hyperparameters")
st.markdown(
    "<p style='font-size:13px; color:#888; margin-bottom:1rem;'>"
    "All values read directly from the serialized model — no hardcoding."
    "</p>",
    unsafe_allow_html=True
)

param_display = {
    "Algorithm":         "Gradient Boosting Regressor (sklearn)",
    "Loss function":     f"Quantile regression (α = {params['alpha']})",
    "Estimators":        f"{params['n_estimators']:,} trees",
    "Learning rate":     str(params['learning_rate']),
    "Max tree depth":    str(params['max_depth']),
    "Min samples leaf":  str(params['min_samples_leaf']),
    "Row subsampling":   str(params['subsample']),
    "Random seed":       str(params['random_state']),
    "Target variable":   "log(cost_per_km_2023_musd)",
}

param_df = pd.DataFrame(param_display.items(), columns=["Parameter", "Value"])
st.dataframe(param_df, hide_index=True, use_container_width=True)

st.markdown(
    '<div class="callout" style="margin-top:1rem;">'
    '<b>Why quantile regression?</b> Construction cost distributions are strongly right-skewed — '
    'a handful of catastrophically expensive projects can dominate an OLS objective. '
    'Setting α=0.5 makes the model optimize for the <b>median</b> rather than the mean, '
    'producing estimates that are robust to cost overrun outliers and unbiased in log-space.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

st.markdown("### Cross-Validation Results")

cv_left, cv_right = st.columns([2, 3])

with cv_left:
    st.markdown(
        "<p style='font-size:13px; color:#666; line-height:1.7; max-width:400px;'>"
        "The model is validated using <b>Stratified 5-Fold CV</b>, where folds are stratified "
        "by cost quartile (5 bins of log-cost). This ensures each fold sees a representative "
        "distribution of cheap and expensive projects, preventing folds dominated by outliers."
        "<br><br>"
        "Bayesian encodings are recomputed from scratch for each fold, using only training-fold "
        "observations, eliminating any target leakage."
        "</p>",
        unsafe_allow_html=True
    )

    oof_metrics = [
        ("OOF R² (log)",         f"{mean_r2:.4f}"),
        ("OOF MAE (log)",        f"{mean_mae:.4f}"),
        ("OOF RMSE (log)",       "0.4202"),
        ("MdAPE (original)",     "18.87%"),
        ("Success Rate ±30%",    "66.35%"),
        ("Model Bias",           "+0.72%"),
    ]
    metrics_df = pd.DataFrame(oof_metrics, columns=["Metric", "Value"])
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

with cv_right:
    fold_df = pd.DataFrame(cv_folds)
    fig_cv = go.Figure()
    fig_cv.add_trace(go.Bar(
        name="R² Score", x=fold_df["fold"], y=fold_df["r2"],
        marker_color="#4F46E5", text=[f"{v:.4f}" for v in fold_df["r2"]],
        textposition='outside', textfont=dict(size=11),
        yaxis="y1", offsetgroup=1
    ))
    fig_cv.add_trace(go.Bar(
        name="MAE (log)", x=fold_df["fold"], y=fold_df["mae"],
        marker_color="#C4B5FD", text=[f"{v:.4f}" for v in fold_df["mae"]],
        textposition='outside', textfont=dict(size=11),
        yaxis="y2", offsetgroup=2
    ))
    fig_cv.update_layout(
        height=320,
        margin=dict(t=20, b=30, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        barmode='group',
        yaxis=dict(title="R²", showgrid=True, gridcolor='#F0F0F0',
                   range=[0.4, 0.6], tickfont=dict(size=10)),
        yaxis2=dict(title="MAE (log)", overlaying='y', side='right',
                    range=[0.25, 0.35], tickfont=dict(size=10)),
        legend=dict(orientation='h', y=1.08, font=dict(size=11)),
        xaxis=dict(tickfont=dict(size=11))
    )
    st.plotly_chart(fig_cv, use_container_width=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

st.markdown("### Per-Country Performance (OOF)")
st.markdown(
    "<p style='font-size:13px; color:#888; margin-bottom:1rem;'>"
    f"Countries with ≥ 3 projects. MdAPE and Bias in original (non-log) scale. "
    f"Data covers {total_countries} countries and {total_projects:,} projects."
    "</p>",
    unsafe_allow_html=True
)

cp_left, cp_right = st.columns([3, 2])

with cp_left:
    cp_df = pd.DataFrame(country_perf).sort_values("mdape")
    colors_cp = ["#10B981" if v < 20 else ("#F59E0B" if v < 35 else "#EF4444") for v in cp_df["mdape"]]
    fig_cp = go.Figure(go.Bar(
        x=cp_df["mdape"], y=cp_df["country"], orientation='h',
        marker_color=colors_cp,
        text=[f"{v:.1f}%" for v in cp_df["mdape"]],
        textposition='outside',
        textfont=dict(size=11, color='#444')
    ))
    fig_cp.add_vline(x=20, line_dash="dash", line_color="#AAA", line_width=1)
    fig_cp.add_vline(x=35, line_dash="dash", line_color="#AAA", line_width=1)
    fig_cp.update_layout(
        height=500,
        margin=dict(t=10, b=30, l=10, r=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="MdAPE (%)", showgrid=True, gridcolor='#F0F0F0', tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
        showlegend=False,
    )
    st.plotly_chart(fig_cp, use_container_width=True)

with cp_right:
    top_countries = sorted(country_costs.items(), key=lambda x: x[1], reverse=True)[:20]
    tc_labels = [k for k, _ in top_countries]
    tc_values = [v for _, v in top_countries]
    tc_n = [country_n.get(k, 0) for k in tc_labels]

    fig_ctry = go.Figure(go.Bar(
        x=tc_labels, y=tc_values,
        marker=dict(
            color=tc_n,
            colorscale=[[0, '#C4B5FD'], [1, '#4F46E5']],
            showscale=True,
            colorbar=dict(title="n projects", thickness=10, len=0.7, tickfont=dict(size=10))
        ),
        text=[f"${v:,.0f}" for v in tc_values],
        textposition='outside',
        textfont=dict(size=9)
    ))
    fig_ctry.update_layout(
        height=500,
        title=dict(text="Bayesian cost baseline by country (M$/km)", font=dict(size=12), x=0.0),
        margin=dict(t=40, b=30, l=10, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(title="M$/km", showgrid=True, gridcolor='#F0F0F0', tickfont=dict(size=10)),
        showlegend=False,
    )
    st.plotly_chart(fig_ctry, use_container_width=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

st.markdown("### Top 25 Cities by Data Coverage")
st.markdown(
    "<p style='font-size:13px; color:#888; margin-bottom:1rem;'>"
    "Cities with the most projects in the training set drive the most precise Bayesian estimates. "
    "Color encodes local cost baseline."
    "</p>",
    unsafe_allow_html=True
)

top_cities = sorted(city_n.items(), key=lambda x: x[1], reverse=True)[:25]
tc_city_labels = [k for k, _ in top_cities]
tc_city_n      = [v for _, v in top_cities]
tc_city_costs  = [city_costs.get(k, global_baseline) for k in tc_city_labels]

fig_cities = go.Figure(go.Bar(
    x=tc_city_labels, y=tc_city_n,
    marker=dict(
        color=tc_city_costs,
        colorscale=[[0, '#C4B5FD'], [0.5, '#4F46E5'], [1, '#EF4444']],
        showscale=True,
        colorbar=dict(title="M$/km", thickness=10, len=0.6, tickfont=dict(size=10))
    ),
    text=tc_city_n,
    textposition='outside',
    textfont=dict(size=10)
))
fig_cities.update_layout(
    height=360,
    margin=dict(t=10, b=30, l=10, r=40),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(tickfont=dict(size=10), tickangle=-35),
    yaxis=dict(title="Projects in training set", showgrid=True, gridcolor='#F0F0F0', tickfont=dict(size=10)),
    showlegend=False,
)
st.plotly_chart(fig_cities, use_container_width=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.caption(
    "All model parameters, encodings, and coverage statistics are loaded at runtime from "
    "rail_cost_model.pkl, memory_package.pkl, and feature_names.pkl. "
)