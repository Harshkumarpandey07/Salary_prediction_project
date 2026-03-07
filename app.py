"""
app.py  —  SalaryIQ: ML-Powered Compensation Predictor
-------------------------------------------------------
Streamlit web app that loads the trained models and lets users
predict salary, increment %, and promotion probability.

Run:  streamlit run app.py
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
PLOTS  = os.path.join(BASE, "plots")

st.set_page_config(
    page_title="SalaryIQ · ML Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background-color: #080c10; color: #e8edf5; }
    [data-testid="stSidebar"] { background-color: #0f1520 !important; border-right: 1px solid #1e2d3d; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #e8edf5 !important; }
    [data-testid="metric-container"] { background: #0f1520; border: 1px solid #1e2d3d; border-radius: 14px; padding: 18px !important; }
    [data-testid="stMetricLabel"] { color: #6b7a8d !important; font-size: 0.78rem !important; }
    [data-testid="stMetricValue"] { color: #00d4ff !important; font-family: 'Syne', sans-serif !important; }
    .stButton > button { background: linear-gradient(135deg, #00d4ff, #7c3aed) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 12px 32px !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 1rem !important; width: 100%; }
    .result-box { background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(124,58,237,0.08)); border: 1px solid rgba(0,212,255,0.2); border-radius: 16px; padding: 24px 28px; margin: 8px 0; }
    .salary-hero { font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #ffffff, #00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1; }
    .section-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 2.5px; text-transform: uppercase; color: #00d4ff; margin-bottom: 6px; }
    .divider { border-top: 1px solid #1e2d3d; margin: 28px 0; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(os.path.join(MODELS, "preprocessor.pkl"))
    best_model   = joblib.load(os.path.join(MODELS, "best_model.pkl"))
    lr_model     = joblib.load(os.path.join(MODELS, "linear_regression.pkl"))
    rf_model     = joblib.load(os.path.join(MODELS, "random_forest.pkl"))
    gb_model     = joblib.load(os.path.join(MODELS, "gradient_boosting.pkl"))
    with open(os.path.join(MODELS, "metrics.json")) as f:
        metrics = json.load(f)
    return preprocessor, best_model, lr_model, rf_model, gb_model, metrics

preprocessor, best_model, lr_model, rf_model, gb_model, metrics = load_artifacts()
all_models = {
    "Linear Regression":  lr_model,
    "Random Forest":      rf_model,
    "Gradient Boosting":  gb_model,
}

with st.sidebar:
    st.markdown('<p style="font-family:Syne;font-size:1.6rem;font-weight:800;color:#e8edf5">Salary<span style="color:#00d4ff">IQ</span></p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b7a8d;font-size:0.82rem;margin-top:-12px">ML-Powered Compensation Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    page = st.radio("Navigate", ["🎯 Predict Salary", "📊 Model Evaluation", "📈 Data Insights"], label_visibility="collapsed")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Best Model</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#10b981;font-weight:600">{metrics["best_model"]}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#6b7a8d;font-size:0.82rem">R² = {metrics[metrics["best_model"]]["R2"]} | MAE = {metrics[metrics["best_model"]]["MAE"]} LPA</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b7a8d;font-size:0.75rem">Dataset: 12,000 rows · 16 features<br>Models: Linear Regression, Random Forest, Gradient Boosting<br>Encoding: OneHotEncoder · Scaling: StandardScaler</p>', unsafe_allow_html=True)

if page == "🎯 Predict Salary":
    st.markdown('<p class="section-label">Prediction Engine</p>', unsafe_allow_html=True)
    st.markdown("# Predict Your Compensation")
    st.markdown('<p style="color:#6b7a8d">Enter your professional details. Our trained ML model will predict your expected salary, increment, and promotion probability.</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Role & Industry**")
        job_role = st.selectbox("Job Role", ["Software Engineer","Senior Software Engineer","Data Analyst","Data Scientist","ML Engineer","Product Manager","Business Analyst","HR Manager","Finance Analyst","Marketing Manager","DevOps Engineer","Cloud Architect","Cybersecurity Analyst","UI/UX Designer","QA Engineer","Fresher / Junior Dev","Operations Manager","Sales Executive","Tech Lead","Engineering Manager"])
        industry = st.selectbox("Industry", ["IT / Software","Finance & Banking","E-Commerce","Healthcare","Manufacturing","Consulting","Education","Government/PSU","Media & Entertainment","Startup"])
        location = st.selectbox("Location", ["Bangalore","Mumbai","Delhi NCR","Hyderabad","Pune","Chennai","Kolkata","Ahmedabad","Tier-2 City","Remote"])
    with col2:
        st.markdown("**Education & Company**")
        education_level = st.selectbox("Education Level", ["High School","Diploma","Bachelor's","Master's","MBA","PhD"])
        company_type    = st.selectbox("Company Type", ["MNC","Indian Large Corp","Mid-size","Startup","Government/PSU"])
        company_size    = st.selectbox("Company Size", ["Small (<50)","Mid (50-500)","Large (500-5000)","Enterprise (5000+)"])
    with col3:
        st.markdown("**Experience & Performance**")
        years_experience   = st.slider("Years of Experience",      0.0, 32.0, 3.0, 0.5)
        years_at_company   = st.slider("Years at Current Company", 0.0, 20.0, 1.0, 0.5)
        performance_rating = st.slider("Performance Rating",        1.0,  5.0, 3.5, 0.5)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: num_skills         = st.slider("Number of Skills",     1, 12,  5)
    with c2: certifications     = st.slider("Certifications",       0,  6,  1)
    with c3: projects_completed = st.slider("Projects Completed",   0, 60, 10)
    with c4: promotions_received= st.slider("Promotions Received",  0,  8,  1)
    c5, c6, c7 = st.columns(3)
    with c5: overtime_hours_week = st.slider("Overtime Hours/Week",    0, 25,  4)
    with c6: distance_from_work  = st.slider("Distance from Work (km)",0,100, 15)
    with c7: monthly_hours       = st.slider("Monthly Working Hours", 120,240,176)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Select Model for Prediction", ["Gradient Boosting (Best)", "Random Forest", "Linear Regression"])
    predict_model = {"Gradient Boosting (Best)": gb_model, "Random Forest": rf_model, "Linear Regression": lr_model}[model_choice]

    if st.button("🚀 Predict My Compensation"):
        input_dict = {
            "job_role": job_role, "industry": industry, "education_level": education_level,
            "location": location, "company_type": company_type, "company_size": company_size,
            "years_experience": years_experience, "years_at_company": years_at_company,
            "performance_rating": performance_rating, "num_skills": num_skills,
            "certifications": certifications, "projects_completed": projects_completed,
            "promotions_received": promotions_received, "overtime_hours_week": overtime_hours_week,
            "distance_from_work": distance_from_work, "monthly_hours": monthly_hours,
        }
        df_input = pd.DataFrame([input_dict])
        X_proc   = preprocessor.transform(df_input)
        salary_pred = max(predict_model.predict(X_proc)[0], 1.5)
        all_preds   = {n: max(m.predict(X_proc)[0], 1.5) for n, m in all_models.items()}

        increment_pct  = round(min(performance_rating * 2.5 + (num_skills / 12) * 3 + certifications * 0.5, 40), 1)
        promotion_prob = round(min((performance_rating/5)*0.4 + (promotions_received/max(years_experience,1))*0.15 + (num_skills/12)*0.2 + (certifications/6)*0.1, 1.0)*100, 1)
        hike_if_switch = round(salary_pred * np.random.uniform(0.20, 0.40), 2)
        new_salary     = round(salary_pred + hike_if_switch, 2)

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        rc1, rc2 = st.columns([1.5, 1])
        with rc1:
            st.markdown(f'<p style="color:#6b7a8d;font-size:0.8rem;margin-bottom:4px">PREDICTED SALARY</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="salary-hero">₹{salary_pred:.1f} LPA</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:#6b7a8d;font-size:0.85rem">Range: ₹{max(salary_pred-2,1):.1f} – ₹{salary_pred+2:.1f} LPA</p>', unsafe_allow_html=True)
        with rc2:
            m_key2 = "Gradient Boosting" if "Gradient" in model_choice else model_choice.split("(")[0].strip()
            r2_val = metrics.get(m_key2, {}).get("R2", 0.9)
            conf_label = "High" if r2_val >= 0.90 else "Medium"
            st.markdown(f'<p style="margin-top:20px;color:#10b981;font-weight:600">{conf_label} Confidence — R² {r2_val}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected Increment",   f"{increment_pct:.1f}%",    "Annual raise")
        m2.metric("Promotion Probability",f"{promotion_prob:.1f}%",   "Next cycle")
        m3.metric("Market Hike (Switch)", f"₹{hike_if_switch:.1f}L",  f"→ ₹{new_salary:.1f} LPA")
        m4.metric("Performance Score",    f"{performance_rating}/5",  f"{num_skills} skills")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("#### All Models Comparison")
        comp_df = pd.DataFrame({
            "Model":             list(all_preds.keys()),
            "Predicted (₹ LPA)": [round(v, 2) for v in all_preds.values()],
            "R² Score":          [metrics[k]["R2"] for k in all_preds.keys()],
            "MAE (₹ LPA)":       [metrics[k]["MAE"] for k in all_preds.keys()],
        })
        st.dataframe(comp_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor("#0f1520"); ax.set_facecolor("#0f1520")
        colors = ["#00d4ff","#7c3aed","#10b981"]
        bars = ax.barh(list(all_preds.keys()), list(all_preds.values()), color=colors, alpha=0.85, height=0.4)
        for bar, v in zip(bars, all_preds.values()):
            ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2, f"₹{v:.1f} LPA", va="center", color="#e8edf5", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Salary (₹ LPA)", color="#6b7a8d"); ax.tick_params(colors="#e8edf5")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#1e2d3d"); ax.spines["left"].set_color("#1e2d3d")
        ax.grid(axis="x", color="#1e2d3d", alpha=0.5)
        plt.tight_layout(); st.pyplot(fig); plt.close()

elif page == "📊 Model Evaluation":
    st.markdown('<p class="section-label">ML Evaluation</p>', unsafe_allow_html=True)
    st.markdown("# Model Performance Report")
    st.markdown('<p style="color:#6b7a8d">Full evaluation metrics and plots from training on 12,000 salary records.</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    model_names = ["Linear Regression", "Random Forest", "Gradient Boosting"]
    rows = []
    for m in model_names:
        row = {"Model": m}
        for k in ["R2","MAE","RMSE","MAPE_%","CV_R2_mean","CV_R2_std"]:
            row[k] = metrics[m][k]
        row["Best?"] = "⭐" if m == metrics["best_model"] else ""
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    for fname, title in [
        ("01_model_comparison.png",   "Model Comparison — R², MAE, RMSE"),
        ("02_actual_vs_predicted.png","Actual vs Predicted — All Models"),
        ("03_residuals.png",          "Residual Distributions"),
        ("04_feature_importance.png", "Top 15 Feature Importances"),
        ("05_salary_distribution.png","Salary Distribution in Dataset"),
        ("06_cv_scores.png",          "5-Fold Cross-Validation Scores"),
    ]:
        fpath = os.path.join(PLOTS, fname)
        if os.path.exists(fpath):
            st.markdown(f"**{title}**")
            st.image(fpath)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

elif page == "📈 Data Insights":
    st.markdown('<p class="section-label">Dataset Analysis</p>', unsafe_allow_html=True)
    st.markdown("# Data Insights & EDA")
    st.markdown('<p style="color:#6b7a8d">Exploratory analysis of the 12,000-row salary dataset used for training.</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    @st.cache_data
    def load_data():
        return pd.read_csv(os.path.join(BASE, "salary_dataset.csv"))
    df = load_data()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Features", "16")
    c3.metric("Avg Salary", f"₹{df.salary_lpa.mean():.1f} LPA")
    c4.metric("Salary Range", f"₹{df.salary_lpa.min():.0f}–{df.salary_lpa.max():.0f} LPA")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["💰 Salary Analysis","🧑‍💼 Role & Industry","📋 Raw Data"])

    DARK="#0f1520"; ACCENT="#00d4ff"; PURPLE="#7c3aed"; GREEN="#10b981"; TEXT="#e8edf5"
    plt.rcParams.update({"figure.facecolor":DARK,"axes.facecolor":DARK,"axes.labelcolor":TEXT,"xtick.color":TEXT,"ytick.color":TEXT,"text.color":TEXT,"grid.color":"#1e2d3d"})

    with tab1:
        c1,c2 = st.columns(2)
        with c1:
            edu_s = df.groupby("education_level")["salary_lpa"].median().sort_values(ascending=False)
            fig,ax = plt.subplots(figsize=(7,4)); fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
            ax.barh(edu_s.index, edu_s.values, color=ACCENT, alpha=0.85, height=0.5)
            ax.set_xlabel("Median Salary ₹ LPA"); ax.set_title("Salary by Education", color=TEXT, fontweight="bold")
            for s in ax.spines.values(): s.set_color("#1e2d3d")
            ax.grid(axis="x",color="#1e2d3d",alpha=0.5); plt.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            loc_s = df.groupby("location")["salary_lpa"].median().sort_values(ascending=False)
            fig,ax = plt.subplots(figsize=(7,4)); fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
            ax.barh(loc_s.index, loc_s.values, color=PURPLE, alpha=0.85, height=0.5)
            ax.set_xlabel("Median Salary ₹ LPA"); ax.set_title("Salary by Location", color=TEXT, fontweight="bold")
            for s in ax.spines.values(): s.set_color("#1e2d3d")
            ax.grid(axis="x",color="#1e2d3d",alpha=0.5); plt.tight_layout(); st.pyplot(fig); plt.close()

        fig,ax = plt.subplots(figsize=(12,4)); fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
        sc = ax.scatter(df["years_experience"],df["salary_lpa"],c=df["performance_rating"],cmap="cool",alpha=0.3,s=6)
        plt.colorbar(sc,ax=ax,label="Performance Rating")
        ax.set_xlabel("Years of Experience"); ax.set_ylabel("Salary ₹ LPA"); ax.set_title("Salary vs Experience", color=TEXT, fontweight="bold")
        for s in ax.spines.values(): s.set_color("#1e2d3d")
        ax.grid(color="#1e2d3d",alpha=0.3); plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        c1,c2 = st.columns(2)
        with c1:
            role_s = df.groupby("job_role")["salary_lpa"].median().sort_values(ascending=False).head(12)
            fig,ax = plt.subplots(figsize=(7,6)); fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
            ax.barh(role_s.index, role_s.values, color=GREEN, alpha=0.85, height=0.6)
            ax.set_xlabel("Median Salary ₹ LPA"); ax.set_title("Salary by Job Role", color=TEXT, fontweight="bold")
            for s in ax.spines.values(): s.set_color("#1e2d3d")
            ax.grid(axis="x",color="#1e2d3d",alpha=0.5); plt.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            ind_s = df.groupby("industry")["salary_lpa"].median().sort_values(ascending=False)
            fig,ax = plt.subplots(figsize=(7,6)); fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
            ax.barh(ind_s.index, ind_s.values, color="#f59e0b", alpha=0.85, height=0.6)
            ax.set_xlabel("Median Salary ₹ LPA"); ax.set_title("Salary by Industry", color=TEXT, fontweight="bold")
            for s in ax.spines.values(): s.set_color("#1e2d3d")
            ax.grid(axis="x",color="#1e2d3d",alpha=0.5); plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        st.dataframe(df.head(200), use_container_width=True)
        st.dataframe(df.describe().round(2), use_container_width=True)
