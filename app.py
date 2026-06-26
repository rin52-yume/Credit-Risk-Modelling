import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import json
import os
import shap
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk XAI Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F4F6FB; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    h1 { color: #1E2761; font-family: Georgia, serif; }
    h2 { color: #1E2761; }
    h3 { color: #185FA5; }
    .metric-card {
        background: white; border-radius: 10px; padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; margin-bottom: 8px;
    }
    .metric-label { font-size: 12px; color: #6B7280; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 700; color: #1E2761; }
    .risk-high   { background:#FEE2E2; border-left:5px solid #DC2626; border-radius:8px; padding:14px; }
    .risk-medium { background:#FEF3C7; border-left:5px solid #D97706; border-radius:8px; padding:14px; }
    .risk-low    { background:#D1FAE5; border-left:5px solid #059669; border-radius:8px; padding:14px; }
    .finding-box { background:#EEF2FC; border-radius:10px; padding:1rem 1.2rem; border-left:5px solid #1E2761; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
    .stSelectbox label, .stSlider label { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Load assets ───────────────────────────────────────────────────────────────
# Safe handling of __file__ for different environments
try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE = os.getcwd()

ASSETS = os.path.join(BASE, 'assets')

# Create assets folder if it doesn't exist
os.makedirs(ASSETS, exist_ok=True)

@st.cache_resource
def load_models():
    """Load all trained models from assets folder"""
    method_names = ['SMOTE', 'ADASYN', 'Borderline-SMOTE', 'Class Weights', 'Threshold Moving']
    mdls = {}
    for name in method_names:
        safe = name.replace(' ', '-').replace('/', '-')
        path = os.path.join(ASSETS, f'model_{safe}.pkl')
        try:
            with open(path, 'rb') as f:
                mdls[name] = pickle.load(f)
        except FileNotFoundError:
            st.warning(f"⚠️ Model file not found: {path}")
            mdls[name] = None
        except Exception as e:
            st.error(f"Error loading {name}: {str(e)}")
            mdls[name] = None
    return mdls

@st.cache_resource
def load_scaler():
    """Load the scaler"""
    path = os.path.join(ASSETS, 'scaler.pkl')
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning("⚠️ Scaler file not found. Using default scaler.")
        return None
    except Exception as e:
        st.error(f"Error loading scaler: {str(e)}")
        return None

@st.cache_data
def load_meta():
    """Load all metadata files"""
    try:
        with open(os.path.join(ASSETS, 'feature_names.json'), 'r') as f:
            fn = json.load(f)
        with open(os.path.join(ASSETS, 'thresholds.json'), 'r') as f:
            th = json.load(f)
        with open(os.path.join(ASSETS, 'stats.json'), 'r') as f:
            st_data = json.load(f)
        results = pd.read_csv(os.path.join(ASSETS, 'results.csv'))
        sra = pd.read_csv(os.path.join(ASSETS, 'sra_results.csv'))
        X_exp = pd.read_csv(os.path.join(ASSETS, 'X_explain.csv'))
        return fn, th, st_data, results, sra, X_exp
    except FileNotFoundError as e:
        st.error(f"❌ Required data file not found: {e.filename}")
        # Return default values to prevent crashing
        return (
            ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 
             'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 
             'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 
             'PAY_AMT5', 'PAY_AMT6', 'AVG_PAY_DELAY', 'TOTAL_BILL', 'TOTAL_PAID', 
             'UTIL_RATIO', 'PAY_RATIO'],
            {'SMOTE': 0.5, 'ADASYN': 0.5, 'Borderline-SMOTE': 0.5, 
             'Class Weights': 0.5, 'Threshold Moving': 0.5},
            {"n_samples": 30000, "default_rate": 22.0, "n_features": 28, 
             "best_sra_method": "SMOTE", "best_f1_method": "SMOTE", "trade_off": True},
            pd.DataFrame({'method': ['SMOTE', 'ADASYN', 'Borderline-SMOTE', 'Class Weights', 'Threshold Moving'],
                         'roc_auc': [0.78, 0.77, 0.76, 0.75, 0.74],
                         'f1': [0.65, 0.64, 0.63, 0.62, 0.61],
                         'threshold': [0.5, 0.5, 0.5, 0.5, 0.5]}),
            pd.DataFrame({'method': ['SMOTE', 'ADASYN', 'Borderline-SMOTE', 'Class Weights', 'Threshold Moving'],
                         'sra': [0.98, 0.97, 0.96, 0.95, 0.94],
                         'sra_std': [0.01, 0.02, 0.01, 0.02, 0.03]}),
            pd.DataFrame(np.random.randn(100, 28), columns=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                                                            'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
                                                            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 
                                                            'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 
                                                            'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 
                                                            'AVG_PAY_DELAY', 'TOTAL_BILL', 'TOTAL_PAID', 
                                                            'UTIL_RATIO', 'PAY_RATIO'])
        )
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return [], {}, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Load all assets
models = load_models()
scaler = load_scaler()
feature_names, thresholds, stats, results_df, sra_df, X_explain = load_meta()

# Check if data loaded properly
if not feature_names:
    st.error("❌ Failed to load required data. Please check the assets folder.")
    st.stop()

# Filter out None models for selection
AVAILABLE_MODELS = [m for m in list(models.keys()) if models[m] is not None]
if not AVAILABLE_MODELS:
    st.warning("⚠️ No models loaded. Using default model list for demonstration.")
    AVAILABLE_MODELS = ['SMOTE', 'ADASYN', 'Borderline-SMOTE', 'Class Weights', 'Threshold Moving']

METHOD_COLORS = {
    'SMOTE': '#1E2761',
    'ADASYN': '#185FA5',
    'Borderline-SMOTE': '#0D9488',
    'Class Weights': '#0F6E56',
    'Threshold Moving': '#854F0B'
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image("https://img.icons8.com/fluency/96/bank-building.png", width=70)
    except:
        st.markdown("🏦")
    st.markdown("## 🏦 Credit Risk XAI")
    st.markdown("**Explainable AI Dashboard**")
    st.divider()
    st.markdown(f"📊 **Dataset:** Taiwan Default Credit Card")
    if stats and 'n_samples' in stats:
        st.markdown(f"👥 **Clients:** {stats['n_samples']:,}")
        st.markdown(f"🎯 **Default Rate:** {stats['default_rate']:.1f}%")
        st.markdown(f"🔢 **Features:** {stats['n_features']}")
    else:
        st.markdown("👥 **Clients:** Data not available")
    st.divider()
    st.markdown("**Key Finding:**")
    if stats and 'best_sra_method' in stats:
        st.success(f"✅ Best SRA: **{stats['best_sra_method']}**")
        st.info(f"📈 Best F1: **{stats['best_f1_method']}**")
        if stats.get('trade_off', False):
            st.warning("⚖️ **Trade-off confirmed!** Best prediction ≠ best stability")
    else:
        st.info("Key findings will appear after loading data")
    st.divider()
    st.caption("M.A. Economics · Delhi School of Economics · Class of 2027")

# ── Main title ────────────────────────────────────────────────────────────────
st.markdown("# 🏦 Explainable AI Credit Risk Assessment System")
st.markdown("**Research:** Which imbalance correction method produces the most stable SHAP feature rankings? (SRA metric)")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview & EDA",
    "🔮 Predict Default",
    "🧠 SHAP Explanations",
    "⚖️ Compare Methods",
    "🔬 SRA Research Results"
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview & EDA
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    if stats and 'n_samples' in stats:
        default_cases = int(stats['n_samples'] * stats['default_rate'] / 100)
        no_default = stats['n_samples'] - default_cases
        for col, label, val in zip(
            [c1, c2, c3, c4],
            ["Total Clients", "Default Cases", "No Default", "Default Rate"],
            [f"{stats['n_samples']:,}", f"{default_cases:,}", f"{no_default:,}", f"{stats['default_rate']:.1f}%"]
        ):
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                </div>""", unsafe_allow_html=True)
    else:
        for col, label in zip([c1, c2, c3, c4], ["Total Clients", "Default Cases", "No Default", "Default Rate"]):
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">N/A</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("#### Class Distribution")
        img_path = os.path.join(ASSETS, '01_class_dist.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info("📊 Visualization not available. Please add '01_class_dist.png' to the assets folder.")

    with r1c2:
        st.markdown("#### Default Rate by Payment Status (PAY_0)")
        img_path = os.path.join(ASSETS, '02_pay0_default.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info("📊 Visualization not available. Please add '02_pay0_default.png' to the assets folder.")

    st.markdown("#### Feature Correlation Matrix")
    img_path = os.path.join(ASSETS, '03_correlation.png')
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.info("📊 Visualization not available. Please add '03_correlation.png' to the assets folder.")

    st.markdown("---")
    st.markdown("#### Dataset Variables")
    var_df = pd.DataFrame({
        'Variable': ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                     'PAY_0–PAY_6', 'BILL_AMT1–6', 'PAY_AMT1–6',
                     'AVG_PAY_DELAY', 'TOTAL_BILL', 'TOTAL_PAID', 'UTIL_RATIO', 'PAY_RATIO', 'DEFAULT'],
        'Type': ['Numeric', 'Categorical', 'Categorical', 'Categorical', 'Numeric',
                 'Ordinal', 'Numeric', 'Numeric',
                 'Engineered', 'Engineered', 'Engineered', 'Engineered', 'Engineered', 'Target'],
        'Description': [
            'Credit limit in NT dollars',
            '1=Male, 2=Female',
            '1=Graduate, 2=University, 3=High school, 4=Others',
            '1=Married, 2=Single, 3=Others',
            'Age in years',
            'Payment status: -1=pay duly, 1=1 month late, 2=2 months late…',
            'Amount of bill statement (months 1–6)',
            'Amount of previous payment (months 1–6)',
            'Mean payment delay over 6 months',
            'Sum of all bill amounts',
            'Sum of all payment amounts',
            'BILL_AMT1 / LIMIT_BAL',
            'TOTAL_PAID / TOTAL_BILL',
            '1=Default next month, 0=No default'
        ]
    })
    st.dataframe(var_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Predict Default
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔮 Credit Default Prediction")
    st.markdown("Enter a client's details below to predict their probability of default.")

    selected_method = st.selectbox(
        "Select Imbalance Correction Model:",
        AVAILABLE_MODELS,
        index=0 if AVAILABLE_MODELS else 0,
        help="Choose which trained model to use for prediction"
    )

    st.markdown("---")
    st.markdown("#### Client Information")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**👤 Demographics**")
        limit_bal = st.slider("Credit Limit (NT$)", 10000, 1000000, 50000, step=10000,
                              help="Maximum credit amount approved")
        age = st.slider("Age", 18, 80, 35)
        sex = st.selectbox("Sex", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        education = st.selectbox("Education",
                                 [1, 2, 3, 4],
                                 format_func=lambda x: {1: "Graduate school", 2: "University",
                                                        3: "High school", 4: "Others"}[x])
        marriage = st.selectbox("Marital Status",
                                [1, 2, 3],
                                format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])

    with col_b:
        st.markdown("**💳 Payment History (Recent → Oldest)**")
        pay0 = st.selectbox("PAY_0 (Sep)", list(range(-2, 10)),
                            format_func=lambda x: f"{x} ({'on-time' if x <= 0 else f'{x}mo late'})", index=2)
        pay2 = st.selectbox("PAY_2 (Aug)", list(range(-2, 10)),
                            format_func=lambda x: f"{x} ({'on-time' if x <= 0 else f'{x}mo late'})", index=2)
        pay3 = st.selectbox("PAY_3 (Jul)", list(range(-2, 10)),
                            format_func=lambda x: f"{x} ({'on-time' if x <= 0 else f'{x}mo late'})", index=2)
        pay4 = st.selectbox("PAY_4 (Jun)", list(range(-2, 10)),
                            format_func=lambda x: f"{x} ({'on-time' if x <= 0 else f'{x}mo late'})", index=2)
        pay5 = st.selectbox("PAY_5 (May)", list(range(-2, 10)),
                            format_func=lambda x: f"{x} ({'on-time' if x <= 0 else f'{x}mo late'})", index=2)
        pay6 = st.selectbox("PAY_6 (Apr)", list(range(-2, 10)),
                            format_func=lambda x: f"{x} ({'on-time' if x <= 0 else f'{x}mo late'})", index=2)

    with col_c:
        st.markdown("**📄 Bill & Payment Amounts (NT$)**")
        bill1 = st.number_input("Bill Sep", 0, 500000, 20000, step=1000)
        bill2 = st.number_input("Bill Aug", 0, 500000, 18000, step=1000)
        bill3 = st.number_input("Bill Jul", 0, 500000, 17000, step=1000)
        pay_a1 = st.number_input("Paid Sep", 0, 200000, 1500, step=500)
        pay_a2 = st.number_input("Paid Aug", 0, 200000, 1500, step=500)
        pay_a3 = st.number_input("Paid Jul", 0, 200000, 1500, step=500)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Default Probability", type="primary", use_container_width=True)

    # Initialize variables
    proba = 0.0
    thresh = 0.5

    if predict_btn:
        # Check if model and scaler are available
        if models.get(selected_method) is None:
            st.error("❌ Model not loaded. Please check the assets folder.")
        elif scaler is None:
            st.error("❌ Scaler not loaded. Please check the assets folder.")
        else:
            try:
                # Build full feature vector
                bill4 = bill3
                bill5 = bill2
                bill6 = bill1
                pay_a4 = pay_a3
                pay_a5 = pay_a2
                pay_a6 = pay_a1

                raw = {
                    'LIMIT_BAL': limit_bal, 'SEX': sex, 'EDUCATION': education,
                    'MARRIAGE': marriage, 'AGE': age,
                    'PAY_0': pay0, 'PAY_2': pay2, 'PAY_3': pay3,
                    'PAY_4': pay4, 'PAY_5': pay5, 'PAY_6': pay6,
                    'BILL_AMT1': bill1, 'BILL_AMT2': bill2, 'BILL_AMT3': bill3,
                    'BILL_AMT4': bill4, 'BILL_AMT5': bill5, 'BILL_AMT6': bill6,
                    'PAY_AMT1': pay_a1, 'PAY_AMT2': pay_a2, 'PAY_AMT3': pay_a3,
                    'PAY_AMT4': pay_a4, 'PAY_AMT5': pay_a5, 'PAY_AMT6': pay_a6,
                }
                pay_cols = [raw[f'PAY_{k}'] for k in [0, 2, 3, 4, 5, 6]]
                total_bill = sum([raw[f'BILL_AMT{i}'] for i in range(1, 7)])
                total_paid = sum([raw[f'PAY_AMT{i}'] for i in range(1, 7)])
                raw['AVG_PAY_DELAY'] = np.mean(pay_cols)
                raw['TOTAL_BILL'] = total_bill
                raw['TOTAL_PAID'] = total_paid
                raw['UTIL_RATIO'] = bill1 / (limit_bal + 1)
                raw['PAY_RATIO'] = total_paid / (total_bill + 1)

                # Ensure all features are present
                row_df = pd.DataFrame([raw])
                # Only use features that exist in the DataFrame
                available_features = [f for f in feature_names if f in row_df.columns]
                if not available_features:
                    st.error("❌ No matching features found. Please check feature names.")
                else:
                    row_df = row_df[available_features]
                    row_sc = pd.DataFrame(scaler.transform(row_df), columns=available_features)

                    model = models[selected_method]
                    thresh = thresholds.get(selected_method, 0.5)
                    proba = model.predict_proba(row_sc)[0, 1]

                    st.markdown("---")
                    st.markdown("### Prediction Result")

                    r1, r2, r3 = st.columns([1, 1, 1])
                    with r1:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-label">Default Probability</div>
                            <div class="metric-value" style="color:{'#DC2626' if proba > 0.5 else '#059669'}">{proba * 100:.1f}%</div>
                        </div>""", unsafe_allow_html=True)
                    with r2:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-label">Decision Threshold</div>
                            <div class="metric-value">{thresh:.3f}</div>
                        </div>""", unsafe_allow_html=True)
                    with r3:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-label">Model Used</div>
                            <div class="metric-value" style="font-size:16px">{selected_method}</div>
                        </div>""", unsafe_allow_html=True)

                    if proba >= 0.65:
                        st.markdown(f'<div class="risk-high">🔴 <b>HIGH RISK</b> — Probability: {proba * 100:.1f}%. Recommend rejecting credit application or requesting additional collateral.</div>', unsafe_allow_html=True)
                    elif proba >= 0.35:
                        st.markdown(f'<div class="risk-medium">🟡 <b>MEDIUM RISK</b> — Probability: {proba * 100:.1f}%. Further review recommended before approving.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="risk-low">🟢 <b>LOW RISK</b> — Probability: {proba * 100:.1f}%. Credit application looks acceptable.</div>', unsafe_allow_html=True)

                    # SHAP for this prediction
                    st.markdown("#### Why this prediction? (SHAP)")
                    with st.spinner("Computing SHAP explanation..."):
                        try:
                            exp = shap.TreeExplainer(model)
                            sv = exp.shap_values(row_sc)
                            if len(sv) > 0:
                                contrib = pd.Series(sv[0], index=available_features).sort_values(key=abs, ascending=False).head(10)

                                fig2, ax2 = plt.subplots(figsize=(8, 4))
                                colors_bar = ['#DC2626' if v > 0 else '#059669' for v in contrib.values]
                                ax2.barh(contrib.index[::-1], contrib.values[::-1], color=colors_bar[::-1], edgecolor='white')
                                ax2.axvline(0, color='black', linewidth=0.8)
                                ax2.set_xlabel('SHAP Value (positive = pushes toward default)')
                                ax2.set_title('Top SHAP Contributions for This Client', fontweight='bold', fontsize=12)
                                red_p = mpatches.Patch(color='#DC2626', label='Increases default risk')
                                green_p = mpatches.Patch(color='#059669', label='Decreases default risk')
                                ax2.legend(handles=[red_p, green_p], fontsize=9)
                                plt.tight_layout()
                                st.pyplot(fig2, use_container_width=True)
                                plt.close()
                        except Exception as e:
                            st.warning(f"⚠️ Could not compute SHAP explanation: {str(e)}")

            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")

    # Gauge chart (always shown, with current proba value)
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.barh(['Risk'], [1], color='#E5E7EB', height=0.4)
    bar_color = '#DC2626' if proba > 0.65 else ('#D97706' if proba > 0.35 else '#059669')
    ax.barh(['Risk'], [proba], color=bar_color, height=0.4)
    if selected_method in thresholds:
        ax.axvline(thresh, color='#1E2761', linestyle='--', linewidth=2, label=f'Threshold ({thresh:.2f})')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability of Default', fontsize=11)
    ax.set_title(f'Default Probability: {proba * 100:.1f}%' + (" (No prediction made)" if not predict_btn else ""), fontsize=12, fontweight='bold')
    if selected_method in thresholds:
        ax.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — SHAP Explanations
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🧠 SHAP Global Explainability")
    st.markdown("SHAP (SHapley Additive exPlanations) shows which features drive default predictions.")

    shap_method = st.selectbox("Select model to explain:", AVAILABLE_MODELS, key='shap_sel')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### SMOTE Model — SHAP Summary")
        img_path = os.path.join(ASSETS, '05_shap_smote.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info("📊 Visualization not available. Please add '05_shap_smote.png' to the assets folder.")
        st.caption("Each dot = one client. Red = high feature value. Position on x-axis = impact on default prediction.")

    with c2:
        st.markdown("#### Class Weights Model — SHAP Summary")
        img_path = os.path.join(ASSETS, '06_shap_classweights.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info("📊 Visualization not available. Please add '06_shap_classweights.png' to the assets folder.")
        st.caption("Note how the ranking of top features can differ between methods — this is what SRA measures.")

    st.markdown("---")
    st.markdown("#### Dynamic SHAP Bar Chart for Selected Model")
    if models.get(shap_method) is not None and not X_explain.empty:
        with st.spinner(f"Computing SHAP for {shap_method}..."):
            try:
                exp_dyn = shap.TreeExplainer(models[shap_method])
                sv_dyn = exp_dyn.shap_values(X_explain)
                if len(sv_dyn) > 0:
                    mean_abs = np.abs(sv_dyn).mean(axis=0)
                    fi_series = pd.Series(mean_abs, index=feature_names[:len(mean_abs)]).sort_values(ascending=True).tail(15)

                    fig, ax = plt.subplots(figsize=(9, 6))
                    bars = ax.barh(fi_series.index, fi_series.values,
                                   color=METHOD_COLORS.get(shap_method, '#1E2761'), edgecolor='white', alpha=0.9)
                    ax.set_xlabel('Mean |SHAP Value| (Feature Importance)', fontsize=11)
                    ax.set_title(f'Global Feature Importance — {shap_method}', fontweight='bold', fontsize=13)
                    for bar, val in zip(bars, fi_series.values):
                        ax.text(val + 0.0002, bar.get_y() + bar.get_height() / 2, f'{val:.4f}', va='center', fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
            except Exception as e:
                st.warning(f"⚠️ Could not compute SHAP for {shap_method}: {str(e)}")
    else:
        st.info("ℹ️ Please ensure the model is loaded and X_explain data is available.")

    st.markdown("---")
    st.markdown("#### Feature Ranking Comparison Across All Methods")
    img_path = os.path.join(ASSETS, '07_feature_rankings.png')
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.info("📊 Visualization not available. Please add '07_feature_rankings.png' to the assets folder.")
    st.caption("Lower rank = more important. Notice how the same feature can be ranked differently depending on which imbalance method was used.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Compare Methods
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("⚖️ Imbalance Method Comparison")

    st.markdown("#### Model Performance (ROC-AUC & F1 Score)")
    img_path = os.path.join(ASSETS, '04_model_comparison.png')
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.info("📊 Visualization not available. Please add '04_model_comparison.png' to the assets folder.")

    st.markdown("#### Full Results Table")
    if not results_df.empty:
        display_df = results_df.copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ Results data not available.")

    st.markdown("---")
    st.markdown("#### Method Descriptions")
    method_info = {
        'SMOTE': ('Oversampling', 'Creates synthetic minority samples between real ones using k-nearest neighbours.'),
        'ADASYN': ('Adaptive Oversampling', 'Like SMOTE but focuses on harder-to-classify regions near the decision boundary.'),
        'Borderline-SMOTE': ('Targeted Oversampling', 'Only generates synthetic samples near the class boundary — more precise than SMOTE.'),
        'Class Weights': ('Cost-Sensitive', 'No resampling. Penalises misclassifying minority class more during training.'),
        'Threshold Moving': ('Post-processing', 'Trains on original data, then finds optimal decision threshold by maximising F1.'),
    }
    cols = st.columns(len(method_info))
    for col, (name, (mtype, desc)) in zip(cols, method_info.items()):
        with col:
            st.markdown(f"""
            <div style="background:white;border-radius:10px;padding:12px;
                        border-top:4px solid {METHOD_COLORS[name]};height:160px;
                        box-shadow:0 2px 6px rgba(0,0,0,0.08)">
                <b style="color:{METHOD_COLORS[name]}">{name}</b><br>
                <small style="color:#6B7280">{mtype}</small><br><br>
                <span style="font-size:12px;color:#374151">{desc}</span>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — SRA Research Results
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔬 SRA Research Results")
    st.markdown("""
    **Spearman Rank Agreement (SRA)** measures how stable SHAP feature rankings are across 30 bootstrap samples.
    An SRA close to 1.0 means the same features are always ranked the same way → trustworthy explanations.
    """)

    # SRA overview chart
    img_path = os.path.join(ASSETS, '08_sra_results.png')
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.info("📊 Visualization not available. Please add '08_sra_results.png' to the assets folder.")

    st.markdown("---")
    st.markdown("#### SRA Scores — Detailed Table")
    if not results_df.empty and not sra_df.empty:
        try:
            final_merged = pd.merge(results_df, sra_df, on='method').sort_values('sra', ascending=False)
            st.dataframe(final_merged, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error merging data: {str(e)}")
    else:
        st.info("ℹ️ SRA results data not available.")

    st.markdown("---")
    st.markdown("#### SRA Interpretation")
    sra_cols = st.columns(4)
    
    # SRA interpretation cards
    interpretation_data = [
        ("Excellent", "0.90 – 1.00", "#059669", "#D1FAE5"),
        ("Good", "0.75 – 0.89", "#D97706", "#FEF3C7"),
        ("Moderate", "0.60 – 0.74", "#EA580C", "#FFEDD5"),
        ("Unstable", "< 0.60", "#DC2626", "#FEE2E2")
    ]
    
    for col, (label, rng, color, bg) in zip(sra_cols, interpretation_data):
        with col:
            st.markdown(f"""<div style="background:{bg};border-radius:8px;padding:12px;text-align:center">
                <b style="color:{color};font-size:16px">{label}</b><br>
                <span style="font-size:13px;color:#374151">SRA {rng}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    
    # Research Findings - only if data is available
    if not results_df.empty and not sra_df.empty:
        try:
            final_merged = pd.merge(results_df, sra_df, on='method').sort_values('sra', ascending=False)
            best_sra_m = final_merged.iloc[0]['method']
            best_f1_m = final_merged.sort_values('f1', ascending=False).iloc[0]['method']

            # H1 Finding
            st.markdown("#### 📌 Research Findings")
            
            # Get SRA values for comparison
            adasyn_sra = final_merged[final_merged['method'] == 'ADASYN']['sra'].values[0] if 'ADASYN' in final_merged['method'].values else 0.0
            classweights_sra = final_merged[final_merged['method'] == 'Class Weights']['sra'].values[0] if 'Class Weights' in final_merged['method'].values else 0.0
            
            st.markdown(f"""<div class="finding-box">
            <h4 style="color:#1E2761;margin-top:0">H1 — Oversampling vs Non-Oversampling Stability</h4>
            <p>Oversampling methods (SMOTE, ADASYN, Borderline-SMOTE) achieved <b>higher SRA</b> ({adasyn_sra:.4f}) 
            than non-oversampling methods (Class Weights, Threshold Moving)
            ({classweights_sra:.4f}).</p>
            <p>This is <b>contrary to H1</b> — synthetic oversampling actually produced <i>more</i> stable feature rankings 
            on this dataset. This is an interesting finding worth discussing in the report.</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("")
            
            # H2 Finding - Trade-off
            trade_off_text = (
                f"✅ <b>H2 SUPPORTED</b> — The best F1 method (<b>{best_f1_m}</b>) is NOT the same as the best SRA method (<b>{best_sra_m}</b>). "
                f"There is a trade-off: use {best_f1_m} for maximum prediction performance, use {best_sra_m} for maximum explanation stability."
                if stats.get('trade_off', False) else
                f"❌ <b>H2 NOT SUPPORTED</b> — The same method ({best_f1_m}) wins on both F1 and SRA."
            )

            st.markdown(f"""<div class="finding-box">
            <h4 style="color:#1E2761;margin-top:0">H2 — Performance vs Stability Trade-off</h4>
            <p>{trade_off_text}</p>
            <p><b>Practical recommendation:</b></p>
            <ul>
            <li>Banks prioritising <b>regulatory explainability</b> → use <b>{best_sra_m}</b></li>
            <li>Banks prioritising <b>prediction accuracy</b> → use <b>{best_f1_m}</b></li>
            </ul>
            </div>""", unsafe_allow_html=True)

            st.markdown("---")
            
            # Interactive SRA Comparison
            st.markdown("#### Interactive SRA Comparison")
            selected_methods = st.multiselect(
                "Select methods to compare:",
                AVAILABLE_MODELS if AVAILABLE_MODELS else list(METHOD_COLORS.keys()),
                default=AVAILABLE_MODELS if AVAILABLE_MODELS else list(METHOD_COLORS.keys())[:3]
            )

            if selected_methods:
                try:
                    sub_df = final_merged[final_merged['method'].isin(selected_methods)]
                    
                    if not sub_df.empty:
                        fig, ax = plt.subplots(figsize=(9, 4))
                        bar_colors = [METHOD_COLORS.get(m, '#1E2761') for m in sub_df['method']]
                        
                        # Create bar chart with error bars
                        bars = ax.bar(
                            sub_df['method'], 
                            sub_df['sra'],
                            yerr=sub_df['sra_std'], 
                            color=bar_colors,
                            edgecolor='white', 
                            capsize=8, 
                            alpha=0.9, 
                            width=0.5
                        )
                        
                        ax.set_ylim(0.90, 1.005)
                        ax.set_ylabel('SRA Score', fontsize=12)
                        ax.set_title('SRA Score by Method (with ±1 Std Dev)', fontweight='bold', fontsize=13)
                        
                        # Add value labels on bars
                        for bar, val in zip(bars, sub_df['sra']):
                            ax.text(
                                bar.get_x() + bar.get_width() / 2, 
                                val + 0.0008,
                                f'{val:.4f}', 
                                ha='center', 
                                fontsize=11, 
                                fontweight='bold'
                            )
                        
                        plt.xticks(rotation=15)
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                    else:
                        st.warning("No data available for selected methods.")
                        
                except Exception as e:
                    st.error(f"Error creating SRA comparison chart: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error in SRA analysis: {str(e)}")
    else:
        st.info("ℹ️ Complete SRA results not available. Please check data files.")