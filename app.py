import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# ==========================================
# 1. PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Advanced Credit Card Fraud Detection System")
st.markdown("This system uses a **Soft-Voting Ensemble** (Keras Neural Network + XGBoost) to predict the probability of a fraudulent transaction in real-time.")

# 📖 NEW: Collapsible User Guide for New Users
with st.expander("📖 New User? Click here to see What to Do", expanded=False):
    st.markdown("""
    ### 🚀 Quick Start Guide
    Welcome to the Fraud Monitoring Dashboard! This tool simulates how an enterprise banking system evaluates credit card transaction risks using Machine Learning. Follow these simple steps to run a test:
    
    1. **Choose a Scenario (Sidebar):** On the left panel under **🧪 Quick Test Presets**, choose a preset. 
       * Selective profiles like **'Fraudulent Transaction Preset'** will automatically shift the sliders to reflect extreme anomalies.
       * Selecting **'Normal Transaction Preset'** will populate typical safe banking patterns.
    2. **Review or Modify Features (Main Panel):** You can manually adjust the **Transaction Amount ($)** or tweak any of the **V1 to V28 Latent Features** (which represent hidden, PCA-anonymized transaction characteristics like location variance, velocity, and device signatures).
    3. **Run the AI Analysis:** Scroll to the bottom and click the big red **🔍 Analyze Transaction Profile** button.
    4. **Read the Diagnostic Report:** The app will instantly generate a breakdown showing individual risk percentages from our Deep Neural Network, our XGBoost Classifier, and the final blended **Ensemble Engine Verdict**.
    
    *💡 Tip: You can play with the **Model Decision Threshold sliders** in the sidebar to control how strict or sensitive the AI engines are when flagging anomalies!*
    """)

st.write("---")

# ==========================================
# 2. CACHED MODEL LOADING (Prevents Reloading Spikes)
# ==========================================
@st.cache_resource
def load_artifacts():
    # Load Scaler and XGBoost using joblib
    scaler = joblib.load('scaler.pkl')
    xgb_model = joblib.load('xgb_model.pkl')
    
    # Load Keras Neural Network
    keras_model = tf.keras.models.load_model('keras_model.keras')
    
    return scaler, xgb_model, keras_model

try:
    scaler, xgb_model, keras_model = load_artifacts()
    st.sidebar.success("✅ All 3 Model Artifacts Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading models: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR CONTROLS & THRESHOLDS
# ==========================================
st.sidebar.header("⚙️ Model Decision Thresholds")
# Pre-set to balanced decision boundaries
keras_th = st.sidebar.slider("Keras Threshold", 0.0, 1.0, 0.50, 0.01)
xgb_th = st.sidebar.slider("XGBoost Threshold", 0.0, 1.0, 0.50, 0.01)
ensemble_th = st.sidebar.slider("Ensemble Threshold", 0.0, 1.0, 0.50, 0.01)

# Quick Sample Data injector for Testing
st.sidebar.write("---")
st.sidebar.header("🧪 Quick Test Presets")
preset = st.sidebar.selectbox("Choose a sample scenario:", ["Manual Input", "Normal Transaction Preset", "Fraudulent Transaction Preset"])

# Default initializations for features
default_features = [0.0] * 28
time_val = 0.0
amount_val = 10.0

if preset == "Normal Transaction Preset":
    time_val = 406.0
    amount_val = 1.98
    default_features = [-0.33, 1.12, -0.21, -0.62, 0.57, -0.12, 0.22, -0.01, -0.11, -0.42, -0.85, 0.33, 0.12, -0.42, 0.92, 0.42, -0.12, -0.22, 0.15, -0.05, -0.12, -0.32, 0.15, -0.05, 0.12, -0.21, 0.05, 0.01]
elif preset == "Fraudulent Transaction Preset":
    time_val = 406.0
    amount_val = 0.00  
    default_features = [-2.31, 1.66, -5.64, 4.12, -2.44, -1.98, -4.42, 1.52, -2.41, -6.31, 3.44, -7.12, 1.22, -8.11, -0.52, -4.33, -7.21, -2.55, 0.45, 0.12, 0.52, -0.12, -0.45, 0.12, 0.21, 0.52, 0.35, 0.15]

# ==========================================
# 4. INPUT INTERFACE GENERATION (GRID LAYOUT)
# ==========================================
st.header("📥 Transaction Details Input")

col1, col2 = st.columns(2)
with col1:
    Time = st.number_input("Transaction Time (Seconds elapsed since first txn)", min_value=0.0, value=float(time_val))
with col2:
    Amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=float(amount_val))

st.markdown("### V1 to V28 Latent Features (PCA Transformed components)")

v_inputs = []
v_cols = st.columns(4)
for i in range(28):
    with v_cols[i % 4]:
        val = st.slider(f"Feature V{i+1}", min_value=-30.0, max_value=30.0, value=float(default_features[i]), step=0.01)
        v_inputs.append(val)

# ==========================================
# 5. REAL-TIME PREDICTION ENGINE (VERIFIED)
# ==========================================
if st.button("🔍 Analyze Transaction Profile", type="primary"):
    
    # 📑 CROSS-VERIFICATION CHECKPOINT 1: Scaler Dimension Match
    # Passing data as a 2D array [[value]] to protect the 1-feature scaler configuration.
    scaled_amount_value = float(scaler.transform([[Amount]])[0][0])
    
    # 📑 CROSS-VERIFICATION CHECKPOINT 2: Feature Matrix Structural Integrity
    # We construct the 29 features matching the exact training columns order: V1 -> V28, then normAmount.
    # The 'Time' feature is excluded entirely to ensure a clean (1, 29) shape matrix.
    feature_names = [f'V{i}' for i in range(1, 29)] + ['normAmount']
    processed_row_data = v_inputs + [scaled_amount_value]
    
    prepared_df = pd.DataFrame([processed_row_data], columns=feature_names)
    
    # Generate explicit probability predictions
    keras_p = float(keras_model.predict(prepared_df)[0][0])
    xgb_p = float(xgb_model.predict_proba(prepared_df)[:, 1][0])
    
    # Soft-voting blended calculation
    ensemble_p = (keras_p + xgb_p) / 2.0
    
    # Assign statuses based on custom sidebar sliders
    keras_status = "🔴 FRAUD DETECTED" if keras_p > keras_th else "🟢 NORMAL"
    xgb_status = "🔴 FRAUD DETECTED" if xgb_p > xgb_th else "🟢 NORMAL"
    ensemble_status = "🚨 CRITICAL RISK: FRAUD DETECTED" if ensemble_p > ensemble_th else "✅ APPROVED: NORMAL TRANSACTION"

    # Display Results Dashboard
    st.write("---")
    st.header("📊 Multi-Model Diagnostic Report")
    
    if ensemble_p > ensemble_th:
        st.error(f"### Final Ensemble Verdict: {ensemble_status}")
    else:
        st.success(f"### Final Ensemble Verdict: {ensemble_status}")
        
    st.write(f"**Combined Risk Factor Probability:** `{ensemble_p * 100:.2f}%`")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.subheader("🧠 Keras Deep ANN")
        st.metric(label="Fraud Probability", value=f"{keras_p * 100:.2f}%")
        st.write(f"Status Verdict: **{keras_status}**")
        st.progress(keras_p)
        
    with m_col2:
        st.subheader("🌲 XGBoost Classifier")
        st.metric(label="Fraud Probability", value=f"{xgb_p * 100:.2f}%")
        st.write(f"Status Verdict: **{xgb_status}**")
        st.progress(xgb_p)
        
    with m_col3:
        st.subheader("⛓️ Ensemble Engine")
        st.metric(label="Blended Probability", value=f"{ensemble_p * 100:.2f}%")
        st.write(f"Decision Boundary: `{ensemble_th}`")
        st.progress(ensemble_p)