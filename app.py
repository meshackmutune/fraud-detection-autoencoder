# app.py - FULL WITH VISUALIZATIONS
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import firebase_admin
from firebase_admin import credentials, auth, firestore
from model_utils import load_model_and_assets, predict_transaction, INPUT_DIM

# ---------------------------------------------------------
# 0. PAGE CONFIG + THEME
# ---------------------------------------------------------
st.set_page_config(page_title="Secure Bank", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    :root { --primary-green: #10B981; --dark-green: #047857; }
    .stButton>button { background: var(--primary-green) !important; color: white !important; }
    .stButton>button:hover { background: var(--dark-green) !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. FIREBASE & MODEL
# ---------------------------------------------------------
def init_firebase():
    if not firebase_admin._apps:
        try:
            # Convert TOML section to plain dict
            cred_dict = dict(st.secrets["firebase"])
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Firebase init failed: {e}")
            st.stop()
    if "db" not in st.session_state:
        st.session_state.db = firestore.client()

init_firebase()
MODEL, SCALER, THRESHOLD = load_model_and_assets()
if MODEL is None:
    st.stop()

# ---------------------------------------------------------
# 2. AUTH
# ---------------------------------------------------------
def login(email, pwd):
    try:
        user = auth.get_user_by_email(email)
        st.session_state.update(uid=user.uid, logged_in=True, is_admin=(email=="admin@securebank.com"))
        st.success("Logged in!")
    except: st.error("Login failed.")

def register(email, pwd):
    try:
        user = auth.create_user(email=email, password=pwd)
        st.session_state.update(uid=user.uid, logged_in=True, is_admin=False)
        st.success("Registered!")
    except: st.error("Register failed.")

def logout():
    for k in ["uid", "logged_in", "is_admin"]: st.session_state.pop(k, None)
    st.success("Logged out.")

# ---------------------------------------------------------
# 3. LOGIN PAGE WITH VISUAL WELCOME
# ---------------------------------------------------------
if not st.session_state.get("logged_in", False):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:15px;">
            <img src="https://img.icons8.com/fluency/96/bank-building.png" width="70">
            <h2 style="color:#10B981; margin:8px 0 0;">Secure Bank</h2>
            <p style="color:#10B981; font-size:0.95rem;">Fraud Detection Portal</p>
            <hr style="border-top:2px solid #10B981;">
        </div>
        """, unsafe_allow_html=True)
        email = st.text_input("Email", key="email")
        pwd = st.text_input("Password", type="password", key="pwd")
        c1, c2 = st.columns(2)
        if c1.button("Login"): login(email, pwd); st.rerun()
        if c2.button("Register"): register(email, pwd); st.rerun()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://img.icons8.com/fluency/256/bank-building.png", width=200)
    with col2:
        st.markdown("""
        <h1 style='color:#10B981;'>Welcome to Secure Bank</h1>
        <p style='font-size:1.2rem; color:#047857;'>
            <b>Deep Autoencoder AI</b> protects your money.<br>
            <i>86% fraud detection • 4% false alerts</i>
        </p>
        """, unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------
# 4. LOGGED IN UI
# ---------------------------------------------------------
st.sidebar.success(f"User: {st.session_state.uid[:8]}...")
if st.sidebar.button("Logout"): logout(); st.rerun()

pages = ["Customer Interface"]
if st.session_state.get("is_admin"): pages.insert(0, "Administrator Dashboard")
page = st.sidebar.radio("Menu", pages)

# ---------------------------------------------------------
# 5. CUSTOMER INTERFACE + VISUALIZATIONS
# ---------------------------------------------------------
if page == "Customer Interface":
    st.title("Transaction Verification")
    
    colA, colB = st.columns([1, 3])
    with colA:
        amt = st.text_input("Amount (USD)", "100.00")
        try: amount = float(amt)
        except: amount = 0.0; st.warning("Invalid")

        if st.button("Verify", type="primary", use_container_width=True):
            vec = np.zeros(INPUT_DIM); vec[29] = amount
            with st.spinner("Analyzing..."):
                err, fraud = predict_transaction(MODEL, SCALER, THRESHOLD, vec)
            st.session_state.last_err = err
            st.session_state.last_fraud = fraud

    with colB:
        if 'last_err' in st.session_state:
            err = st.session_state.last_err
            fraud = st.session_state.last_fraud

            # 1. GAUGE CHART
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=err, domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [0, 2]}, 'bar': {'color': "red" if fraud else "green"},
                       'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': THRESHOLD}},
                title={'text': "Reconstruction Error (MSE)"}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # 2. RESULT
            if fraud:
                st.error("FRAUD DETECTED")
                st.metric("Status", "BLOCKED", delta="HIGH RISK")
            else:
                st.success("APPROVED")
                st.metric("Status", "APPROVED", delta="LOW RISK")

    # 3. TRANSACTION HISTORY (MOCK)
    st.markdown("### Your Recent Activity")
    history = pd.DataFrame({
        "Time": pd.date_range("2025-04-01", periods=10, freq="H"),
        "Amount": np.random.uniform(10, 1000, 10),
        "MSE": np.random.uniform(0.1, 1.5, 10),
        "Status": ["Approved"]*8 + ["Blocked", "Approved"]
    })
    history["Fraud"] = history["MSE"] > THRESHOLD
    fig_line = px.line(history, x="Time", y="MSE", color="Status", markers=True,
                       title="Transaction Risk Over Time")
    fig_line.add_hline(y=THRESHOLD, line_dash="dash", line_color="red", annotation_text="Threshold")
    st.plotly_chart(fig_line, use_container_width=True)

    # 4. HEATMAP
    st.markdown("### Fraud Risk by Amount & Time")
    heatmap = pd.pivot_table(history, values="MSE", index=history["Time"].dt.hour, columns=pd.cut(history["Amount"], 5), aggfunc="mean")
    fig_heat = px.imshow(heatmap, text_auto=True, color_continuous_scale="RdYlGn_r", aspect="auto")
    st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------
# 6. ADMIN DASHBOARD + VISUALIZATIONS
# ---------------------------------------------------------
elif page == "Administrator Dashboard":
    st.title("Fraud Operations Center")

    # Mock test data
    np.random.seed(42)
    normal_mse = np.random.normal(0.35, 0.15, 5000)
    fraud_mse = np.random.normal(1.2, 0.4, 400)
    all_mse = np.concatenate([normal_mse, fraud_mse])
    labels = ["Normal"]*5000 + ["Fraud"]*400
    df_test = pd.DataFrame({"MSE": all_mse, "Class": labels})

    col1, col2 = st.columns(2)
    with col1:
        # 1. MSE DISTRIBUTION
        fig_hist = px.histogram(df_test, x="MSE", color="Class", nbins=50, barmode="overlay",
                                title="Reconstruction Error Distribution", opacity=0.7)
        fig_hist.add_vline(x=THRESHOLD, line_dash="dash", line_color="red")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # 2. THRESHOLD IMPACT
        thresholds = np.linspace(0.3, 1.5, 100)
        recall, fpr = [], []
        for t in thresholds:
            pred = (df_test[df_test["Class"]=="Fraud"]["MSE"] > t).mean()
            fp = (df_test[df_test["Class"]=="Normal"]["MSE"] > t).mean()
            recall.append(pred)
            fpr.append(fp)
        fig_impact = go.Figure()
        fig_impact.add_trace(go.Scatter(x=fpr, y=recall, mode="lines", name="Threshold Curve"))
        fig_impact.add_scatter(x=[(df_test[df_test["Class"]=="Normal"]["MSE"] > THRESHOLD).mean()],
                               y=[(df_test[df_test["Class"]=="Fraud"]["MSE"] > THRESHOLD).mean()],
                               mode="markers", marker=dict(color="red", size=10), name="Current")
        fig_impact.update_layout(title="Recall vs False Positive Rate", xaxis_title="FPR", yaxis_title="Recall")
        st.plotly_chart(fig_impact, use_container_width=True)

    # 3. LIVE FRAUD MAP (MOCK)
    st.markdown("### Global Fraud Activity (Last 24h)")
    geo_df = pd.DataFrame({
        "lat": np.random.uniform(-50, 50, 50),
        "lon": np.random.uniform(-120, 120, 50),
        "amount": np.random.uniform(100, 10000, 50),
        "risk": np.random.choice(["Low", "Medium", "High"], 50)
    })
    fig_map = px.scatter_geo(geo_df, lat="lat", lon="lon", size="amount", color="risk",
                             title="Fraud Heatmap", color_discrete_map={"Low":"green", "Medium":"orange", "High":"red"})
    st.plotly_chart(fig_map, use_container_width=True)

    # 4. THRESHOLD TUNER
    st.markdown("### Tune Detection Threshold")
    new_thr = st.slider("MSE Threshold", 0.3, 1.5, THRESHOLD, 0.01)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.metric("Current Threshold", f"{THRESHOLD:.3f}")
    with col_t2:
        st.metric("Proposed", f"{new_thr:.3f}", delta=f"{new_thr - THRESHOLD:+.3f}")

