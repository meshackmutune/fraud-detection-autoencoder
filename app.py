# app.py - COLORFUL THEME + EASY VISUALS
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import firebase_admin
from firebase_admin import credentials, auth, firestore
from model_utils import load_model_and_assets, predict_transaction, INPUT_DIM

# ---------------------------------------------------------
# 0. PAGE CONFIG + COLORFUL THEME
# ---------------------------------------------------------
st.set_page_config(page_title="Secure Bank", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #6366F1 0%, #10B981 100%);
        background-attachment: fixed;
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Big Numbers */
    .big-number { font-size: 48px; font-weight: 700; margin: 0; }
    .label { font-size: 16px; opacity: 0.9; margin-top: 5px; }
    
    /* Buttons */
    .stButton > button {
        border-radius: 12px !important; padding: 12px 24px !important;
        font-weight: 600 !important; text-transform: uppercase;
        background: linear-gradient(45deg, #6366F1, #10B981) !important;
        color: white !important; border: none !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
    }
    
    /* Traffic Light */
    .light { font-size: 80px; line-height: 1; }
    .green-glow { text-shadow: 0 0 20px #10B981; }
    .red-glow { text-shadow: 0 0 20px #EF4444; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. FIREBASE & MODEL
# ---------------------------------------------------------
def init_firebase():
    if not firebase_admin._apps:
        cred_dict = dict(st.secrets["firebase"])
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
    if "db" not in st.session_state:
        st.session_state.db = firestore.client()

init_firebase()
MODEL, SCALER, THRESHOLD = load_model_and_assets()
if MODEL is None: st.stop()

# ---------------------------------------------------------
# 2. AUTH
# ---------------------------------------------------------
def login(email, pwd):
    try:
        user = auth.get_user_by_email(email)
        st.session_state.update(uid=user.uid, logged_in=True, is_admin=(email=="admin@securebank.com"))
        st.balloons()
    except: st.error("Login failed.")

def register(email, pwd):
    try:
        user = auth.create_user(email=email, password=pwd)
        st.session_state.update(uid=user.uid, logged_in=True, is_admin=False)
        st.success("Welcome!")
    except: st.error("Register failed.")

def logout():
    for k in ["uid", "logged_in", "is_admin"]: st.session_state.pop(k, None)
    st.success("Logged out.")

# ---------------------------------------------------------
# 3. LOGIN PAGE – COLORFUL WELCOME
# ---------------------------------------------------------
if not st.session_state.get("logged_in", False):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://img.icons8.com/fluency/256/bank-building.png", width=200)
    with col2:
        st.markdown("""
        <h1 style='color: white; text-shadow: 0 2px 10px rgba(0,0,0,0.3);'>
            Secure Bank
        </h1>
        <p style='color: white; font-size: 1.3rem; opacity: 0.9;'>
            AI-Powered Fraud Protection
        </p>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<h3 style='color: white;'>Login</h3>", unsafe_allow_html=True)
        email = st.text_input("Email", placeholder="you@securebank.com")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"): login(email, pwd); st.rerun()
        if st.button("Register"): register(email, pwd); st.rerun()
    st.stop()

# ---------------------------------------------------------
# 4. LOGGED IN – COLORFUL DASHBOARD
# ---------------------------------------------------------
st.sidebar.markdown(f"""
<div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 12px; color: white;">
    <p><b>User:</b> {st.session_state.uid[:8]}...</p>
</div>
""", unsafe_allow_html=True)
if st.sidebar.button("Logout"): logout(); st.rerun()

pages = ["Check Transaction"]
if st.session_state.get("is_admin"): pages.append("Admin Center")
page = st.sidebar.radio("Menu", pages)

# ---------------------------------------------------------
# 5. CUSTOMER: RAINBOW TRAFFIC LIGHT + METER
# ---------------------------------------------------------
if page == "Check Transaction":
    st.markdown("<h1 style='color: white; text-align: center;'>Check Your Transaction</h1>", unsafe_allow_html=True)

    amount = st.number_input("Amount (USD)", min_value=0.0, value=100.0, step=10.0, help="Enter any amount")
    
    if st.button("Verify Now", use_container_width=True):
        vec = np.zeros(INPUT_DIM); vec[29] = amount
        err, fraud = predict_transaction(MODEL, SCALER, THRESHOLD, vec)
        st.session_state.result = (err, fraud, amount)

    if 'result' in st.session_state:
        err, fraud, amount = st.session_state.result

        # TRAFFIC LIGHT
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="light {'green-glow' if not fraud else 'red-glow'}">
                    ●
                </div>
                <p class="big-number" style="color: {'#10B981' if not fraud else '#EF4444'};">
                    {'APPROVED' if not fraud else 'BLOCKED'}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # RISK METER
            risk = min(100, int(err * 100))
            color = "green" if not fraud else "red"
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk,
                delta={'reference': 20},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': color},
                       'steps': [{'range': [0, 30], 'color': '#10B981'},
                                 {'range': [30, 70], 'color': '#F59E0B'},
                                 {'range': [70, 100], 'color': '#EF4444'}]},
                title={'text': "Risk Level"}
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            # COMPARE
            avg = 18
            fig = go.Figure(go.Bar(
                x=['You', 'Average User'],
                y=[risk, avg],
                marker_color=['#6366F1' if risk > avg else '#10B981', '#94A3B8'],
                text=[f"{risk}%", f"{avg}%"],
                textposition='outside'
            ))
            fig.update_layout(title="Your Risk vs Average", yaxis_title="Risk %", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div style='text-align: center; color: white; margin-top: 20px;'>
            <p><b>Amount:</b> ${amount:,.2f} | <b>AI Score:</b> {err:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 6. ADMIN: RAINBOW DASHBOARD
# ---------------------------------------------------------
elif page == "Admin Center":
    st.markdown("<h1 style='color: white; text-align: center;'>Fraud Control Center</h1>", unsafe_allow_html=True)

    # Mock Data
    total = 12500
    fraud = 480
    false = 420
    normal = total - fraud - false

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Fraud Caught", fraud, "#EF4444"),
        ("False Alerts", false, "#F59E0B"),
        ("Safe Txns", normal, "#10B981"),
        ("Total", total, "#6366F1")
    ]
    for col, (label, value, color) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="glass-card">
                <p class="big-number" style="color: {color};">{value:,}</p>
                <p class="label" style="color: white;">{label}</p>
            </div>
            """, unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        # PIE CHART
        fig = px.pie(
            values=[normal, fraud, false],
            names=['Safe', 'Fraud', 'False Alert'],
            hole=0.4,
            color_discrete_sequence=['#10B981', '#EF4444', '#F59E0B'],
            title="Transaction Breakdown"
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        # LINE CHART
        dates = pd.date_range("2025-04-01", periods=7).strftime("%a")
        frauds = [60, 85, 70, 110, 65, 95, 75]
        fig = px.area(x=dates, y=frauds, title="Daily Fraud Attempts",
                      color_discrete_sequence=['#EF4444'])
        fig.update_layout(yaxis_title="Fraud Count")
        st.plotly_chart(fig, use_container_width=True)

    # THRESHOLD TUNER
    st.markdown("### AI Sensitivity Control")
    new_thr = st.slider("Risk Threshold", 0.3, 1.5, THRESHOLD, 0.01, help="Lower = more sensitive")
    colT1, colT2 = st.columns(2)
    with colT1:
        st.metric("Current", f"{THRESHOLD:.3f}", delta=None)
    with colT2:
        st.metric("New", f"{new_thr:.3f}", delta=f"{new_thr - THRESHOLD:+.3f}")
