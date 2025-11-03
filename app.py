# app.py - EASY VISUALIZATIONS
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import firebase_admin
from firebase_admin import credentials, auth, firestore
from model_utils import load_model_and_assets, predict_transaction, INPUT_DIM

# ---------------------------------------------------------
# 0. PAGE CONFIG + SIMPLE STYLE
# ---------------------------------------------------------
st.set_page_config(page_title="Secure Bank", layout="wide")
st.markdown("""
<style>
    .big-font { font-size: 50px !important; font-weight: bold; text-align: center; }
    .medium-font { font-size: 24px !important; text-align: center; }
    .green { color: #10B981; }
    .red { color: #EF4444; }
    .card {
        padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. FIREBASE INIT (FIXED)
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
        st.success("Logged in!")
    except: st.error("Login failed.")

def register(email, pwd):
    try:
        user = auth.create_user(email=email, password=pwd)
        st.session_state.update(uid=user.uid, logged_in=True, is_admin=False)
        st.success("Account created!")
    except: st.error("Register failed.")

def logout():
    for k in ["uid", "logged_in", "is_admin"]: st.session_state.pop(k, None)
    st.success("Logged out.")

# ---------------------------------------------------------
# 3. LOGIN PAGE – SIMPLE WELCOME
# ---------------------------------------------------------
if not st.session_state.get("logged_in", False):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://img.icons8.com/fluency/256/bank-building.png", width=180)
    with col2:
        st.markdown("<h1 class='green'>Welcome to Secure Bank</h1>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>Your money is safe with AI protection.</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/bank-building.png", width=70)
        st.markdown("<h3 class='green'>Login</h3>", unsafe_allow_html=True)
        email = st.text_input("Email")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"): login(email, pwd); st.rerun()
        if st.button("Register"): register(email, pwd); st.rerun()
    st.stop()

# ---------------------------------------------------------
# 4. LOGGED IN
# ---------------------------------------------------------
st.sidebar.success(f"User: {st.session_state.uid[:8]}...")
if st.sidebar.button("Logout"): logout(); st.rerun()

pages = ["Check Transaction"]
if st.session_state.get("is_admin"): pages.append("Admin Dashboard")
page = st.sidebar.radio("Go to", pages)

# ---------------------------------------------------------
# 5. CUSTOMER: TRAFFIC LIGHT + RISK METER
# ---------------------------------------------------------
if page == "Check Transaction":
    st.markdown("<h1 class='green'>Check Your Transaction</h1>", unsafe_allow_html=True)

    amount = st.number_input("Amount (USD)", min_value=0.0, value=100.0, step=10.0)
    if st.button("Verify Now", type="primary", use_container_width=True):
        vec = np.zeros(INPUT_DIM); vec[29] = amount
        err, fraud = predict_transaction(MODEL, SCALER, THRESHOLD, vec)
        st.session_state.result = (err, fraud, amount)

    if 'result' in st.session_state:
        err, fraud, amount = st.session_state.result

        col1, col2, col3 = st.columns(3)
        with col1:
            # TRAFFIC LIGHT
            color = "red" if fraud else "green"
            st.markdown(f"""
            <div class="card" style="background-color: {'#FEE2E2' if fraud else '#DCFCE7'};">
                <div class="big-font" style="color: {color};">●</div>
                <p><b>{'BLOCKED' if fraud else 'APPROVED'}</b></p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # RISK METER (0-100)
            risk_score = min(100, int(err * 100))
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_score,
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "red" if fraud else "green"}},
                title={'text': "Risk Score (%)"}
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            # COMPARE TO AVERAGE
            avg_risk = 15  # mock
            fig = go.Figure(go.Bar(
                x=['Your Risk', 'Average'],
                y=[risk_score, avg_risk],
                marker_color=['red' if fraud else 'green', 'lightgray']
            ))
            fig.update_layout(title="Your Risk vs Average", yaxis_title="Risk %")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Amount:** ${amount:,.2f} | **Error:** {err:.4f} | **Threshold:** {THRESHOLD:.4f}")

# ---------------------------------------------------------
# 6. ADMIN: PIE + BIG NUMBERS + LINE
# ---------------------------------------------------------
elif page == "Admin Dashboard":
    st.markdown("<h1 class='green'>Fraud Control Center</h1>", unsafe_allow_html=True)

    # Mock data
    total_tx = 10000
    fraud_caught = 430
    false_alerts = 380
    normal_tx = total_tx - fraud_caught - false_alerts

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="card" style="background-color: #DCFCE7;">
            <p class="big-font green">{fraud_caught}</p>
            <p>Fraud Caught</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card" style="background-color: #FEE2E2;">
            <p class="big-font red">{false_alerts}</p>
            <p>False Alerts</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="card" style="background-color: #E0E7FF;">
            <p class="big-font" style="color: #4F46E5;">{total_tx:,}</p>
            <p>Total Checked</p>
        </div>
        """, unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        # PIE CHART
        fig = px.pie(values=[normal_tx, fraud_caught, false_alerts],
                     names=['Normal', 'Fraud', 'False Alert'],
                     color_discrete_sequence=['#10B981', '#EF4444', '#F59E0B'],
                     title="Transaction Breakdown")
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        # LINE CHART (mock daily fraud)
        dates = pd.date_range("2025-04-01", periods=7).strftime("%b %d")
        fraud_daily = [50, 70, 60, 90, 55, 80, 65]
        fig = px.line(x=dates, y=fraud_daily, markers=True, title="Daily Fraud Attempts")
        fig.update_traces(line_color="#EF4444")
        st.plotly_chart(fig, use_container_width=True)

    # THRESHOLD SLIDER
    st.markdown("### Adjust AI Sensitivity")
    new_thr = st.slider("Risk Threshold", 0.5, 1.5, THRESHOLD, 0.05)
    st.write(f"Current: **{THRESHOLD:.3f}** → New: **{new_thr:.3f}**")
