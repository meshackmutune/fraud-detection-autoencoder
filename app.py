# app.py - EASY VISUALS (NO CITY MAP)
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    .big-light { font-size: 120px; text-align: center; line-height: 1; }
    .big-number { font-size: 48px; font-weight: bold; text-align: center; }
    .label { font-size: 18px; text-align: center; color: #555; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. FIREBASE & MODEL (FIXED)
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
if MODEL is None:
    st.stop()

# ---------------------------------------------------------
# 2. AUTH
# ---------------------------------------------------------
def login(email, pwd):
    try:
        user = auth.get_user_by_email(email)
        st.session_state.update(uid=user.uid, logged_in=True,
                                is_admin=(email == "admin@securebank.com"))
        st.success("Logged in!")
    except:
        st.error("Login failed.")

def register(email, pwd):
    try:
        user = auth.create_user(email=email, password=pwd)
        st.session_state.update(uid=user.uid, logged_in=True, is_admin=False)
        st.success("Registered!")
    except:
        st.error("Register failed.")

def logout():
    for k in ["uid", "logged_in", "is_admin"]:
        st.session_state.pop(k, None)
    st.success("Logged out.")

# ---------------------------------------------------------
# 3. LOGIN PAGE
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
            <b>AI protects your money 24/7</b><br>
            <i>86% fraud caught • Only 4% false alerts</i>
        </p>
        """, unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------
# 4. LOGGED IN
# ---------------------------------------------------------
st.sidebar.success(f"User: {st.session_state.uid[:8]}...")
if st.sidebar.button("Logout"): logout(); st.rerun()

pages = ["Check Transaction"]
if st.session_state.get("is_admin"):
    pages.insert(0, "Admin Dashboard")
page = st.sidebar.radio("Menu", pages)

# ---------------------------------------------------------
# 5. CUSTOMER: TRAFFIC LIGHT + BAR
# ---------------------------------------------------------
if page == "Check Transaction":
    st.title("Check Your Transaction")

    colA, colB = st.columns([1, 3])
    with colA:
        amt = st.text_input("Amount (USD)", "100.00")
        try: amount = float(amt)
        except: amount = 0.0; st.warning("Invalid")

        if st.button("Verify", type="primary", use_container_width=True):
            vec = np.zeros(INPUT_DIM); vec[29] = amount
            with st.spinner("Checking..."):
                err, fraud = predict_transaction(MODEL, SCALER, THRESHOLD, vec)
            st.session_state.last_err = err
            st.session_state.last_fraud = fraud

    with colB:
        if 'last_err' in st.session_state:
            fraud = st.session_state.last_fraud

            # 1. TRAFFIC LIGHT
            light = "Green Light" if not fraud else "Red Light"
            color = "#10B981" if not fraud else "#EF4444"
            st.markdown(f"""
            <div class="big-light" style="color: {color};">
                ●
            </div>
            <p class="big-number" style="color: {color};">
                {light}
            </p>
            <p class="label">Transaction is <b>{'SAFE' if not fraud else 'BLOCKED'}</b></p>
            """, unsafe_allow_html=True)

            # 2. SIMPLE BAR: Your Risk vs Average
            your_risk = min(100, int(err * 100))
            avg_risk = 15
            fig = go.Figure(go.Bar(
                x=['Your Risk', 'Average Risk'],
                y=[your_risk, avg_risk],
                marker_color=[color, '#94A3B8'],
                text=[f"{your_risk}%", f"{avg_risk}%"],
                textposition='outside'
            ))
            fig.update_layout(title="How Risky Is This?", yaxis_title="Risk Level (%)")
            st.plotly_chart(fig, use_container_width=True)

    # 3. RECENT ACTIVITY (SIMPLE TABLE)
    st.markdown("### Your Last 5 Transactions")
    history = pd.DataFrame({
        "Amount": [120, 450, 89, 2000, 75],
        "Status": ["Approved", "Approved", "Approved", "Blocked", "Approved"],
        "Risk": [12, 18, 10, 88, 15]
    })
    history["Status"] = history["Risk"].apply(lambda x: "Blocked" if x > 70 else "Approved")
    st.dataframe(history.style.applymap(
        lambda x: f"background-color: {'#FEE2E2' if x=='Blocked' else '#DCFCE7'}",
        subset=["Status"]
    ), use_container_width=True)

# ---------------------------------------------------------
# 6. ADMIN: PIE + CARDS + TOP 5 (NO MAP)
# ---------------------------------------------------------
elif page == "Admin Dashboard":
    st.title("Fraud Control Center")

    # Mock Summary
    total = 12500
    fraud = 480
    false = 420
    normal = total - fraud - false

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style="text-align:center; padding:20px; background:#DCFCE7; border-radius:12px;">
            <p class="big-number" style="color:#10B981;">{fraud}</p>
            <p class="label">Fraud Caught</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="text-align:center; padding:20px; background:#FEE2E2; border-radius:12px;">
            <p class="big-number" style="color:#EF4444;">{false}</p>
            <p class="label">False Alerts</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="text-align:center; padding:20px; background:#E0E7FF; border-radius:12px;">
            <p class="big-number" style="color:#6366F1;">{total:,}</p>
            <p class="label">Total Checked</p>
        </div>
        """, unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        # PIE CHART
        fig = px.pie(
            values=[normal, fraud, false],
            names=['Safe', 'Fraud', 'False'],
            color_discrete_sequence=['#10B981', '#EF4444', '#F59E0B'],
            hole=0.4,
            title="All Transactions"
        )
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        # TOP 5 RISKY CITIES (MOCK) – BAR CHART
        cities = pd.DataFrame({
            "City": ["Mumbai", "Delhi", "Lagos", "São Paulo", "Jakarta"],
            "Fraud Count": [95, 82, 78, 65, 60]
        })
        fig = px.bar(cities, x="City", y="Fraud Count",
                     color="Fraud Count", color_continuous_scale="Reds",
                     title="Top 5 Risky Cities")
        st.plotly_chart(fig, use_container_width=True)

    # THRESHOLD TUNER
    st.markdown("### AI Sensitivity")
    new_thr = st.slider("Risk Threshold", 0.5, 1.5, THRESHOLD, 0.05)
    st.write(f"Current: **{THRESHOLD:.2f}** → New: **{new_thr:.2f}**")
