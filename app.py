# app.py - COLORFUL + FULLY VISIBLE (NO MAP, NO TOP 5)
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import firebase_admin
from firebase_admin import credentials, auth, firestore
from model_utils import load_model_and_assets, predict_transaction, INPUT_DIM

# ---------------------------------------------------------
# 0. PAGE CONFIG + VISIBLE THEME
# ---------------------------------------------------------
st.set_page_config(page_title="Secure Bank", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    * { font-family: 'Poppins', sans-serif; }

    /* Soft Animated Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #1E3A8A, #1E40AF, #1D4ED8, #2563EB);
        background-size: 300% 300%;
        animation: gradient 12s ease infinite;
        color: white;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* High-Contrast Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(12px);
        border-radius: 18px;
        padding: 24px;
        margin: 16px 0;
        border: 1.5px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        text-align: center;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.4);
    }

    /* Bright Rainbow Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #10B981, #14B8A6, #0EA5E9);
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 6px 18px rgba(16, 185, 129, 0.5);
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.7);
    }

    /* Glowing Traffic Light */
    .light {
        font-size: 130px;
        line-height: 1;
        text-shadow: 0 0 25px currentColor;
    }

    /* Big Numbers */
    .big-number {
        font-size: 52px;
        font-weight: 700;
        margin: 0;
        color: white;
        text-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    .label {
        font-size: 17px;
        color: #E0E7FF;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Table */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        background: rgba(255,255,255,0.1);
    }

    /* Sidebar */
    .css-1d391kg { background: rgba(30, 58, 138, 0.9); }
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
        st.success("Welcome!")
    except:
        st.error("Register failed.")

def logout():
    for k in ["uid", "logged_in", "is_admin"]:
        st.session_state.pop(k, None)
    st.success("Logged out.")

# ---------------------------------------------------------
# 3. LOGIN PAGE – VISIBLE
# ---------------------------------------------------------
if not st.session_state.get("logged_in", False):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://img.icons8.com/fluency/256/bank-building.png", width=220)
    with col2:
        st.markdown("""
        <h1 style='color: white; font-weight: 700;'>
            Secure Bank
        </h1>
        <p style='color: #C7D2FE; font-size: 1.3rem;'>
            <b>AI-Powered Fraud Protection</b>
        </p>
        <p style='color: #A5B4FC;'>
            86% fraud caught • Only 4% false alerts
        </p>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:20px; background: rgba(255,255,255,0.15); border-radius: 16px;">
            <img src="https://img.icons8.com/fluency/96/bank-building.png" width="70">
            <h3 style="color: white; margin: 10px 0;">Login</h3>
        </div>
        """, unsafe_allow_html=True)
        email = st.text_input("Email", placeholder="you@securebank.com")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"): login(email, pwd); st.rerun()
        if st.button("Register"): register(email, pwd); st.rerun()
    st.stop()

# ---------------------------------------------------------
# 4. LOGGED IN
# ---------------------------------------------------------
st.sidebar.markdown(f"""
<div style="background: rgba(255,255,255,0.2); padding: 16px; border-radius: 14px; text-align: center;">
    <p style="color: white; margin: 0; font-weight: 600;">
        User: {st.session_state.uid[:8]}...
    </p>
</div>
""", unsafe_allow_html=True)
if st.sidebar.button("Logout"): logout(); st.rerun()

pages = ["Check Transaction"]
if st.session_state.get("is_admin"):
    pages.insert(0, "Admin Dashboard")
page = st.sidebar.radio("Menu", pages)

# ---------------------------------------------------------
# 5. CUSTOMER: TRAFFIC LIGHT + BAR
# ---------------------------------------------------------
if page == "Check Transaction":
    st.markdown("<h1 style='color: white; text-align: center; font-weight: 700;'>Check Your Transaction</h1>", unsafe_allow_html=True)

    colA, colB = st.columns([1, 3])
    with colA:
        amt = st.text_input("Amount (USD)", "100.00", help="Enter any amount")
        try: amount = float(amt)
        except: amount = 0.0; st.warning("Invalid")

        if st.button("Verify Now", use_container_width=True):
            vec = np.zeros(INPUT_DIM); vec[29] = amount
            with st.spinner("AI is scanning..."):
                err, fraud = predict_transaction(MODEL, SCALER, THRESHOLD, vec)
            st.session_state.last_err = err
            st.session_state.last_fraud = fraud

    with colB:
        if 'last_err' in st.session_state:
            fraud = st.session_state.last_fraud
            color = "#10B981" if not fraud else "#EF4444"
            status = "SAFE" if not fraud else "BLOCKED"

            # TRAFFIC LIGHT
            st.markdown(f"""
            <div class="glass-card">
                <div class="light" style="color: {color};">
                    Circle
                </div>
                <p class="big-number" style="color: {color};">{status}</p>
                <p class="label">Transaction is <b>{status}</b></p>
            </div>
            """, unsafe_allow_html=True)

            # RISK BAR
            your_risk = min(100, int(err * 100))
            avg_risk = 15
            fig = go.Figure(go.Bar(
                x=['Your Risk', 'Average'],
                y=[your_risk, avg_risk],
                marker_color=[color, '#64748B'],
                text=[f"{your_risk}%", f"{avg_risk}%"],
                textposition='outside'
            ))
            fig.update_layout(
                title="Risk Level",
                yaxis_title="Risk %",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color="white"
            )
            st.plotly_chart(fig, use_container_width=True)

    # RECENT ACTIVITY
    st.markdown("### Your Last 5 Transactions")
    history = pd.DataFrame({
        "Amount": [120, 450, 89, 2000, 75],
        "Status": ["Approved", "Approved", "Approved", "Blocked", "Approved"],
        "Risk": [12, 18, 10, 88, 15]
    })
    st.dataframe(history.style.applymap(
        lambda x: "background-color: #FCA5A5" if x == "Blocked" else "background-color: #86EFAC",
        subset=["Status"]
    ).set_properties(**{
        'color': 'black', 'font-weight': 'bold', 'text-align': 'center'
    }), use_container_width=True)

# ---------------------------------------------------------
# 6. ADMIN: CARDS + PIE
# ---------------------------------------------------------
elif page == "Admin Dashboard":
    st.markdown("<h1 style='color: white; text-align: center; font-weight: 700;'>Fraud Control Center</h1>", unsafe_allow_html=True)

    total = 12500
    fraud = 480
    false = 420
    normal = total - fraud - false

    col1, col2, col3 = st.columns(3)
    cards = [
        ("Fraud Caught", fraud, "#EF4444"),
        ("False Alerts", false, "#F59E0B"),
        ("Total Checked", total, "#0EA5E9")
    ]
    for col, (label, value, color) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div class="glass-card">
                <p class="big-number" style="color: {color};">{value:,}</p>
                <p class="label">{label}</p>
            </div>
            """, unsafe_allow_html=True)

    # PIE CHART
    st.markdown("### Transaction Breakdown")
    fig = px.pie(
        values=[normal, fraud, false],
        names=['Safe', 'Fraud', 'False'],
        hole=0.5,
        color_discrete_sequence=['#10B981', '#EF4444', '#F59E0B']
    )
    fig.update_traces(textinfo='percent+label', textfont_size=15)
    fig.update_layout(font_color="white", paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    # THRESHOLD
    st.markdown("### AI Sensitivity")
    new_thr = st.slider("Risk Threshold", 0.5, 1.5, THRESHOLD, 0.05)
    st.markdown(f"""
    <div style="text-align: center; color: #E0E7FF; font-size: 1.1rem;">
        Current: <b style="color: #F59E0B;">{THRESHOLD:.2f}</b> → 
        New: <b style="color: #0EA5E9;">{new_thr:.2f}</b>
    </div>
    """, unsafe_allow_html=True)
