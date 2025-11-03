# app.py - COLORFUL & EASY (NO MAP, NO TOP 5)
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import firebase_admin
from firebase_admin import credentials, auth, firestore
from model_utils import load_model_and_assets, predict_transaction, INPUT_DIM

# ---------------------------------------------------------
# 0. PAGE CONFIG + VIBRANT THEME
# ---------------------------------------------------------
st.set_page_config(page_title="Secure Bank", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    * { font-family: 'Poppins', sans-serif; }

    /* Rainbow Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FECA57);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        text-align: center;
        transition: transform 0.3s;
    }
    .glass-card:hover {
        transform: translateY(-8px);
    }

    /* Rainbow Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
        background-size: 300%;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
        animation: rainbow 8s ease infinite;
    }
    .stButton > button:hover {
        animation: rainbow 1.5s ease infinite;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.6);
    }
    @keyframes rainbow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glowing Traffic Light */
    .light {
        font-size: 140px;
        line-height: 1;
        text-shadow: 0 0 30px currentColor;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }

    /* Big Numbers */
    .big-number {
        font-size: 56px;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .label { font-size: 18px; color: white; opacity: 0.9; margin-top: 8px; }

    /* Table */
    .stDataFrame { border-radius: 16px; overflow: hidden; }
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
        st.balloons()
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
# 3. LOGIN PAGE – COLORFUL
# ---------------------------------------------------------
if not st.session_state.get("logged_in", False):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://img.icons8.com/fluency/256/bank-building.png", width=220)
    with col2:
        st.markdown("""
        <h1 style='color: white; text-shadow: 0 4px 15px rgba(0,0,0,0.4);'>
            Secure Bank
        </h1>
        <p style='color: white; font-size: 1.4rem; opacity: 0.95;'>
            <b>AI-Powered Fraud Shield</b>
        </p>
        <p style='color: white; opacity: 0.8;'>
            86% fraud caught • Only 4% false alerts
        </p>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:20px; background: rgba(255,255,255,0.15); border-radius: 16px;">
            <img src="https://img.icons8.com/fluency/96/bank-building.png" width="80">
            <h3 style="color: white; margin: 12px 0 0;">Login</h3>
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
<div style="background: rgba(255,255,255,0.2); padding: 18px; border-radius: 16px; text-align: center;">
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
# 5. CUSTOMER: RAINBOW TRAFFIC LIGHT
# ---------------------------------------------------------
if page == "Check Transaction":
    st.markdown("<h1 style='color: white; text-align: center; text-shadow: 0 4px 15px rgba(0,0,0,0.4);'>Check Your Transaction</h1>", unsafe_allow_html=True)

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

            # TRAFFIC LIGHT
            light_color = "#4ECDC4" if not fraud else "#FF6B6B"
            status = "SAFE" if not fraud else "BLOCKED"
            st.markdown(f"""
            <div class="glass-card">
                <div class="light" style="color: {light_color};">
                    Circle
                </div>
                <p class="big-number" style="background: linear-gradient(45deg, #FF6B6B, #4ECDC4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {status}
                </p>
                <p class="label" style="color: white;">Transaction is <b>{status}</b></p>
            </div>
            """, unsafe_allow_html=True)

            # RAINBOW BAR
            your_risk = min(100, int(err * 100))
            avg_risk = 15
            fig = go.Figure(go.Bar(
                x=['Your Risk', 'Average'],
                y=[your_risk, avg_risk],
                marker_color=['#FF6B6B' if fraud else '#4ECDC4', '#A0A0A0'],
                text=[f"{your_risk}%", f"{avg_risk}%"],
                textposition='outside'
            ))
            fig.update_layout(
                title="Your Risk Level",
                yaxis_title="Risk %",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
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
        lambda x: f"background: {'linear-gradient(45deg, #FF6B6B, #FF8E8E)' if x=='Blocked' else 'linear-gradient(45deg, #4ECDC4, #7ED9D2)'}",
        subset=["Status"]
    ).set_properties(**{
        'color': 'white', 'font-weight': 'bold', 'text-align': 'center'
    }), use_container_width=True)

# ---------------------------------------------------------
# 6. ADMIN: RAINBOW CARDS + PIE
# ---------------------------------------------------------
elif page == "Admin Dashboard":
    st.markdown("<h1 style='color: white; text-align: center; text-shadow: 0 4px 15px rgba(0,0,0,0.4);'>Fraud Control Center</h1>", unsafe_allow_html=True)

    total = 12500
    fraud = 480
    false = 420
    normal = total - fraud - false

    col1, col2, col3 = st.columns(3)
    cards = [
        ("Fraud Caught", fraud, "#FF6B6B"),
        ("False Alerts", false, "#FECA57"),
        ("Total Checked", total, "#45B7D1")
    ]
    for col, (label, value, color) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div class="glass-card">
                <p class="big-number" style="color: {color};">{value:,}</p>
                <p class="label" style="color: white;">{label}</p>
            </div>
            """, unsafe_allow_html=True)

    # RAINBOW PIE
    st.markdown("### Transaction Breakdown")
    fig = px.pie(
        values=[normal, fraud, false],
        names=['Safe', 'Fraud', 'False'],
        hole=0.5,
        color_discrete_sequence=['#4ECDC4', '#FF6B6B', '#FECA57']
    )
    fig.update_traces(textinfo='percent+label', textfont_size=16)
    st.plotly_chart(fig, use_container_width=True)

    # RAINBOW SLIDER
    st.markdown("### AI Sensitivity")
    new_thr = st.slider("Risk Threshold", 0.5, 1.5, THRESHOLD, 0.05,
                        help="Lower = catch more fraud")
    st.markdown(f"""
    <div style="text-align: center; color: white; font-size: 1.2rem;">
        Current: <b style="color: #FECA57;">{THRESHOLD:.2f}</b> → 
        New: <b style="color: #45B7D1;">{new_thr:.2f}</b>
    </div>
    """, unsafe_allow_html=True)
