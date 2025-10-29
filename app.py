# app.py
import streamlit as st
import numpy as np
import os
import json
import firebase_admin
from firebase_admin import credentials, auth, firestore

from model_utils import load_model_and_assets, predict_transaction, INPUT_DIM

# ---------------------------------------------------------
# 0. PAGE CONFIG + GREEN THEME
# ---------------------------------------------------------
st.set_page_config(page_title="Secure Bank", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    :root {
        --primary-green: #10B981;
        --dark-green: #047857;
        --light-green: #D1FAE5;
    }
    .stButton>button {
        background-color: var(--primary-green) !important;
        color: white !important;
        border: 1px solid var(--dark-green) !important;
    }
    .stButton>button:hover {
        background-color: var(--dark-green) !important;
        border-color: var(--primary-green) !important;
    }
    .stSuccess {
        background-color: var(--light-green) !important;
        color: var(--dark-green) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. FIREBASE INIT
# ---------------------------------------------------------
def init_firebase():
    if not firebase_admin._apps:
        firebase_cfg = st.secrets["firebase"]
        cred = credentials.Certificate(firebase_cfg)
        firebase_admin.initialize_app(cred)
    if "db" not in st.session_state:
        st.session_state.db = firestore.client()

init_firebase()

# ---------------------------------------------------------
# 2. AUTH HELPERS
# ---------------------------------------------------------
def login(email, pwd):
    try:
        user = auth.get_user_by_email(email)
        st.session_state.update(
            user=user, uid=user.uid,
            logged_in=True,
            is_admin=(email == "admin@securebank.com")
        )
        st.success(f"Welcome back, {email}!")
    except Exception as e:
        st.error(f"Login failed: {e}")

def register(email, pwd):
    try:
        user = auth.create_user(email=email, password=pwd)
        st.session_state.update(
            user=user, uid=user.uid,
            logged_in=True, is_admin=False
        )
        firestore.client().collection("users").document(user.uid).set({"email": email})
        st.success("Account created!")
    except Exception as e:
        st.error(f"Register error: {e}")

def logout():
    for k in ["user", "uid", "logged_in", "is_admin"]:
        st.session_state.pop(k, None)
    st.success("Logged out.")

# ---------------------------------------------------------
# 3. LOAD MODEL
# ---------------------------------------------------------
MODEL, SCALER, THRESHOLD = load_model_and_assets()
if MODEL is None:
    st.error("Model failed to load.")
    st.stop()

# ---------------------------------------------------------
# 4. LOGIN PAGE – WELCOME IN MAIN AREA
# ---------------------------------------------------------
if not st.session_state.get("logged_in", False):
    # === SIDEBAR: Login Form ===
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding:15px;">
                <img src="https://img.icons8.com/fluency/96/bank-building.png" width="70">
                <h2 style="color:#10B981; margin:8px 0 0; font-weight:bold;">Secure Bank</h2>
                <p style="color:#10B981; margin:4px 0 0; font-size:0.95rem;">
                    Fraud Detection Portal
                </p>
                <hr style="border-top:2px solid #10B981; margin:12px 0;">
            </div>
            """,
            unsafe_allow_html=True,
        )
        email = st.text_input("Email", placeholder="you@securebank.com", key="login_email")
        pwd   = st.text_input("Password", type="password", key="login_pwd")
        c1, c2 = st.columns(2)
        if c1.button("Login", use_container_width=True):
            login(email, pwd)
            st.rerun()
        if c2.button("Register", use_container_width=True):
            register(email, pwd)
            st.rerun()

    # === MAIN AREA: Big Welcome + Logo ===
    col_logo, col_text = st.columns([1, 2])
    with col_logo:
        st.image(
            "https://img.icons8.com/fluency/256/bank-building.png",
            width=200,
            caption="Secure Bank"
        )
    with col_text:
        st.markdown(
            """
            <h1 style='color:#10B981; margin-top:30px;'>
                Welcome to Secure Bank
            </h1>
            <p style='font-size:1.3rem; color:#047857; line-height:1.6;'>
                Your transactions are protected by our <b>Deep Autoencoder AI</b>.<br>
                <i>86% fraud detection • Only 4% false alerts</i>
            </p>
            <p style='color:#1F2937; margin-top:20px;'>
                Please sign in to verify your transactions in real-time.
            </p>
            """,
            unsafe_allow_html=True,
        )
    st.stop()

# ---------------------------------------------------------
# 5. LOGGED-IN UI
# ---------------------------------------------------------
st.sidebar.success(f"Signed in: {st.session_state.user.email}")
if st.sidebar.button("Logout"):
    logout()
    st.rerun()

# Role-based navigation
pages = ["Customer Interface"]
if st.session_state.get("is_admin"):
    pages.insert(0, "Administrator Dashboard")
page = st.sidebar.radio("Menu", pages, index=0)

# ---------------------------------------------------------
# 6. CUSTOMER INTERFACE
# ---------------------------------------------------------
if page == "Customer Interface":
    st.title("Secure Banking Portal")
    st.write(f"**User ID:** `{st.session_state.uid}`")

    colA, colB = st.columns([1, 2])
    with colA:
        amt = st.text_input("Amount (USD)", "100.00")
        try:
            amount = float(amt)
        except:
            amount = 0.0
            st.warning("Invalid number")

        if st.button("Verify Transaction", type="primary", use_container_width=True):
            vec = np.zeros(INPUT_DIM)
            vec[29] = amount
            with st.spinner("Analyzing..."):
                err, fraud = predict_transaction(MODEL, SCALER, THRESHOLD, vec)

            with colB:
                st.markdown("### Result")
                if fraud:
                    st.error("FRAUD DETECTED")
                    st.metric("Status", "BLOCKED", delta="HIGH RISK", delta_color="inverse")
                else:
                    st.success("APPROVED")
                    st.metric("Status", "APPROVED", delta="LOW RISK")
                st.write(f"**MSE:** `{err:.6f}` | **Threshold:** `{THRESHOLD:.6f}`")

# ---------------------------------------------------------
# 7. ADMIN DASHBOARD
# ---------------------------------------------------------
elif page == "Administrator Dashboard":
    st.title("Fraud Analyst Center")
    col1, col2, col3 = st.columns(3)
    col1.metric("Threshold", f"{THRESHOLD:.6f}")
    col2.metric("Total Users", len(auth.list_users().users))
    col3.metric("Avg Inference", "30 ms")

    st.markdown("### Adjust Sensitivity")
    new_thr = st.slider(
        "Operational Threshold",
        THRESHOLD * 0.5, THRESHOLD * 1.5,
        THRESHOLD, step=1e-4, format="%.6f"
    )
    if new_thr != THRESHOLD:
        st.info(f"New threshold: **{new_thr:.6f}** → {'More sensitive' if new_thr > THRESHOLD else 'More precise'}")
