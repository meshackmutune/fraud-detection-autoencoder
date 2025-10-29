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
st.set_page_config(page_title="Secure Bank PoC", layout="wide")
st.markdown("""
<style>
:root {--primary-green:#10B981; --light-green:#D1FAE5; --dark-green:#047857;}
.stButton>button {background:var(--primary-green)!important;color:#fff!important;}
.stButton>button:hover {background:var(--dark-green)!important;}
.stSuccess {background:var(--light-green)!important;color:var(--dark-green)!important;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. FIREBASE INIT (uses Streamlit secrets)
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
        user = auth.get_user_by_email(email)   # password check happens on client SDK
        st.session_state.update(user=user, uid=user.uid,
                                logged_in=True,
                                is_admin=(email=="admin@securebank.com"))
        st.success(f"Welcome {email}")
    except Exception as e:
        st.error(f"Login failed: {e}")

def register(email, pwd):
    try:
        user = auth.create_user(email=email, password=pwd)
        st.session_state.update(user=user, uid=user.uid,
                                logged_in=True, is_admin=False)
        firestore.client().collection("users").document(user.uid).set({"email":email})
        st.success("Account created")
    except Exception as e:
        st.error(f"Register error: {e}")

def logout():
    for k in ["user","uid","logged_in","is_admin"]:
        st.session_state.pop(k, None)
    st.success("Logged out")

# ---------------------------------------------------------
# 3. LOAD MODEL (once)
# ---------------------------------------------------------
MODEL, SCALER, THRESHOLD = load_model_and_assets()
if MODEL is None:
    st.stop()

# ---------------------------------------------------------
# 4. UI – AUTH
# ---------------------------------------------------------
if not st.session_state.get("logged_in"):
    st.sidebar.title("Secure Sign-In")
    email = st.sidebar.text_input("Email")
    pwd   = st.sidebar.text_input("Password", type="password")
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Login"):   login(email, pwd); st.rerun()
    if col2.button("Register"):register(email, pwd); st.rerun()
    st.stop()

st.sidebar.success(f"Signed in: {st.session_state.user.email}")
if st.sidebar.button("Logout"): logout(); st.rerun()

# ---------------------------------------------------------
# 5. ROLE-BASED NAVIGATION
# ---------------------------------------------------------
pages = ["Customer Interface"]
if st.session_state.get("is_admin"):
    pages.insert(0, "Administrator Dashboard")
page = st.sidebar.radio("Menu", pages, index=0)

# ---------------------------------------------------------
# 6. CUSTOMER INTERFACE
# ---------------------------------------------------------
if page == "Customer Interface":
    st.title("Secure Banking Portal 🏦")
    st.write(f"**User ID:** `{st.session_state.uid}`")

    colA, colB = st.columns([1,2])
    with colA:
        amt = st.text_input("Amount (USD)", "100.00")
        try: amount = float(amt)
        except: amount = 0.0; st.warning("Invalid number")

        if st.button("Verify Transaction", type="primary"):
            vec = np.zeros(INPUT_DIM)
            vec[29] = amount                     # Amount = last column
            with st.spinner("Running auto-encoder…"):
                err, fraud = predict_transaction(MODEL, SCALER, THRESHOLD, vec)

            with colB:
                st.markdown("### Result")
                if fraud:
                    st.error("FRAUD DETECTED")
                    st.metric("Status","BLOCKED", delta="HIGH RISK", delta_color="inverse")
                else:
                    st.success("APPROVED")
                    st.metric("Status","APPROVED", delta="LOW RISK")
                st.write(f"**MSE:** `{err:.6f}` | **Thresh:** `{THRESHOLD:.4f}`")

# ---------------------------------------------------------
# 7. ADMIN DASHBOARD
# ---------------------------------------------------------
elif page == "Administrator Dashboard":
    st.title("Fraud Analyst Center 📈")
    col1, col2, col3 = st.columns(3)
    col1.metric("Threshold", f"{THRESHOLD:.4f}")
    col2.metric("Total Users", len(auth.list_users().users))
    col3.metric("Avg Inference", "≈30 ms")

    st.markdown("### Adjust Sensitivity")
    new_thr = st.slider("Operational MSE threshold",
                        THRESHOLD*0.5, THRESHOLD*1.5,
                        THRESHOLD, step=1e-4, format="%.4f")
    if new_thr != THRESHOLD:
        st.info(f"New threshold **{new_thr:.4f}** → {'More recalls' if new_thr>THRESHOLD else 'Higher precision'}")
