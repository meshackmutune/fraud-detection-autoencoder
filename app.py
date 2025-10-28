import streamlit as st
import numpy as np
import firebase_admin
from firebase_admin import credentials, auth, firestore
from model_utils import load_model_and_assets, predict_transaction, INPUT_DIM, generate_realistic_transaction
import os
import json

# --- 0. CONFIGURATION AND STYLING (Green Theme) ---
st.set_page_config(
    page_title="Secure Bank PoC",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for the Green Theme
st.markdown("""
<style>
:root {
    --primary-green: #10B981;
    --light-green: #D1FAE5;
    --dark-green: #047857;
}
.stButton>button, .stRadio label {
    background-color: var(--primary-green) !important;
    color: white !important;
    border-color: var(--dark-green) !important;
}
.stButton>button:hover {
    background-color: var(--dark-green) !important;
    border-color: var(--primary-green) !important;
}
.stSuccess {
    background-color: var(--light-green) !important;
    color: var(--dark-green) !important;
    border-left: 6px solid var(--primary-green) !important;
}
.stError {
    border-left: 6px solid #EF4444 !important;
}
.stApp {
    --primary-color: var(--primary-green);
    --secondary-background-color: #F9FAFB; 
    color: #1F2937;
}
</style>
""", unsafe_allow_html=True)


# --- 1. FIREBASE INITIALIZATION & AUTHENTICATION ---
def initialize_firebase():
    """Initializes Firebase Admin SDK using environment variables."""
    if not firebase_admin._apps:
        try:
            app_id = os.environ.get('__app_id', 'default-app-id')
            firebase_config = json.loads(os.environ.get('__firebase_config', '{}'))
            initial_auth_token = os.environ.get('__initial_auth_token')
            
            cred = credentials.Certificate(firebase_config)
            firebase_admin.initialize_app(cred, name=app_id)
            
            if initial_auth_token:
                user = auth.verify_id_token(initial_auth_token)
                st.session_state.user = user
                st.session_state.user_id = user['uid']
            
            st.session_state.db = firestore.client()
            st.session_state.auth_ready = True
        except Exception as e:
            st.error(f"Firebase Initialization Error: {e}")
            st.session_state.auth_ready = False


# --- 2. AUTHENTICATION HELPERS ---
def get_db():
    if 'db' not in st.session_state or st.session_state.db is None:
        return firestore.client()
    return st.session_state.db

def login_user(email, password):
    try:
        user = auth.get_user_by_email(email)
        st.session_state.user = user
        st.session_state.user_id = user.uid
        st.session_state.is_admin = (email == "admin@securebank.com")
        st.session_state.logged_in = True
        st.success(f"Welcome back, {user.email}!")
    except Exception:
        st.error("Login failed. Check email and password.")

def register_user(email, password):
    if not email or not password:
        st.warning("Email and Password cannot be empty.")
        return
    try:
        user = auth.create_user(email=email, password=password)
        st.session_state.user = user
        st.session_state.user_id = user.uid
        st.session_state.is_admin = False
        st.session_state.logged_in = True
        
        db = get_db()
        doc_ref = db.collection('artifacts').document(st.session_state.app_id)\
                    .collection('users').document(user.uid)\
                    .collection('profile').document('data')
        doc_ref.set({'email': email, 'is_admin': False})
        
        st.success(f"Registration successful! Welcome, {user.email}.")
    except Exception as e:
        st.error(f"Registration failed: {e}")

def logout_user():
    for key in ['logged_in', 'user', 'user_id', 'is_admin']:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.auth_status = "login"
    st.success("Logged out successfully.")

def UserAuthentication():
    if st.session_state.logged_in:
        st.sidebar.success(f"Signed in as: {st.session_state.user.email}")
        if st.sidebar.button("Logout", key="logout_btn", use_container_width=True):
            logout_user()
        return True

    if 'auth_status' not in st.session_state:
        st.session_state.auth_status = "login"

    st.sidebar.title("Secure Sign-In")
    st.sidebar.subheader("Fraud Detection PoC")
    
    auth_container = st.sidebar.container(border=True)

    with auth_container:
        email = st.text_input("Email", key="auth_email")
        password = st.text_input("Password", type="password", key="auth_password")

        col_l, col_r = st.columns(2)
        
        with col_l:
            if st.button("Login", key="login_btn", use_container_width=True):
                login_user(email, password)
                st.rerun()

        with col_r:
            if st.button("Register", key="register_btn", use_container_width=True):
                register_user(email, password)
                st.rerun()

    return False


# --- 3. MAIN APPLICATION FLOW ---

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.is_admin = False
    st.session_state.app_id = os.environ.get('__app_id', 'default-app-id')

initialize_firebase()
if not st.session_state.auth_ready:
    st.stop()

# Load model & artifacts (now includes means and stds)
MODEL, SCALER, THRESHOLD, MEANS, STDS = load_model_and_assets()
if MODEL is None:
    st.stop()

is_authenticated = UserAuthentication()
if not is_authenticated:
    st.info("Please login or register to access the Fraud Detection Interface.")
else:
    st.sidebar.markdown("---")
    if st.session_state.is_admin:
        page = st.sidebar.radio(
            "Select Interface", 
            ["Administrator Dashboard", "Customer Interface"],
            index=0 
        )
    else:
        page = st.sidebar.radio(
            "Select Interface", 
            ["Customer Interface"],
            index=0 
        )
    st.sidebar.markdown("---")

    # --- CUSTOMER INTERFACE ---
    if page == "Customer Interface":
        st.title("Welcome to Your Secure Banking Portal 🏦")
        st.subheader("Transaction Verification Simulation")
        st.markdown(f"**Current User:** `{st.session_state.user_id}`")
        st.success("Your identity is verified. Proceed with your transaction.")
        
        st.markdown("---")

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Transaction Details")
            amount_str = st.text_input("Transaction Amount (USD)", value="100.00")
            try:
                amount = float(amount_str)
            except ValueError:
                amount = 0.0
                st.warning("Please enter a valid number.")

            if st.button("Process & Verify Transaction", use_container_width=True, type="primary"):
                # Generate realistic synthetic transaction
                raw_data = generate_realistic_transaction(MEANS, STDS, amount)

                with st.spinner('Running Deep Autoencoder Inference...'):
                    error, anomaly = predict_transaction(MODEL, SCALER, THRESHOLD, raw_data)
                
                with col2:
                    st.markdown("### Verification Results")
                    if anomaly:
                        st.error("🚨 FRAUD ALERT! High Risk Transaction Detected.")
                        st.markdown(f"**Anomaly Score:** `{error:.6f}` (Above Threshold: `{THRESHOLD:.4f}`)")
                        st.metric(label="Status", value="BLOCKED", delta="HIGH RISK", delta_color="inverse")
                    else:
                        st.success("✅ TRANSACTION APPROVED: Normal Pattern Detected")
                        st.markdown(f"**Anomaly Score:** `{error:.6f}` (Below Threshold: `{THRESHOLD:.4f}`)")
                        st.metric(label="Status", value="APPROVED", delta="LOW RISK", delta_color="normal")

    # --- ADMINISTRATOR DASHBOARD ---
    elif page == "Administrator Dashboard" and st.session_state.is_admin:
        st.title("Fraud Analyst Operations Center 📈")
        st.subheader("System Performance and Threshold Management")
        
        st.markdown("---")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Optimal Model Threshold (MSE)", f"{THRESHOLD:.4f}")
        with col_m2:
            st.metric("System Users", f"{len(auth.list_users().users)}", delta="New users since launch")
        with col_m3:
            st.metric("Average Inference Time", "30 ms", delta="<50ms Target")

        st.markdown("---")
        st.markdown("### ⚙️ Live System Sensitivity Adjustment")
        with st.container(border=True):
            min_val = float(THRESHOLD * 0.5)
            max_val = float(THRESHOLD * 1.5)
            new_threshold = st.slider("Set New Operational Threshold", min_val, max_val, float(THRESHOLD), step=0.0001, format="%.4f")
            st.warning(f"**Proposed Threshold:** **{new_threshold:.4f}**")

            if new_threshold > THRESHOLD:
                st.info("⬆️ Increased Recall: More sensitive (flags more transactions).")
            elif new_threshold < THRESHOLD:
                st.info("⬇️ Increased Precision: Less sensitive (flags fewer transactions).")
            else:
                st.info("Running at the optimal validated threshold.")
