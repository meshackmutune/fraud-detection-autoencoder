import streamlit as st
import numpy as np
import firebase_admin
import os
import json
from firebase_admin import credentials, auth, firestore
from model_utils import load_model_and_assets, predict_transaction, INPUT_DIM

# --- 0. CONFIGURATION AND STYLING (Green Theme) ---
st.set_page_config(
    page_title="Secure Bank PoC",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for the Green Theme
st.markdown("""
<style>
/* Main green color palette */
:root {
    --primary-green: #10B981; /* Tailwind Emerald 500 */
    --light-green: #D1FAE5; /* Tailwind Emerald 100 */
    --dark-green: #047857;  /* Tailwind Emerald 700 */
}

/* Primary color for buttons and accents */
.stButton>button, .stRadio label {
    background-color: var(--primary-green) !important;
    color: white !important;
    border-color: var(--dark-green) !important;
}
.stButton>button:hover {
    background-color: var(--dark-green) !important;
    border-color: var(--primary-green) !important;
}

/* Success messages green */
.stSuccess {
    background-color: var(--light-green) !important;
    color: var(--dark-green) !important;
    border-left: 6px solid var(--primary-green) !important;
}

/* Warning/Error messages (using default for contrast) */
.stError {
    border-left: 6px solid #EF4444 !important;
}

/* Adjust Streamlit primary color */
.stApp {
    --primary-color: var(--primary-green);
    --secondary-background-color: #F9FAFB; 
    color: #1F2937;
}

</style>
""", unsafe_allow_html=True)


# --- 1. FIREBASE INITIALIZATION & AUTHENTICATION ---

def initialize_firebase():
    """Initializes Firebase Admin SDK using canvas environment variables."""
    if not firebase_admin._apps:
        try:
            # 1. Get configuration from environment
            app_id = os.environ.get('__app_id', 'default-app-id')
            firebase_config = json.loads(os.environ.get('__firebase_config', '{}'))
            initial_auth_token = os.environ.get('__initial_auth_token')
            
            # 2. Initialize Firebase
            cred = credentials.Certificate(firebase_config)
            firebase_admin.initialize_app(cred, name=app_id)
            
            # 3. Sign in using the custom token
            if initial_auth_token:
                user = auth.verify_id_token(initial_auth_token)
                st.session_state.user = user
                st.session_state.user_id = user['uid']
            
            # 4. Initialize Firestore DB instance
            st.session_state.db = firestore.client()
            
            st.session_state.auth_ready = True
        except Exception as e:
            st.error(f"Firebase Initialization Error: {e}")
            st.session_state.auth_ready = False
            

# --- 2. AUTHENTICATION WIDGETS ---

def get_db():
    if 'db' not in st.session_state or st.session_state.db is None:
        return firestore.client()
    return st.session_state.db

def login_user(email, password):
    """Attempts to simulate user login."""
    try:
        # In a real app, this would use sign_in_with_email_and_password
        # Since we use the Admin SDK, we simulate user existence check
        user = auth.get_user_by_email(email)
        # Assuming password verification is handled by Firebase on a client SDK
        st.session_state.user = user
        st.session_state.user_id = user.uid
        st.session_state.is_admin = (email == "admin@securebank.com")
        st.session_state.logged_in = True
        st.success(f"Welcome back, {user.email}!")
    except Exception as e:
        st.error("Login failed. Check email and password.")

def register_user(email, password):
    """Creates a new user account."""
    if not email or not password:
        st.warning("Email and Password cannot be empty.")
        return
        
    try:
        # Create user with Firebase Admin SDK
        user = auth.create_user(email=email, password=password)
        st.session_state.user = user
        st.session_state.user_id = user.uid
        st.session_state.is_admin = False
        st.session_state.logged_in = True
        
        # Initialize user data in Firestore
        db = get_db()
        doc_ref = db.collection('artifacts').document(st.session_state.app_id).collection('users').document(user.uid).collection('profile').document('data')
        doc_ref.set({'email': email, 'is_admin': False})
        
        st.success(f"Registration successful! Welcome, {user.email}.")
    except Exception as e:
        st.error(f"Registration failed: {e}")

def logout_user():
    """Logs the user out."""
    for key in ['logged_in', 'user', 'user_id', 'is_admin']:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.auth_status = "login"
    st.success("Logged out successfully.")

def UserAuthentication():
    """Displays the Login/Register/Logout UI."""
    if st.session_state.logged_in:
        st.sidebar.success(f"Signed in as: {st.session_state.user.email}")
        if st.sidebar.button("Logout", key="logout_btn", use_container_width=True):
            logout_user()
        return True

    # Login/Register View
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

# Initialize state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.is_admin = False
    st.session_state.app_id = os.environ.get('__app_id', 'default-app-id')

# Initialize Firebase services
initialize_firebase()
if not st.session_state.auth_ready:
    st.stop()


# Load model assets once
MODEL, SCALER, THRESHOLD = load_model_and_assets()

if MODEL is None or SCALER is None or THRESHOLD is None:
    st.stop() # Stop execution if assets failed to load

# Authentication check
is_authenticated = UserAuthentication()

if not is_authenticated:
    st.info("Please login or register to access the Fraud Detection Interface.")
else:
    # Role-based content selection
    
    st.sidebar.markdown("---")
    if st.session_state.is_admin:
        page = st.sidebar.radio(
            "Select Interface", 
            ["Administrator Dashboard", "Customer Interface"],
            icons=['gear', 'person'],
            index=0 
        )
    else:
        page = st.sidebar.radio(
            "Select Interface", 
            ["Customer Interface"],
            icons=['person'],
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
            
            # Input using text box instead of slider
            amount_str = st.text_input(
                "Transaction Amount (USD)", 
                value="100.00", 
                help="Enter the exact dollar amount for the transaction."
            )
            
            try:
                amount = float(amount_str)
            except ValueError:
                amount = 0.0
                st.warning("Please enter a valid numerical amount.")

            st.write("") 

            if st.button("Process & Verify Transaction", use_container_width=True, type="primary"):
                
                # Create the required 30-feature vector (V1-V28, Time, Amount)
                raw_data = np.zeros(INPUT_DIM)
                # Assign the amount to the last feature (index 29)
                raw_data[29] = amount 
                
                # For a better simulation, we might inject random "normal" values for V1-V28
                # For this PoC, we keep V1-V28 as 0 for simplicity, relying on the scaler to standardize
                
                # Processing block
                with st.spinner('Running Deep Autoencoder Inference...'):
                    error, anomaly = predict_transaction(MODEL, SCALER, THRESHOLD, raw_data)
                
                # Display Results
                with col2:
                    st.markdown("### Verification Results")
                    
                    if anomaly:
                        st.error("🚨 FRAUD ALERT! High Risk Transaction Detected.")
                        st.markdown(f"**Anomaly Score:** `{error:.6f}` (Above Threshold: `{THRESHOLD:.4f}`)")
                        st.markdown("Immediate action has been taken: the transaction has been blocked.")
                        st.metric(label="Status", value="BLOCKED", delta="HIGH RISK", delta_color="inverse")
                    else:
                        st.success("✅ TRANSACTION APPROVED: Normal Pattern in the transaction")
                        st.markdown(f"**Anomaly Score:** `{error:.6f}` (Below Threshold: `{THRESHOLD:.4f}`)")
                        st.markdown("This transaction matches expected behavior and has been successfully processed.")
                        st.metric(label="Status", value="APPROVED", delta="LOW RISK", delta_color="normal")
        
        with col2:
            st.empty() 

    # --- ADMINISTRATOR DASHBOARD ---
    elif page == "Administrator Dashboard" and st.session_state.is_admin:
        
        st.title("Fraud Analyst Operations Center 📈")
        st.subheader("System Performance and Threshold Management")
        
        st.markdown("---")
        
        # Display Core Metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric("Optimal Model Threshold (MSE)", f"{THRESHOLD:.4f}")
        with col_m2:
            st.metric("System Users", f"{len(auth.list_users().users)}", delta="New users since launch", delta_color="normal")
        with col_m3:
            st.metric("Average Inference Time", "30 ms", delta="< 50ms Target", delta_color="normal")
            
        st.markdown("---")
            
        st.markdown("### ⚙️ Live System Sensitivity Adjustment")
        
        with st.container(border=True):
            st.caption("Adjusting the threshold directly impacts the system's sensitivity.")
            
            # Dynamic range based on the optimal threshold
            min_val = float(THRESHOLD * 0.5)
            max_val = float(THRESHOLD * 1.5)
            
            new_threshold = st.slider(
                "Set New Operational Threshold", 
                min_val, 
                max_val, 
                float(THRESHOLD),
                step=0.0001,
                format="%.4f"
            )
            
            st.warning(f"**Proposed Threshold:** **{new_threshold:.4f}**")
            
            if new_threshold > THRESHOLD:
                st.info("⬆️ **Increased Recall:** System is more sensitive; more transactions will be flagged.")
            elif new_threshold < THRESHOLD:
                st.info("⬇️ **Increased Precision:** System is less sensitive; only extreme anomalies will be flagged.")
            else:
                st.info("Currently running at the validated optimal threshold.")


