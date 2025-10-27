import streamlit as st
import numpy as np
import firebase_admin
from firebase_admin import credentials, auth, firestore
from model_utils import load_model_and_assets, predict_transaction, INPUT_DIM
import os
import json # Essential for parsing the new secret format

# --- CONFIGURATION ---
ADMIN_EMAIL = "admin@securebank.com"
DB_TIMEOUT_SECONDS = 5
THEME_COLOR = "#10B981" # Green color for the interface

# --- FIREBASE INITIALIZATION ---

@st.cache_resource
def initialize_firebase():
    """Initializes the Firebase Admin SDK using the securely encoded JSON string from st.secrets."""
    if not firebase_admin._apps:
        try:
            # 1. Attempt to load the JSON-encoded string from the new secret key
            if "FIREBASE_CREDENTIALS_JSON" in st.secrets:
                # Get the string and parse it back into a Python dictionary
                credentials_string = st.secrets["FIREBASE_CREDENTIALS_JSON"]
                service_account_info = json.loads(credentials_string)
            
            # Fallback (in case the user accidentally kept the old key name but formatted it as JSON)
            elif "FIREBASE_SERVICE_ACCOUNT" in st.secrets and isinstance(st.secrets["FIREBASE_SERVICE_ACCOUNT"], str):
                 credentials_string = st.secrets["FIREBASE_SERVICE_ACCOUNT"]
                 service_account_info = json.loads(credentials_string)
                 st.warning("Using fallback secret key. Please update to FIREBASE_CREDENTIALS_JSON.")

            # If no secret key is found, raise an error
            else:
                raise KeyError("Neither 'FIREBASE_CREDENTIALS_JSON' nor 'FIREBASE_SERVICE_ACCOUNT' found in Streamlit Secrets.")

            # --- CRITICAL FIX: Ensure Newline Characters are Correct ---
            # If the Streamlit/TOML parser is over-escaping the newlines, we force a fix here.
            if 'private_key' in service_account_info:
                private_key_value = service_account_info['private_key']
                # Replace literal \n sequences with actual newline characters
                if '\\n' in private_key_value:
                    service_account_info['private_key'] = private_key_value.replace('\\n', '\n')
            # ---------------------------------------------------------
            
            # 2. Convert to Firebase Credentials object
            cred = credentials.Certificate(service_account_info)
            
            # 3. Initialize the app
            firebase_admin.initialize_app(cred)
            
            st.success("Firebase initialized successfully.")
            
            # Return Firestore and Auth instances
            db = firestore.client()
            return db, auth
            
        except Exception as e:
            st.error(f"Firebase Initialization Error: Check Streamlit Cloud Secrets for correct credentials. Error: {e}")
            return None, None
    else:
        # App is already initialized
        db = firestore.client()
        return db, auth

# --- APPLICATION STATE MANAGEMENT ---

def init_session_state(db, fb_auth):
    """Initializes all necessary session state variables."""
    if 'db' not in st.session_state:
        st.session_state.db = db
        st.session_state.fb_auth = fb_auth
        st.session_state.user = None
        st.session_state.is_admin = False
        st.session_state.app_id = 'default-app-id' # Since we use st.secrets, we default this.
        st.session_state.model_loaded = False
        st.session_state.model = None
        st.session_state.scaler = None
        st.session_state.threshold = None

def load_assets_and_set_state():
    """Loads model assets and updates session state."""
    if not st.session_state.model_loaded:
        model, scaler, threshold = load_model_and_assets()
        if model and scaler and threshold:
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.threshold = threshold
            st.session_state.model_loaded = True
            st.success("AI Model and assets loaded.")
        else:
            st.error("Could not load all necessary AI assets. Check logs.")

# --- FIREBASE HELPER FUNCTIONS ---

def get_user_data_path(user_id):
    """Generates the Firestore collection path for a user's private data."""
    app_id = st.session_state.app_id
    # Path: artifacts/{appId}/users/{userId}/transactions
    return st.session_state.db.collection('artifacts').document(app_id).collection('users').document(user_id).collection('transactions')

def write_transaction_to_db(user_id, transaction_data, prediction_result):
    """Writes the transaction and prediction result to Firestore."""
    
    transaction_ref = get_user_data_path(user_id)
    
    data = {
        'timestamp': firestore.SERVER_TIMESTAMP,
        'user_id': user_id,
        'amount': float(transaction_data[-1]), # Last feature is 'Amount'
        'features': [float(x) for x in transaction_data], # Store raw features
        'prediction': {
            'is_fraud': prediction_result['is_fraud'],
            'error_score': prediction_result['error_score'],
            'threshold': prediction_result['threshold']
        }
    }
    
    try:
        transaction_ref.add(data, timeout=DB_TIMEOUT_SECONDS)
    except Exception as e:
        st.error(f"Database Write Error: Could not save transaction history. {e}")


# --- AUTHENTICATION UI COMPONENTS ---

def login_form():
    """Renders the login form."""
    with st.form("Login"):
        st.markdown(f"<h2 style='color: {THEME_COLOR};'>Login to SecureBank PoC</h2>", unsafe_allow_html=True)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")

        if submitted:
            try:
                user = st.session_state.fb_auth.get_user_by_email(email)
                # In a real app, you'd use the client SDK for secure login. 
                # Here, we verify the user exists for demonstration purposes.
                if user and password: 
                    st.session_state.user = user
                    st.session_state.is_admin = (email == ADMIN_EMAIL)
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials.")
            except Exception as e:
                st.error(f"Login failed. Check email and password. Error: {e}")

def register_form():
    """Renders the registration form."""
    with st.form("Register"):
        st.markdown(f"<h2 style='color: {THEME_COLOR};'>New User Registration</h2>", unsafe_allow_html=True)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Register", type="primary")

        if submitted:
            try:
                user = st.session_state.fb_auth.create_user(
                    email=email,
                    password=password
                )
                st.success(f"Account created successfully for {user.email}! Please log in.")
                st.info("Note: The password is visible to Streamlit's backend in this demo. Use a dummy password.")
            except Exception as e:
                st.error(f"Registration failed: {e}")

def logout_button():
    """Renders the logout button."""
    if st.session_state.user:
        if st.sidebar.button("Logout"):
            st.session_state.user = None
            st.session_state.is_admin = False
            st.experimental_rerun()

# --- MAIN APP UI SCREENS ---

def admin_dashboard():
    """UI for the Admin user."""
    st.markdown(f"<h1 style='color: {THEME_COLOR};'>Admin Dashboard</h1>", unsafe_allow_html=True)
    st.write(f"Welcome, Admin: {st.session_state.user.email}")
    st.info("Here you would see aggregated fraud metrics, user activity logs, and system health checks.")
    
    # Placeholder for viewing all transactions
    st.subheader("Recent System Transactions (Demo)")
    try:
        # For simplicity, we'll just show a mock message.
        st.warning("Fetching all transactions is resource-intensive. Using a placeholder for demonstration.")
        st.dataframe({
            'User': ['user123', 'admin@sec...'],
            'Amount': [45.99, 1000.00],
            'Prediction': ['Non-Fraud', 'Fraud'],
            'Score': [0.001, 0.089]
        })
    except Exception as e:
        st.error(f"Error fetching admin data: {e}")


def customer_portal():
    """UI for a standard Customer user."""
    st.markdown(f"<h1 style='color: {THEME_COLOR};'>Customer Transaction Checker</h1>", unsafe_allow_html=True)
    st.write(f"Welcome, {st.session_state.user.email}")
    
    # 1. Load Assets
    load_assets_and_set_state()
    
    if st.session_state.model_loaded:
        st.subheader("Simulate a Transaction")
        
        with st.form("transaction_form"):
            # Replace slider with text input for amount (V1 feature)
            transaction_amount = st.text_input(
                "Transaction Amount ($)", 
                value="50.00", 
                help="Input the dollar amount of the transaction."
            )
            
            # Use placeholder inputs for the 29 V-features (V1 to V28) + Time
            st.markdown("---")
            st.info("The remaining 29 features (Time, V1-V28) are auto-populated for this demo.")
            
            # Generate mock features (V1-V28) and Time. Time is always the first feature.
            # V-features are typically normalized/PCA'd, so they are close to 0.
            mock_features = np.random.normal(loc=0.0, scale=1.0, size=INPUT_DIM - 2) # 28 V-features
            time_feature = np.array([45000]) # Mock Time feature
            
            # The final feature vector structure MUST match the training data (30 features: Time, V1-V28, Amount)
            try:
                amount_feature = np.array([float(transaction_amount)])
                raw_transaction_data = np.concatenate([time_feature, mock_features, amount_feature])
                
            except ValueError:
                st.error("Please enter a valid number for the Transaction Amount.")
                return

            submitted = st.form_submit_button("Check for Fraud", type="primary")

        if submitted:
            # Run Prediction
            with st.spinner("Analyzing transaction for anomalies..."):
                model = st.session_state.model
                scaler = st.session_state.scaler
                threshold = st.session_state.threshold
                
                error_score, is_anomaly = predict_transaction(model, scaler, threshold, raw_transaction_data)
                
            # Display Result
            if is_anomaly:
                st.error(f"FRAUD ALERT! ANOMALY DETECTED.")
                st.metric(label="Anomaly Score", value=f"{error_score:.4f}", delta=f"Threshold: {threshold:.4f}", delta_color="inverse")
                st.markdown("⚠️ **This transaction is flagged as suspicious and requires manual review.**")
            else:
                st.success("Transaction is LIKELY LEGITIMATE.")
                st.metric(label="Anomaly Score", value=f"{error_score:.4f}", delta=f"Threshold: {threshold:.4f}", delta_color="normal")
                st.markdown("✅ **Transaction cleared based on reconstruction error.**")

            # Save to Database
            prediction_result = {
                'is_fraud': bool(is_anomaly), 
                'error_score': float(error_score),
                'threshold': float(threshold)
            }
            write_transaction_to_db(st.session_state.user.uid, raw_transaction_data, prediction_result)
            st.sidebar.success("Transaction saved to your history.")
            
        st.subheader("Your Transaction History (Demo)")
        st.write("This section would typically pull data from the user's private Firestore collection.")
        # Mock history data
        st.dataframe({
            'Time': ['2025-01-01', '2025-01-02'],
            'Amount': [25.50, 890.00],
            'Status': ['Cleared', 'Suspicious']
        })


def authentication_ui():
    """Handles the display of login/register forms."""
    st.sidebar.markdown(f"<h3 style='color: {THEME_COLOR};'>Account</h3>", unsafe_allow_html=True)
    
    # Simple navigation toggle
    auth_mode = st.sidebar.radio("Mode", ["Login", "Register"])
    
    if auth_mode == "Login":
        login_form()
    elif auth_mode == "Register":
        register_form()

# --- MAIN APPLICATION LOGIC ---

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="SecureBank PoC",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Apply Green Theme (using st.markdown and direct style tags)
    st.markdown(f"""
    <style>
    .stButton>button {{
        background-color: {THEME_COLOR};
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.06);
    }}
    .stButton>button:hover {{
        background-color: #059669; /* Darker green on hover */
    }}
    </style>
    """, unsafe_allow_html=True)

    # 1. Initialize Firebase and Get Instances
    db, fb_auth = initialize_firebase()
    
    if not db:
        st.warning("Application halted. Check Streamlit logs for Firebase initialization error.")
        return
        
    # 2. Initialize Session State
    init_session_state(db, fb_auth)
    
    # 3. Main Content Rendering
    st.sidebar.title("SecureBank PoC")
    logout_button() 
    
    if st.session_state.user:
        if st.session_state.is_admin:
            admin_dashboard()
        else:
            customer_portal()
    else:
        authentication_ui()

if __name__ == '__main__':
    main()
