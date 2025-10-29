import streamlit as st
import numpy as np
import firebase_admin
from firebase_admin import credentials, auth, firestore
from model_utils import load_model_and_assets, predict_transaction, INPUT_DIM
import json

# --- CONFIGURATION ---
ADMIN_EMAIL = "admin@securebank.com"
DB_TIMEOUT_SECONDS = 5
THEME_COLOR = "#10B981"

# --- FIREBASE INITIALIZATION ---

@st.cache_resource
def initialize_firebase():
    """Initializes the Firebase Admin SDK using credentials from st.secrets."""
    if not firebase_admin._apps:
        try:
            # Check if using JSON string format
            if "FIREBASE_CREDENTIALS_JSON" in st.secrets:
                credentials_string = st.secrets["FIREBASE_CREDENTIALS_JSON"]
                service_account_info = json.loads(credentials_string)
            
            # Check if using TOML dictionary format
            elif "FIREBASE_SERVICE_ACCOUNT" in st.secrets:
                service_account_info = dict(st.secrets["FIREBASE_SERVICE_ACCOUNT"])
            
            else:
                raise KeyError("Neither 'FIREBASE_CREDENTIALS_JSON' nor 'FIREBASE_SERVICE_ACCOUNT' found in Streamlit Secrets.")

            # Fix newline characters in private key
            if 'private_key' in service_account_info:
                private_key_value = service_account_info['private_key']
                private_key_value = private_key_value.replace('\\\\n', '\n')
                private_key_value = private_key_value.replace('\\n', '\n')
                service_account_info['private_key'] = private_key_value
            
            # Convert to Firebase Credentials object
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)
            
            # Return Firestore and Auth instances
            db = firestore.client()
            return db, auth
            
        except json.JSONDecodeError as je:
            st.error(f"Firebase Initialization Error: Invalid JSON format. {je}")
            st.stop()
            return None, None
        except KeyError as ke:
            st.error(f"Firebase Initialization Error: {ke}")
            st.stop()
            return None, None
        except Exception as e:
            st.error(f"Firebase Initialization Error: {str(e)}")
            st.stop()
            return None, None
    else:
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
        st.session_state.app_id = 'default-app-id'  # Fixed: removed os.environ reference
        st.session_state.model_loaded = False
        st.session_state.model = None
        st.session_state.scaler = None
        st.session_state.threshold = None
        st.session_state.original_threshold = None

def load_assets_and_set_state():
    """Loads model assets and updates session state."""
    if not st.session_state.model_loaded:
        try:
            model, scaler, threshold = load_model_and_assets()
            if model and scaler and threshold:
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.original_threshold = threshold
                
                # Use reasonable threshold
                if threshold > 0.5:
                    st.session_state.threshold = 0.08
                elif threshold < 0.001:
                    st.session_state.threshold = 0.005
                else:
                    st.session_state.threshold = threshold
                
                st.session_state.model_loaded = True
                st.success("✅ AI Model and assets loaded successfully!")
            else:
                st.error("❌ Could not load AI assets. Check deployment files.")
        except Exception as e:
            st.error(f"Error loading model: {e}")

# --- FIREBASE HELPER FUNCTIONS ---

def get_user_data_path(user_id):
    """Generates the Firestore collection path for a user's private data."""
    app_id = st.session_state.app_id
    return st.session_state.db.collection('artifacts').document(app_id).collection('users').document(user_id).collection('transactions')

def write_transaction_to_db(user_id, transaction_data, prediction_result):
    """Writes the transaction and prediction result to Firestore."""
    transaction_ref = get_user_data_path(user_id)
    
    data = {
        'timestamp': firestore.SERVER_TIMESTAMP,
        'user_id': user_id,
        'amount': float(transaction_data[-1]),
        'features': [float(x) for x in transaction_data],
        'prediction': {
            'is_fraud': prediction_result['is_fraud'],
            'error_score': prediction_result['error_score'],
            'threshold': prediction_result['threshold']
        }
    }
    
    try:
        transaction_ref.add(data, timeout=DB_TIMEOUT_SECONDS)
    except Exception as e:
        st.error(f"Database Write Error: {e}")

# --- AUTHENTICATION UI COMPONENTS ---

def login_form():
    """Renders the login form."""
    with st.form("Login"):
        st.markdown(f"<h2 style='color: {THEME_COLOR};'>Login to SecureBank</h2>", unsafe_allow_html=True)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")

        if submitted:
            if not email or not password:
                st.error("Please enter both email and password.")
                return
                
            try:
                user = st.session_state.fb_auth.get_user_by_email(email)
                if user:
                    st.session_state.user = user
                    st.session_state.is_admin = (email == ADMIN_EMAIL)
                    st.rerun()
            except auth.UserNotFoundError:
                st.error("User not found. Please check your email or register.")
            except Exception as e:
                st.error(f"Login failed: {e}")

def register_form():
    """Renders the registration form."""
    with st.form("Register"):
        st.markdown(f"<h2 style='color: {THEME_COLOR};'>New User Registration</h2>", unsafe_allow_html=True)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Register", type="primary")

        if submitted:
            if not email or not password:
                st.error("Please enter both email and password.")
                return
                
            if len(password) < 6:
                st.error("Password must be at least 6 characters long.")
                return
                
            try:
                user = st.session_state.fb_auth.create_user(email=email, password=password)
                st.success(f"✅ Account created for {user.email}! Please log in.")
                st.info("Note: Use a dummy password for this demo.")
            except auth.EmailAlreadyExistsError:
                st.error("An account with this email already exists.")
            except Exception as e:
                st.error(f"Registration failed: {e}")

def logout_button():
    """Renders the logout button."""
    if st.session_state.user:
        if st.sidebar.button("Logout", type="secondary"):
            st.session_state.user = None
            st.session_state.is_admin = False
            st.rerun()

# --- MAIN APP UI SCREENS ---

def admin_dashboard():
    """UI for the Admin user."""
    st.markdown(f"<h1 style='color: {THEME_COLOR};'>Admin Dashboard</h1>", unsafe_allow_html=True)
    st.write(f"Welcome, Admin: **{st.session_state.user.email}**")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Transactions", "👥 Users", "⚙️ Settings"])
    
    with tab1:
        st.subheader("System Overview")
        
        try:
            all_users = st.session_state.fb_auth.list_users()
            total_users = len(all_users.users)
        except:
            total_users = "N/A"
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", total_users)
        with col2:
            st.metric("System Status", "🟢 Online")
        with col3:
            if st.session_state.threshold:
                st.metric("Current Threshold", f"{st.session_state.threshold:.6f}")
        with col4:
            if st.session_state.model_loaded:
                st.metric("AI Model", "✅ Loaded")
        
        st.info("💡 Use the tabs above to manage users, view transactions, and configure settings.")
    
    with tab2:
        st.subheader("All System Transactions")
        
        col1, col2 = st.columns(2)
        with col1:
            filter_user = st.text_input("Filter by User Email")
        with col2:
            filter_fraud = st.selectbox("Filter", ["All", "Fraud Only", "Clear Only"])
        
        if st.button("Load Transactions", type="primary"):
            with st.spinner("Fetching..."):
                try:
                    app_id = st.session_state.app_id
                    users_ref = st.session_state.db.collection('artifacts').document(app_id).collection('users')
                    
                    transactions_data = []
                    users_docs = users_ref.stream()
                    
                    for user_doc in users_docs:
                        user_id = user_doc.id
                        
                        if filter_user:
                            try:
                                user_info = st.session_state.fb_auth.get_user(user_id)
                                if filter_user.lower() not in user_info.email.lower():
                                    continue
                            except:
                                continue
                        
                        transactions_ref = users_ref.document(user_id).collection('transactions')
                        transactions = transactions_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(50).stream()
                        
                        for trans in transactions:
                            trans_data = trans.to_dict()
                            is_fraud = trans_data.get('prediction', {}).get('is_fraud', False)
                            
                            if filter_fraud == "Fraud Only" and not is_fraud:
                                continue
                            elif filter_fraud == "Clear Only" and is_fraud:
                                continue
                            
                            try:
                                user_info = st.session_state.fb_auth.get_user(user_id)
                                user_email = user_info.email
                            except:
                                user_email = user_id
                            
                            transactions_data.append({
                                'User': user_email,
                                'Amount': f"${trans_data.get('amount', 0):.2f}",
                                'Status': '🚨 FRAUD' if is_fraud else '✅ Clear',
                                'Score': f"{trans_data.get('prediction', {}).get('error_score', 0):.6f}",
                                'Timestamp': trans_data.get('timestamp', 'N/A')
                            })
                    
                    if transactions_data:
                        st.success(f"Found {len(transactions_data)} transactions")
                        st.dataframe(transactions_data, use_container_width=True)
                    else:
                        st.warning("No transactions found.")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab3:
        st.subheader("User Management")
        
        if st.button("Load All Users", type="primary"):
            with st.spinner("Fetching users..."):
                try:
                    all_users = st.session_state.fb_auth.list_users()
                    users_data = []
                    for user in all_users.users:
                        users_data.append({
                            'Email': user.email,
                            'UID': user.uid,
                            'Admin': '✅' if user.email == ADMIN_EMAIL else '❌'
                        })
                    st.success(f"Found {len(users_data)} users")
                    st.dataframe(users_data, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with st.expander("⚠️ Delete User"):
            user_email_to_delete = st.text_input("User Email to Delete")
            confirm = st.checkbox("I confirm deletion")
            if st.button("Delete User", type="secondary"):
                if not confirm:
                    st.error("Please confirm deletion.")
                elif user_email_to_delete == ADMIN_EMAIL:
                    st.error("Cannot delete admin!")
                elif user_email_to_delete:
                    try:
                        user = st.session_state.fb_auth.get_user_by_email(user_email_to_delete)
                        st.session_state.fb_auth.delete_user(user.uid)
                        st.success(f"Deleted {user_email_to_delete}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    with tab4:
        st.subheader("System Settings")
        
        st.info("🔧 Model Configuration")
        if st.session_state.threshold:
            st.write(f"**Current Threshold:** {st.session_state.threshold:.6f}")
        if st.session_state.original_threshold:
            st.write(f"**Original Threshold:** {st.session_state.original_threshold:.6f}")
        
        st.write(f"**App ID:** `{st.session_state.app_id}`")
        st.write(f"**Input Dimension:** {INPUT_DIM} features")

def customer_portal():
    """UI for a standard Customer user."""
    st.markdown(f"<h1 style='color: {THEME_COLOR};'>Transaction Fraud Checker</h1>", unsafe_allow_html=True)
    st.write(f"Welcome, **{st.session_state.user.email}**")
    
    load_assets_and_set_state()
    
    if st.session_state.model_loaded:
        # Threshold Settings
        with st.expander("⚙️ Detection Sensitivity", expanded=False):
            current = st.session_state.threshold
            
            if st.session_state.original_threshold and st.session_state.original_threshold > 0.5:
                st.warning(f"⚠️ Original threshold ({st.session_state.original_threshold:.6f}) was too high.")
                st.success(f"✅ Auto-corrected to: {current:.6f}")
            
            new_threshold = st.slider(
                "Adjust Threshold",
                min_value=0.001,
                max_value=0.100,
                value=float(current),
                step=0.001,
                format="%.4f",
                help="Lower = more sensitive (more fraud detected, more false alarms)\nHigher = less sensitive (fewer false alarms, may miss fraud)"
            )
            
            if new_threshold != current:
                st.session_state.threshold = new_threshold
                st.success(f"✅ Threshold updated to {new_threshold:.4f}")
            
            if new_threshold < 0.003:
                st.warning("⚠️ Very sensitive - expect many false alarms")
            elif new_threshold > 0.020:
                st.warning("⚠️ Less sensitive - may miss fraud")
            else:
                st.info("✅ Balanced sensitivity")
        
        # Quick Test Scenarios
        st.markdown("---")
        st.subheader("Quick Test Scenarios")
        col1, col2, col3, col4 = st.columns(4)
        
        test_scenario = None
        with col1:
            if st.button("🛒 Small ($25)", use_container_width=True):
                test_scenario = "small"
        with col2:
            if st.button("💳 Medium ($150)", use_container_width=True):
                test_scenario = "medium"
        with col3:
            if st.button("🚨 Large ($2500)", use_container_width=True):
                test_scenario = "large"
        with col4:
            if st.button("🧪 Zero Test", use_container_width=True):
                test_scenario = "zero"
        
        st.markdown("---")
        
        # Transaction Form
        with st.form("transaction_form"):
            st.subheader("Custom Transaction")
            transaction_amount = st.text_input(
                "Transaction Amount ($)",
                value="50.00",
                help="Enter the dollar amount"
            )
            
            st.info("ℹ️ The remaining 29 features (Time, V1-V28) are auto-generated for this demo.")
            submitted = st.form_submit_button("Check for Fraud", type="primary")
        
        # Process Transaction
        if submitted or test_scenario:
            try:
                if test_scenario:
                    scenario_config = {
                        "small": 25.0,
                        "medium": 150.0,
                        "large": 2500.0,
                        "zero": 50.0
                    }
                    amount_value = scenario_config[test_scenario]
                    st.info(f"Testing: ${amount_value}")
                else:
                    amount_value = float(transaction_amount)
                    if amount_value <= 0:
                        st.error("Please enter a positive amount.")
                        st.stop()
            except ValueError:
                st.error("Please enter a valid number.")
                st.stop()
            
            # Generate features (deterministic based on amount)
            amount_seed = int(amount_value * 1000) % 10000
            np.random.seed(amount_seed)
            
            if test_scenario == "zero":
                mock_v_features = np.zeros(INPUT_DIM - 2)
                time_feature = np.array([0.0])
            elif test_scenario == "small":
                np.random.seed(1000)
                mock_v_features = np.random.normal(0.0, 0.3, INPUT_DIM - 2)
                time_feature = np.array([43200.0])
            elif test_scenario == "medium":
                np.random.seed(2000)
                mock_v_features = np.random.normal(0.0, 0.4, INPUT_DIM - 2)
                time_feature = np.array([54000.0])
            elif test_scenario == "large":
                np.random.seed(3000)
                mock_v_features = np.random.normal(0.0, 1.8, INPUT_DIM - 2)
                mock_v_features[0] = 3.5
                mock_v_features[1] = -3.2
                time_feature = np.array([3600.0])
            else:
                if amount_value < 50:
                    scale = 0.3
                    time_base = 43200.0
                elif amount_value < 100:
                    scale = 0.4
                    time_base = 50400.0
                elif amount_value < 500:
                    scale = 0.6
                    time_base = 57600.0
                elif amount_value < 1000:
                    scale = 1.0
                    time_base = 72000.0
                else:
                    scale = 1.8
                    time_base = 3600.0
                
                mock_v_features = np.random.normal(0.0, scale, INPUT_DIM - 2)
                
                if amount_value >= 1000:
                    mock_v_features[0] = 3.2 + (amount_value / 1000.0) * 0.1
                    mock_v_features[2] = -3.0 - (amount_value / 1000.0) * 0.1
                
                time_feature = np.array([time_base])
            
            amount_feature = np.array([amount_value])
            raw_transaction_data = np.concatenate([time_feature, mock_v_features, amount_feature])
            
            # Predict
            with st.spinner("Analyzing transaction..."):
                try:
                    model = st.session_state.model
                    scaler = st.session_state.scaler
                    threshold = st.session_state.threshold
                    
                    error_score, is_anomaly = predict_transaction(model, scaler, threshold, raw_transaction_data)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.stop()
            
            # Display Results
            st.markdown("---")
            st.subheader("Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Error Score", f"{error_score:.6f}")
            with col2:
                st.metric("Threshold", f"{threshold:.6f}")
            with col3:
                diff = error_score - threshold
                st.metric("Difference", f"{diff:.6f}")
            
            if is_anomaly:
                st.error("🚨 FRAUD ALERT - ANOMALY DETECTED")
                st.markdown(f"**This transaction is flagged as suspicious.**")
                st.caption(f"Error ({error_score:.6f}) exceeds threshold ({threshold:.6f}) by {diff:.6f}")
            else:
                st.success("✅ TRANSACTION CLEARED")
                st.markdown(f"**This transaction appears legitimate.**")
                st.caption(f"Error ({error_score:.6f}) is below threshold ({threshold:.6f}) by {abs(diff):.6f}")
            
            # Debug Info
            with st.expander("🔍 Technical Details"):
                st.write(f"**Amount:** ${amount_value:.2f}")
                st.write(f"**Feature fingerprint:** {np.sum(raw_transaction_data):.6f}")
                st.write(f"**First 5 features:** {raw_transaction_data[:5]}")
                st.write(f"**Is Anomaly:** {is_anomaly}")
            
            # Save to DB
            prediction_result = {
                'is_fraud': bool(is_anomaly),
                'error_score': float(error_score),
                'threshold': float(threshold)
            }
            write_transaction_to_db(st.session_state.user.uid, raw_transaction_data, prediction_result)

def authentication_ui():
    """Handles login/register forms."""
    st.sidebar.markdown(f"<h3 style='color: {THEME_COLOR};'>Account</h3>", unsafe_allow_html=True)
    auth_mode = st.sidebar.radio("Mode", ["Login", "Register"])
    
    if auth_mode == "Login":
        login_form()
    else:
        register_form()

# --- MAIN APPLICATION LOGIC ---

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="SecureBank Fraud Detection",
        page_icon="🔒",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Apply theme
    st.markdown(f"""
    <style>
    .stButton>button {{
        background-color: {THEME_COLOR};
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
    }}
    .stButton>button:hover {{
        background-color: #059669;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Initialize Firebase
    db, fb_auth = initialize_firebase()
    
    if not db or not fb_auth:
        st.error("❌ Application halted. Check Firebase configuration.")
        return
        
    # Initialize Session State
    init_session_state(db, fb_auth)
    
    # Sidebar
    st.sidebar.title("🔒 SecureBank")
    
    if st.session_state.user:
        st.sidebar.success(f"Logged in: {st.session_state.user.email}")
        st.sidebar.info(f"Admin: {'✅' if st.session_state.is_admin else '❌'}")
    
    logout_button()
    
    # Main Content
    if st.session_state.user:
        if st.session_state.is_admin:
            admin_dashboard()
        else:
            customer_portal()
    else:
        authentication_ui()

if __name__ == '__main__':
    main()
