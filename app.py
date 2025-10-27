import streamlit as st
import numpy as np
import firebase_admin
from firebase_admin import credentials, auth, firestore
from model_utils import load_model_and_assets, predict_transaction, INPUT_DIM
import json

# --- CONFIGURATION ---
ADMIN_EMAIL = "admin@securebank.com"
DB_TIMEOUT_SECONDS = 5
THEME_COLOR = "#10B981"  # Green color for the interface

# --- FIREBASE INITIALIZATION ---

@st.cache_resource
def initialize_firebase():
    """Initializes the Firebase Admin SDK using the securely encoded JSON string from st.secrets."""
    if not firebase_admin._apps:
        try:
            # 1. Attempt to load the JSON-encoded string from the new secret key
            if "FIREBASE_CREDENTIALS_JSON" in st.secrets:
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
            if 'private_key' in service_account_info:
                private_key_value = service_account_info['private_key']
                # Handle multiple possible escape sequences
                # Replace escaped newlines with actual newlines
                private_key_value = private_key_value.replace('\\\\n', '\n')  # Double-escaped
                private_key_value = private_key_value.replace('\\n', '\n')     # Single-escaped
                service_account_info['private_key'] = private_key_value
            # ---------------------------------------------------------
            
            # 2. Convert to Firebase Credentials object
            cred = credentials.Certificate(service_account_info)
            
            # 3. Initialize the app
            firebase_admin.initialize_app(cred)
            
            st.success("Firebase initialized successfully.")
            
            # Return Firestore and Auth instances
            db = firestore.client()
            return db, auth
            
        except json.JSONDecodeError as je:
            st.error(f"Firebase Initialization Error: Invalid JSON format in secrets. Error: {je}")
            st.stop()
            return None, None
        except KeyError as ke:
            st.error(f"Firebase Initialization Error: Missing required secret key. Error: {ke}")
            st.stop()
            return None, None
        except Exception as e:
            st.error(f"Firebase Initialization Error: Check Streamlit Cloud Secrets for correct credentials.")
            st.error(f"Error details: {str(e)}")
            st.stop()
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
        st.session_state.app_id = 'default-app-id'
        st.session_state.model_loaded = False
        st.session_state.model = None
        st.session_state.scaler = None
        st.session_state.threshold = None

def load_assets_and_set_state():
    """Loads model assets and updates session state."""
    if not st.session_state.model_loaded:
        try:
            model, scaler, threshold = load_model_and_assets()
            if model and scaler and threshold:
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.threshold = threshold
                st.session_state.model_loaded = True
                st.success("AI Model and assets loaded.")
            else:
                st.error("Could not load all necessary AI assets. Check logs.")
        except Exception as e:
            st.error(f"Error loading model assets: {e}")

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
        'amount': float(transaction_data[-1]),  # Last feature is 'Amount'
        'features': [float(x) for x in transaction_data],  # Store raw features
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
            if not email or not password:
                st.error("Please enter both email and password.")
                return
                
            try:
                user = st.session_state.fb_auth.get_user_by_email(email)
                # In a real app, you'd use the client SDK for secure login. 
                # Here, we verify the user exists for demonstration purposes.
                if user:
                    st.session_state.user = user
                    st.session_state.is_admin = (email == ADMIN_EMAIL)
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
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
                user = st.session_state.fb_auth.create_user(
                    email=email,
                    password=password
                )
                st.success(f"Account created successfully for {user.email}! Please log in.")
                st.info("Note: The password is visible to Streamlit's backend in this demo. Use a dummy password.")
            except auth.EmailAlreadyExistsError:
                st.error("An account with this email already exists. Please login.")
            except Exception as e:
                st.error(f"Registration failed: {e}")

def logout_button():
    """Renders the logout button."""
    if st.session_state.user:
        if st.sidebar.button("Logout"):
            st.session_state.user = None
            st.session_state.is_admin = False
            st.rerun()

# --- MAIN APP UI SCREENS ---

def admin_dashboard():
    """UI for the Admin user."""
    st.markdown(f"<h1 style='color: {THEME_COLOR};'>Admin Dashboard</h1>", unsafe_allow_html=True)
    st.write(f"Welcome, Admin: {st.session_state.user.email}")
    
    # Create tabs for different admin functions
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 All Transactions", "👥 User Management", "⚙️ System Settings"])
    
    with tab1:
        st.subheader("System Overview")
        
        # Fetch aggregated statistics
        try:
            app_id = st.session_state.app_id
            users_ref = st.session_state.db.collection('artifacts').document(app_id).collection('users')
            
            # Count total users
            try:
                all_users = st.session_state.fb_auth.list_users()
                total_users = len(all_users.users)
            except Exception:
                total_users = "N/A"
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Users", total_users)
            with col2:
                st.metric("Total Transactions", "---", help="Requires aggregation query")
            with col3:
                st.metric("Fraud Detected", "---", help="Requires aggregation query")
            with col4:
                st.metric("System Status", "✅ Online")
            
            st.markdown("---")
            st.info("💡 **Tip**: Use the tabs above to manage users, view transactions, and configure system settings.")
            
        except Exception as e:
            st.error(f"Error loading overview: {e}")
    
    with tab2:
        st.subheader("All System Transactions")
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            filter_user = st.text_input("Filter by User Email (optional)")
        with col2:
            filter_fraud = st.selectbox("Filter by Status", ["All", "Fraud Only", "Non-Fraud Only"])
        
        if st.button("Load Transactions", type="primary"):
            with st.spinner("Fetching transactions from database..."):
                try:
                    app_id = st.session_state.app_id
                    users_ref = st.session_state.db.collection('artifacts').document(app_id).collection('users')
                    
                    transactions_data = []
                    
                    # Get all users
                    users_docs = users_ref.stream()
                    
                    for user_doc in users_docs:
                        user_id = user_doc.id
                        
                        # Skip if filtering by specific user
                        if filter_user:
                            try:
                                user_info = st.session_state.fb_auth.get_user(user_id)
                                if filter_user.lower() not in user_info.email.lower():
                                    continue
                            except:
                                continue
                        
                        # Get transactions for this user
                        transactions_ref = users_ref.document(user_id).collection('transactions')
                        transactions = transactions_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(50).stream()
                        
                        for trans in transactions:
                            trans_data = trans.to_dict()
                            
                            # Apply fraud filter
                            is_fraud = trans_data.get('prediction', {}).get('is_fraud', False)
                            if filter_fraud == "Fraud Only" and not is_fraud:
                                continue
                            elif filter_fraud == "Non-Fraud Only" and is_fraud:
                                continue
                            
                            # Get user email
                            try:
                                user_info = st.session_state.fb_auth.get_user(user_id)
                                user_email = user_info.email
                            except:
                                user_email = user_id
                            
                            transactions_data.append({
                                'User': user_email,
                                'Amount': f"${trans_data.get('amount', 0):.2f}",
                                'Status': '🚨 FRAUD' if is_fraud else '✅ Clear',
                                'Score': f"{trans_data.get('prediction', {}).get('error_score', 0):.4f}",
                                'Threshold': f"{trans_data.get('prediction', {}).get('threshold', 0):.4f}",
                                'Timestamp': trans_data.get('timestamp', 'N/A')
                            })
                    
                    if transactions_data:
                        st.success(f"Found {len(transactions_data)} transactions")
                        st.dataframe(transactions_data, use_container_width=True)
                        
                        # Summary statistics
                        fraud_count = sum(1 for t in transactions_data if '🚨' in t['Status'])
                        st.metric("Fraud Detected in Results", f"{fraud_count} / {len(transactions_data)}")
                    else:
                        st.warning("No transactions found matching your filters.")
                        
                except Exception as e:
                    st.error(f"Error fetching transactions: {e}")
    
    with tab3:
        st.subheader("User Management")
        
        # List all users
        if st.button("Load All Users", type="primary"):
            with st.spinner("Fetching users..."):
                try:
                    all_users = st.session_state.fb_auth.list_users()
                    
                    users_data = []
                    for user in all_users.users:
                        users_data.append({
                            'Email': user.email,
                            'UID': user.uid,
                            'Created': user.user_metadata.creation_timestamp if hasattr(user, 'user_metadata') else 'N/A',
                            'Admin': '✅' if user.email == ADMIN_EMAIL else '❌'
                        })
                    
                    st.success(f"Found {len(users_data)} users")
                    st.dataframe(users_data, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error fetching users: {e}")
        
        st.markdown("---")
        
        # Delete user section
        with st.expander("⚠️ Delete User (Dangerous)"):
            user_email_to_delete = st.text_input("User Email to Delete")
            confirm_delete = st.checkbox("I confirm I want to delete this user")
            
            if st.button("Delete User", type="secondary"):
                if not confirm_delete:
                    st.error("Please check the confirmation box.")
                elif user_email_to_delete == ADMIN_EMAIL:
                    st.error("Cannot delete the admin account!")
                elif user_email_to_delete:
                    try:
                        user = st.session_state.fb_auth.get_user_by_email(user_email_to_delete)
                        st.session_state.fb_auth.delete_user(user.uid)
                        st.success(f"User {user_email_to_delete} deleted successfully.")
                    except Exception as e:
                        st.error(f"Error deleting user: {e}")
                else:
                    st.error("Please enter a user email.")
    
    with tab4:
        st.subheader("System Settings")
        
        st.info("🔧 **Model Configuration**")
        st.write(f"Current Fraud Detection Threshold: **{st.session_state.threshold if st.session_state.threshold else 'Not loaded'}**")
        
        if st.session_state.model_loaded:
            st.success("✅ AI Model is loaded and ready")
        else:
            st.warning("⚠️ AI Model not loaded. Load it from the customer portal first.")
        
        st.markdown("---")
        
        st.info("📊 **Database Configuration**")
        st.write(f"App ID: `{st.session_state.app_id}`")
        st.write(f"Database Timeout: `{DB_TIMEOUT_SECONDS}s`")
        
        st.markdown("---")
        
        # Clear all data (dangerous operation)
        with st.expander("🗑️ Clear All Transaction Data (Very Dangerous)"):
            st.warning("This will delete ALL transaction data for ALL users. This action cannot be undone!")
            confirm_clear = st.checkbox("I understand this will delete all transaction data")
            
            if st.button("Clear All Data", type="secondary"):
                if confirm_clear:
                    try:
                        app_id = st.session_state.app_id
                        users_ref = st.session_state.db.collection('artifacts').document(app_id).collection('users')
                        
                        # This is a simplified version - in production, use batch deletes
                        st.warning("Data clearing not fully implemented for safety. Contact system administrator.")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("Please confirm the action.")


def customer_portal():
    """UI for a standard Customer user."""
    st.markdown(f"<h1 style='color: {THEME_COLOR};'>Customer Transaction Checker</h1>", unsafe_allow_html=True)
    st.write(f"Welcome, {st.session_state.user.email}")
    
    # 1. Load Assets
    load_assets_and_set_state()
    
    if st.session_state.model_loaded:
        st.subheader("Simulate a Transaction")
        
        # Add threshold adjustment option
        with st.expander("⚙️ Advanced Settings"):
            st.write("**Fraud Detection Sensitivity**")
            
            # Show current threshold from config
            original_threshold = st.session_state.threshold if st.session_state.threshold else 0.1
            st.warning(f"⚠️ Current threshold from config.json: {original_threshold:.4f}")
            
            if original_threshold > 0.5:
                st.error("🚨 Your threshold is EXTREMELY HIGH! This will cause almost all transactions to be marked as legitimate.")
                st.info("💡 Recommended: Use the override below to set a more reasonable threshold (0.01 - 0.15)")
            
            use_custom_threshold = st.checkbox("Override default threshold", value=True)  # Default to True
            if use_custom_threshold:
                custom_threshold = st.slider(
                    "Custom Threshold (higher = less sensitive)", 
                    min_value=0.0, 
                    max_value=0.5, 
                    value=0.05,  # Start with a reasonable default
                    step=0.001,
                    format="%.4f",
                    help="Increase to reduce false positives, decrease to catch more fraud. Typical range: 0.01-0.15"
                )
                st.session_state.threshold = custom_threshold
                st.success(f"✅ Using custom threshold: {custom_threshold:.4f}")
            else:
                st.warning(f"Using original threshold: {original_threshold:.4f}")
        
        # Add pre-defined test scenarios
        st.markdown("---")
        st.write("**Quick Test Scenarios** (Optional)")
        col1, col2, col3 = st.columns(3)
        
        test_scenario = None
        with col1:
            if st.button("🛒 Small Purchase ($25)"):
                test_scenario = "small"
        with col2:
            if st.button("💳 Medium Purchase ($150)"):
                test_scenario = "medium"
        with col3:
            if st.button("🚨 Large Purchase ($2500)"):
                test_scenario = "large"
        
        st.markdown("---")
        
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
            
            submitted = st.form_submit_button("Check for Fraud", type="primary")

        if submitted or test_scenario:
            # Validate amount input
            try:
                if test_scenario:
                    # Use predefined amounts for test scenarios
                    scenario_amounts = {"small": 25.0, "medium": 150.0, "large": 2500.0}
                    amount_value = scenario_amounts[test_scenario]
                    st.info(f"Testing with {test_scenario} purchase: ${amount_value}")
                else:
                    amount_value = float(transaction_amount)
                    if amount_value <= 0:
                        st.error("Please enter a positive transaction amount.")
                        return
            except ValueError:
                st.error("Please enter a valid number for the Transaction Amount.")
                return
            
            # IMPORTANT: Generate realistic mock features
            # Strategy: Use smaller variance for lower amounts (more typical), higher for large amounts
            np.random.seed()
            
            if test_scenario == "small":
                # Small purchase - very typical pattern
                mock_v_features = np.random.normal(loc=0.0, scale=0.3, size=INPUT_DIM - 2)
                time_feature = np.array([43200])  # Midday
            elif test_scenario == "medium":
                # Medium purchase - normal pattern
                mock_v_features = np.random.normal(loc=0.0, scale=0.5, size=INPUT_DIM - 2)
                time_feature = np.array([54000])  # Afternoon
            elif test_scenario == "large":
                # Large purchase - suspicious pattern (higher variance)
                mock_v_features = np.random.normal(loc=0.0, scale=1.2, size=INPUT_DIM - 2)
                mock_v_features[0] = 2.5  # Abnormal V1 value
                mock_v_features[2] = -2.8  # Abnormal V3 value
                time_feature = np.array([3600])  # Late night (suspicious)
            else:
                # Manual entry - use moderate variance
                if amount_value < 100:
                    mock_v_features = np.random.normal(loc=0.0, scale=0.4, size=INPUT_DIM - 2)
                elif amount_value < 500:
                    mock_v_features = np.random.normal(loc=0.0, scale=0.6, size=INPUT_DIM - 2)
                else:
                    mock_v_features = np.random.normal(loc=0.0, scale=0.9, size=INPUT_DIM - 2)
                time_feature = np.array([np.random.uniform(0, 86400)])
            
            # Amount feature
            amount_feature = np.array([amount_value])
            
            # CRITICAL: The raw transaction data should be UNSCALED
            raw_transaction_data = np.concatenate([time_feature, mock_v_features, amount_feature])
            
            # Run Prediction
            with st.spinner("Analyzing transaction for anomalies..."):
                try:
                    model = st.session_state.model
                    scaler = st.session_state.scaler
                    threshold = st.session_state.threshold
                    
                    error_score, is_anomaly = predict_transaction(model, scaler, threshold, raw_transaction_data)
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.exception(e)
                    return
                
            # Display Result
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Anomaly Score", f"{error_score:.4f}")
            with col2:
                st.metric("Threshold", f"{threshold:.4f}")
            with col3:
                difference = error_score - threshold
                st.metric("Difference", f"{difference:.4f}", delta=f"{'Over' if difference > 0 else 'Under'} threshold")
            
            if is_anomaly:
                st.error(f"🚨 FRAUD ALERT! ANOMALY DETECTED.")
                st.markdown("⚠️ **This transaction is flagged as suspicious and requires manual review.**")
                st.caption(f"The reconstruction error ({error_score:.4f}) exceeds the threshold ({threshold:.4f}) by {difference:.4f}")
            else:
                st.success("✅ Transaction is LIKELY LEGITIMATE.")
                st.markdown("**Transaction cleared based on reconstruction error.**")
                st.caption(f"The reconstruction error ({error_score:.4f}) is below the threshold ({threshold:.4f}) by {abs(difference):.4f}")
            
            # Debug information (expandable)
            with st.expander("🔍 Debug Information"):
                st.write("**Transaction Features (first 5):**")
                st.code(raw_transaction_data[:5])
                st.write("**Model Threshold:**", threshold)
                st.write("**Calculated Error:**", error_score)
                st.write("**Is Anomaly:**", is_anomaly)

            # Save to Database
            prediction_result = {
                'is_fraud': bool(is_anomaly), 
                'error_score': float(error_score),
                'threshold': float(threshold)
            }
            write_transaction_to_db(st.session_state.user.uid, raw_transaction_data, prediction_result)
            st.sidebar.success("Transaction saved to your history.")
            
        st.markdown("---")
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
        background-color: #059669;
    }}
    </style>
    """, unsafe_allow_html=True)

    # 1. Initialize Firebase and Get Instances
    db, fb_auth = initialize_firebase()
    
    if not db or not fb_auth:
        st.warning("Application halted. Check Streamlit logs for Firebase initialization error.")
        return
        
    # 2. Initialize Session State
    init_session_state(db, fb_auth)
    
    # 3. Main Content Rendering
    st.sidebar.title("SecureBank PoC")
    
    # Debug information in sidebar
    if st.session_state.user:
        st.sidebar.success(f"Logged in as: {st.session_state.user.email}")
        st.sidebar.info(f"Admin status: {'✅ Yes' if st.session_state.is_admin else '❌ No'}")
        st.sidebar.caption(f"User ID: {st.session_state.user.uid}")
    
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
