# app.py

import streamlit as st
import numpy as np
from model_utils import load_model_and_assets, predict_transaction, INPUT_DIM

# --- LOAD CACHED ASSETS (Model, Scaler, Threshold) ---
MODEL, SCALER, THRESHOLD = load_model_and_assets()

# --- STREAMLIT APPLICATION ROUTING ---

# Only proceed if assets were successfully loaded
if MODEL and SCALER and THRESHOLD is not None:
    st.sidebar.title("💳 Fraud Detection PoC")
    
    # Role-based selection for Admin/Customer (U.R. fulfillment)
    page = st.sidebar.radio("Go to", ["Customer Interface", "Administrator Dashboard"])

    if page == "Customer Interface":
        st.header("Customer Transaction Interface")
        st.markdown("*(Simulating U.R. 2.1 - Secure Access and U.R. 2.2 - Notification)*")
        
        st.subheader("Simulate New Transaction")
        
        # --- Simple user input simulation ---
        # We only take the Amount as input for simplicity, assuming others are zero or default.
        amount = st.slider("Transaction Amount", 0.0, 2500.0, 100.0)
        
        # Create the required 30-feature vector
        # NOTE: This assumes 'Amount' is the last feature (index 29) after the 28 PCA features and 'Time'.
        raw_data = np.zeros(INPUT_DIM)
        raw_data[29] = amount 
        
        if st.button("Process Transaction"):
            with st.spinner('Processing transaction...'):
                # Call the service layer function
                error, anomaly = predict_transaction(MODEL, SCALER, THRESHOLD, raw_data)
            
            st.write(f"Reconstruction Error (MSE): **{error:.6f}**")
            
            if anomaly:
                st.error("🚨 ANOMALY DETECTED! This transaction is flagged as high risk. Please confirm immediately.")
            else:
                st.success("✅ Transaction verified as normal. Proceeding.")

    elif page == "Administrator Dashboard":
        st.header("Fraud Analyst Dashboard")
        st.markdown("*(Simulating U.R. 1.1 - Admin Access and U.R. 1.2 - Threshold Tuning)*")
        
        st.metric(label="Current Operational Threshold (MSE)", value=f"{THRESHOLD:.4f}")

        # U.R. 1.2: Threshold Tuning Simulation
        st.subheader("Threshold Tuning Interface")
        st.caption("Adjust this slider to see the impact on fraud sensitivity (in a live system).")
        
        # Create a slider centered around the optimal THRESHOLD
        new_threshold = st.slider(
            "Adjust Anomaly Threshold", 
            float(THRESHOLD * 0.5), # Min value
            float(THRESHOLD * 2),   # Max value
            float(THRESHOLD),       # Default value
            step=0.0001,
            format="%.4f"
        )
        
        st.warning(f"Proposed new threshold for system test: **{new_threshold:.4f}**")
        st.caption("Analyst would use this control to optimize the Precision/Recall trade-off.")

else:
    st.error("Application cannot start. Please resolve the file loading error shown above.")