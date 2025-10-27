import torch
import torch.nn as nn
import json
import pickle
import streamlit as st
import numpy as np
import os # Required for file existence check
# Note: You MUST add 'gdown' to your requirements.txt for this to work on Streamlit Cloud!

# --- CONFIGURATION (Must match your training setup) ---
INPUT_DIM = 30
DEVICE = torch.device("cpu") # Use CPU for deployment simulation simplicity

# --- 0. DATA CONFIGURATION ---
# REPLACE THIS with the FILE ID from your shareable Google Drive link
GDRIVE_FILE_ID = "1FL-f5uoJZQDx7d5vaonI1cUdNyZOaxi2" 
DATA_PATH = "creditcard.csv" 

# --- 1. MODEL ARCHITECTURE ---
class DeepAutoencoder(nn.Module):
    """Symmetric Deep Autoencoder architecture for fraud detection."""
    def __init__(self, input_dim):
        super(DeepAutoencoder, self).__init__()
        latent_dim = 10 
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20), nn.ReLU(),
            nn.Linear(20, 15), nn.ReLU(),
            nn.Linear(15, latent_dim), nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 15), nn.ReLU(),
            nn.Linear(15, 20), nn.ReLU(),
            nn.Linear(20, input_dim), nn.Identity()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- 2. CACHED ASSET LOADER (@st.cache_resource) ---

@st.cache_resource
def load_and_cache_data():
    """Downloads the large dataset from Drive if not found, and caches it."""
    try:
        import gdown # Use gdown for reliable Google Drive download
    except ImportError:
        st.error("The 'gdown' library is missing. Please add it to requirements.txt.")
        return None

    # 1. Check if the file is already downloaded in the Streamlit Cloud temp folder
    if not os.path.exists(DATA_PATH):
        st.info(f"Downloading dataset from Google Drive (approx 144MB) using File ID: {GDRIVE_FILE_ID}")
        try:
            # Construct the direct download URL for gdown
            url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
            gdown.download(url, DATA_PATH, quiet=True)
            st.success("Dataset download complete.")
        except Exception as e:
            st.error(f"Failed to download data from Google Drive. Check the File ID and permissions (must be 'Anyone with the link'). Error: {e}")
            return None
    
    # We return the path, even though it's not strictly used in this PoC, 
    # to confirm the download finished.
    return DATA_PATH


@st.cache_resource
def load_model_and_assets():
    """Loads the model, scaler, and threshold only once upon startup."""
    # NOTE: We DO NOT need the large CSV file here, only the small artifacts.
    try:
        # Load Model
        model = DeepAutoencoder(INPUT_DIM)
        # Load state dictionary, mapping any CUDA tensors to CPU
        model.load_state_dict(torch.load('fraud_autoencoder_model.pth', map_location=DEVICE))
        model.eval()
        
        # Load Scaler (Required for preprocessing new data)
        with open('scaler_params.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        # Load Threshold (Required for classification)
        with open('config.json', 'r') as f:
            config = json.load(f)
            threshold = config['anomaly_threshold']
            
        return model, scaler, threshold
    except FileNotFoundError as e:
        st.error(f"FATAL ERROR: Required artifact file not found: {e}. Ensure all three artifact files (.pth, .pkl, .json) are present on GitHub.")
        return None, None, None
    except Exception as e:
        st.error(f"ERROR LOADING ASSETS: {e}")
        return None, None, None

# --- 3. INFERENCE FUNCTION (The Service Logic) ---
# ... (This function remains unchanged)
def predict_transaction(model, scaler, threshold, raw_transaction_data):
    """Processes a raw vector through the model and returns the result."""
    
    # 1. Scale the input vector using the loaded scaler
    scaled_data = scaler.transform(raw_transaction_data.reshape(1, -1))
    
    # 2. Convert to PyTorch Tensor
    tensor_data = torch.tensor(scaled_data, dtype=torch.float32).to(DEVICE)
    
    # 3. Perform Inference
    with torch.no_grad():
        reconstruction = model(tensor_data)
        
    # 4. Calculate Reconstruction Error (MSE)
    mse_error = torch.mean((reconstruction - tensor_data) ** 2, dim=1).cpu().numpy()[0]
    
    # 5. Classify
    is_anomaly = mse_error > threshold
    
    return mse_error, is_anomaly