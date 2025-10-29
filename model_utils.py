import torch
import torch.nn as nn
import json
import pickle
import streamlit as st
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_DIM = 30
DEVICE = torch.device("cpu")   # Streamlit Cloud has no GPU

# --- 0. DATA CONFIGURATION ---
GDRIVE_FILE_ID = "1FL-f5uoJZQDx7d5vaonI1cUdNyZOaxi2"
DATA_PATH = "creditcard.csv"

# --- 1. MODEL ARCHITECTURE ---
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        latent_dim = 10
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20), nn.ReLU(),
            nn.Linear(20, 15), nn.ReLU(),
            nn.Linear(15, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 15), nn.ReLU(),
            nn.Linear(15, 20), nn.ReLU(),
            nn.Linear(20, input_dim), nn.Identity()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- 2. CACHED ASSET LOADER ---
@st.cache_resource
def load_and_cache_data():
    if not os.path.exists(DATA_PATH):
        import gdown
        st.info(f"Downloading dataset (~144 MB)…")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, DATA_PATH, quiet=True)
        st.success("Dataset ready.")
    return DATA_PATH

@st.cache_resource
def load_model_and_assets():
    model = DeepAutoencoder(INPUT_DIM)
    model.load_state_dict(torch.load('fraud_autoencoder_model.pth', map_location=DEVICE))
    model.eval()

    with open('scaler_params.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('config.json', 'r') as f:
        threshold = json.load(f)['anomaly_threshold']

    return model, scaler, threshold

# --- 3. INFERENCE ---
def predict_transaction(model, scaler, threshold, raw_transaction_data):
    scaled = scaler.transform(raw_transaction_data.reshape(1, -1))
    tensor = torch.tensor(scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        recon = model(tensor)

    mse = torch.mean((recon - tensor) ** 2, dim=1).cpu().numpy()[0]
    return mse, mse > threshold
