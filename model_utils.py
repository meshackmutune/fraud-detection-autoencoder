import torch
import torch.nn as nn
import json
import pickle
import streamlit as st
import numpy as np
import os

INPUT_DIM = 30
DEVICE = torch.device("cpu")

class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DeepAutoencoder, self).__init__()
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

@st.cache_resource
def load_model_and_assets():
    try:
        model = DeepAutoencoder(INPUT_DIM)
        model.load_state_dict(torch.load('fraud_autoencoder_model.pth', map_location=DEVICE))
        model.eval()

        with open('scaler_params.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open('config.json', 'r') as f:
            config = json.load(f)
            threshold = config['anomaly_threshold']
            means = np.array(config.get('scaler_means', np.zeros(INPUT_DIM)))
            stds = np.array(config.get('scaler_stds', np.ones(INPUT_DIM)))

        return model, scaler, threshold, means, stds
    except Exception as e:
        st.error(f"Asset loading error: {e}")
        return None, None, None, None, None

def generate_realistic_transaction(means, stds, amount):
    """Generate a realistic transaction vector based on training stats."""
    raw_data = np.random.normal(means, stds)
    raw_data[29] = amount  # override amount
    return raw_data

def predict_transaction(model, scaler, threshold, raw_transaction_data):
    scaled_data = scaler.transform(raw_transaction_data.reshape(1, -1))
    tensor_data = torch.tensor(scaled_data, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        reconstruction = model(tensor_data)
    mse_error = torch.mean((reconstruction - tensor_data) ** 2, dim=1).cpu().numpy()[0]
    is_anomaly = mse_error > threshold
    return mse_error, is_anomaly
