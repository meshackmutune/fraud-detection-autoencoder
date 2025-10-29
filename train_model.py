# train_model.py - FINAL VERSION THAT WORKS
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, classification_report
import pickle
import json
import os
import gdown

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------
CSV = "creditcard.csv"
if not os.path.exists(CSV):
    import gdown
    gdown.download("https://drive.google.com/uc?id=1FL-f5uoJZQDx7d5vaonI1cUdNyZOaxi2", CSV)

# ---------------------------------------------------------
# LOAD & SPLIT - EXACT ORIGINAL LOGIC
# ---------------------------------------------------------
print("Loading data...")
df = pd.read_csv(CSV)
normal_df = df[df['Class'] == 0]
fraud_df = df[df['Class'] == 1]

print(f"Normal: {len(normal_df):,}, Fraud: {len(fraud_df):,}")

# CRITICAL: EXACT ORIGINAL SPLIT
X_train_normal, X_temp = train_test_split(
    normal_df.drop('Class', axis=1), 
    test_size=0.2, 
    random_state=SEED
)

# THIS IS THE KEY FIX: NO SECOND SPLIT - USE ALL OF X_temp AS TEST_NORMAL
X_test_normal = X_temp  # ← NOT split further!

# Build test set: ALL X_temp_normal + ALL fraud
X_test = pd.concat([X_test_normal, fraud_df.drop('Class', axis=1)])
y_test = pd.concat([
    pd.Series([0]*len(X_test_normal)), 
    pd.Series([1]*len(fraud_df))
]).sample(frac=1, random_state=SEED).reset_index(drop=True)

X_test = X_test.sample(frac=1, random_state=SEED).reset_index(drop=True)
X_train_normal = X_train_normal.values
X_test = X_test.values
y_test = y_test.values

print(f"Train normal: {len(X_train_normal):,}")
print(f"Test  mixed : {len(X_test):,} (fraud: {sum(y_test):,})")

# ---------------------------------------------------------
# SCALE
# ---------------------------------------------------------
scaler = StandardScaler()
scaler.fit(X_train_normal)
X_train_s = scaler.transform(X_train_normal)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------
train_tensor = torch.tensor(X_train_s, dtype=torch.float32).to(DEVICE)
loader = DataLoader(TensorDataset(train_tensor, train_tensor), 
                    batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20), nn.ReLU(),
            nn.Linear(20, 15), nn.ReLU(),
            nn.Linear(15, 10), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 15), nn.ReLU(),
            nn.Linear(15, 20), nn.ReLU(),
            nn.Linear(20, input_dim), nn.Identity()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = DeepAutoencoder(X_train_normal.shape[1]).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------
print("\nTraining...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    for xb, _ in loader:
        optimizer.zero_grad()
        recon = model(xb)
        loss = criterion(recon, xb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch in [1, 10, 20, 30, 40, 50]:
        print(f"  Epoch {epoch:2d} | Loss: {epoch_loss/len(loader):.6f}")

# ---------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------
model.eval()
with torch.no_grad():
    test_tensor = torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE)
    recon = model(test_tensor)
    mse = torch.mean((recon - test_tensor)**2, dim=1).cpu().numpy()

# ---------------------------------------------------------
# THRESHOLD
# ---------------------------------------------------------
threshold = np.percentile(mse, 95)
y_pred = (mse > threshold).astype(int)

print("\n" + "="*60)
print(f"THRESHOLD (95th %ile): {threshold:.6f}")
print("="*60)
print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

from sklearn.metrics import auc
precision, recall, _ = precision_recall_curve(y_test, mse)
print(f"AUC-PR: {auc(recall, precision):.4f}")

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------
torch.save(model.state_dict(), "fraud_autoencoder_model.pth")
with open("scaler_params.pkl", "wb") as f:
    pickle.dump(scaler, f)

config = {
    "anomaly_threshold": float(threshold),
    "threshold_percentile": 95,
    "fraud_detection_rate": float(np.mean(y_pred[y_test==1]))*100,
    "false_positive_rate": float(np.mean(y_pred[y_test==0]))*100,
    "auc_pr": float(auc(recall, precision))
}
with open("config.json", "w") as f:
    json.dump(config, f, indent=4)

print("\n✅ Artifacts saved!")s
