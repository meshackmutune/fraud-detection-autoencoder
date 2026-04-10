# train_model.py
import io, json, pickle, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, auc

# Configurations
DATA_PATH   = "creditcard.csv"
OUTPUT_ZIP  = "fraud_detector.zip"
SEED        = 42
EPOCHS      = 50
BATCH_SIZE  = 128
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)


# Load data
print("Loading data ...")
df        = pd.read_csv(DATA_PATH)
features  = df.drop("Class", axis=1)
INPUT_DIM = features.shape[1]

normal_df = df[df["Class"] == 0]
fraud_df  = df[df["Class"] == 1]


# Split & scale
# Training set = NORMAL transactions only (UAD constraint)
X_train, X_temp = train_test_split(
    normal_df.drop("Class", axis=1), test_size=0.2, random_state=SEED
)
X_test_normal, _ = train_test_split(X_temp, test_size=0.5, random_state=SEED)

X_test = pd.concat([X_test_normal, fraud_df.drop("Class", axis=1)]).sample(
    frac=1, random_state=SEED
)
y_test = pd.concat([
    pd.Series([0] * len(X_test_normal)),
    pd.Series([1] * len(fraud_df)),
]).sample(frac=1, random_state=SEED).values

scaler        = StandardScaler().fit(X_train)
X_train_sc    = scaler.transform(X_train)
X_test_sc     = scaler.transform(X_test)

print(f"  Train size : {len(X_train_sc):,}  (normal only)")
print(f"  Test  size : {len(X_test_sc):,}  (fraud: {y_test.sum():,})")


# --- 1. Deep Autoencoder Architecture ---
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20), nn.ReLU(),
            nn.Linear(20, 15),        nn.ReLU(),
            nn.Linear(15, 10),        nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 15),        nn.ReLU(),
            nn.Linear(15, 20),        nn.ReLU(),
            nn.Linear(20, input_dim), nn.Identity(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Training the Autoencoder
print("\nTraining Autoencoder ...")
ae_model  = DeepAutoencoder(INPUT_DIM).to(DEVICE)
criterion = nn.MSELoss()
ae_optimizer = optim.Adam(ae_model.parameters(), lr=LR)

X_tensor = torch.tensor(X_train_sc, dtype=torch.float32).to(DEVICE)
loader   = DataLoader(TensorDataset(X_tensor, X_tensor),
                      batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(1, EPOCHS + 1):
    ae_model.train()
    total_loss = 0
    for x, y in loader:
        ae_optimizer.zero_grad()
        loss = criterion(ae_model(x), y)
        loss.backward()
        ae_optimizer.step()
        total_loss += loss.item() * x.size(0)
    if epoch % 10 == 0:
        print(f"  AE Epoch {epoch}/{EPOCHS}  loss: {total_loss/len(X_tensor):.5f}")

# Evaluate the Autoencoder
ae_model.eval()
with torch.no_grad():
    X_test_t  = torch.tensor(X_test_sc, dtype=torch.float32).to(DEVICE)
    recon     = ae_model(X_test_t)
    ae_scores = torch.mean((recon - X_test_t) ** 2, dim=1).cpu().numpy()

ae_threshold = float(np.percentile(ae_scores, 95))
ae_pred      = (ae_scores > ae_threshold).astype(int)

p_ae, r_ae, _ = precision_recall_curve(y_test, ae_scores)
ae_auc_pr     = auc(r_ae, p_ae)

print(f"\nAutoencoder  |  AUC-PR: {ae_auc_pr:.4f}  |  threshold: {ae_threshold:.5f}")
print(classification_report(y_test, ae_pred, target_names=["Normal", "Fraud"]))


# --- 2. LSTM Anomaly Detector ---
print("\nTraining LSTM ...")

# Reshape data for LSTM: (samples, time_steps, features)
X_train_lstm = X_train_sc.reshape((X_train_sc.shape[0], 1, X_train_sc.shape[1]))
X_test_lstm  = X_test_sc.reshape((X_test_sc.shape[0], 1, X_test_sc.shape[1]))

class LSTMDetector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=20, 
                            num_layers=1, batch_first=True)
        self.linear = nn.Linear(20, input_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out)

lstm_model = LSTMDetector(INPUT_DIM).to(DEVICE)
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=LR)

X_train_t_lstm = torch.tensor(X_train_lstm, dtype=torch.float32).to(DEVICE)
lstm_loader = DataLoader(TensorDataset(X_train_t_lstm, X_train_t_lstm),
                         batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(1, EPOCHS + 1):
    lstm_model.train()
    total_loss = 0
    for x, y in lstm_loader:
        lstm_optimizer.zero_grad()
        loss = criterion(lstm_model(x), y)
        loss.backward()
        lstm_optimizer.step()
        total_loss += loss.item() * x.size(0)
    if epoch % 10 == 0:
        print(f"  LSTM Epoch {epoch}/{EPOCHS}  loss: {total_loss/len(X_train_t_lstm):.5f}")

# Evaluate the LSTM
lstm_model.eval()
with torch.no_grad():
    X_test_t_lstm = torch.tensor(X_test_lstm, dtype=torch.float32).to(DEVICE)
    lstm_recon = lstm_model(X_test_t_lstm)
    lstm_scores = torch.mean((lstm_recon - X_test_t_lstm) ** 2, dim=(1, 2)).cpu().numpy()

lstm_threshold = float(np.percentile(lstm_scores, 95))
lstm_pred = (lstm_scores > lstm_threshold).astype(int)

p_lstm, r_lstm, _ = precision_recall_curve(y_test, lstm_scores)
lstm_auc_pr = auc(r_lstm, p_lstm)

print(f"\nLSTM  |  AUC-PR: {lstm_auc_pr:.4f}  |  threshold: {lstm_threshold:.5f}")
print(classification_report(y_test, lstm_pred, target_names=["Normal", "Fraud"]))


# --- 3. Pick the best model ---
if ae_auc_pr >= lstm_auc_pr:
    best_name  = "Autoencoder"
    best_obj   = ae_model
    best_auc   = ae_auc_pr
    threshold  = ae_threshold
else:
    best_name  = "LSTM"
    best_obj   = lstm_model
    best_auc   = lstm_auc_pr
    threshold  = lstm_threshold

print(f"\nBest model: {best_name}  (AUC-PR = {best_auc:.4f})")


# --- 4. Package & save  ->  fraud_detector.zip ---
manifest = {
    "model_type"   : best_name,
    "input_dim"    : int(INPUT_DIM),
    "feature_names": list(features.columns),
    "auc_pr"       : round(best_auc, 6),
    "threshold"    : float(threshold),
}

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    scaler_buf = io.BytesIO()
    pickle.dump(scaler, scaler_buf)
    zf.writestr("scaler.pkl", scaler_buf.getvalue())

    buf = io.BytesIO()
    torch.save(best_obj.state_dict(), buf)
    zf.writestr("model.pth", buf.getvalue())

print(f"Saved -> {OUTPUT_ZIP}")
