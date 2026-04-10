# train_model.py
import io, json, pickle, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import IsolationForest
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
# Training set = NORMAL transactions only  (UAD constraint)
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


# Deep Autoencoder
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
model     = DeepAutoencoder(INPUT_DIM).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

X_tensor = torch.tensor(X_train_sc, dtype=torch.float32).to(DEVICE)
loader   = DataLoader(TensorDataset(X_tensor, X_tensor),
                      batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    if epoch % 10 == 0:
        print(f"  Epoch {epoch}/{EPOCHS}  loss: {total_loss/len(X_tensor):.5f}")

# Evaluate the Autoencoder
model.eval()
with torch.no_grad():
    X_test_t  = torch.tensor(X_test_sc, dtype=torch.float32).to(DEVICE)
    recon     = model(X_test_t)
    ae_scores = torch.mean((recon - X_test_t) ** 2, dim=1).cpu().numpy()

ae_threshold = float(np.percentile(ae_scores, 95))
ae_pred      = (ae_scores > ae_threshold).astype(int)

p, r, _   = precision_recall_curve(y_test, ae_scores)
ae_auc_pr = auc(r, p)

print(f"\nAutoencoder  |  AUC-PR: {ae_auc_pr:.4f}  |  threshold: {ae_threshold:.5f}")
print(classification_report(y_test, ae_pred, target_names=["Normal", "Fraud"]))

# Isolation Forest
print("Training Isolation Forest ...")
iso_model = IsolationForest(n_estimators=100, contamination="auto",
                            random_state=SEED, n_jobs=-1)
iso_model.fit(X_train_sc)

if_scores = -iso_model.score_samples(X_test_sc)
if_pred   = np.where(iso_model.predict(X_test_sc) == -1, 1, 0)

p, r, _   = precision_recall_curve(y_test, if_scores)
if_auc_pr = auc(r, p)

print(f"\nIsolation Forest  |  AUC-PR: {if_auc_pr:.4f}")
print(classification_report(y_test, if_pred, target_names=["Normal", "Fraud"]))


# Pick the best model
if ae_auc_pr >= if_auc_pr:
    best_name  = "Autoencoder"
    best_obj   = model
    best_auc   = ae_auc_pr
    threshold  = ae_threshold
else:
    best_name  = "Isolation Forest"
    best_obj   = iso_model
    best_auc   = if_auc_pr
    threshold  = None

print(f"\nBest model: {best_name}  (AUC-PR = {best_auc:.4f})")


# Package & save  ->  fraud_detector.zip
manifest = {
    "model_type"   : best_name,
    "input_dim"    : int(INPUT_DIM),
    "feature_names": list(features.columns),
    "auc_pr"       : round(best_auc, 6),
    "threshold"    : float(threshold) if threshold else None,
}

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    scaler_buf = io.BytesIO()
    pickle.dump(scaler, scaler_buf)
    zf.writestr("scaler.pkl", scaler_buf.getvalue())

    if best_name == "Autoencoder":
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        zf.writestr("model.pth", buf.getvalue())
    else:
        buf = io.BytesIO()
        pickle.dump(iso_model, buf)
        zf.writestr("model.pkl", buf.getvalue())

print(f"Saved -> {OUTPUT_ZIP}")
