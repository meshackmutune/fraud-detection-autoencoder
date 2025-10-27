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

# --- CONFIGURATION ---
RANDOM_SEED = 42
BATCH_SIZE = 128
N_EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)

# --- 1. DATA LOADING AND PREPROCESSING ---
print("1. Loading and Preprocessing Data...")
# NOTE: Ensure 'creditcard.csv' is accessible by the script.
try:
    df = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    print("FATAL ERROR: 'creditcard.csv' not found. Please place the dataset file in the execution directory.")
    exit()

features = df.drop('Class', axis=1)
INPUT_DIM = features.shape[1] 

# Separate Normal and Fraud transactions
normal_df = df[df['Class'] == 0]
fraud_df = df[df['Class'] == 1]

# Partitioning: Train ONLY on normal data (UAD Constraint)
X_train_normal, X_temp = train_test_split(
    normal_df.drop('Class', axis=1), 
    test_size=0.2, 
    random_state=RANDOM_SEED
)

# Create Test Set (imbalanced)
X_test_normal, _ = train_test_split(X_temp, test_size=0.5, random_state=RANDOM_SEED) 
X_test = pd.concat([X_test_normal, fraud_df.drop('Class', axis=1)]).sample(frac=1, random_state=RANDOM_SEED)
y_test = pd.concat([pd.Series([0]*len(X_test_normal)), pd.Series([1]*len(fraud_df))]).sample(frac=1, random_state=RANDOM_SEED).values

# Scaling: Fit ONLY on the normal training data
scaler = StandardScaler()
scaler.fit(X_train_normal)

# Apply Standardization
X_train_scaled = scaler.transform(X_train_normal)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# --- 2. MODEL ARCHITECTURE ---
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

model = DeepAutoencoder(INPUT_DIM).to(DEVICE)

# --- 3. TRAINING PROCEDURE ---
criterion = nn.MSELoss(reduction='mean') 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("2. Starting Model Training...")
for epoch in range(N_EPOCHS):
    model.train()
    running_loss = 0.0
    for data in train_loader:
        inputs, targets = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(X_train_tensor)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{N_EPOCHS}], Loss: {epoch_loss:.6f}')

print("Training complete.")


# --- 4. EVALUATION AND THRESHOLD DETERMINATION ---
model.eval()
with torch.no_grad():
    reconstructions = model(X_test_tensor)
    mse_error = torch.mean((reconstructions - X_test_tensor) ** 2, dim=1).cpu().numpy()

# Determine Anomaly Threshold (e.g., 95th percentile of errors)
threshold = np.percentile(mse_error, 95) 
y_pred = (mse_error > threshold).astype(int)

print(f"\n3. Evaluation Metrics (Threshold: {threshold:.4f}):")
print("-" * 40)
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fraud (1)']))

precision, recall, _ = precision_recall_curve(y_test, mse_error)
auc_pr = auc(recall, precision)
print(f"Area Under the Precision-Recall Curve (AUC-PR): {auc_pr:.4f}")


# --- 5. ARTIFACT SAVING (CRUCIAL FOR STREAMLIT DEPLOYMENT) ---
print("\n--- 4. Saving Deployment Artifacts ---")

# A. Save Model Weights
torch.save(model.state_dict(), 'fraud_autoencoder_model.pth')
print("✅ 1/3 Model weights saved to 'fraud_autoencoder_model.pth'")

# B. Save Fitted Scaler
with open('scaler_params.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ 2/3 StandardScaler saved to 'scaler_params.pkl'")

# C. Save Anomaly Threshold
threshold_float = float(threshold)
config_data = {'anomaly_threshold': threshold_float}

with open('config.json', 'w') as f:
    json.dump(config_data, f, indent=4)
print("✅ 3/3 Anomaly Threshold saved to 'config.json'")

print("------------------------------------")
print("\nDeployment setup complete. You can now run 'streamlit run app.py'")