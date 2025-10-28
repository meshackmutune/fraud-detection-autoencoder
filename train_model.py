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

# Create Test Set - SEPARATE normal and fraud for proper threshold calculation
X_test_normal, X_val_normal = train_test_split(X_temp, test_size=0.5, random_state=RANDOM_SEED)
X_test_fraud = fraud_df.drop('Class', axis=1)

# Combined test set for evaluation
X_test = pd.concat([X_test_normal, X_test_fraud]).sample(frac=1, random_state=RANDOM_SEED)
y_test = pd.concat([pd.Series([0]*len(X_test_normal)), pd.Series([1]*len(X_test_fraud))]).sample(frac=1, random_state=RANDOM_SEED).values

# Scaling: Fit ONLY on the normal training data
scaler = StandardScaler()
scaler.fit(X_train_normal)

# Apply Standardization
X_train_scaled = scaler.transform(X_train_normal)
X_test_scaled = scaler.transform(X_test)
X_test_normal_scaled = scaler.transform(X_test_normal)  # CRITICAL: Separate normal test data

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
X_test_normal_tensor = torch.tensor(X_test_normal_scaled, dtype=torch.float32).to(DEVICE)  # CRITICAL

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
print("\n3. Calculating Threshold on NORMAL transactions only...")
model.eval()

# CRITICAL FIX: Calculate threshold on NORMAL test data ONLY
with torch.no_grad():
    reconstructions_normal = model(X_test_normal_tensor)
    mse_error_normal = torch.mean((reconstructions_normal - X_test_normal_tensor) ** 2, dim=1).cpu().numpy()

# Determine Anomaly Threshold - 95th percentile of NORMAL transaction errors
threshold = np.percentile(mse_error_normal, 95)

print(f"\nThreshold Statistics (Normal Transactions):")
print(f"  Mean Error: {np.mean(mse_error_normal):.6f}")
print(f"  Median Error: {np.median(mse_error_normal):.6f}")
print(f"  Std Dev: {np.std(mse_error_normal):.6f}")
print(f"  95th Percentile (THRESHOLD): {threshold:.6f}")

# Now evaluate on the full test set (normal + fraud)
with torch.no_grad():
    reconstructions = model(X_test_tensor)
    mse_error = torch.mean((reconstructions - X_test_tensor) ** 2, dim=1).cpu().numpy()

y_pred = (mse_error > threshold).astype(int)

print(f"\n4. Evaluation Metrics (Threshold: {threshold:.6f}):")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fraud (1)']))

precision, recall, _ = precision_recall_curve(y_test, mse_error)
auc_pr = auc(recall, precision)
print(f"Area Under the Precision-Recall Curve (AUC-PR): {auc_pr:.4f}")

# Additional statistics
fraud_indices = y_test == 1
normal_indices = y_test == 0
print(f"\nReconstruction Error Statistics:")
print(f"  Normal transactions - Mean: {np.mean(mse_error[normal_indices]):.6f}, Max: {np.max(mse_error[normal_indices]):.6f}")
print(f"  Fraud transactions  - Mean: {np.mean(mse_error[fraud_indices]):.6f}, Max: {np.max(mse_error[fraud_indices]):.6f}")


# --- 5. ARTIFACT SAVING (CRUCIAL FOR STREAMLIT DEPLOYMENT) ---
print("\n--- 5. Saving Deployment Artifacts ---")

# A. Save Model Weights
torch.save(model.state_dict(), 'fraud_autoencoder_model.pth')
print("✅ 1/3 Model weights saved to 'fraud_autoencoder_model.pth'")

# B. Save Fitted Scaler
with open('scaler_params.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ 2/3 StandardScaler saved to 'scaler_params.pkl'")

# C. Save Anomaly Threshold with metadata
threshold_float = float(threshold)
config_data = {
    'anomaly_threshold': threshold_float,
    'threshold_percentile': 95,
    'normal_mean_error': float(np.mean(mse_error_normal)),
    'normal_std_error': float(np.std(mse_error_normal)),
    'fraud_detection_rate': float(np.sum(mse_error[fraud_indices] > threshold) / len(mse_error[fraud_indices]) * 100)
}

with open('config.json', 'w') as f:
    json.dump(config_data, f, indent=4)
print("✅ 3/3 Anomaly Threshold saved to 'config.json'")

print("-" * 60)
print(f"\n🎉 Deployment setup complete!")
print(f"   Threshold: {threshold_float:.6f}")
print(f"   This should give you a working fraud detection model.")
print("\nYou can now run 'streamlit run app.py'")
