import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve, auc
import pickle
import json

# Configuration and device setup
RANDOM_SEED = 42
BATCH_SIZE = 128
N_EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)

print("=" * 60)
print("FRAUD DETECTION MODEL TRAINING")
print("=" * 60)

# --- 1. LOADING THE DATASET ---
print("\n1. Loading dataset...")
try:
    df = pd.read_csv('creditcard.csv')
    print(f"✅ Dataset loaded: {df.shape[0]} transactions, {df.shape[1]} features")
except FileNotFoundError:
    print("❌ FATAL ERROR: 'creditcard.csv' not found.")
    print("   Please ensure the dataset is in the same directory as this script.")
    exit()

# Check for missing values
if df.isnull().sum().sum() > 0:
    print("⚠️  Warning: Dataset contains missing values. Dropping them...")
    df = df.dropna()

# Display class distribution
normal_count = len(df[df['Class'] == 0])
fraud_count = len(df[df['Class'] == 1])
print(f"\nClass Distribution:")
print(f"  Normal transactions: {normal_count} ({normal_count/len(df)*100:.2f}%)")
print(f"  Fraud transactions:  {fraud_count} ({fraud_count/len(df)*100:.2f}%)")

# --- 2. DATA PREPROCESSING AND PARTITIONING ---
print("\n2. Preprocessing and splitting data...")

features = df.drop('Class', axis=1)
INPUT_DIM = features.shape[1]
print(f"   Input dimension: {INPUT_DIM} features")

# Separate Normal and Fraud transactions
normal_df = df[df['Class'] == 0]
fraud_df = df[df['Class'] == 1]

# CRITICAL: Split data properly for threshold calculation
# Train on 80% of normal data
X_train_normal, X_temp_normal = train_test_split(
    normal_df.drop('Class', axis=1), 
    test_size=0.2, 
    random_state=RANDOM_SEED
)

# Split remaining normal data: 50% for validation (threshold calculation), 50% for testing
X_val_normal, X_test_normal = train_test_split(
    X_temp_normal, 
    test_size=0.5, 
    random_state=RANDOM_SEED
)

print(f"   Training set (normal only): {len(X_train_normal)} samples")
print(f"   Validation set (normal only): {len(X_val_normal)} samples")
print(f"   Test set (normal): {len(X_test_normal)} samples")
print(f"   Test set (fraud): {len(fraud_df)} samples")

# Create test set with both normal and fraud (for final evaluation)
X_test_fraud = fraud_df.drop('Class', axis=1)
X_test_combined = pd.concat([X_test_normal, X_test_fraud]).sample(frac=1, random_state=RANDOM_SEED)
y_test_combined = pd.concat([
    pd.Series([0]*len(X_test_normal)), 
    pd.Series([1]*len(X_test_fraud))
]).sample(frac=1, random_state=RANDOM_SEED).values

# --- 3. FEATURE SCALING ---
print("\n3. Applying StandardScaler (fit on training data only)...")
scaler = StandardScaler()
scaler.fit(X_train_normal)

# Transform all datasets
X_train_scaled = scaler.transform(X_train_normal)
X_val_scaled = scaler.transform(X_val_normal)  # CRITICAL: Separate validation set
X_test_combined_scaled = scaler.transform(X_test_combined)

print(f"   Scaler fitted on {len(X_train_normal)} normal transactions")
print(f"   Mean: {scaler.mean_[0]:.4f}, Std: {scaler.scale_[0]:.4f} (for first feature)")

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)
X_test_tensor = torch.tensor(X_test_combined_scaled, dtype=torch.float32).to(DEVICE)

# Create DataLoader for training
train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 4. MODEL ARCHITECTURE ---
print("\n4. Initializing model architecture...")

class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DeepAutoencoder, self).__init__()
        latent_dim = 10
        
        # Encoder (Compression)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20), nn.ReLU(),
            nn.Linear(20, 15), nn.ReLU(),
            nn.Linear(15, latent_dim), nn.ReLU()
        )
        
        # Decoder (Reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 15), nn.ReLU(),
            nn.Linear(15, 20), nn.ReLU(),
            nn.Linear(20, input_dim), nn.Identity()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = DeepAutoencoder(INPUT_DIM).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"   Model created with {total_params} trainable parameters")
print(f"   Device: {DEVICE}")

# --- 5. MODEL TRAINING ---
print("\n5. Training model...")
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"   Epochs: {N_EPOCHS}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")
print("-" * 60)

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
        print(f'   Epoch [{epoch+1:2d}/{N_EPOCHS}] - Loss: {epoch_loss:.6f}')

print("-" * 60)
print("✅ Training complete!")

# --- 6. THRESHOLD CALCULATION (CRITICAL FIX) ---
print("\n6. Calculating anomaly threshold...")
print("   ⚠️  IMPORTANT: Calculating threshold on VALIDATION SET (normal transactions only)")

model.eval()
with torch.no_grad():
    # Calculate reconstruction errors on VALIDATION SET (normal only)
    val_reconstructions = model(X_val_tensor)
    val_errors = torch.mean((val_reconstructions - X_val_tensor) ** 2, dim=1).cpu().numpy()

# Calculate threshold as 95th percentile of NORMAL validation errors
threshold = np.percentile(val_errors, 95)

print(f"\n   Validation Set Statistics (Normal Transactions):")
print(f"   ├─ Mean error:   {np.mean(val_errors):.6f}")
print(f"   ├─ Median error: {np.median(val_errors):.6f}")
print(f"   ├─ Std dev:      {np.std(val_errors):.6f}")
print(f"   ├─ Min error:    {np.min(val_errors):.6f}")
print(f"   ├─ Max error:    {np.max(val_errors):.6f}")
print(f"   └─ 95th percentile (THRESHOLD): {threshold:.6f}")

# --- 7. EVALUATION ON TEST SET ---
print("\n7. Evaluating on test set (normal + fraud)...")

with torch.no_grad():
    test_reconstructions = model(X_test_tensor)
    test_errors = torch.mean((test_reconstructions - X_test_tensor) ** 2, dim=1).cpu().numpy()

# Classify based on threshold
y_pred = (test_errors > threshold).astype(int)

# Separate errors by class for analysis
normal_mask = y_test_combined == 0
fraud_mask = y_test_combined == 1

normal_errors = test_errors[normal_mask]
fraud_errors = test_errors[fraud_mask]

print(f"\n   Test Set Reconstruction Errors:")
print(f"   Normal transactions:")
print(f"   ├─ Mean:  {np.mean(normal_errors):.6f}")
print(f"   └─ Max:   {np.max(normal_errors):.6f}")
print(f"   Fraud transactions:")
print(f"   ├─ Mean:  {np.mean(fraud_errors):.6f}")
print(f"   └─ Max:   {np.max(fraud_errors):.6f}")

# Classification metrics
print("\n" + "=" * 60)
print("EVALUATION METRICS")
print("=" * 60)
print(f"Threshold: {threshold:.6f}")
print("-" * 60)
print(classification_report(y_test_combined, y_pred, target_names=['Normal (0)', 'Fraud (1)']))

# Calculate AUC-PR
precision, recall, _ = precision_recall_curve(y_test_combined, test_errors)
auc_pr = auc(recall, precision)
print(f"Area Under Precision-Recall Curve (AUC-PR): {auc_pr:.4f}")

# Detection statistics
fraud_detected = np.sum(test_errors[fraud_mask] > threshold)
fraud_total = len(fraud_errors)
detection_rate = (fraud_detected / fraud_total) * 100

normal_flagged = np.sum(test_errors[normal_mask] > threshold)
normal_total = len(normal_errors)
false_positive_rate = (normal_flagged / normal_total) * 100

print(f"\nDetection Statistics:")
print(f"├─ Fraud detected: {fraud_detected}/{fraud_total} ({detection_rate:.2f}%)")
print(f"└─ False positives: {normal_flagged}/{normal_total} ({false_positive_rate:.2f}%)")

# --- 8. SAVE ARTIFACTS FOR DEPLOYMENT ---
print("\n" + "=" * 60)
print("SAVING DEPLOYMENT ARTIFACTS")
print("=" * 60)

# Save model weights
model_path = 'fraud_autoencoder_model.pth'
torch.save(model.state_dict(), model_path)
print(f"✅ 1/3 Model weights saved to '{model_path}'")

# Save scaler
scaler_path = 'scaler_params.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✅ 2/3 StandardScaler saved to '{scaler_path}'")

# Save configuration with metadata
config_data = {
    'anomaly_threshold': float(threshold),
    'threshold_percentile': 95,
    'validation_mean_error': float(np.mean(val_errors)),
    'validation_std_error': float(np.std(val_errors)),
    'fraud_detection_rate': float(detection_rate),
    'false_positive_rate': float(false_positive_rate),
    'auc_pr': float(auc_pr),
    'training_samples': len(X_train_normal),
    'input_dim': INPUT_DIM
}

config_path = 'config.json'
with open(config_path, 'w') as f:
    json.dump(config_data, f, indent=4)
print(f"✅ 3/3 Configuration saved to '{config_path}'")

print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE!")
print("=" * 60)
print(f"\nKey Metrics:")
print(f"├─ Threshold: {threshold:.6f}")
print(f"├─ Fraud Detection Rate: {detection_rate:.2f}%")
print(f"├─ False Positive Rate: {false_positive_rate:.2f}%")
print(f"└─ AUC-PR: {auc_pr:.4f}")

print(f"\n📦 Files ready for deployment:")
print(f"   1. {model_path}")
print(f"   2. {scaler_path}")
print(f"   3. {config_path}")

print("\n💡 Next steps:")
print("   1. Upload these 3 files to your GitHub repository")
print("   2. Streamlit Cloud will automatically redeploy your app")
print("   3. Test with the threshold: {:.6f}".format(threshold))
print("\n" + "=" * 60)
