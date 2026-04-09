
# train_model.py
import argparse
import io
import json
import logging
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture  (must match app.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
class DeepAutoencoder(nn.Module):
    """
    Symmetric deep autoencoder for unsupervised anomaly detection.

    Architecture
    ────────────
    Encoder: input_dim → 20 → 15 → 10  (ReLU activations)
    Decoder: 10 → 15 → 20 → input_dim  (ReLU + Identity)

    Anomaly score = per-sample mean squared reconstruction error.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        latent_dim = 10

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20), nn.ReLU(),
            nn.Linear(20, 15),        nn.ReLU(),
            nn.Linear(15, latent_dim),nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 15), nn.ReLU(),
            nn.Linear(15, 20),         nn.ReLU(),
            nn.Linear(20, input_dim),  nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    log.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path)
    log.info("Shape: %s  |  Fraud rate: %.4f%%",
             df.shape, df["Class"].mean() * 100)
    assert "Class" in df.columns, "Dataset must contain a 'Class' column (0=normal, 1=fraud)."
    return df


def split_and_scale(df: pd.DataFrame, random_seed: int):
    """
    UAD constraint: training set contains ONLY normal transactions.
    Test set retains the original class imbalance.

    Returns
    -------
    X_train_scaled, X_test_scaled, y_test, scaler, feature_names
    """
    feature_cols = [c for c in df.columns if c != "Class"]
    normal_df = df[df["Class"] == 0]
    fraud_df  = df[df["Class"] == 1]

    # 80 % of normal transactions → train; 10 % → test (other 10 % discarded)
    X_train_normal, X_temp = train_test_split(
        normal_df[feature_cols], test_size=0.2, random_state=random_seed
    )
    X_test_normal, _ = train_test_split(
        X_temp, test_size=0.5, random_state=random_seed
    )

    # Test set = normal slice + all fraud, shuffled
    X_test = pd.concat([X_test_normal, fraud_df[feature_cols]]).sample(
        frac=1, random_state=random_seed
    )
    y_test = pd.concat([
        pd.Series([0] * len(X_test_normal)),
        pd.Series([1] * len(fraud_df)),
    ]).sample(frac=1, random_state=random_seed).values

    # Fit scaler on training normals only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled  = scaler.transform(X_test)

    log.info("Train size (normal only): %d  |  Test size: %d  (fraud: %d)",
             len(X_train_scaled), len(X_test_scaled), int(y_test.sum()))

    return X_train_scaled, X_test_scaled, y_test, scaler, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Autoencoder training
# ─────────────────────────────────────────────────────────────────────────────
def train_autoencoder(
    X_train: np.ndarray,
    *,
    input_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    random_seed: int,
) -> tuple[DeepAutoencoder, list[float]]:

    torch.manual_seed(random_seed)
    model = DeepAutoencoder(input_dim).to(device)

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    dataset  = TensorDataset(X_tensor, X_tensor)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history   = []

    log.info("Training Autoencoder for %d epochs on %s …", epochs, device)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(X_tensor)
        history.append(epoch_loss)
        if epoch % 10 == 0:
            log.info("  Epoch [%d/%d]  loss: %.6f", epoch, epochs, epoch_loss)

    log.info("Autoencoder training complete.")
    return model, history


def evaluate_autoencoder(
    model: DeepAutoencoder,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Returns: mse_errors, y_pred, threshold, auc_pr
    Threshold = 95th percentile of reconstruction errors on the test set.
    """
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon  = model(X_tensor)
        errors = torch.mean((recon - X_tensor) ** 2, dim=1).cpu().numpy()

    threshold = float(np.percentile(errors, 95))
    y_pred    = (errors > threshold).astype(int)

    precision, recall, _ = precision_recall_curve(y_test, errors)
    auc_pr = auc(recall, precision)

    log.info("Autoencoder  →  threshold: %.5f  |  AUC-PR: %.4f", threshold, auc_pr)
    log.info("\n%s", classification_report(y_test, y_pred,
             target_names=["Normal (0)", "Fraud (1)"]))

    return errors, y_pred, threshold, auc_pr


# ─────────────────────────────────────────────────────────────────────────────
# Isolation Forest training
# ─────────────────────────────────────────────────────────────────────────────
def train_isolation_forest(
    X_train: np.ndarray,
    *,
    n_estimators: int = 100,
    contamination: str | float = "auto",
    random_seed: int = 42,
) -> IsolationForest:

    log.info("Training Isolation Forest (%d trees, contamination=%s) …",
             n_estimators, contamination)
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_seed,
        n_jobs=-1,
    )
    iso.fit(X_train)
    log.info("Isolation Forest training complete.")
    return iso


def evaluate_isolation_forest(
    iso: IsolationForest,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns: if_scores, if_y_pred, auc_pr
    """
    if_scores   = -iso.score_samples(X_test)
    if_raw_pred = iso.predict(X_test)
    if_y_pred   = np.where(if_raw_pred == -1, 1, 0)

    precision, recall, _ = precision_recall_curve(y_test, if_scores)
    auc_pr = auc(recall, precision)

    log.info("Isolation Forest  →  AUC-PR: %.4f", auc_pr)
    log.info("\n%s", classification_report(y_test, if_y_pred,
             target_names=["Normal (0)", "Fraud (1)"]))

    return if_scores, if_y_pred, auc_pr


# ─────────────────────────────────────────────────────────────────────────────
# Packaging
# ─────────────────────────────────────────────────────────────────────────────
def package_best_model(
    *,
    best_name: str,
    model_obj,               # DeepAutoencoder | IsolationForest
    scaler: StandardScaler,
    feature_names: list[str],
    input_dim: int,
    auc_pr: float,
    threshold: float | None,
    output_path: str,
) -> None:
    """
    Write a self-contained ZIP bundle:
      manifest.json   – metadata for inference
      scaler.pkl      – fitted StandardScaler
      model.pth       – Autoencoder state_dict  (if AE wins)
      model.pkl       – IsolationForest object  (if IF wins)
    """
    manifest = {
        "model_type"   : best_name,
        "input_dim"    : int(input_dim),
        "feature_names": list(feature_names),
        "auc_pr"       : round(auc_pr, 6),
        "threshold"    : float(threshold) if threshold is not None else None,
    }

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        scaler_buf = io.BytesIO()
        pickle.dump(scaler, scaler_buf)
        zf.writestr("scaler.pkl", scaler_buf.getvalue())

        if best_name == "Autoencoder":
            model_buf = io.BytesIO()
            torch.save(model_obj.state_dict(), model_buf)
            zf.writestr("model.pth", model_buf.getvalue())
        else:
            if_buf = io.BytesIO()
            pickle.dump(model_obj, if_buf)
            zf.writestr("model.pkl", if_buf.getvalue())

    size_kb = os.path.getsize(output_path) / 1024
    log.info("✅  Package saved → %s  (%.1f KB)", output_path, size_kb)
    log.info("   Model type : %s", best_name)
    log.info("   AUC-PR     : %.6f", auc_pr)
    if threshold is not None:
        log.info("   Threshold  : %.6f", threshold)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Autoencoder + Isolation Forest for fraud detection "
                    "and package the best model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",            default="creditcard.csv",   help="Path to CSV dataset")
    p.add_argument("--output",          default="fraud_detector.zip", help="Output ZIP path")
    p.add_argument("--epochs",    type=int,   default=50,     help="Autoencoder training epochs")
    p.add_argument("--batch-size", type=int,  default=128,    help="Autoencoder batch size")
    p.add_argument("--lr",         type=float, default=1e-3,  help="Autoencoder learning rate")
    p.add_argument("--if-trees",   type=int,  default=100,    help="Isolation Forest n_estimators")
    p.add_argument("--if-contamination", default="auto",
                   help="Isolation Forest contamination  (float or 'auto')")
    p.add_argument("--seed",       type=int,  default=42,     help="Random seed")
    p.add_argument("--saved-models-dir", default="saved_models",
                   help="Directory for individual model artefacts")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = load_data(args.data)

    # ── 2. Split & scale ──────────────────────────────────────────────────────
    X_train, X_test, y_test, scaler, feature_names = split_and_scale(
        df, random_seed=args.seed
    )
    input_dim = X_train.shape[1]

    # ── 3. Autoencoder ────────────────────────────────────────────────────────
    ae_model, ae_history = train_autoencoder(
        X_train,
        input_dim=input_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        random_seed=args.seed,
    )
    ae_errors, ae_y_pred, ae_threshold, ae_auc_pr = evaluate_autoencoder(
        ae_model, X_test, y_test, device
    )

    # ── 4. Isolation Forest ───────────────────────────────────────────────────
    contamination = args.if_contamination
    if contamination != "auto":
        contamination = float(contamination)

    iso_model = train_isolation_forest(
        X_train,
        n_estimators=args.if_trees,
        contamination=contamination,
        random_seed=args.seed,
    )
    if_scores, if_y_pred, if_auc_pr = evaluate_isolation_forest(
        iso_model, X_test, y_test
    )

    # ── 5. Pick winner ────────────────────────────────────────────────────────
    log.info("─" * 50)
    log.info("AUC-PR summary")
    log.info("  Autoencoder      : %.4f", ae_auc_pr)
    log.info("  Isolation Forest : %.4f", if_auc_pr)

    if ae_auc_pr >= if_auc_pr:
        best_name  = "Autoencoder"
        best_obj   = ae_model
        best_auc   = ae_auc_pr
        best_thr   = ae_threshold
    else:
        best_name  = "Isolation Forest"
        best_obj   = iso_model
        best_auc   = if_auc_pr
        best_thr   = None

    log.info("🏆  Winner: %s", best_name)
    log.info("─" * 50)

    # ── 6. Save individual artefacts ──────────────────────────────────────────
    os.makedirs(args.saved_models_dir, exist_ok=True)

    # Always save the scaler (needed by both models at inference time)
    with open(f"{args.saved_models_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Autoencoder
    torch.save(ae_model.state_dict(),
               f"{args.saved_models_dir}/autoencoder.pth")
    with open(f"{args.saved_models_dir}/autoencoder_threshold.pkl", "wb") as f:
        pickle.dump(ae_threshold, f)

    # Isolation Forest
    with open(f"{args.saved_models_dir}/isolation_forest.pkl", "wb") as f:
        pickle.dump(iso_model, f)

    log.info("Individual artefacts saved to '%s/'", args.saved_models_dir)

    # ── 7. Package best model ─────────────────────────────────────────────────
    package_best_model(
        best_name=best_name,
        model_obj=best_obj,
        scaler=scaler,
        feature_names=feature_names,
        input_dim=input_dim,
        auc_pr=best_auc,
        threshold=best_thr,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
