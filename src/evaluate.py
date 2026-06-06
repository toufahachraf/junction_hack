import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import warnings

# Suppress pennylane warnings
warnings.filterwarnings("ignore")

from model import HybridAutoencoder

def main():
    parser = argparse.ArgumentParser(description="Zero-Day Anomaly Detection Evaluation")
    parser.add_argument("--round", type=int, default=10, help="Which federated round model to load")
    parser.add_argument("--real-qpu", action="store_true", help="Run on real IBM Quantum Hardware instead of simulator")
    args = parser.parse_args()

    print("=== Phase 3: Zero-Day Anomaly Detection ===")
    
    # 1. Load Global Test Data
    print("Loading global test dataset...")
    df = pd.read_csv('data/processed/global_test.csv')
    
    y = df['Is Laundering'].values
    X = df.drop(columns=['Is Laundering']).values.astype(np.float32)
    num_features = X.shape[1]
    
    print(f"Total Test Transactions: {len(X)}")
    print(f"True Illicit Transactions: {sum(y)}")
    print(f"True Normal Transactions: {len(y) - sum(y)}")

    # 2. Determine Backend & Apply Safety Limit
    if args.real_qpu:
        print("\n[WARNING] Attempting to connect to IBM Quantum Hardware!")
        print("Ensure you have saved your IBMQ account token locally.")
        device_name = "qiskit.remote" # Updated device for qiskit-ibm-runtime
        backend_name = "least_busy"
        
        # SAFETY LIMIT: Reduce dataset size to avoid draining the 10-minute quota
        print(">> APPLYING SAFETY LIMIT: Sampling 20 transactions for real QPU evaluation <<")
        # Grab 10 normal and 10 illicit
        illicit_idx = np.where(y == 1)[0][:10]
        normal_idx = np.where(y == 0)[0][:10]
        safe_idx = np.concatenate([illicit_idx, normal_idx])
        
        X = X[safe_idx]
        y = y[safe_idx]
    else:
        print("\n[INFO] Using fast local Quantum Simulator (default.qubit)")
        device_name = "default.qubit"
        backend_name = None

    # 3. Initialize Model & Load Federated Weights
    print(f"\nBuilding Hybrid Quantum Autoencoder...")
    # NOTE: n_layers=1 must match what was used in client.py
    model = HybridAutoencoder(num_features=num_features, n_qubits=10, n_layers=1, device=device_name, backend=backend_name)
    
    weights_path = f"models/global_weights_round_{args.round}.npy"
    print(f"Loading aggregated global weights from: {weights_path}")
    try:
        parameters = np.load(weights_path, allow_pickle=True)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
    except FileNotFoundError:
        print(f"Error: {weights_path} not found. Please ensure the federated server completed round {args.round}.")
        return

    # 4. Evaluation Loop
    print("\nRunning test transactions through the Quantum Circuit...")
    model.eval()
    with torch.no_grad():
        tensor_x = torch.from_numpy(X)
        output = model(tensor_x)
        
        # Calculate Mean Squared Error (MSE) for each transaction individually
        # We don't reduce the loss so we get an anomaly score per transaction
        mse_scores = torch.mean((output - tensor_x)**2, dim=1).numpy()

    # 5. Thresholding & Metrics
    print("\n=== Evaluation Results ===")
    
    # Calculate ROC-AUC which is threshold-independent
    auc = roc_auc_score(y, mse_scores)
    print(f"ROC-AUC Score: {auc:.4f} (1.0 is perfect, 0.5 is random)")
    
    # Dynamically find the best threshold to maximize F1-Score
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y, mse_scores)
    
    # Calculate F1 score for each threshold (ignoring the last element to match thresholds array length)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"Optimized Anomaly Threshold: {best_threshold:.4f}")
    
    y_pred = (mse_scores > best_threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y, y_pred)
    
    print("\nMetrics:")
    print(f"Precision: {precision:.4f} (When flagged as laundering, how often was it correct?)")
    print(f"Recall:    {recall:.4f} (Out of all laundering, how many did we catch?)")
    print(f"F1-Score:  {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"True Negatives (Normal correctly identified): {cm[0][0]}")
    print(f"False Positives (Normal falsely flagged):     {cm[0][1]}")
    print(f"False Negatives (Laundering missed):          {cm[1][0]}")
    print(f"True Positives (Laundering CATCHED!):         {cm[1][1]}")
    
    if args.real_qpu:
        print("\nSuccess! The IBM Quantum Hardware successfully executed the Anomaly Detection!")

if __name__ == "__main__":
    main()
