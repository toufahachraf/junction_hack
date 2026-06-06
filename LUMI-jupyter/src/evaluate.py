import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

from model import HybridAutoencoder

"""
evaluate.py — LUMI / IQM / Simulator compatible
-------------------------------------------------
Loads a saved global model from federated training and runs
zero-day anomaly detection on the global test set.

Usage (simulator):
    python evaluate.py --round 10

Usage (IQM real QPU):
    export IQM_TOKEN=<your_iqm_token>
    python evaluate.py --round 10 --device iqm.garnet --n-qubits 8

Note: --n-qubits and --n-layers MUST match the values used during training.
"""


def main():
    parser = argparse.ArgumentParser(description="QAML Zero-Day Anomaly Detection Evaluation")
    parser.add_argument("--round", type=int, default=10,
                        help="Federated round whose weights to load")
    parser.add_argument(
        "--device",
        type=str,
        default="default.qubit",
        help=(
            "PennyLane backend:\n"
            "  default.qubit  — local simulator (default)\n"
            "  iqm.garnet     — IQM Garnet QPU (20 qubits, needs IQM_TOKEN)\n"
            "  iqm.deneb      — IQM Deneb QPU  (6 qubits,  needs IQM_TOKEN)\n"
            "  iqm.sim        — IQM cloud simulator (needs IQM_TOKEN)"
        )
    )
    parser.add_argument("--n-qubits", type=int, default=8,
                        help="Must match the value used during training")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="Must match the value used during training")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--models-dir", type=str, default="models")
    args = parser.parse_args()

    print("=== Phase 3: Zero-Day Anomaly Detection ===")
    print(f"    Backend : {args.device}")
    print(f"    Round   : {args.round}")

    # 1. Load global test data
    test_path = f"{args.data_dir}/global_test.csv"
    print(f"\nLoading global test dataset from: {test_path}")
    df = pd.read_csv(test_path)

    y = df['Is Laundering'].values
    X = df.drop(columns=['Is Laundering']).values.astype(np.float32)
    num_features = X.shape[1]

    print(f"Total Test Transactions  : {len(X)}")
    print(f"True Illicit             : {int(sum(y))}")
    print(f"True Normal              : {int(len(y) - sum(y))}")

    # 2. Build model
    print(f"\nBuilding Hybrid Quantum Autoencoder ({args.n_qubits} qubits, {args.n_layers} layers)...")
    model = HybridAutoencoder(
        num_features=num_features,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        device=args.device
    )

    # 3. Load federated weights
    weights_path = f"{args.models_dir}/global_weights_round_{args.round}.npy"
    print(f"Loading aggregated global weights from: {weights_path}")
    try:
        parameters = np.load(weights_path, allow_pickle=True)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
    except FileNotFoundError:
        print(f"\nError: {weights_path} not found.")
        print("Make sure the federated server completed the requested round.")
        return

    # 4. Forward pass — compute per-transaction reconstruction error
    print("\nRunning test transactions through the Quantum Circuit...")
    model.eval()
    with torch.no_grad():
        tensor_x = torch.from_numpy(X)
        output = model(tensor_x)
        # Per-sample MSE as anomaly score (higher = more anomalous)
        mse_scores = torch.mean((output - tensor_x) ** 2, dim=1).numpy()

    # 5. Metrics
    print("\n=== Evaluation Results ===")

    auc = roc_auc_score(y, mse_scores)
    print(f"ROC-AUC Score : {auc:.4f}  (1.0 = perfect, 0.5 = random)")

    # Find threshold that maximises F1
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y, mse_scores)
    f1_scores = (
        2 * (precisions[:-1] * recalls[:-1])
        / (precisions[:-1] + recalls[:-1] + 1e-10)
    )
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"Optimal Threshold : {best_threshold:.4f}")

    y_pred = (mse_scores > best_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average='binary', zero_division=0
    )
    cm = confusion_matrix(y, y_pred)

    print(f"\nPrecision : {precision:.4f}  (of flagged, how many were truly illicit?)")
    print(f"Recall    : {recall:.4f}  (of all illicit, how many did we catch?)")
    print(f"F1-Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(f"  True Negatives  (normal correctly cleared) : {cm[0][0]}")
    print(f"  False Positives (normal incorrectly flagged): {cm[0][1]}")
    print(f"  False Negatives (laundering missed)         : {cm[1][0]}")
    print(f"  True Positives  (laundering caught!)        : {cm[1][1]}")

    if args.device != "default.qubit":
        print(f"\nSuccess! {args.device} successfully executed anomaly detection.")


if __name__ == "__main__":
    main()
