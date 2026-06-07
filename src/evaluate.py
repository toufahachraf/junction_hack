import argparse
import numpy as np
import os
import pandas as pd
import torch
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

from model import HybridAutoencoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=10)
    parser.add_argument("--real-qpu", action="store_true")
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'processed', 'global_test.csv')
    df = pd.read_csv(data_path)
    
    y = df['Is Laundering'].values
    X = df.drop(columns=['Is Laundering']).values.astype(np.float32)
    num_features = X.shape[1]

    if args.real_qpu:
        device_name = "qiskit.remote" 
        from qiskit_ibm_runtime import QiskitRuntimeService
        
        if args.token:
            service = QiskitRuntimeService(token=args.token)
        else:
            service = QiskitRuntimeService()
            
        backend_name = service.least_busy(simulator=False, operational=True, min_num_qubits=10)
        
        illicit_idx = np.where(y == 1)[0][:15]
        normal_idx = np.where(y == 0)[0][:15]
        safe_idx = np.concatenate([illicit_idx, normal_idx])
        
        X = X[safe_idx]
        y = y[safe_idx]
    else:
        device_name = "default.qubit"
        backend_name = None

    model = HybridAutoencoder(num_features=num_features, n_qubits=10, n_layers=1, device=device_name, backend=backend_name)
    
    weights_path = os.path.join(script_dir, 'models', f'global_weights_round_{args.round}.npy')
    try:
        parameters = np.load(weights_path, allow_pickle=True)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
    except FileNotFoundError:
        return

    model.eval()
    with torch.no_grad():
        tensor_x = torch.from_numpy(X)
        output = model(tensor_x)
        mse_scores = torch.mean((output - tensor_x)**2, dim=1).numpy()

    auc = roc_auc_score(y, mse_scores)
    
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y, mse_scores)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    y_pred = (mse_scores > best_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y, y_pred)
    
    metrics_str = "\nMetrics:\n"
    metrics_str += f"Precision: {precision:.4f}\n"
    metrics_str += f"Recall:    {recall:.4f}\n"
    metrics_str += f"F1-Score:  {f1:.4f}\n"
    metrics_str += "\nConfusion Matrix:\n"
    metrics_str += f"True Negatives:  {cm[0][0]}\n"
    metrics_str += f"False Positives: {cm[0][1]}\n"
    metrics_str += f"False Negatives: {cm[1][0]}\n"
    metrics_str += f"True Positives:  {cm[1][1]}\n"
    
    print(metrics_str)
        
    with open(os.path.join(script_dir, "results.txt"), "w") as f:
        f.write(f"ROC-AUC Score: {auc:.4f}\n")
        f.write(f"Optimized Anomaly Threshold: {best_threshold:.4f}\n")
        f.write(metrics_str)

if __name__ == "__main__":
    main()
