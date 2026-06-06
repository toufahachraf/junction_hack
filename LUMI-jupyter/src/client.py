import argparse
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import warnings
from collections import OrderedDict

warnings.filterwarnings("ignore")

from model import HybridAutoencoder

"""
client.py — LUMI / IQM / Simulator compatible
----------------------------------------------
Run one client per bank node. On LUMI, launch each in a separate
terminal or SLURM job after the server is running.

Usage (simulator):
    python client.py --bank 70
    python client.py --bank 10

Usage (IQM real QPU — set IQM_TOKEN env var first):
    export IQM_TOKEN=<your_iqm_token>
    python client.py --bank 70 --device iqm.garnet

Usage (IQM cloud simulator):
    export IQM_TOKEN=<your_iqm_token>
    python client.py --bank 70 --device iqm.sim

Data is read from local paths (output of preprocess.py).
Set --data-dir if your processed CSVs are in a non-default location.
"""


def load_data(bank_id, data_dir="data/processed"):
    path = f"{data_dir}/bank_{bank_id}_train.csv"
    df = pd.read_csv(path)
    X = df.values.astype(np.float32)
    tensor_x = torch.from_numpy(X)
    dataset = TensorDataset(tensor_x, tensor_x)
    # Larger batch size to speed up quantum simulation
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    return dataloader, X.shape[1]


class QuantumBankClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader):
        self.model = model
        self.trainloader = trainloader
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("\n--- Received Global Parameters. Starting Local Quantum Training ---")
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # 1 local epoch per round to prevent client drift
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(self.trainloader)}  Loss: {loss.item():.4f}")
        avg_loss = running_loss / len(self.trainloader)
        print(f"Local Training Finished. Avg Loss: {avg_loss:.4f}\n")
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        with torch.no_grad():
            for data, target in self.trainloader:
                output = self.model(data)
                loss += self.criterion(output, target).item()
        avg_loss = loss / len(self.trainloader)
        return float(avg_loss), len(self.trainloader.dataset), {"mse": float(avg_loss)}


def main():
    parser = argparse.ArgumentParser(description="QAML Bank Client Node")
    parser.add_argument("--bank", type=str, required=True, help="Bank ID (e.g. 70 or 10)")
    parser.add_argument(
        "--device",
        type=str,
        default="default.qubit",
        help=(
            "PennyLane backend. Options:\n"
            "  default.qubit  — local simulator (default)\n"
            "  iqm.garnet     — IQM Garnet QPU (20 qubits, needs IQM_TOKEN)\n"
            "  iqm.deneb      — IQM Deneb QPU  (6 qubits,  needs IQM_TOKEN)\n"
            "  iqm.sim        — IQM cloud simulator (needs IQM_TOKEN)"
        )
    )
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080",
                        help="Flower server address (host:port)")
    parser.add_argument("--n-qubits", type=int, default=8,
                        help="Number of qubits. Use <=6 for Deneb, <=20 for Garnet.")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="PQC depth. Use 1 for real QPU to limit circuit depth.")
    args = parser.parse_args()

    print(f"=== Initializing Bank {args.bank} Node ===")
    print(f"    Backend : {args.device}")
    print(f"    Qubits  : {args.n_qubits}  |  Layers: {args.n_layers}")

    trainloader, num_features = load_data(args.bank, args.data_dir)
    print(f"    Features: {num_features}  |  Batches: {len(trainloader)}")

    print("Building Hybrid Quantum Autoencoder...")
    model = HybridAutoencoder(
        num_features=num_features,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        device=args.device
    )

    print(f"Connecting to Flower server at {args.server} ...")
    fl.client.start_numpy_client(
        server_address=args.server,
        client=QuantumBankClient(model, trainloader),
    )


if __name__ == "__main__":
    main()
