import argparse
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import warnings
from collections import OrderedDict

# Suppress pennylane warnings for cleaner terminal output
warnings.filterwarnings("ignore")

from model import HybridAutoencoder

def load_data(bank_id):
    """Load local bank data and create PyTorch DataLoader"""
    df = pd.read_csv(f'data/processed/bank_{bank_id}_train.csv')
    
    # Features only (Unsupervised autoencoder)
    X = df.values.astype(np.float32)
    
    # Create DataLoader (input and target are the same for autoencoder)
    tensor_x = torch.from_numpy(X)
    dataset = TensorDataset(tensor_x, tensor_x)
    # Using a relatively large batch size to speed up quantum simulation
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    return dataloader, X.shape[1]

# Define Flower client
class QuantumBankClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader):
        self.model = model
        self.trainloader = trainloader
        self.criterion = nn.MSELoss()
        # Adam optimizer works well for Hybrid Quantum models
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        """Extract PyTorch/Quantum parameters to send to the Server"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Load aggregated parameters from the Server into the Local Model"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model locally"""
        print("\n--- Received Global Parameters. Starting Local Quantum Training ---")
        self.set_parameters(parameters)
        
        self.model.train()
        for epoch in range(1): # 1 local epoch per federated round to prevent Client Drift
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}/{len(self.trainloader)} Loss: {loss.item():.4f}")
                    
        avg_loss = running_loss / len(self.trainloader)
        print(f"Local Training Finished. Avg Loss: {avg_loss:.4f}\n")
        
        # Return updated parameters, dataset size, and metrics
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model locally"""
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
    parser = argparse.ArgumentParser(description="Quantum Bank Client Node")
    parser.add_argument("--bank", type=str, required=True, help="Bank ID (e.g., 70 or 10)")
    args = parser.parse_args()

    print(f"=== Initializing Bank {args.bank} Node ===")
    
    # Load data
    trainloader, num_features = load_data(args.bank)
    print(f"Data loaded. Features (Qubits): {num_features}. Total Batches: {len(trainloader)}")
    
    # Initialize model
    print("Building Hybrid Quantum Autoencoder (Simulator Backend)...")
    model = HybridAutoencoder(num_features=num_features, n_layers=1, device="default.qubit")
    
    # Start Flower client
    print("Connecting to Central Server (127.0.0.1:8081)...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=QuantumBankClient(model, trainloader),
    )

if __name__ == "__main__":
    main()
