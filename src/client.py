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

def load_data(bank_id):
    df = pd.read_csv(f'data/processed/bank_{bank_id}_train.csv')
    X = df.values.astype(np.float32)
    
    tensor_x = torch.from_numpy(X)
    dataset = TensorDataset(tensor_x, tensor_x)
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
        self.set_parameters(parameters)
        self.model.train()
        
        for epoch in range(1):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                    
        avg_loss = running_loss / len(self.trainloader)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=str, required=True)
    args = parser.parse_args()

    trainloader, num_features = load_data(args.bank)
    model = HybridAutoencoder(num_features=num_features, n_layers=1, device="default.qubit")
    
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=QuantumBankClient(model, trainloader),
    )

if __name__ == "__main__":
    main()
