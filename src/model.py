import torch
import torch.nn as nn
import pennylane as qml

class HybridAutoencoder(nn.Module):
    def __init__(self, num_features, n_qubits=10, n_layers=1, device="default.qubit", backend=None):
        super(HybridAutoencoder, self).__init__()
        self.num_features = num_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, n_qubits),
            nn.Tanh() 
        )
        
        if backend:
            self.dev = qml.device(device, wires=n_qubits, backend=backend)
        else:
            self.dev = qml.device(device, wires=n_qubits)
        
        @qml.qnode(self.dev, interface="torch")
        def qcircuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes)
        
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, num_features)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        encoded_scaled = encoded * torch.pi
        q_out = self.qlayer(encoded_scaled)
        reconstruction = self.decoder(q_out)
        
        return reconstruction
