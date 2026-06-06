import torch
import torch.nn as nn
import pennylane as qml

class HybridAutoencoder(nn.Module):
    def __init__(self, num_features, n_qubits=8, n_layers=2, device="default.qubit", backend=None):
        """
        Hybrid Quantum-Classical Autoencoder.
        
        Args:
            num_features (int): Number of input features (e.g., 39).
            n_qubits (int): Number of qubits to simulate. This acts as the bottleneck.
            n_layers (int): Number of layers in the StronglyEntanglingLayers PQC.
            device (str): The Pennylane backend to use (e.g., "default.qubit" for simulation,
                          "qiskit.ibmq" for real IBM hardware).
        """
        super(HybridAutoencoder, self).__init__()
        self.num_features = num_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Classical Encoder: Compresses 39 features down to 8 features (or whatever n_qubits is)
        # This replaces PCA and avoids the 8 Terabyte memory crash by limiting the qubit count.
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, n_qubits),
            nn.Tanh() # Outputs values between -1 and 1
        )
        
        # Initialize quantum device
        if backend:
            self.dev = qml.device(device, wires=n_qubits, backend=backend)
        else:
            self.dev = qml.device(device, wires=n_qubits)
        
        # Define the quantum circuit
        @qml.qnode(self.dev, interface="torch")
        def qcircuit(inputs, weights):
            # 1. Quantum Embedding: Encode classical compressed data into quantum state
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            
            # 2. Parametrized Quantum Circuit (PQC)
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            
            # 3. Measurement: Return expectation value of PauliZ for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        # Determine weight shape for StronglyEntanglingLayers
        # Shape is (n_layers, n_wires, 3) because each layer applies 3 rotations (RX, RY, RZ) per wire
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        # Wrap the QNode in a PyTorch layer
        self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes)
        
        # Classical Decoder: Maps the quantum expectation values back to the 39 feature space
        # This allows for simple MSE loss computation against the original data.
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, num_features)
        )

    def forward(self, x):
        """
        Forward pass through the Hybrid Autoencoder.
        """
        # 1. Classical Compression (batch_size, num_features) -> (batch_size, n_qubits)
        encoded = self.encoder(x)
        
        # Scale the outputs from [-1, 1] to [-pi, pi] to act as angles for the embedding
        encoded_scaled = encoded * torch.pi
        
        # 2. Quantum Bottleneck (batch_size, n_qubits) -> (batch_size, n_qubits)
        q_out = self.qlayer(encoded_scaled)
        
        # 3. Classical Reconstruction (batch_size, n_qubits) -> (batch_size, num_features)
        reconstruction = self.decoder(q_out)
        
        return reconstruction
