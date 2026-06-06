import torch
import torch.nn as nn
import pennylane as qml

class HybridAutoencoder(nn.Module):
    def __init__(self, num_features, n_layers=2, device="default.qubit"):
        """
        Hybrid Quantum-Classical Autoencoder.
        
        Args:
            num_features (int): Number of input features. Since we aren't using PCA, 
                                this equals the number of qubits.
            n_layers (int): Number of layers in the StronglyEntanglingLayers PQC.
            device (str): The Pennylane backend to use (e.g., "default.qubit" for simulation,
                          "qiskit.ibmq" for real IBM hardware).
        """
        super(HybridAutoencoder, self).__init__()
        self.num_features = num_features
        self.n_layers = n_layers
        
        # Initialize quantum device
        self.dev = qml.device(device, wires=num_features)
        
        # Define the quantum circuit
        @qml.qnode(self.dev, interface="torch")
        def qcircuit(inputs, weights):
            # 1. Quantum Embedding: Encode classical data into quantum state
            qml.AngleEmbedding(inputs, wires=range(self.num_features))
            
            # 2. Parametrized Quantum Circuit (PQC)
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_features))
            
            # 3. Measurement: Return expectation value of PauliZ for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_features)]
        
        # Determine weight shape for StronglyEntanglingLayers
        # Shape is (n_layers, n_wires, 3) because each layer applies 3 rotations (RX, RY, RZ) per wire
        weight_shapes = {"weights": (n_layers, num_features, 3)}
        
        # Wrap the QNode in a PyTorch layer
        self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes)
        
        # Classical Decoder: Maps the quantum expectation values back to the original feature space
        # This solves the Barren Plateau problem and allows for simple MSE loss computation.
        self.decoder = nn.Sequential(
            nn.Linear(num_features, num_features)
        )

    def forward(self, x):
        """
        Forward pass through the Hybrid Autoencoder.
        """
        # x is of shape (batch_size, num_features)
        q_out = self.qlayer(x)
        
        # q_out is of shape (batch_size, num_features)
        reconstruction = self.decoder(q_out)
        
        return reconstruction
