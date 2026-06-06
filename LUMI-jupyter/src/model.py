import torch
import torch.nn as nn
import pennylane as qml

"""
model.py — LUMI / IQM / Simulator compatible
---------------------------------------------
Backend options (pass as `device` argument):
  "default.qubit"     — Fast local simulator (default, no credentials needed)
  "iqm.garnet"        — IQM Garnet QPU (20 qubits, requires IQM credentials)
  "iqm.deneb"         — IQM Deneb QPU (6 qubits, requires IQM credentials)
  "iqm.sim"           — IQM cloud simulator (requires IQM credentials)

IQM credentials:
  Set the environment variable IQM_TOKEN before running:
    export IQM_TOKEN=<your_token>
  Or pass tokens_file= to qml.device() — see pennylane-iqm docs.

Note: IQM devices have a star-topology connectivity constraint.
StronglyEntanglingLayers uses CNOT gates which may be decomposed
automatically by PennyLane, but expect higher gate counts on real QPU.
For real QPU runs, prefer n_qubits <= 6 (Deneb) or <= 20 (Garnet)
and n_layers=1 to limit circuit depth.
"""


class HybridAutoencoder(nn.Module):
    def __init__(self, num_features, n_qubits=8, n_layers=1, device="default.qubit"):
        """
        Hybrid Quantum-Classical Autoencoder.

        Args:
            num_features (int): Number of input features (determined by preprocessing).
            n_qubits (int): Number of qubits — acts as the bottleneck dimension.
                            Keep <= 6 for IQM Deneb, <= 20 for IQM Garnet.
            n_layers (int): Depth of StronglyEntanglingLayers PQC.
                            Use 1 for real QPU runs to limit circuit depth.
            device (str): PennyLane backend string. See header for options.
        """
        super(HybridAutoencoder, self).__init__()
        self.num_features = num_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical Encoder: compresses num_features → n_qubits
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, n_qubits),
            nn.Tanh()  # Output in [-1, 1] → scaled to [-pi, pi] before embedding
        )

        # Initialise the PennyLane device
        # For IQM backends, pennylane-iqm reads IQM_TOKEN from the environment.
        # Example for Garnet:
        #   dev = qml.device("iqm.garnet", wires=n_qubits)
        # Example for local sim:
        #   dev = qml.device("default.qubit", wires=n_qubits)
        self.dev = qml.device(device, wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def qcircuit(inputs, weights):
            # 1. Encode compressed classical vector as rotation angles
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            # 2. Parametrized Quantum Circuit
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            # 3. Measure expectation value of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes)

        # Classical Decoder: maps quantum output back to feature space
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, num_features)
        )

    def forward(self, x):
        # 1. Classical compression
        encoded = self.encoder(x)
        # Scale [-1, 1] → [-pi, pi] for AngleEmbedding
        encoded_scaled = encoded * torch.pi
        # 2. Quantum bottleneck
        q_out = self.qlayer(encoded_scaled)
        # 3. Classical reconstruction
        reconstruction = self.decoder(q_out)
        return reconstruction
