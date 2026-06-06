import flwr as fl
import os
import numpy as np
import argparse

"""
server.py — LUMI / Local compatible
-------------------------------------
The Flower server is backend-agnostic — it only aggregates numpy arrays
and never touches quantum hardware. No changes needed from the GCP version
other than removing cloud dependencies.

Usage:
    python server.py
    python server.py --rounds 10 --port 8080

On LUMI, run the server on a login or compute node, then point each
client.py at the server's hostname:port with --server <hostname>:8080.
"""


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """FedAvg with per-round weight saving to local disk."""

    def __init__(self, save_dir="models", **kwargs):
        super().__init__(**kwargs)
        self.save_dir = save_dir

    def aggregate_fit(self, server_round, results, failures):
        aggregated_weights, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_weights is not None:
            weights = fl.common.parameters_to_ndarrays(aggregated_weights)
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = f"{self.save_dir}/global_weights_round_{server_round}.npy"
            np.save(save_path, np.array(weights, dtype=object), allow_pickle=True)
            print(f"  Round {server_round} weights saved → {save_path}")
        return aggregated_weights, aggregated_metrics


def main():
    parser = argparse.ArgumentParser(description="QAML Flower Server")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--min-clients", type=int, default=2,
                        help="Minimum clients before training starts")
    parser.add_argument("--save-dir", type=str, default="models")
    args = parser.parse_args()

    print("Starting Flower Central Server for Quantum Federated Learning...")
    print(f"  Rounds      : {args.rounds}")
    print(f"  Min clients : {args.min_clients}")
    print(f"  Save dir    : {args.save_dir}/")

    strategy = SaveModelStrategy(
        save_dir=args.save_dir,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
    )

    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
