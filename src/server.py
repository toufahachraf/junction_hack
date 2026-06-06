import flwr as fl
import os
import numpy as np

# Custom Strategy to save the global model weights after each round
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Call the original FedAvg aggregate_fit
        aggregated_weights, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_weights is not None:
            # Convert `Parameters` to list of numpy arrays
            weights = fl.common.parameters_to_ndarrays(aggregated_weights)
            
            # Save aggregated weights locally
            print(f"Saving round {server_round} aggregated weights...")
            os.makedirs("models", exist_ok=True)
            # Save as numpy object
            np.save(f"models/global_weights_round_{server_round}.npy", np.array(weights, dtype=object), allow_pickle=True)
            
        return aggregated_weights, aggregated_metrics

def main():
    print("Starting Flower Central Server for Quantum Federated Learning...")
    
    # Define strategy
    strategy = SaveModelStrategy(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=2,  # Never start training without 2 clients
        min_evaluate_clients=2,
        min_available_clients=2,  # Wait until 2 clients are connected
    )

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )
    
if __name__ == "__main__":
    main()
