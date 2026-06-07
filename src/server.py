import flwr as fl
import os
import numpy as np

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_weights, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_weights is not None:
            weights = fl.common.parameters_to_ndarrays(aggregated_weights)
            
            os.makedirs("models", exist_ok=True)
            np.save(f"models/global_weights_round_{server_round}.npy", np.array(weights, dtype=object), allow_pickle=True)
            
        return aggregated_weights, aggregated_metrics

def main():
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )
    
if __name__ == "__main__":
    main()
