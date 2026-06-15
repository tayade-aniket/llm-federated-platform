# server/federated_server.py
"""
Federated Learning Server using Flower Framework
Coordinates distributed training across multiple clients
"""

import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Callable, Union
from flwr.common import (
    Parameters, 
    Scalar, 
    NDArrays, 
    FitRes,
    EvaluateRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedAdam, FedYogi
import json
import os
import yaml
from pathlib import Path
from datetime import datetime
import hashlib
import pickle

class PersonalizedFedAvg(FedAvg):
    """Extended Federated Averaging with personalization support"""
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable[[int, NDArrays, Dict[str, Scalar]], 
                                       Optional[Tuple[float, Dict[str, Scalar]]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[Callable[[List[Tuple[int, Dict[str, Scalar]]]], 
                                                      Dict[str, Scalar]]] = None,
        evaluate_metrics_aggregation_fn: Optional[Callable[[List[Tuple[int, Dict[str, Scalar]]]], 
                                                           Dict[str, Scalar]]] = None,
        server_learning_rate: float = 1.0,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.server_learning_rate = server_learning_rate
        self.round_history = []
        self.client_contributions = {}
        
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average"""
        
        if not results:
            return None, {}
        
        # Extract weights from results
        weights_results = []
        for client, fit_res in results:
            # Convert parameters to numpy arrays
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            
            weights_results.append((client_weights, num_examples))
            
            # Track client contributions
            client_id = str(client.cid) if hasattr(client, 'cid') else "unknown"
            if client_id not in self.client_contributions:
                self.client_contributions[client_id] = []
            self.client_contributions[client_id].append({
                'round': rnd,
                'num_examples': num_examples,
                'timestamp': datetime.now().isoformat()
            })
        
        # Perform aggregation
        aggregated_weights = self._aggregate(weights_results)
        
        # Apply server learning rate
        if aggregated_weights is not None and self.server_learning_rate != 1.0:
            aggregated_weights = [w * self.server_learning_rate for w in aggregated_weights]
        
        # Convert back to Parameters
        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)
        
        # Save round history
        self.round_history.append({
            'round': rnd,
            'num_clients': len(results),
            'total_examples': sum(num for _, num in weights_results),
            'timestamp': datetime.now().isoformat()
        })
        
        # Save checkpoint
        self._save_checkpoint(rnd, aggregated_weights)
        
        return parameters_aggregated, {}
    
    def _aggregate(self, weights_results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute weighted average"""
        if not weights_results:
            return None
        
        # Sum weights
        total_weight = sum(num_examples for _, num_examples in weights_results)
        
        # Initialize aggregated weights with zeros
        aggregated_weights = [
            np.zeros_like(weights) for weights in weights_results[0][0]
        ]
        
        if total_weight == 0:
            # Fall back to uniform average if total_weight is zero
            num_clients = len(weights_results)
            for client_weights, _ in weights_results:
                for i, layer_weights in enumerate(client_weights):
                    aggregated_weights[i] += (layer_weights / num_clients)
            return aggregated_weights
            
        # Weighted sum
        for client_weights, num_examples in weights_results:
            for i, layer_weights in enumerate(client_weights):
                aggregated_weights[i] += (layer_weights * (num_examples / total_weight))
        
        return aggregated_weights
    
    def _save_checkpoint(self, round_num: int, weights: NDArrays):
        """Save model checkpoint"""
        checkpoint_dir = Path("models/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"round_{round_num}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'round': round_num,
                'weights': weights,
                'timestamp': datetime.now().isoformat(),
                'metadata': self.round_history[-1] if self.round_history else {}
            }, f)
        
        print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    def save_metrics(self):
        """Save training metrics to file"""
        metrics = {
            'round_history': self.round_history,
            'client_contributions': self.client_contributions,
            'total_rounds': len(self.round_history),
            'total_clients': len(self.client_contributions)
        }
        
        metrics_dir = Path("models/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_path = metrics_dir / f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"📊 Saved metrics: {metrics_path}")

class FederatedServer:
    """Main federated learning server coordinator"""
    
    def __init__(self, config_path: str = "server/config.yaml"):
        self.config = self._load_config(config_path)
        self.strategy = None
        self.client_manager = None
        
    def _load_config(self, config_path: str) -> dict:
        """Load server configuration"""
        config = {}
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Ensure default structure is present
        if 'server' not in config:
            config['server'] = {}
        if 'aggregation' not in config:
            config['aggregation'] = {}
        if 'privacy' not in config:
            config['privacy'] = {}

        # Set default values if not present
        config['server'].setdefault('host', '0.0.0.0')
        config['server'].setdefault('port', 8080)
        config['server'].setdefault('num_rounds', 5)
        config['server'].setdefault('min_fit_clients', 1)
        config['server'].setdefault('min_available_clients', 1)
        config['server'].setdefault('min_evaluate_clients', 1)
        config['server'].setdefault('fraction_evaluate', 0.0) # Default to 0.0 to disable client evaluation
        
        config['aggregation'].setdefault('strategy', 'fedavg')
        config['aggregation'].setdefault('fraction_fit', 1.0)
        config['aggregation'].setdefault('server_learning_rate', 1.0)
        
        config['privacy'].setdefault('differential_privacy', False)
        config['privacy'].setdefault('clip_norm', 1.0)
        config['privacy'].setdefault('noise_multiplier', 1.0)

        # Environment variable overrides
        env_host = os.environ.get("FL_SERVER_HOST") or os.environ.get("HOST")
        if env_host:
            config['server']['host'] = env_host
            
        env_port = os.environ.get("FL_SERVER_PORT") or os.environ.get("PORT")
        if env_port:
            try:
                config['server']['port'] = int(env_port)
            except ValueError:
                pass
                
        env_rounds = os.environ.get("FL_NUM_ROUNDS")
        if env_rounds:
            try:
                config['server']['num_rounds'] = int(env_rounds)
            except ValueError:
                pass
                
        env_min_fit = os.environ.get("FL_MIN_FIT_CLIENTS")
        if env_min_fit:
            try:
                config['server']['min_fit_clients'] = int(env_min_fit)
            except ValueError:
                pass
                
        env_min_avail = os.environ.get("FL_MIN_AVAILABLE_CLIENTS")
        if env_min_avail:
            try:
                config['server']['min_available_clients'] = int(env_min_avail)
            except ValueError:
                pass

        env_min_eval = os.environ.get("FL_MIN_EVALUATE_CLIENTS")
        if env_min_eval:
            try:
                config['server']['min_evaluate_clients'] = int(env_min_eval)
            except ValueError:
                pass
                
        env_frac_eval = os.environ.get("FL_FRACTION_EVALUATE")
        if env_frac_eval:
            try:
                config['server']['fraction_evaluate'] = float(env_frac_eval)
            except ValueError:
                pass

        # Enforce strict types to prevent configuration parsing type errors
        try:
            config['server']['port'] = int(config['server']['port'])
            config['server']['num_rounds'] = int(config['server']['num_rounds'])
            config['server']['min_fit_clients'] = int(config['server']['min_fit_clients'])
            config['server']['min_available_clients'] = int(config['server']['min_available_clients'])
            config['server']['min_evaluate_clients'] = int(config['server']['min_evaluate_clients'])
            config['server']['fraction_evaluate'] = float(config['server']['fraction_evaluate'])
        except Exception:
            pass

        return config
    
    def _create_strategy(self, initial_parameters: Optional[Parameters] = None):
        """Create federated learning strategy"""
        strategy_config = self.config['aggregation']
        
        if strategy_config['strategy'].lower() == 'fedavg':
            strategy = PersonalizedFedAvg(
                fraction_fit=strategy_config.get('fraction_fit', 1.0),
                fraction_evaluate=self.config['server'].get('fraction_evaluate', 0.0),
                min_fit_clients=self.config['server']['min_fit_clients'],
                min_evaluate_clients=self.config['server']['min_evaluate_clients'],
                min_available_clients=self.config['server']['min_available_clients'],
                server_learning_rate=strategy_config.get('server_learning_rate', 1.0),
                initial_parameters=initial_parameters
            )
        elif strategy_config['strategy'].lower() == 'fedadam':
            strategy = FedAdam(
                fraction_fit=strategy_config.get('fraction_fit', 1.0),
                fraction_evaluate=self.config['server'].get('fraction_evaluate', 0.0),
                min_fit_clients=self.config['server']['min_fit_clients'],
                min_evaluate_clients=self.config['server']['min_evaluate_clients'],
                min_available_clients=self.config['server']['min_available_clients'],
                initial_parameters=initial_parameters
            )
        else:
            strategy = PersonalizedFedAvg(
                fraction_fit=strategy_config.get('fraction_fit', 1.0),
                fraction_evaluate=self.config['server'].get('fraction_evaluate', 0.0),
                min_fit_clients=self.config['server']['min_fit_clients'],
                min_evaluate_clients=self.config['server']['min_evaluate_clients'],
                min_available_clients=self.config['server']['min_available_clients'],
                initial_parameters=initial_parameters
            )
        
        return strategy
    
    def start(self):
        """Start the federated learning server"""
        print("🚀 Starting Federated Learning Server")
        print(f"📡 Listening on {self.config['server']['host']}:{self.config['server']['port']}")
        print(f"🔄 Number of rounds: {self.config['server']['num_rounds']}")
        print(f"⚙️ Aggregation strategy: {self.config['aggregation']['strategy']}")
        
        # Create strategy
        self.strategy = self._create_strategy()
        
        # Start server
        fl.server.start_server(
            server_address=f"{self.config['server']['host']}:{self.config['server']['port']}",
            config=fl.server.ServerConfig(num_rounds=self.config['server']['num_rounds']),
            strategy=self.strategy
        )
        
        # Save final metrics
        if hasattr(self.strategy, 'save_metrics'):
            self.strategy.save_metrics()
        
        print("✅ Federated Learning completed")

class FederatedClient(fl.client.NumPyClient):
    """Client implementation for federated learning"""
    
    def __init__(self, model, train_data, val_data):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        
    def fit(self, parameters, config):
        """Train model on client data"""
        # Set model parameters
        self.model.set_weights(parameters)
        
        # Train locally
        history = self.model.fit(
            self.train_data,
            epochs=config.get('local_epochs', 1),
            batch_size=config.get('batch_size', 32),
            verbose=0
        )
        
        # Return updated parameters and number of examples
        return self.model.get_weights(), len(self.train_data), {}
    
    def evaluate(self, parameters, config):
        """Evaluate model on client data"""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.val_data, verbose=0)
        return loss, len(self.val_data), {"accuracy": accuracy}

def start_server():
    """Entry point for starting the server"""
    server = FederatedServer()
    server.start()

def load_checkpoint(checkpoint_path: str):
    """Load a saved checkpoint"""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    print(f"Loaded checkpoint from round {checkpoint['round']}")
    return checkpoint

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--config", type=str, default="server/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        print("Resume functionality would continue training from checkpoint")
    
    # Start server
    start_server()