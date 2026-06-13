# client/flower_client.py
"""
Flower Federated Learning Client for PyTorch/HuggingFace PEFT Models.
Connects the local trainer to the federated server.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["USE_TORCH"] = "1"

import sys
from pathlib import Path

# Add project root directory to sys.path to allow absolute imports
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import flwr as fl
import argparse
import os
import json
from client.local_trainer import LocalTrainer

class PyTorchFlowerClient(fl.client.NumPyClient):
    def __init__(self, trainer: LocalTrainer, data_path: str):
        self.trainer = trainer
        self.data_path = Path(data_path)
        
    def get_parameters(self, config):
        """Get current local LoRA weights"""
        return self.trainer.get_model_weights_list()

    def fit(self, parameters, config):
        """Receive global weights, train locally, and return updated weights"""
        print("\n📥 Received aggregated parameters from server")
        # Load parameters from server into our model
        self.trainer.set_model_weights_list(parameters)
        
        # Train locally
        print("🏋️ Starting local training round...")
        self.trainer.train(str(self.data_path))
        
        # Get updated local weights
        updated_weights = self.trainer.get_model_weights_list()
        
        # Count training examples
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                num_examples = len(json.load(f))
        except Exception:
            num_examples = 2
            
        print("📤 Sending trained local parameters back to server")
        return updated_weights, num_examples, {}

    def evaluate(self, parameters, config):
        """Evaluate parameters (optional, we return dummy values for causal LM)"""
        self.trainer.set_model_weights_list(parameters)
        # In a full deployment, we can evaluate on a validation set.
        # Returning loss=0.0 and examples=0 for now.
        return 0.0, 0, {}

def run_fl_client(config_path="client/config.yaml", data_path="data/user_data.json"):
    print("🤖 Launching Federated Learning Client...")
    
    # Initialize local trainer
    trainer = LocalTrainer(config_path)
    
    server_address = trainer.config['federated']['server_address']
    print(f"📡 Connecting to Federated Server at {server_address}...")
    
    # Convert NumPyClient to Client and start
    client = PyTorchFlowerClient(trainer, data_path)
    
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client()
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--config", type=str, default="client/config.yaml")
    parser.add_argument("--data", type=str, default="data/user_data.json")
    
    args = parser.parse_args()
    run_fl_client(args.config, args.data)
