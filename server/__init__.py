server:
  host: "0.0.0.0"
  port: 8080
  num_rounds: 5
  min_fit_clients: 1
  min_available_clients: 1
  
aggregation:
  strategy: "fedavg"  # Federated Averaging
  fraction_fit: 1.0
  
model:
  base_model: "microsoft/phi-2"
  
storage:
  global_model_path: "./models/global_model"
  adapters_dir: "./adapters"