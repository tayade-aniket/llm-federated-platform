import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["USE_TORCH"] = "1"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import json
import os
import shutil

class LocalTrainer:
    def __init__(self, config_path="client/config.yaml"):
        import yaml
        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
        # Ensure default dictionary structures exist
        if 'model' not in self.config:
            self.config['model'] = {}
        if 'training' not in self.config:
            self.config['training'] = {}
        if 'federated' not in self.config:
            self.config['federated'] = {}
            
        self.config['model'].setdefault('name', 'sshleifer/tiny-gpt2')
        self.config['model'].setdefault('use_quantization', False)
        self.config['model'].setdefault('load_in_4bit', False)
        self.config['training'].setdefault('lora_r', 4)
        self.config['training'].setdefault('lora_alpha', 8)
        self.config['training'].setdefault('target_modules', ["c_attn"])
        self.config['training'].setdefault('lora_dropout', 0.1)
        self.config['training'].setdefault('batch_size', 1)
        self.config['training'].setdefault('learning_rate', 1e-3)
        self.config['training'].setdefault('num_epochs', 1)
        self.config['training'].setdefault('max_seq_length', 32)
        self.config['federated'].setdefault('server_address', '127.0.0.1:8080')
        self.config['federated'].setdefault('client_id', 'user_001')
        
        # Override with environment variables
        env_server_addr = os.environ.get("FL_SERVER_ADDRESS")
        if env_server_addr:
            self.config['federated']['server_address'] = env_server_addr
            
        env_client_id = os.environ.get("FL_CLIENT_ID")
        if env_client_id:
            self.config['federated']['client_id'] = env_client_id
            
        env_model_name = os.environ.get("FL_MODEL_NAME")
        if env_model_name:
            self.config['model']['name'] = env_model_name
            
        # Enforce strict types to prevent TypeError: can't multiply sequence by non-int of type 'float'
        self.config['training']['lora_r'] = int(self.config['training']['lora_r'])
        self.config['training']['lora_alpha'] = int(self.config['training']['lora_alpha'])
        self.config['training']['lora_dropout'] = float(self.config['training']['lora_dropout'])
        self.config['training']['batch_size'] = int(self.config['training']['batch_size'])
        self.config['training']['learning_rate'] = float(self.config['training']['learning_rate'])
        self.config['training']['num_epochs'] = int(self.config['training']['num_epochs'])
        self.config['training']['max_seq_length'] = int(self.config['training']['max_seq_length'])
        if isinstance(self.config['training']['target_modules'], str):
            self.config['training']['target_modules'] = [self.config['training']['target_modules']]

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """Load model with optional 4-bit quantization for CPU/low memory"""
        from transformers import BitsAndBytesConfig
        
        # Quantization is only supported with CUDA on Windows
        cuda_available = torch.cuda.is_available()
        use_quant = self.config['model']['use_quantization'] and cuda_available
        
        if use_quant:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config['model']['load_in_4bit'],
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_use_double_quant=True,
            )
            device_map = "auto"
        else:
            bnb_config = None
            device_map = None
        
        model_name = self.config['model']['name']
        print(f"Loading model '{model_name}' (Quantization: {use_quant}, Device: {self.device})")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        
        if device_map is None:
            self.model = self.model.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Prepare model for PEFT if quantized
        if use_quant:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config['training']['lora_r'],
            lora_alpha=self.config['training']['lora_alpha'],
            target_modules=self.config['training']['target_modules'],
            lora_dropout=self.config['training']['lora_dropout'],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def prepare_dataset(self, data_path):
        """Load and tokenize local dataset"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Format prompts
        formatted_data = []
        for item in data:
            text = f"### Instruction: {item['instruction']}\n### Input: {item.get('input', '')}\n### Response: {item['output']}"
            formatted_data.append({"text": text})
        
        dataset = Dataset.from_list(formatted_data)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.config['training']['max_seq_length']
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        return tokenized_dataset
    
    def train(self, data_path):
        """Run local training"""
        dataset = self.prepare_dataset(data_path)
        
        training_args = TrainingArguments(
            output_dir="./adapters/temp",
            per_device_train_batch_size=self.config['training']['batch_size'],
            num_train_epochs=self.config['training']['num_epochs'],
            learning_rate=self.config['training']['learning_rate'],
            save_strategy="no",
            logging_steps=1,
            remove_unused_columns=True,
            use_cpu=(self.device.type == "cpu"),
            report_to="none",
            optim="adamw_torch"
        )
        
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )
        
        trainer.train()
        
        # Save LoRA adapters
        import time
        client_id = self.config['federated']['client_id']
        timestamp = int(time.time())
        adapter_path = f"./adapters/{client_id}_{timestamp}"
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)
        
        # Save pointer to latest adapter path to avoid Windows file lock issues
        latest_ref_path = "./adapters/latest_path.txt"
        try:
            with open(latest_ref_path, "w", encoding="utf-8") as f:
                f.write(adapter_path)
        except Exception as e:
            print(f"⚠️ Warning: Could not write latest adapter pointer: {e}")

        # Try to also update client_id fallback (catches locking errors on Windows gracefully)
        fallback_path = f"./adapters/{client_id}"
        try:
            self.model.save_pretrained(fallback_path)
            self.tokenizer.save_pretrained(fallback_path)
        except Exception:
            pass
            
        print(f"✅ Training complete. Adapters saved to {adapter_path}")
        return adapter_path
    
    def get_model_weights_list(self):
        """Extract LoRA weights as a list of numpy arrays (Flower format)"""
        weights_list = []
        # Sort by parameter name to ensure consistent ordering across nodes
        for name, param in sorted(self.model.named_parameters()):
            if param.requires_grad:
                weights_list.append(param.detach().cpu().numpy())
        return weights_list

    def set_model_weights_list(self, weights_list):
        """Set LoRA weights from a list of numpy arrays (Flower format)"""
        trainable_params = [param for name, param in sorted(self.model.named_parameters()) if param.requires_grad]
        if len(weights_list) != len(trainable_params):
            raise ValueError(f"Weights list length mismatch: expected {len(trainable_params)}, got {len(weights_list)}")
        
        with torch.no_grad():
            for param, ndarray in zip(trainable_params, weights_list):
                param.copy_(torch.from_numpy(ndarray).to(param.device))
        print("✅ Model weights loaded from aggregated parameters")
