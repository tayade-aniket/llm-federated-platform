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

class LocalTrainer:
    def __init__(self, config_path="client/config.yaml"):
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model with quantization for memory efficiency
        self.load_model()
        
    def load_model(self):
        """Load model with 4-bit quantization for CPU/low memory"""
        from transformers import BitsAndBytesConfig
        
        if self.config['model']['use_quantization']:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config['model']['load_in_4bit'],
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True
        )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Prepare model for PEFT
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
        with open(data_path, 'r') as f:
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
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def train(self, data_path):
        """Run local training"""
        dataset = self.prepare_dataset(data_path)
        
        training_args = TrainingArguments(
            output_dir="./adapters/temp",
            per_device_train_batch_size=self.config['training']['batch_size'],
            num_train_epochs=self.config['training']['num_epochs'],
            learning_rate=self.config['training']['learning_rate'],
            save_steps=100,
            logging_steps=10,
            remove_unused_columns=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )
        
        trainer.train()
        
        # Save LoRA adapters
        adapter_path = f"./adapters/{self.config['federated']['client_id']}"
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)
        
        return adapter_path
    
    def get_model_weights(self):
        """Extract LoRA weights for federated learning"""
        lora_weights = {}
        for name, param in self.model.named_parameters():
            if 'lora' in name and param.requires_grad:
                lora_weights[name] = param.detach().cpu().numpy()
        return lora_weights
