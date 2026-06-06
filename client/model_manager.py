# client/model_manager.py
"""
Manage model loading, saving, caching, and versioning
Handles model lifecycle for on-device fine-tuning
"""

import os
import shutil
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

class ModelManager:
    """Manage LLM models and adapters on device"""
    
    def __init__(self, base_path: str = "./models", cache_path: str = "./cache"):
        self.base_path = Path(base_path)
        self.cache_path = Path(cache_path)
        self.adapters_path = Path("./adapters")
        
        # Create directories
        for path in [self.base_path, self.cache_path, self.adapters_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        self.current_model = None
        self.current_tokenizer = None
        self.model_metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load model metadata from file"""
        metadata_file = self.base_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {"models": {}, "active_model": None}
    
    def _save_metadata(self):
        """Save model metadata to file"""
        metadata_file = self.base_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
    
    def download_model(self, model_name: str, quantize: bool = True) -> bool:
        """Download and cache a model"""
        try:
            print(f"📥 Downloading model: {model_name}")
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_path),
                trust_remote_code=True
            )
            
            # Download model with optional quantization
            if quantize:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float32,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    cache_dir=str(self.cache_path),
                    trust_remote_code=True,
                    device_map="auto"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=str(self.cache_path),
                    trust_remote_code=True
                )
            
            # Save metadata
            self.model_metadata["models"][model_name] = {
                "downloaded_at": datetime.now().isoformat(),
                "quantized": quantize,
                "path": str(self.cache_path / model_name.replace("/", "_"))
            }
            self.model_metadata["active_model"] = model_name
            self._save_metadata()
            
            self.current_model = model
            self.current_tokenizer = tokenizer
            
            print(f"✅ Model {model_name} downloaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            return False
    
    def load_model(self, model_name: str, use_lora: bool = False, 
                   adapter_path: Optional[str] = None) -> bool:
        """Load model from cache or download if not available"""
        try:
            # Check if model exists in cache
            if model_name not in self.model_metadata["models"]:
                print(f"Model {model_name} not found locally, downloading...")
                if not self.download_model(model_name):
                    return False
            
            # Load tokenizer
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_path),
                trust_remote_code=True
            )
            
            # Load model
            self.current_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(self.cache_path),
                trust_remote_code=True,
                device_map="auto"
            )
            
            # Load LoRA adapter if specified
            if use_lora and adapter_path:
                self.current_model = PeftModel.from_pretrained(
                    self.current_model,
                    adapter_path
                )
                print(f"✅ Loaded LoRA adapter from {adapter_path}")
            
            print(f"✅ Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def save_adapter(self, model, adapter_name: str, metadata: Optional[Dict] = None) -> str:
        """Save LoRA adapter to disk"""
        adapter_dir = self.adapters_path / adapter_name
        adapter_dir.mkdir(parents=True, exist_ok=True)
        
        # Save adapter weights
        model.save_pretrained(adapter_dir)
        self.current_tokenizer.save_pretrained(adapter_dir)
        
        # Save metadata
        meta = {
            "name": adapter_name,
            "created_at": datetime.now().isoformat(),
            "base_model": self.model_metadata.get("active_model", "unknown"),
            "metadata": metadata or {}
        }
        
        with open(adapter_dir / "metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        # Update global adapters registry
        self._update_adapter_registry(adapter_name, meta)
        
        print(f"✅ Adapter saved to {adapter_dir}")
        return str(adapter_dir)
    
    def _update_adapter_registry(self, adapter_name: str, metadata: Dict):
        """Update the global adapters registry"""
        registry_file = self.adapters_path / "registry.json"
        
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                registry = json.load(f)
        else:
            registry = {"adapters": {}}
        
        registry["adapters"][adapter_name] = metadata
        registry["last_updated"] = datetime.now().isoformat()
        
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def list_adapters(self) -> List[Dict]:
        """List all saved adapters"""
        registry_file = self.adapters_path / "registry.json"
        
        if not registry_file.exists():
            return []
        
        with open(registry_file, 'r') as f:
            registry = json.load(f)
        
        adapters = []
        for name, metadata in registry.get("adapters", {}).items():
            adapters.append({
                "name": name,
                "created_at": metadata.get("created_at"),
                "base_model": metadata.get("base_model"),
                "path": str(self.adapters_path / name)
            })
        
        return sorted(adapters, key=lambda x: x["created_at"], reverse=True)
    
    def load_adapter(self, adapter_name: str) -> Optional[PeftModel]:
        """Load a specific adapter for inference"""
        adapter_path = self.adapters_path / adapter_name
        
        if not adapter_path.exists():
            print(f"❌ Adapter {adapter_name} not found")
            return None
        
        if self.current_model is None:
            print("❌ No base model loaded")
            return None
        
        try:
            self.current_model = PeftModel.from_pretrained(
                self.current_model,
                str(adapter_path)
            )
            print(f"✅ Loaded adapter: {adapter_name}")
            return self.current_model
        except Exception as e:
            print(f"❌ Failed to load adapter: {e}")
            return None
    
    def delete_adapter(self, adapter_name: str) -> bool:
        """Delete a saved adapter"""
        adapter_path = self.adapters_path / adapter_name
        
        if not adapter_path.exists():
            print(f"❌ Adapter {adapter_name} not found")
            return False
        
        try:
            shutil.rmtree(adapter_path)
            
            # Update registry
            registry_file = self.adapters_path / "registry.json"
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    registry = json.load(f)
                
                if adapter_name in registry.get("adapters", {}):
                    del registry["adapters"][adapter_name]
                    
                    with open(registry_file, 'w') as f:
                        json.dump(registry, f, indent=2)
            
            print(f"✅ Deleted adapter: {adapter_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to delete adapter: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about current model"""
        if self.current_model is None:
            return {"loaded": False}
        
        total_params = sum(p.numel() for p in self.current_model.parameters())
        trainable_params = sum(p.numel() for p in self.current_model.parameters() 
                              if p.requires_grad)
        
        return {
            "loaded": True,
            "model_name": self.model_metadata.get("active_model"),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100,
            "device": str(next(self.current_model.parameters()).device),
            "dtype": str(next(self.current_model.parameters()).dtype)
        }
    
    def clear_cache(self):
        """Clear model cache to free disk space"""
        cache_size = sum(f.stat().st_size for f in self.cache_path.rglob('*') 
                        if f.is_file())
        
        confirm = input(f"Cache size: {cache_size / 1e9:.2f} GB. Clear? (y/N): ")
        if confirm.lower() == 'y':
            shutil.rmtree(self.cache_path)
            self.cache_path.mkdir(parents=True, exist_ok=True)
            print("✅ Cache cleared")
    
    def export_to_onnx(self, adapter_name: str, output_path: str):
        """Export model with adapter to ONNX format"""
        try:
            import torch.onnx
            
            # Load model with adapter
            self.load_adapter(adapter_name)
            
            # Dummy input for tracing
            dummy_input = self.current_tokenizer("Test input", return_tensors="pt")
            
            # Export
            torch.onnx.export(
                self.current_model,
                tuple(dummy_input.values()),
                output_path,
                input_names=['input_ids', 'attention_mask'],
                output_names=['output'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence_length'}
                },
                opset_version=14
            )
            
            print(f"✅ Model exported to ONNX: {output_path}")
            return True
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            return False

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Management CLI")
    parser.add_argument("command", choices=["list", "download", "delete", "info", "cache"],
                       help="Command to execute")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--adapter", type=str, help="Adapter name")
    
    args = parser.parse_args()
    manager = ModelManager()
    
    if args.command == "list":
        adapters = manager.list_adapters()
        print("\n📦 Saved Adapters:")
        for adapter in adapters:
            print(f"  - {adapter['name']} (created: {adapter['created_at']})")
    
    elif args.command == "download":
        if args.model:
            manager.download_model(args.model)
        else:
            print("Please specify --model")
    
    elif args.command == "delete":
        if args.adapter:
            manager.delete_adapter(args.adapter)
        else:
            print("Please specify --adapter")
    
    elif args.command == "info":
        info = manager.get_model_info()
        print("\n📊 Model Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    elif args.command == "cache":
        manager.clear_cache()