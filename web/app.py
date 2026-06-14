import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["USE_TORCH"] = "1"

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import uuid
import yaml
from pathlib import Path

app = FastAPI(title="On-device LLM Fine-tuning Platform")

# Setup templates and static files (search templates inside web/templates)
templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Store training sessions
sessions = {}

# Lazy loaded models cache
cached_model = None
cached_tokenizer = None
loaded_adapter_path = None
is_cached_model_personalized = False

def get_model_and_tokenizer(use_personalized: bool):
    """Lazy load and cache model and tokenizer to prevent RAM bloat and fast inference"""
    global cached_model, cached_tokenizer, loaded_adapter_path, is_cached_model_personalized
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    # Read config
    with open("client/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model_name = config["model"]["name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter_path = "adapters/latest"
    adapter_exists = os.path.exists(adapter_path)
    
    # Determine if we should load the personalized adapter
    should_load_personalized = use_personalized and adapter_exists
    
    # If not cached yet, or configuration mismatch, reload
    if (cached_model is None or 
        cached_tokenizer is None or 
        is_cached_model_personalized != should_load_personalized or 
        (should_load_personalized and loaded_adapter_path != adapter_path)):
        
        print(f"Loading tokenizer for model '{model_name}'...")
        cached_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if cached_tokenizer.pad_token is None:
            cached_tokenizer.pad_token = cached_tokenizer.eos_token
            
        print(f"Loading base model '{model_name}' on {device}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(device)
        
        if should_load_personalized:
            print(f"Applying PEFT LoRA adapters from {adapter_path}...")
            cached_model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
            loaded_adapter_path = adapter_path
            is_cached_model_personalized = True
        else:
            cached_model = base_model
            loaded_adapter_path = None
            is_cached_model_personalized = False
            
        print("✅ Model loaded and cached successfully")
        
    return cached_model, cached_tokenizer

def run_training_task(session_id: str, data_path: str):
    """Worker function for running fine-tuning in a background thread"""
    try:
        from client.local_trainer import LocalTrainer
        print(f"Background worker: Starting training for session {session_id}")
        trainer = LocalTrainer("client/config.yaml")
        adapter_path = trainer.train(data_path)
        
        sessions[session_id]["status"] = "completed"
        sessions[session_id]["adapter_path"] = adapter_path
        print(f"Background worker: Completed training for session {session_id}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sessions[session_id]["status"] = "failed"
        sessions[session_id]["error"] = str(e)
        print(f"Background worker: Training failed for session {session_id}: {e}")

# Create necessary directories at startup
os.makedirs("data", exist_ok=True)
os.makedirs("adapters", exist_ok=True)
os.makedirs("models/cache", exist_ok=True)

def get_system_specs():
    """Gather actual hardware configurations of the local node"""
    import platform
    import torch
    import shutil

    # Get CPU
    cpu_name = platform.processor()
    if not cpu_name:
        cpu_name = platform.machine()
    
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except Exception:
            pass

    # Get GPU
    gpu_name = "None"
    gpu_status = "No GPU detected"
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_status = "CUDA Active"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_name = "Apple Silicon GPU"
            gpu_status = "MPS Active"
        else:
            gpu_name = "None (CPU fallback active)"
            gpu_status = "CPU fallback training active"
    except Exception:
        pass

    # Get RAM
    ram_gb = "Unknown"
    try:
        import psutil
        ram_gb = f"{psutil.virtual_memory().total / (1024**3):.2f} GB RAM"
    except ImportError:
        # Fallback for Windows
        if platform.system() == "Windows":
            try:
                import subprocess
                out = subprocess.check_output("wmic ComputerSystem get TotalPhysicalMemory", shell=True).decode()
                lines = [line.strip() for line in out.splitlines() if line.strip()]
                if len(lines) > 1:
                    bytes_ram = int(lines[1])
                    ram_gb = f"{bytes_ram / (1024**3):.2f} GB RAM"
            except Exception:
                pass
        # Fallback for Linux
        else:
            try:
                import subprocess
                out = subprocess.check_output("free -b", shell=True).decode()
                for line in out.splitlines():
                    if line.startswith("Mem:"):
                        bytes_ram = int(line.split()[1])
                        ram_gb = f"{bytes_ram / (1024**3):.2f} GB RAM"
            except Exception:
                pass
    if ram_gb != "Unknown" and "RAM" not in ram_gb:
        ram_gb = f"{ram_gb} RAM"

    # Get Disk free space
    disk_free = "Unknown"
    try:
        total, used, free = shutil.disk_usage(".")
        disk_free = f"{free / (1024**3):.2f} GB free space"
    except Exception:
        pass

    python_version = platform.python_version()

    return {
        "cpu": cpu_name,
        "gpu": f"{gpu_name} ({gpu_status})",
        "ram": ram_gb,
        "disk": f"{disk_free} | Python {python_version} (PyTorch Mode)"
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    specs = get_system_specs()
    server_address = "127.0.0.1:8080"
    try:
        with open("client/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            server_address = config.get("federated", {}).get("server_address", "127.0.0.1:8080")
    except Exception:
        pass
    return templates.TemplateResponse("dashboard.html", {
        "request": request, 
        "specs": specs,
        "server_address": server_address
    })

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """Upload custom training data"""
    session_id = str(uuid.uuid4())
    os.makedirs("data", exist_ok=True)
    data_path = f"data/session_{session_id}.json"
    
    content = await file.read()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON file format")
    
    # Validate format
    if not isinstance(data, list) or not all("instruction" in item and "output" in item for item in data):
        raise HTTPException(400, "JSON must be a list of objects with 'instruction' and 'output' fields")
    
    with open(data_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    sessions[session_id] = {
        "data_path": data_path, 
        "status": "uploaded",
        "size_kb": len(content) / 1024
    }
    
    return {"session_id": session_id, "message": "Data uploaded successfully"}

@app.post("/start-training/{session_id}")
async def start_training(session_id: str, background_tasks: BackgroundTasks):
    """Start local fine-tuning in background"""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    
    if sessions[session_id]["status"] == "training":
        return {"status": "training", "message": "Training already in progress"}
        
    sessions[session_id]["status"] = "training"
    
    # Run training in background task to prevent event loop blocking
    background_tasks.add_task(run_training_task, session_id, sessions[session_id]["data_path"])
    
    return {"status": "started", "message": "Training started in the background"}

@app.get("/training-status/{session_id}")
async def get_training_status(session_id: str):
    """Endpoint for the client UI to poll training progress"""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    return sessions[session_id]

@app.get("/model-status")
async def model_status():
    """Check if personal model is loaded"""
    adapter_path = "adapters/latest"
    if os.path.exists(adapter_path):
        return {"loaded": True, "path": adapter_path}
    return {"loaded": False}

@app.post("/generate")
async def generate_text(
    prompt: str = Form(...),
    use_personalized: str = Form("true")
):
    """Generate text using the trained model"""
    use_pers = (use_personalized.lower() == "true")
    
    try:
        import torch
        model, tokenizer = get_model_and_tokenizer(use_pers)
        
        inputs = tokenizer(prompt, return_tensors="pt")
        # Move to correct device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the generated response (excluding the prompt)
        input_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        return {"response": response.strip()}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Generation failed: {str(e)}")