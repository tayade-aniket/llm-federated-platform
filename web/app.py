from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import subprocess
import json
import os
import uuid
from pathlib import Path

app = FastAPI(title="On-device LLM Fine-tuning Platform")

# Setup templates and static files
templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Store training sessions
sessions = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """Upload custom training data"""
    session_id = str(uuid.uuid4())
    data_path = f"data/session_{session_id}.json"
    
    content = await file.read()
    data = json.loads(content)
    
    # Validate format
    if not all("instruction" in item and "output" in item for item in data):
        raise HTTPException(400, "Missing 'instruction' or 'output' fields")
    
    with open(data_path, "w") as f:
        json.dump(data, f)
    
    sessions[session_id] = {"data_path": data_path, "status": "uploaded"}
    
    return {"session_id": session_id, "message": "Data uploaded successfully"}

@app.post("/start-training/{session_id}")
async def start_training(session_id: str):
    """Start local fine-tuning"""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    
    # Run training in background
    from client.local_trainer import LocalTrainer
    
    trainer = LocalTrainer()
    adapter_path = trainer.train(sessions[session_id]["data_path"])
    
    sessions[session_id]["status"] = "trained"
    sessions[session_id]["adapter_path"] = adapter_path
    
    return {"status": "completed", "adapter_path": adapter_path}

@app.get("/model-status")
async def model_status():
    """Check if model is loaded"""
    model_path = "models/global_model"
    if os.path.exists(model_path):
        return {"loaded": True, "path": model_path}
    return {"loaded": False}

@app.post("/generate")
async def generate_text(
    prompt: str = Form(...),
    use_personalized: bool = Form(True)
):
    """Generate text using the trained model"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    
    if use_personalized and os.path.exists("adapters/latest"):
        model = PeftModel.from_pretrained(model, "adapters/latest")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"response": response}