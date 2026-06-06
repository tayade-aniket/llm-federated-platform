# run_platform.py (place in root directory)
"""
Main orchestration script for LLM Federated Learning Platform
"""

import subprocess
import sys
import os
import time
import threading
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

def setup_complete_project():
    """Create all necessary files and directories"""
    
    # Create directories
    dirs = ['client', 'server', 'web', 'utils', 'data', 'models', 'adapters',
            'web/static', 'web/templates', 'models/checkpoints', 'models/metrics']
    
    for dir_name in dirs:
        (PROJECT_ROOT / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py files
    init_files = ['client/__init__.py', 'server/__init__.py', 'web/__init__.py', 'utils/__init__.py']
    for init_file in init_files:
        init_path = PROJECT_ROOT / init_file
        if not init_path.exists():
            init_path.write_text("# Auto-generated __init__.py\n")
    
    print("✅ Project structure created")
    
    # Create sample data
    if not (PROJECT_ROOT / 'data' / 'user_data.json').exists():
        from utils.generate_sample_data import DataGenerator
        generator = DataGenerator()
        dataset = generator.generate_general_dataset(20)
        generator.save_dataset(dataset, str(PROJECT_ROOT / 'data' / 'user_data.json'))
    
    print("✅ Sample data generated")

def run_server():
    """Start federated learning server"""
    server_path = PROJECT_ROOT / 'server' / 'federated_server.py'
    if not server_path.exists():
        print(f"❌ Server file not found: {server_path}")
        return
    
    print("\n🚀 Starting Federated Learning Server...")
    subprocess.run([sys.executable, str(server_path)])

def run_web_ui():
    """Start web interface"""
    print("\n🌐 Starting Web UI at http://localhost:8000")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "web.app:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--reload"
    ], cwd=str(PROJECT_ROOT))

def run_client():
    """Run as client (local training)"""
    try:
        from client.local_trainer import LocalTrainer
        
        data_path = PROJECT_ROOT / 'data' / 'user_data.json'
        if not data_path.exists():
            print("❌ No data found. Run setup first.")
            return
        
        print("\n🏋️ Starting Local Training...")
        trainer = LocalTrainer(str(PROJECT_ROOT / 'client' / 'config.yaml'))
        trainer.train(str(data_path))
        
    except Exception as e:
        print(f"❌ Training failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='LLM Federated Learning Platform')
    parser.add_argument('--mode', type=str, choices=['setup', 'server', 'web', 'client', 'all'],
                        default='all', help='Run mode')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🤖 On-device LLM Fine-tuning Platform")
    print("=" * 50)
    
    if args.mode == 'setup':
        setup_complete_project()
    elif args.mode == 'server':
        run_server()
    elif args.mode == 'web':
        run_web_ui()
    elif args.mode == 'client':
        setup_complete_project()
        run_client()
    elif args.mode == 'all':
        setup_complete_project()
        
        print("\n🔄 Starting all components...")
        print("   - Federated Server (port 8080)")
        print("   - Web UI (port 8000)")
        print("\n⚠️ Press Ctrl+C to stop all\n")
        
        # Run server and web UI in parallel
        server_thread = threading.Thread(target=run_server, daemon=True)
        web_thread = threading.Thread(target=run_web_ui, daemon=True)
        
        server_thread.start()
        time.sleep(2)
        web_thread.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")
            sys.exit(0)

if __name__ == "__main__":
    main()