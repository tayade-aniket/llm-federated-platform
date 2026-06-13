# run_platform.py
"""
Main orchestration script for LLM Federated Learning Platform.
Starts the server, web UI, and auto-connecting federated client.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["USE_TORCH"] = "1"

import subprocess
import sys
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
    
    print("✅ Project structure verified")
    
    # Create sample data
    data_path = PROJECT_ROOT / 'data' / 'user_data.json'
    if not data_path.exists():
        print("🎲 Generating synthetic sample dataset...")
        from utils.generate_sample_data import DataGenerator
        generator = DataGenerator()
        dataset = generator.generate_general_dataset(5)
        generator.save_dataset(dataset, str(data_path))
    
    print("✅ Sample data verified")

def run_server():
    """Start federated learning server"""
    server_path = PROJECT_ROOT / 'server' / 'federated_server.py'
    if not server_path.exists():
        print(f"❌ Server file not found: {server_path}")
        return
    
    print("\n🚀 Starting Federated Learning Server...")
    subprocess.run([sys.executable, str(server_path)], cwd=str(PROJECT_ROOT))

def run_web_ui():
    """Start web interface"""
    print("\n🌐 Starting Web UI at http://127.0.0.1:8000")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "web.app:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ], cwd=str(PROJECT_ROOT))

def run_client_node():
    """Run federated learning client process"""
    client_path = PROJECT_ROOT / 'client' / 'flower_client.py'
    if not client_path.exists():
        print(f"❌ Client file not found: {client_path}")
        return
    
    # Wait for server to boot up
    time.sleep(5)
    print("\n🤖 Starting Federated Learning Client connection...")
    subprocess.run([
        sys.executable, "-m", "client.flower_client",
        "--config", str(PROJECT_ROOT / 'client' / 'config.yaml'),
        "--data", str(PROJECT_ROOT / 'data' / 'user_data.json')
    ], cwd=str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description='LLM Federated Learning Platform')
    parser.add_argument('--mode', type=str, choices=['setup', 'server', 'web', 'client', 'all'],
                        default='all', help='Run mode')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 On-device LLM Fine-tuning & Federated Learning Platform")
    print("=" * 60)
    
    setup_complete_project()
    
    if args.mode == 'setup':
        print("✅ Project setup completed successfully")
    elif args.mode == 'server':
        run_server()
    elif args.mode == 'web':
        run_web_ui()
    elif args.mode == 'client':
        run_client_node()
    elif args.mode == 'all':
        print("\n🔄 Starting all components in parallel...")
        print("   - Federated Server (port 8080)")
        print("   - Web UI (port 8000)")
        print("   - Federated Client (auto-connecting in 5 seconds)")
        print("\n⚠️ Press Ctrl+C to stop all components\n")
        
        # Run server, web UI, and client in threads
        server_thread = threading.Thread(target=run_server, daemon=True)
        web_thread = threading.Thread(target=run_web_ui, daemon=True)
        client_thread = threading.Thread(target=run_client_node, daemon=True)
        
        server_thread.start()
        time.sleep(2)
        web_thread.start()
        time.sleep(1)
        client_thread.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down all platform services...")
            sys.exit(0)

if __name__ == "__main__":
    main()