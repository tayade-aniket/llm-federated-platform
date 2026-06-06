import json
import os
from datetime import datetime

class DataProcessor:
    @staticmethod
    def create_sample_data():
        """Create sample instruction data for testing"""
        sample_data = [
            {
                "instruction": "Explain what is machine learning",
                "input": "",
                "output": "Machine learning is a subset of AI that enables systems to learn from data without explicit programming."
            },
            {
                "instruction": "Write a Python function to add two numbers",
                "input": "",
                "output": "def add_numbers(a, b):\n    return a + b"
            }
        ]
        
        os.makedirs("data", exist_ok=True)
        with open("data/user_data.json", "w") as f:
            json.dump(sample_data, f, indent=2)
        
        return "data/user_data.json"