"""
Generate synthetic training data for LLM fine-tuning
Supports multiple domains and custom templates
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

class DataGenerator:
    """Generate instruction-based datasets for fine-tuning"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, List[Dict]]:
        """Initialize domain-specific templates"""
        return {
            "general": [
                {
                    "instruction": "What is {topic}?",
                    "output": "{topic} is a fundamental concept in {field}. It involves {description}."
                },
                {
                    "instruction": "Explain {topic} in simple terms",
                    "output": "Think of {topic} like {analogy}. It helps with {application}."
                },
                {
                    "instruction": "Why is {topic} important?",
                    "output": "{topic} is important because it enables {benefit1} and {benefit2}."
                }
            ],
            "coding": [
                {
                    "instruction": "Write a Python function to {task}",
                    "output": "def {func_name}({params}):\n    \"\"\"\n    {description}\n    \"\"\"\n    # Implementation\n    {implementation}\n    return result"
                },
                {
                    "instruction": "Debug this {language} code: {code_snippet}",
                    "output": "The issue is {issue}. Fix: {solution}"
                }
            ],
            "math": [
                {
                    "instruction": "Solve: {problem}",
                    "output": "Step 1: {step1}\nStep 2: {step2}\nAnswer: {answer}"
                }
            ],
            "science": [
                {
                    "instruction": "Explain {concept}",
                    "output": "{concept} is {definition}. For example, {example}."
                }
            ]
        }
    
    def generate_general_dataset(self, num_samples: int = 50) -> List[Dict]:
        """Generate general knowledge dataset"""
        topics = [
            "machine learning", "artificial intelligence", "neural networks",
            "deep learning", "natural language processing", "computer vision",
            "reinforcement learning", "data science", "big data", "cloud computing"
        ]
        
        fields = [
            "computer science", "mathematics", "engineering", "technology"
        ]
        
        descriptions = [
            "learning from data without explicit programming",
            "recognizing patterns and making predictions",
            "automating complex decision-making processes",
            "extracting insights from large datasets"
        ]
        
        analogies = [
            "a student learning from textbooks",
            "a chef improving recipes through practice",
            "a musician learning to play by ear",
            "a detective solving puzzles"
        ]
        
        applications = [
            "self-driving cars", "medical diagnosis", "fraud detection",
            "recommendation systems", "language translation"
        ]
        
        benefits = [
            "automation of repetitive tasks",
            "improved accuracy and efficiency",
            "discovery of hidden patterns",
            "real-time decision making"
        ]
        
        dataset = []
        for _ in range(num_samples):
            template = random.choice(self.templates["general"])
            topic = random.choice(topics)
            
            sample = {
                "instruction": template["instruction"].format(topic=topic),
                "input": "",
                "output": template["output"].format(
                    topic=topic,
                    field=random.choice(fields),
                    description=random.choice(descriptions),
                    analogy=random.choice(analogies),
                    application=random.choice(applications),
                    benefit1=random.choice(benefits),
                    benefit2=random.choice(benefits)
                )
            }
            dataset.append(sample)
        
        return dataset
    
    def generate_coding_dataset(self, num_samples: int = 30) -> List[Dict]:
        """Generate coding-related dataset"""
        tasks = [
            "sort a list", "find prime numbers", "reverse a string",
            "calculate fibonacci sequence", "check palindrome",
            "merge two dictionaries", "implement binary search"
        ]
        
        func_names = ["sort_list", "find_primes", "reverse_string", 
                      "fibonacci", "is_palindrome", "merge_dicts", "binary_search"]
        
        params_list = [
            "arr", "n", "s", "n", "s", "dict1, dict2", "arr, target"
        ]
        
        implementations = [
            "result = sorted(arr)",
            "primes = [i for i in range(2, n+1) if all(i % j != 0 for j in range(2, int(i**0.5)+1))]",
            "result = s[::-1]",
            "a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b",
            "result = s == s[::-1]",
            "result = {**dict1, **dict2}",
            "left, right = 0, len(arr)-1\n    while left <= right:\n        mid = (left+right)//2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid+1\n        else:\n            right = mid-1\n    return -1"
        ]
        
        dataset = []
        for i in range(num_samples):
            template = random.choice(self.templates["coding"])
            idx = i % len(tasks)
            
            sample = {
                "instruction": template["instruction"].format(
                    task=tasks[idx],
                    language="Python",
                    code_snippet="def buggy_function():\n    pass"
                ),
                "input": "",
                "output": template["output"].format(
                    func_name=func_names[idx],
                    params=params_list[idx],
                    description=f"This function {tasks[idx]}",
                    implementation=implementations[idx],
                    issue="syntax error",
                    solution="proper indentation"
                )
            }
            dataset.append(sample)
        
        return dataset
    
    def generate_conversational_dataset(self, num_samples: int = 40) -> List[Dict]:
        """Generate conversational AI dataset"""
        conversations = [
            {"instruction": "Hello, how are you?", "output": "I'm doing great! How can I help you today?"},
            {"instruction": "Tell me a joke", "output": "Why don't scientists trust atoms? Because they make up everything!"},
            {"instruction": "What's the weather like?", "output": "I don't have real-time weather data, but you can check your local weather service!"},
            {"instruction": "Help me with my homework", "output": "I'd be happy to help! What subject are you working on?"},
            {"instruction": "What can you do?", "output": "I can help with explanations, coding, math problems, and general knowledge questions!"}
        ]
        
        dataset = []
        for i in range(num_samples):
            conv = random.choice(conversations)
            sample = {
                "instruction": conv["instruction"],
                "input": "",
                "output": conv["output"]
            }
            dataset.append(sample)
        
        return dataset
    
    def generate_custom_dataset(self, domain: str, num_samples: int, 
                                custom_topics: List[str] = None) -> List[Dict]:
        """Generate custom dataset for specific domain"""
        if domain == "general":
            return self.generate_general_dataset(num_samples)
        elif domain == "coding":
            return self.generate_coding_dataset(num_samples)
        elif domain == "conversational":
            return self.generate_conversational_dataset(num_samples)
        else:
            # Mixed dataset
            datasets = []
            datasets.extend(self.generate_general_dataset(num_samples // 3))
            datasets.extend(self.generate_coding_dataset(num_samples // 3))
            datasets.extend(self.generate_conversational_dataset(num_samples // 3))
            return datasets
    
    def save_dataset(self, dataset: List[Dict], filepath: str, format: str = "json"):
        """Save dataset to file"""
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        elif format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ Saved {len(dataset)} samples to {output_path}")
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--domain", type=str, default="mixed",
                        choices=["general", "coding", "conversational", "mixed"],
                        help="Domain of dataset")
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data/custom_dataset.json",
                        help="Output file path")
    parser.add_argument("--format", type=str, default="json",
                        choices=["json", "jsonl"],
                        help="Output format")
    
    args = parser.parse_args()
    
    print(f"🎲 Generating {args.samples} samples for domain: {args.domain}")
    
    generator = DataGenerator()
    dataset = generator.generate_custom_dataset(args.domain, args.samples)
    
    generator.save_dataset(dataset, args.output, args.format)
    
    # Print sample
    print("\n📝 Sample generated data:")
    print(json.dumps(dataset[0], indent=2))
    
    # Statistics
    avg_instruction_len = sum(len(d['instruction']) for d in dataset) / len(dataset)
    avg_output_len = sum(len(d['output']) for d in dataset) / len(dataset)
    
    print(f"\n📊 Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Avg instruction length: {avg_instruction_len:.1f} chars")
    print(f"  Avg output length: {avg_output_len:.1f} chars")

if __name__ == "__main__":
    main()