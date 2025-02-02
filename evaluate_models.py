#!/usr/bin/env python3

import os
import torch
import logging
from pathlib import Path
from datetime import datetime
import argparse
import re
from collections import defaultdict
import json

# Import model classes from respective files
from mary_hangman_nv3 import MaryLSTMModel
from mary_hangman_nv4 import MaryBiLSTMModel
from mary_hangman_nv5 import MaryTransformerModel
from mary_hangman_nv5 import run_detailed_validation, load_data

def setup_logging():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'logs'
    Path(log_dir).mkdir(exist_ok=True)
    log_file = f'{log_dir}/model_evaluation_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def get_model_class(model_type):
    if 'nv3' in model_type:
        return MaryLSTMModel
    elif 'nv4' in model_type:
        return MaryBiLSTMModel
    elif 'nv5' in model_type:
        return MaryTransformerModel
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def evaluate_model(model_path, val_words):
    # Extract model type from filename
    model_type = re.search(r'mary_model_(nv[345])_', model_path).group(1)
    
    # Set device - CPU for transformer, MPS/GPU for others
    device = torch.device("cpu") if model_type == 'nv5' else (
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    
    # Initialize appropriate model
    ModelClass = get_model_class(model_type)
    model = ModelClass().to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Run validation
    stats = run_detailed_validation(model, val_words)
    return stats

def main():
    parser = argparse.ArgumentParser(description='Evaluate all Mary Hangman Models')
    parser.add_argument('--model-type', choices=['nv3', 'nv4', 'nv5'], 
                       help='Evaluate only models of this type')
    args = parser.parse_args()
    
    log_file = setup_logging()
    logging.info("Starting model evaluation")
    
    try:
        # Load validation data
        _, _, val_words = load_data()
        if val_words is None:
            raise ValueError("Could not load validation data")
        
        # Find all model files
        data_dir = 'hangman_data'
        model_pattern = f'mary_model_{args.model_type if args.model_type else "nv[345]"}_*.pt'
        model_files = sorted([f for f in os.listdir(data_dir) if re.match(model_pattern, f)])
        
        if not model_files:
            raise ValueError(f"No model files found matching pattern: {model_pattern}")
        
        results = defaultdict(list)
        
        for model_file in model_files:
            model_path = os.path.join(data_dir, model_file)
            logging.info(f"\nEvaluating model: {model_file}")
            
            try:
                stats = evaluate_model(model_path, val_words)
                results[model_file] = stats
                
                # Log summary statistics
                avg_completion = sum(stats['completion_by_length'].values()) / len(stats['completion_by_length'])
                logging.info(f"Average completion rate: {avg_completion:.2%}")
                logging.info("Completion rates by length:")
                for length, rate in stats['completion_by_length'].items():
                    logging.info(f"  Length {length}: {rate:.2%}")
                
            except Exception as e:
                logging.error(f"Failed to evaluate {model_file}: {str(e)}")
                continue
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'{data_dir}/evaluation_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise
    finally:
        logging.info(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    main() 