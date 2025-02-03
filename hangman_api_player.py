import json
import requests
import string
import time
import torch
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import traceback
import sys

# Import your model classes
from mary_hangman_nv3 import MaryLSTMModel
# from mary_hangman_nv4 import MaryBiLSTMModel  # Only uncomment if needed
# from mary_hangman_nv5 import MaryTransformerModel  # Only uncomment if needed

# Import the HangmanAPI class
from hangmanAPI import HangmanAPI

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_file = f'{log_dir}/hangman_player_{timestamp}.log'
    
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging with both file and console output
    logging.basicConfig(
        level=logging.INFO,  # Set root logger level
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)  # Explicitly use stdout
        ]
    )
    
    # Set levels for specific loggers
    logging.getLogger('hangman_player').setLevel(logging.DEBUG)
    
    # Test logging
    logging.info("Logging setup complete")
    logging.info(f"Log file: {log_file}")
    
    return log_file

class ModelHangmanAPI(HangmanAPI):
    """Extended HangmanAPI that uses ML model for guessing"""
    
    def __init__(self, model, access_token=None):
        super().__init__(access_token=access_token)
        self.model = model
        self.guessed_letters = []
        logging.info("ModelHangmanAPI initialized")
        
    def guess(self, word_state):
        """Override the guess method to use our model"""
        try:
            logging.debug(f"Current state: {word_state}")
            logging.debug(f"Guessed letters: {self.guessed_letters}")
            
            # Prepare input
            with time_block("Preparing input"):
                char_indices, guessed, word_length, vowel_ratio, remaining_lives = self.prepare_state(word_state)
            
            # Get model prediction
            with time_block("Model prediction"):
                with torch.no_grad():
                    output = self.model(char_indices.unsqueeze(0), guessed.unsqueeze(0), 
                                      word_length, vowel_ratio, remaining_lives)
            
            # Convert to probabilities and mask already guessed letters
            probs = output[0]
            for letter in self.guessed_letters:
                probs[ord(letter) - ord('a')] = 0
                
            # Get highest probability unguessed letter
            guess_idx = probs.argmax().item()
            guess_letter = chr(guess_idx + ord('a'))
            
            # Update guessed letters
            self.guessed_letters.append(guess_letter)
            
            logging.debug(f"Guessing letter: {guess_letter}")
            return guess_letter
            
        except Exception as e:
            logging.error(f"Error in guess method: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    def prepare_state(self, word_state):
        """Convert API word state to model input"""
        # Remove spaces and convert to list
        clean_state = word_state[::2]
        
        # Convert to model input format
        char_indices = torch.zeros(32, dtype=torch.long)
        for i, char in enumerate(clean_state):
            if char == '_':
                char_indices[i] = 27
            else:
                char_indices[i] = ord(char.lower()) - ord('a') + 1
                
        # Calculate additional features
        word_length = torch.tensor([len(clean_state)], dtype=torch.float32)
        guessed = torch.zeros(26)
        for letter in self.guessed_letters:
            guessed[ord(letter) - ord('a')] = 1
        
        vowel_count = sum(1 for c in clean_state if c in 'aeiou')
        vowel_ratio = torch.tensor([vowel_count / len(clean_state)], dtype=torch.float32)
        remaining_lives = torch.tensor([6 - len([l for l in self.guessed_letters if l not in clean_state])], dtype=torch.float32)
        
        device = next(self.model.parameters()).device
        return (char_indices.to(device), guessed.to(device), word_length.to(device), 
                vowel_ratio.to(device), remaining_lives.to(device))

class time_block:
    """Context manager for timing code blocks"""
    def __init__(self, description):
        self.description = description
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        logging.debug(f"{self.description} took {end - self.start:.3f} seconds")

def play_games(model, access_token, num_games=100, practice=True, verbose=False):
    """Play multiple games using the model"""
    api = ModelHangmanAPI(model=model, access_token=access_token)
    wins = 0
    
    for i in range(num_games):
        if i % 10 == 0:  # Print progress every 10 games
            logging.info(f"\nStarting game {i+1}/{num_games}")
        
        try:
            # Reset guessed letters for new game
            api.guessed_letters = []
            logging.debug("Reset guessed letters")
            
            # Start new game
            logging.debug("Starting new game")
            success = False
            try:
                with time_block(f"Game {i+1}"):
                    success = api.start_game(practice=practice, verbose=False)
            except requests.exceptions.RequestException as e:
                logging.error(f"API request failed: {str(e)}")
                time.sleep(1)  # Wait before retrying
                continue
            
            # Log game result
            if success:
                wins += 1
                if verbose:
                    logging.info(f"Game {i+1}: Won!")
            elif verbose:
                logging.info(f"Game {i+1}: Lost!")
                
            if not practice:
                logging.debug("Waiting between games...")
                time.sleep(0.5)
                
        except Exception as e:
            logging.error(f"Error in game {i+1}: {str(e)}")
            continue
            
    win_rate = wins / num_games
    logging.info(f"\nCompleted {num_games} games")
    logging.info(f"Wins: {wins}")
    logging.info(f"Win rate: {win_rate:.2%}")
    
    return win_rate

def evaluate_models():
    """Evaluate all models in the models directory"""
    log_file = setup_logging()
    logging.info("Starting model evaluation")
    
    # Create results directory if it doesn't exist
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Find all model files
    models_dir = Path('hangman_data/models')
    model_files = list(models_dir.glob('*.pt'))
    
    if not model_files:
        logging.error("No model files found in hangman_data/models/")
        return
    
    results = []
    
    for model_path in model_files:
        try:
            # Extract model type from filename
            model_type = None
            for prefix in ['nv1', 'nv2', 'nv3', 'nv4', 'nv5', 'v1', 'v2']:
                if prefix in model_path.name:
                    model_type = prefix
                    break
            
            if not model_type:
                logging.warning(f"Could not determine model type for {model_path.name}, skipping...")
                continue
                
            logging.info(f"\nEvaluating model: {model_path.name}")
            
            # Initialize device
            device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            
            # Load appropriate model class
            if model_type == 'nv3':
                model = MaryLSTMModel().to(device)
            elif model_type == 'nv4':
                from mary_hangman_nv4 import MaryBiLSTMModel
                model = MaryBiLSTMModel().to(device)
            elif model_type == 'nv5':
                from mary_hangman_nv5 import MaryTransformerModel
                model = MaryTransformerModel().to(device)
            else:
                logging.warning(f"Model type {model_type} not supported yet, skipping...")
                continue
            
            # Load model weights
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            model.eval()
            
            # Play practice games
            logging.info("Playing 100 practice games...")
            win_rate = play_games(model, ACCESS_TOKEN, num_games=100, practice=True, verbose=False)
            
            # Save results
            result = {
                'model_name': model_path.name,
                'model_type': model_type,
                'win_rate': win_rate,
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            }
            results.append(result)
            
            logging.info(f"Win rate: {win_rate:.2%}")
            
        except Exception as e:
            logging.error(f"Error evaluating {model_path.name}: {str(e)}")
            logging.error(traceback.format_exc())
            continue
    
    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'model_evaluation_{timestamp}.txt'
    
    with open(results_file, 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("======================\n\n")
        for result in sorted(results, key=lambda x: x['win_rate'], reverse=True):
            f.write(f"Model: {result['model_name']}\n")
            f.write(f"Type: {result['model_type']}\n")
            f.write(f"Win Rate: {result['win_rate']:.2%}\n")
            f.write(f"Evaluated: {result['timestamp']}\n")
            f.write("-" * 50 + "\n\n")
    
    logging.info(f"\nResults saved to: {results_file}")
    return results

def main():
    try:
        results = evaluate_models()
        if results:
            # Print summary
            logging.info("\nEvaluation Summary:")
            for result in sorted(results, key=lambda x: x['win_rate'], reverse=True):
                logging.info(f"{result['model_name']}: {result['win_rate']:.2%}")
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 