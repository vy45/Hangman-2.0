import json
import requests
import string
import time
import torch
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

# Import your model classes
from mary_hangman_nv3 import MaryLSTMModel
# from mary_hangman_nv4 import MaryBiLSTMModel  # Only uncomment if needed
# from mary_hangman_nv5 import MaryTransformerModel  # Only uncomment if needed

# Import the HangmanAPI class
from hangmanAPI import HangmanAPI

# Setup logging
def setup_logging():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'logs'
    Path(log_dir).mkdir(exist_ok=True)
    log_file = f'{log_dir}/hangman_player_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

class ModelHangmanAPI(HangmanAPI):
    """Extended HangmanAPI that uses ML model for guessing"""
    
    def __init__(self, model, access_token=None):
        super().__init__(access_token=access_token)
        self.model = model
        self.guessed_letters = []
        
    def guess(self, word_state):
        """Override the guess method to use our model"""
        # Prepare input
        char_indices, guessed, word_length, vowel_ratio, remaining_lives = self.prepare_state(word_state)
        
        # Get model prediction
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
        
        return guess_letter
    
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

def play_games(model, access_token, num_games=100, practice=True, verbose=False):
    """Play multiple games using the model"""
    api = ModelHangmanAPI(model=model, access_token=access_token)
    wins = 0
    
    for i in range(num_games):
        if i % 10 == 0:  # Print progress every 10 games regardless of verbose
            logging.info(f"\nStarting game {i+1}/{num_games}")
        
        # Reset guessed letters for new game
        api.guessed_letters = []
        
        # Play game with minimal output
        if verbose:
            logging.info(f"\nGame {i+1}:")
        success = api.start_game(practice=practice, verbose=False)  # Set verbose=False here
        if success:
            wins += 1
            if verbose:
                logging.info("Won!")
        elif verbose:
            logging.info("Lost!")
            
        if not practice:
            time.sleep(0.5)
            
    win_rate = wins / num_games
    logging.info(f"\nCompleted {num_games} games")
    logging.info(f"Wins: {wins}")
    logging.info(f"Win rate: {win_rate:.2%}")
    
    return win_rate

def main():
    log_file = setup_logging()
    
    # Configuration
    MODEL_PATH = 'hangman_data/mary_model_nv3_epochfinal_20250201_173921.pt'
    MODEL_TYPE = 'nv3'
    ACCESS_TOKEN = 'e8f5c563cddaa094b31cb7c6581e47'
    
    # Number of games to play
    PRACTICE_GAMES = 100
    RECORD_GAMES = 0  # Set to positive number when ready to record games
    
    try:
        # Initialize device
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        logging.info(f"Using device: {device}")
        
        # Load model
        if MODEL_TYPE == 'nv3':
            model = MaryLSTMModel().to(device)
        elif MODEL_TYPE == 'nv4':
            raise NotImplementedError("NV4 model not currently imported")
        elif MODEL_TYPE == 'nv5':
            raise NotImplementedError("NV5 model not currently imported")
        else:
            raise ValueError(f"Unknown model type: {MODEL_TYPE}")
            
        # Load weights with safety flag
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.eval()
        logging.info(f"Loaded {MODEL_TYPE} model from {MODEL_PATH}")
        
        # Play practice games
        if PRACTICE_GAMES > 0:
            logging.info(f"\nPlaying {PRACTICE_GAMES} practice games...")
            play_games(model, ACCESS_TOKEN, PRACTICE_GAMES, practice=True, verbose=True)
        
        # Play recorded games
        if RECORD_GAMES > 0:
            logging.info(f"\nPlaying {RECORD_GAMES} recorded games...")
            play_games(model, ACCESS_TOKEN, RECORD_GAMES, practice=False, verbose=True)
            
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise
    finally:
        logging.info(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    main() 