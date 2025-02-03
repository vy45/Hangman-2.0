import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from collections import Counter, defaultdict
import string
import logging
import time
from hangmanAPI import HangmanAPI
from pathlib import Path
from tqdm import tqdm
from mary_hangman_nv2 import (
    MaryLSTMModel, prepare_input, DEVICE, 
    prepare_length_batches, prepare_padded_batch
)
import torch.nn.functional as F

# Constants
QUICK_TEST = True  # Set to True to use only 10% of data
DATA_DIR = 'hangman_data'

# Modify logging setup
def setup_logging():
    """Setup logging configuration"""
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure file handler with detailed logging
    file_handler = logging.FileHandler('training.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    ))
    
    # Configure console handler with minimal logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Configure root logger
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(console_handler)

# Call setup_logging at the start
setup_logging()

class Timer:
    def __init__(self, name):
        self.name = name
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.end = time.time()
        self.duration = self.end - self.start
        logging.info(f"{self.name} took {self.duration:.2f} seconds")

def evaluate_model_with_api(model, num_practice_games=100):
    if QUICK_TEST:
        num_practice_games = 10
        
    logging.info(f"Starting API evaluation with {num_practice_games} games")
    with Timer("API Evaluation"):
        api = ModelHangmanAPI(model=model, access_token="e8f5c563cddaa094b31cb7c6581e47")
        wins = 0
        
        for game_num in tqdm(range(num_practice_games), desc="Evaluating games"):
            try:
                # Reset guessed letters for new game
                api.guessed_letters = []
                logging.debug(f"\nStarting Game {game_num + 1}")
                
                # Start new game
                success = api.start_game(practice=True, verbose=False)
                word_state = api.word_state
                logging.debug(f"Initial word state: {word_state}")
                
                while api.remaining_lives > 0 and '_' in word_state:
                    # Calculate vowel ratio
                    known_vowels = sum(1 for c in word_state if c in 'aeiou' and c != '_')
                    vowel_ratio = known_vowels / len(word_state)
                    
                    # Print current game state (debug level)
                    logging.debug(f"\nCurrent state: {word_state}")
                    logging.debug(f"Guessed letters: {sorted(api.guessed_letters)}")
                    logging.debug(f"Remaining lives: {api.remaining_lives}")
                    logging.debug(f"Vowel ratio: {vowel_ratio:.3f}")
                    
                    # Prepare input
                    state = {
                        'current_state': word_state,
                        'guessed_letters': sorted(list(api.guessed_letters)),
                        'vowel_ratio': vowel_ratio,
                        'remaining_lives': api.remaining_lives
                    }
                    
                    # Get model prediction
                    char_indices, guessed, vowel_ratio, lives = prepare_input(state)
                    char_indices = char_indices.unsqueeze(0).to(DEVICE)
                    guessed = guessed.unsqueeze(0).to(DEVICE)
                    length = torch.tensor([len(word_state)], dtype=torch.float32).to(DEVICE)
                    vowel_ratio = vowel_ratio.unsqueeze(0).to(DEVICE)
                    lives = lives.unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        predictions = model(char_indices, guessed, length, vowel_ratio, lives)
                    
                    # Log Q-values
                    q_values = {chr(i + ord('a')): f"{p.item():.4f}" 
                              for i, p in enumerate(predictions[0])}
                    logging.debug("Q-values for unguessed letters:")
                    for letter, value in sorted(q_values.items()):
                        if letter not in api.guessed_letters:
                            logging.debug(f"  {letter}: {value}")
                    
                    # Choose next letter
                    valid_preds = [(i, p.item()) for i, p in enumerate(predictions[0])
                                  if chr(i + ord('a')) not in api.guessed_letters]
                    next_letter = chr(max(valid_preds, key=lambda x: x[1])[0] + ord('a'))
                    logging.debug(f"Chosen letter: {next_letter}")
                    
                    # Make guess
                    api.guessed_letters.append(next_letter)
                    result = api.make_guess(next_letter)
                    word_state = result['word']
                    
                    if '_' not in word_state:  # Win condition
                        wins += 1
                        logging.debug(f"\nGame {game_num + 1} WON! Final word: {word_state}")
                        break
                
                if '_' in word_state:
                    logging.debug(f"\nGame {game_num + 1} LOST. Final word state: {word_state}")
                
                logging.debug("-" * 50)
                
            except Exception as e:
                logging.error(f"Error in game {game_num}: {str(e)}")
                continue
    
    final_rate = wins / num_practice_games
    logging.info(f"Evaluation complete. Final win rate: {final_rate:.2%}")
    return final_rate

def run_detailed_evaluation(model, val_words, max_words=5):
    """Run detailed evaluation on validation words with comprehensive logging"""
    logging.info("\nStarting Detailed Evaluation")
    stats = {
        'total_games': 0,
        'wins': 0,
        'total_guesses': 0,
        'correct_guesses': 0,
        'wrong_guesses': 0,
        'by_length': defaultdict(lambda: {'total': 0, 'wins': 0, 'guesses': 0})
    }
    
    for word_idx, word in enumerate(val_words):
        if word_idx >= max_words:
            break
            
        logging.info(f"\n{'='*50}")
        logging.info(f"Evaluating word {word_idx + 1}: {word}")
        
        current_state = '_' * len(word)
        guessed_letters = set()
        wrong_guesses = 0
        total_guesses = 0
        word_letters = set(word)
        
        while wrong_guesses < 6 and '_' in current_state:
            # Calculate vowel ratio
            known_vowels = sum(1 for c in word if c in 'aeiou' and c in guessed_letters)
            vowel_ratio = known_vowels / len(word)
            
            # Print current game state
            logging.info(f"\nTurn {total_guesses + 1}")
            logging.info(f"Current state: {current_state}")
            logging.info(f"Guessed letters: {sorted(guessed_letters)}")
            logging.info(f"Remaining lives: {6 - wrong_guesses}")
            logging.info(f"Vowel ratio: {vowel_ratio:.3f}")
            
            # Prepare input
            state = {
                'current_state': current_state,
                'guessed_letters': sorted(list(guessed_letters)),
                'vowel_ratio': vowel_ratio,
                'remaining_lives': 6 - wrong_guesses
            }
            
            # Get model prediction
            with torch.no_grad():
                char_indices, guessed, vowel_ratio, lives = prepare_input(state)
                char_indices = char_indices.unsqueeze(0).to(DEVICE)
                guessed = guessed.unsqueeze(0).to(DEVICE)
                length = torch.tensor([len(current_state)], dtype=torch.float32).to(DEVICE)
                vowel_ratio = vowel_ratio.unsqueeze(0).to(DEVICE)
                lives = lives.unsqueeze(0).to(DEVICE)
                
                predictions = model(char_indices, guessed, length, vowel_ratio, lives)
            
            # Log Q-values
            q_values = {chr(i + ord('a')): f"{p.item():.4f}" 
                       for i, p in enumerate(predictions[0])}
            logging.info("\nQ-values for unguessed letters:")
            for letter, value in sorted(q_values.items()):
                if letter not in guessed_letters:
                    logging.info(f"  {letter}: {value}")
            
            # Choose next letter
            valid_preds = [(i, p.item()) for i, p in enumerate(predictions[0])
                          if chr(i + ord('a')) not in guessed_letters]
            next_letter = chr(max(valid_preds, key=lambda x: x[1])[0] + ord('a'))
            
            logging.info(f"\nChosen letter: {next_letter}")
            
            # Update game state
            guessed_letters.add(next_letter)
            total_guesses += 1
            
            if next_letter in word_letters:
                stats['correct_guesses'] += 1
                current_state = ''.join(c if c in guessed_letters else '_' for c in word)
                logging.info(f"Correct guess! New state: {current_state}")
            else:
                wrong_guesses += 1
                stats['wrong_guesses'] += 1
                logging.info(f"Wrong guess. Lives remaining: {6 - wrong_guesses}")
        
        # Update statistics
        stats['total_games'] += 1
        stats['total_guesses'] += total_guesses
        stats['by_length'][len(word)]['total'] += 1
        stats['by_length'][len(word)]['guesses'] += total_guesses
        
        if '_' not in current_state:
            stats['wins'] += 1
            stats['by_length'][len(word)]['wins'] += 1
            logging.info(f"\nWord completed: {word}")
        else:
            logging.info(f"\nFailed to complete word. Final state: {current_state}")
    
    # Print summary statistics
    logging.info("\n" + "="*50)
    logging.info("Detailed Evaluation Summary:")
    logging.info(f"Total words evaluated: {stats['total_games']}")
    logging.info(f"Win rate: {stats['wins']/stats['total_games']:.2%}")
    logging.info(f"Average guesses per word: {stats['total_guesses']/stats['total_games']:.2f}")
    logging.info(f"Guess accuracy: {stats['correct_guesses']/(stats['correct_guesses'] + stats['wrong_guesses']):.2%}")
    
    return stats

def train_model():
    logging.info("Starting training process")
    logging.info(f"QUICK_TEST mode: {QUICK_TEST}")
    
    # Load dataset
    with Timer("Dataset Loading"):
        data_dir = Path(DATA_DIR)
        data_dir.mkdir(exist_ok=True)

        dataset_files = list(data_dir.glob('hangman_states_nv2_*.pkl'))
        if not dataset_files:
            raise ValueError("No dataset files found!")
        
        latest_file = max(dataset_files, key=lambda p: p.stat().st_mtime)
        logging.info(f"Loading dataset: {latest_file}")
        
        with open(latest_file, 'rb') as f:
            data = pickle.load(f)
    
    # Combine train and validation sets
    with Timer("Dataset Preparation"):
        all_states = data['train_states'] + data['val_states']
        if QUICK_TEST:
            # Use only 10% of data for quick testing
            all_states = all_states[:len(all_states)//10]
        logging.info(f"Total states after combining: {len(all_states)}")
    
    # Initialize model with new dimensions
    with Timer("Model Initialization"):
        model = MaryLSTMModel(hidden_dim=256, embedding_dim=32).to(DEVICE)
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10 if QUICK_TEST else 100
    best_completion_rate = 0
    patience = 5
    patience_counter = 0
    
    # Training loop
    logging.info("Starting training loop")
    for epoch in range(num_epochs):
        with Timer(f"Epoch {epoch + 1}"):
            model.train()
            total_loss = 0
            num_batches = 0
            
            # Prepare batches for this epoch
            with Timer("Batch Preparation"):
                batches = prepare_length_batches(all_states)
                logging.info(f"Number of batches: {len(batches)}")
            
            # Training batches
            progress_bar = tqdm(batches, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch in progress_bar:
                # Prepare padded batch
                word_states, guessed_letters, lengths, vowel_ratios, remaining_lives, targets = prepare_padded_batch(batch)
                
                # Forward pass
                optimizer.zero_grad()
                predictions = model(word_states, guessed_letters, lengths, vowel_ratios, remaining_lives)
                
                loss = F.kl_div(predictions.log(), targets, reduction='batchmean')
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss/num_batches
            logging.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            # Evaluate using HangmanAPI
            model.eval()
            completion_rate = evaluate_model_with_api(model)
            logging.info(f"Epoch {epoch+1} - Completion Rate: {completion_rate:.4f}")
            
            # Run detailed evaluation every 2 epochs
            if (epoch + 1) % 2 == 0:
                detailed_stats = run_detailed_evaluation(model, data['val_words'][:100])
                logging.info(f"Epoch {epoch+1} - Detailed Evaluation Stats: {detailed_stats}")
            
            # Early stopping
            if completion_rate > best_completion_rate:
                best_completion_rate = completion_rate
                model_path = f'best_model_nv2_enhanced_rate{completion_rate:.3f}.pth'
                torch.save(model.state_dict(), model_path)
                logging.info(f"New best model saved: {model_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info("Early stopping triggered")
                    break

class ModelHangmanAPI(HangmanAPI):
    """Extended HangmanAPI that uses ML model for guessing"""
    
    def __init__(self, model, access_token=None):
        super().__init__(access_token=access_token)
        self.model = model
        self.guessed_letters = []
        self.word_state = None
        self.remaining_lives = 6
        logging.debug("ModelHangmanAPI initialized")
    
    def start_game(self, practice=True, verbose=False):
        """Start a new game"""
        self.guessed_letters = []
        self.remaining_lives = 6
        result = self.start_game(practice_game=practice)
        self.word_state = result['word']
        return result
    
    def make_guess(self, letter):
        """Make a guess and update game state"""
        result = self.guess(letter)
        self.word_state = result['word']
        self.remaining_lives = result['remaining_lives']
        return result

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        logging.exception("Training failed with error")
        raise 