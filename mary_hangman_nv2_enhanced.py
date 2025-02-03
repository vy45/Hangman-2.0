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
import os
import pandas as pd

# Constants
QUICK_TEST = False  # Set to True to use only 10% of data
DATA_DIR = 'hangman_data'


# Modify logging setup
def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure file handler with detailed logging
    file_handler = logging.FileHandler(log_file)
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
    
    logging.info(f"Log file created at: {log_file}")
    return log_file

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

def log_game_to_df(game_data):
    """Convert game data to DataFrame row"""
    return pd.DataFrame([game_data])

def evaluate_model_with_api(model, num_practice_games=100, epoch=0):
    if QUICK_TEST:
        num_practice_games = 10
        
    logging.info(f"Starting API evaluation with {num_practice_games} games")
    
    # Create DataFrame to store all game data
    games_data = []
    
    with Timer("API Evaluation"):
        api = ModelHangmanAPI(model=model, access_token="e8f5c563cddaa094b31cb7c6581e47")
        wins = 0
        
        for game_num in tqdm(range(num_practice_games), desc="Evaluating games"):
            try:
                success = api.start_game(practice=True, verbose=False)
                
                # Get game data from API object
                game_data = api.game_data
                game_data.update({
                    'game_num': game_num,
                    'success': success,
                    'epoch': epoch
                })
                games_data.append(game_data)
                
                if success:
                    wins += 1
            except Exception as e:
                logging.error(f"Error in game {game_num}: {str(e)}")
                continue
    
    # Save games data to CSV
    df = pd.DataFrame(games_data)
    csv_dir = os.path.join(os.getcwd(), 'game_logs')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f'api_games_epoch_{epoch}.csv')
    df.to_csv(csv_path, index=False)
    logging.info(f"API games data saved to {csv_path}")
    
    final_rate = wins / num_practice_games
    logging.info(f"Evaluation complete. Final win rate: {final_rate:.2%}")
    return final_rate

def run_detailed_evaluation(model, val_words, max_words=5, epoch=0):
    """Run detailed evaluation on validation words with comprehensive logging"""
    logging.info("\nStarting Detailed Evaluation")
    
    eval_data = []
    
    for word_idx, word in enumerate(val_words):
        if word_idx >= max_words:
            break
            
        current_state = '_' * len(word)
        guessed_letters = set()
        wrong_guesses = 0
        word_letters = set(word)
        
        while wrong_guesses < 6 and '_' in current_state:
            # Calculate features
            known_vowels = sum(1 for c in word if c in 'aeiou' and c in guessed_letters)
            vowel_ratio = known_vowels / len(word)
            
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
            
            # Get Q-values
            q_values = {chr(i + ord('a')): p.item() for i, p in enumerate(predictions[0])}
            
            # Choose next letter
            valid_preds = [(i, p.item()) for i, p in enumerate(predictions[0])
                          if chr(i + ord('a')) not in guessed_letters]
            next_letter = chr(max(valid_preds, key=lambda x: x[1])[0] + ord('a'))
            
            # Store turn data
            turn_data = {
                'epoch': epoch,
                'word': word,
                'word_idx': word_idx,
                'current_state': current_state,
                'guessed_letters': sorted(list(guessed_letters)),
                'q_values': q_values,
                'chosen_letter': next_letter,
                'vowel_ratio': vowel_ratio,
                'word_length': len(word),
                'remaining_lives': 6 - wrong_guesses,
                'is_correct': next_letter in word_letters
            }
            eval_data.append(turn_data)
            
            # Update game state
            guessed_letters.add(next_letter)
            if next_letter in word_letters:
                current_state = ''.join(c if c in guessed_letters else '_' for c in word)
            else:
                wrong_guesses += 1
    
    # Save evaluation data to CSV
    df = pd.DataFrame(eval_data)
    csv_dir = os.path.join(os.getcwd(), 'game_logs')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f'detailed_eval_epoch_{epoch}.csv')
    df.to_csv(csv_path, index=False)
    logging.info(f"Detailed evaluation data saved to {csv_path}")
    
    return df

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
            completion_rate = evaluate_model_with_api(model, epoch=epoch)
            logging.info(f"Epoch {epoch+1} - Completion Rate: {completion_rate:.4f}")
            
            # Run detailed evaluation every 2 epochs
            if (epoch + 1) % 2 == 0:
                detailed_stats = run_detailed_evaluation(model, data['val_words'][:100], epoch=epoch)
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
        self.game_data = {
            'states': [],
            'guessed_letters': [],
            'q_values': [],
            'chosen_letters': [],
            'vowel_ratios': [],
            'word_lengths': [],
            'remaining_lives': [],
            'correct_guesses': []
        }
        logging.debug("ModelHangmanAPI initialized")
    
    def guess(self, word_state):
        """Override the guess method to use our model"""
        try:
            # Calculate vowel ratio
            known_vowels = sum(1 for c in word_state[::2] if c in 'aeiou' and c != '_')
            vowel_ratio = known_vowels / (len(word_state) // 2)
            word_length = len(word_state) // 2
            
            # Prepare state dictionary
            state = {
                'current_state': word_state[::2],  # Remove spaces
                'guessed_letters': sorted(list(self.guessed_letters)),
                'vowel_ratio': vowel_ratio,
                'remaining_lives': 6 - len([l for l in self.guessed_letters if l not in word_state])
            }
            
            # Get model prediction
            with torch.no_grad():
                char_indices, guessed, vowel_ratio, lives = prepare_input(state)
                char_indices = char_indices.unsqueeze(0).to(DEVICE)
                guessed = guessed.unsqueeze(0).to(DEVICE)
                length = torch.tensor([len(state['current_state'])], dtype=torch.float32).to(DEVICE)
                vowel_ratio = vowel_ratio.unsqueeze(0).to(DEVICE)
                lives = lives.unsqueeze(0).to(DEVICE)
                
                predictions = self.model(char_indices, guessed, length, vowel_ratio, lives)
            
            # Get Q-values for all letters
            q_values = {chr(i + ord('a')): p.item() for i, p in enumerate(predictions[0])}
            
            # Choose next letter (highest Q-value for unguessed letter)
            valid_preds = [(i, p.item()) for i, p in enumerate(predictions[0])
                          if chr(i + ord('a')) not in self.guessed_letters]
            next_letter = chr(max(valid_preds, key=lambda x: x[1])[0] + ord('a'))
            
            # Store game state
            self.game_data['states'].append(word_state)
            self.game_data['guessed_letters'].append(list(self.guessed_letters))
            self.game_data['q_values'].append(q_values)
            self.game_data['chosen_letters'].append(next_letter)
            self.game_data['vowel_ratios'].append(vowel_ratio)
            self.game_data['word_lengths'].append(word_length)
            self.game_data['remaining_lives'].append(state['remaining_lives'])
            self.game_data['correct_guesses'].append(next_letter in word_state)
            
            return next_letter
            
        except Exception as e:
            logging.error(f"Error in model guess: {str(e)}")
            # Fallback to parent class guess if model fails
            return super().guess(word_state)

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        logging.exception("Training failed with error")
        raise 