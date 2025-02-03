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
import traceback
import platform

# Setup device
def setup_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and platform.processor() == 'arm':
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = setup_device()
logging.info(f"Using device: {DEVICE}")

# Constants
QUICK_TEST = True  # Set to True to use only 10% of data
DATA_DIR = 'hangman_data'
VALIDATION_WORDS = 10000

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

def run_detailed_evaluation(model, val_words, max_words=VALIDATION_WORDS, epoch=0):
    """Run detailed evaluation on validation words with comprehensive logging"""
    logging.info("\nStarting Detailed Evaluation")
    
    eval_data = []
    num_words = min(len(val_words), max_words)
    
    # Create progress bar for words
    progress_bar = tqdm(range(num_words), desc="Evaluating words")
    
    for word_idx in progress_bar:
        word = val_words[word_idx]
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
            
            # Update progress bar with current word state
            progress_bar.set_postfix({
                'word': word, 
                'state': current_state, 
                'lives': 6 - wrong_guesses
            })
            
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

def prepare_length_batches(states, batch_size=64):
    """Prepare batches once and keep them on device"""
    # Sort by length first
    sorted_states = sorted(states, key=lambda x: len(x['current_state']))
    
    # Pre-process all states at once
    processed_states = []
    for state in sorted_states:
        # Convert word state to indices
        char_indices = torch.tensor([
            (ord(c) - ord('a') if c != '_' else 26)
            for c in state['current_state']
        ], dtype=torch.long)
        
        # Create guessed letters tensor
        guessed = torch.zeros(26, dtype=torch.long)
        for letter in state['guessed_letters']:
            guessed[ord(letter) - ord('a')] = 1
            
        # Create target tensor
        target = torch.zeros(26, dtype=torch.float32)
        if 'target_distribution' in state:
            for letter, prob in state['target_distribution'].items():
                target[ord(letter) - ord('a')] = prob
                
        # Store processed tensors
        processed_states.append({
            'char_indices': char_indices,
            'guessed': guessed,
            'length': len(state['current_state']),
            'vowel_ratio': state['vowel_ratio'],
            'remaining_lives': state['remaining_lives'],
            'target': target
        })
    
    # Create batches
    batches = []
    current_batch = []
    current_length = None
    
    for state in processed_states:
        if current_length is None:
            current_length = state['length']
            
        if state['length'] != current_length or len(current_batch) == batch_size:
            if current_batch:
                # Convert batch to tensors and move to device
                batch_tensors = {
                    'char_indices': torch.stack([s['char_indices'] for s in current_batch]).to(DEVICE),
                    'guessed': torch.stack([s['guessed'] for s in current_batch]).to(DEVICE),
                    'lengths': torch.tensor([s['length'] for s in current_batch], dtype=torch.float32).to(DEVICE),
                    'vowel_ratios': torch.tensor([s['vowel_ratio'] for s in current_batch], dtype=torch.float32).to(DEVICE),
                    'remaining_lives': torch.tensor([s['remaining_lives'] for s in current_batch], dtype=torch.float32).to(DEVICE),
                    'targets': torch.stack([s['target'] for s in current_batch]).to(DEVICE)
                }
                batches.append(batch_tensors)
            current_batch = []
            current_length = state['length']
        
        current_batch.append(state)
    
    # Add last batch if exists
    if current_batch:
        batch_tensors = {
            'char_indices': torch.stack([s['char_indices'] for s in current_batch]).to(DEVICE),
            'guessed': torch.stack([s['guessed'] for s in current_batch]).to(DEVICE),
            'lengths': torch.tensor([s['length'] for s in current_batch], dtype=torch.float32).to(DEVICE),
            'vowel_ratios': torch.tensor([s['vowel_ratio'] for s in current_batch], dtype=torch.float32).to(DEVICE),
            'remaining_lives': torch.tensor([s['remaining_lives'] for s in current_batch], dtype=torch.float32).to(DEVICE),
            'targets': torch.stack([s['target'] for s in current_batch]).to(DEVICE)
        }
        batches.append(batch_tensors)
    
    return batches

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
    
    # Add gradient clipping value
    max_grad_norm = 1.0
    
    # Prepare batches once before training
    logging.info("Preparing batches...")
    with Timer("Batch Preparation"):
        all_batches = prepare_length_batches(all_states)
        logging.info(f"Number of batches: {len(all_batches)}")
    
    # Training loop
    logging.info("Starting training loop")
    for epoch in range(num_epochs):
        with Timer(f"Epoch {epoch + 1}"):
            model.train()
            total_loss = 0
            num_batches = 0
            
            # Training batches
            progress_bar = tqdm(all_batches, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Forward pass
                    optimizer.zero_grad()
                    predictions = model(
                        batch['char_indices'],
                        batch['guessed'],
                        batch['lengths'],
                        batch['vowel_ratios'],
                        batch['remaining_lives']
                    )
                    
                    loss = F.kl_div(predictions.log(), batch['targets'], reduction='batchmean')
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        # Log the state that caused NaN
                        logging.error(f"NaN loss detected in epoch {epoch+1}, batch {batch_idx}")
                        logging.error(f"Input shapes:")
                        logging.error(f"char_indices: {batch['char_indices'].shape}, values: {batch['char_indices'][:2]}")
                        logging.error(f"guessed: {batch['guessed'].shape}, values: {batch['guessed'][:2]}")
                        logging.error(f"lengths: {batch['lengths'].shape}, values: {batch['lengths'][:2]}")
                        logging.error(f"vowel_ratios: {batch['vowel_ratios'].shape}, values: {batch['vowel_ratios'][:2]}")
                        logging.error(f"remaining_lives: {batch['remaining_lives'].shape}, values: {batch['remaining_lives'][:2]}")
                        logging.error(f"predictions: {predictions[:2]}")
                        logging.error(f"targets: {batch['targets'][:2]}")
                        
                        # Save problematic batch for analysis
                        nan_data = {
                            'char_indices': batch['char_indices'].cpu().numpy(),
                            'guessed': batch['guessed'].cpu().numpy(),
                            'lengths': batch['lengths'].cpu().numpy(),
                            'vowel_ratios': batch['vowel_ratios'].cpu().numpy(),
                            'remaining_lives': batch['remaining_lives'].cpu().numpy(),
                            'targets': batch['targets'].cpu().numpy(),
                            'predictions': predictions.detach().cpu().numpy()
                        }
                        nan_file = f'nan_batch_e{epoch}_b{batch_idx}.pkl'
                        with open(nan_file, 'wb') as f:
                            pickle.dump(nan_data, f)
                        logging.error(f"Problematic batch saved to {nan_file}")
                        
                        raise ValueError("NaN loss detected - training stopped")
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    # Check for NaN gradients
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            logging.error(f"NaN gradient detected in {name}")
                            raise ValueError(f"NaN gradient detected in {name} - training stopped")
                    
                    optimizer.step()
                    
                    # Check for NaN parameters
                    for name, param in model.named_parameters():
                        if torch.isnan(param).any():
                            logging.error(f"NaN parameter detected in {name}")
                            raise ValueError(f"NaN parameter detected in {name} - training stopped")
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}: {str(e)}")
                    logging.error(traceback.format_exc())
                    raise  # Re-raise the exception to stop training
            
            avg_loss = total_loss/num_batches
            logging.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            # Evaluate on validation words
            model.eval()
            
            # Evaluate using HangmanAPI
            # completion_rate = evaluate_model_with_api(model, epoch=epoch)
            # logging.info(f"Epoch {epoch+1} - Completion Rate: {completion_rate:.4f}")
            
            # # Run detailed evaluation every 2 epochs
            # if (epoch + 1) % 2 == 0:
            #     detailed_stats = run_detailed_evaluation(model, data['val_words'][:100], epoch=epoch)
            #     logging.info(f"Epoch {epoch+1} - Detailed Evaluation Stats: {detailed_stats}")
            detailed_stats = run_detailed_evaluation(model, data['val_words'], epoch=epoch)
            
            # Calculate completion rate from detailed evaluation
            successful_words = len([
                word for word, group in detailed_stats.groupby('word')
                if group.iloc[-1]['current_state'].count('_') == 0
            ])
            completion_rate = successful_words / len(detailed_stats['word'].unique())
            logging.info(f"Epoch {epoch+1} - Validation Completion Rate: {completion_rate:.4f}")
            
            # Comment out API evaluation for now
            # completion_rate = evaluate_model_with_api(model, epoch=epoch)
            # logging.info(f"Epoch {epoch+1} - API Completion Rate: {completion_rate:.4f}")
            
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
            
            # Convert word state to indices for embedding
            char_indices = torch.zeros(word_length, dtype=torch.long).to(DEVICE)
            for i, c in enumerate(word_state[::2]):  # Skip spaces
                if c == '_':
                    char_indices[i] = 26  # Mask token
                else:
                    char_indices[i] = ord(c) - ord('a')
            
            # Create guessed letters tensor
            guessed = torch.zeros(26, dtype=torch.long).to(DEVICE)
            for letter in self.guessed_letters:
                guessed[ord(letter) - ord('a')] = 1
            
            # Get model prediction
            with torch.no_grad():
                predictions = self.model(
                    char_indices.unsqueeze(0),  # Add batch dimension
                    guessed.unsqueeze(0),
                    torch.tensor([word_length], dtype=torch.float32).to(DEVICE),
                    torch.tensor([vowel_ratio], dtype=torch.float32).to(DEVICE),
                    torch.tensor([6 - len([l for l in self.guessed_letters if l not in word_state])], 
                               dtype=torch.float32).to(DEVICE)
                )
            
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
            self.game_data['remaining_lives'].append(6 - len([l for l in self.guessed_letters if l not in word_state]))
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