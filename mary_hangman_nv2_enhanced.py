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
    prepare_curriculum_batches, prepare_padded_batch, prepare_length_batches,
    BATCH_SIZE  # Also import BATCH_SIZE constant
)
import torch.nn.functional as F
import os
import pandas as pd
import traceback
import platform
from multiprocessing import Pool, cpu_count
from functools import partial
import random

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
VALIDATION_WORDS = 1000
VALIDATION_STATES = 15000

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

def evaluate_single_word(word, model, device):
    """Evaluate a single word - to be used with multiprocessing"""
    current_state = '_' * len(word)
    guessed_letters = set()
    wrong_guesses = 0
    word_letters = set(word)
    word_history = []
    
    # Ensure model is in eval mode
    model.eval()
    
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
            char_indices = char_indices.unsqueeze(0)
            guessed = guessed.unsqueeze(0)
            length = torch.tensor([len(current_state)], dtype=torch.float32)
            vowel_ratio = vowel_ratio.unsqueeze(0)
            lives = lives.unsqueeze(0)
            
            predictions = model(char_indices, guessed, length, vowel_ratio, lives)
        
        # Get Q-values
        q_values = {chr(i + ord('a')): p.item() for i, p in enumerate(predictions[0])}
        
        # Choose next letter
        valid_preds = [(i, p.item()) for i, p in enumerate(predictions[0])
                      if chr(i + ord('a')) not in guessed_letters]
        next_letter = chr(max(valid_preds, key=lambda x: x[1])[0] + ord('a'))
        
        # Store turn data
        turn_data = {
            'word': word,
            'current_state': current_state,
            'guessed_letters': sorted(list(guessed_letters)),
            'q_values': q_values,
            'chosen_letter': next_letter,
            'vowel_ratio': vowel_ratio.item(),
            'word_length': len(word),
            'remaining_lives': 6 - wrong_guesses,
            'is_correct': next_letter in word_letters
        }
        word_history.append(turn_data)
        
        # Update game state
        guessed_letters.add(next_letter)
        if next_letter in word_letters:
            current_state = ''.join(c if c in guessed_letters else '_' for c in word)
        else:
            wrong_guesses += 1
    
    return word_history

def evaluate_validation_states(model, val_states, epoch):
    """Calculate validation loss on validation states"""
    logging.info("\nStarting Validation Loss Calculation")
    model.eval()
    total_loss = 0
    num_batches = 0
    predictions_data = []
    nan_val_batches = 0
    
    # Prepare validation batches (no curriculum needed)
    with Timer("Validation Batch Preparation"):
        val_batches = prepare_length_batches(val_states)
        logging.info(f"Number of validation batches: {len(val_batches)}")
    
    # Create progress bar
    progress_bar = tqdm(val_batches, desc="Calculating validation loss")
    
    with torch.no_grad():
        for batch in progress_bar:
            word_states = []
            guessed_letters = []
            lengths = []
            vowel_ratios = []
            remaining_lives = []
            targets = []
            
            for state in batch:
                char_indices, guessed, vowel_ratio, lives = prepare_input(state)
                word_states.append(char_indices)
                guessed_letters.append(guessed)
                lengths.append(len(state['current_state']))
                vowel_ratios.append(vowel_ratio)
                remaining_lives.append(lives)
                targets.append(torch.tensor(
                    list(state['target_distribution'].values()), 
                    dtype=torch.float32
                ))
            
            # Stack tensors
            word_states = torch.stack(word_states).to(DEVICE)
            guessed_letters = torch.stack(guessed_letters).to(DEVICE)
            lengths = torch.tensor(lengths).to(DEVICE)
            vowel_ratios = torch.stack(vowel_ratios).to(DEVICE)
            remaining_lives = torch.stack(remaining_lives).to(DEVICE)
            targets = torch.stack(targets).to(DEVICE)
            
            predictions = model(word_states, guessed_letters, lengths, vowel_ratios, remaining_lives)
            
            # Add small epsilon to prevent log(0)
            epsilon = 1e-7
            predictions = torch.clamp(predictions, epsilon, 1.0 - epsilon)
            targets = torch.clamp(targets, epsilon, 1.0 - epsilon)
            
            # Normalize predictions and targets
            predictions = F.normalize(predictions, p=1, dim=1)
            targets = F.normalize(targets, p=1, dim=1)
            
            loss = F.kl_div(predictions.log(), targets, reduction='batchmean')
            
            # Check for NaN loss
            if torch.isnan(loss):
                nan_val_batches += 1
                logging.warning(f"NaN validation loss in batch {num_batches}")
                continue
                
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'nan_batches': nan_val_batches
            })
            
            # Store predictions for analysis
            for i in range(len(predictions)):
                pred_data = {
                    'epoch': epoch,
                    'predicted_probs': predictions[i].cpu().numpy(),
                    'target_probs': targets[i].cpu().numpy(),
                    'loss': loss.item()
                }
                predictions_data.append(pred_data)
    
    # Handle case where all batches had NaN loss
    if num_batches == 0:
        logging.error("All validation batches resulted in NaN loss")
        return float('inf')
        
    avg_val_loss = total_loss / num_batches
    logging.info(f"Validation Loss: {avg_val_loss:.4f} (skipped {nan_val_batches} NaN batches)")
    
    # Save predictions to CSV if we have valid predictions
    if predictions_data:
        df = pd.DataFrame(predictions_data)
        csv_dir = os.path.join(os.getcwd(), 'val_predictions')
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f'val_predictions_epoch_{epoch}.csv')
        df.to_csv(csv_path, index=False)
        logging.info(f"Validation predictions saved to {csv_path}")
    
    return avg_val_loss

def run_detailed_evaluation(model, val_words, max_words=VALIDATION_WORDS, epoch=0):
    """Run detailed evaluation on validation words with comprehensive logging"""
    logging.info("\nStarting Detailed Evaluation")
    
    eval_data = []
    num_words = min(len(val_words), max_words)
    logging.info(f"Evaluating {num_words} words")
    
    # Create progress bar for words
    word_progress = tqdm(range(num_words), desc="Evaluating words")
    
    total_correct = 0
    total_guesses = 0
    
    for word_idx in word_progress:
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
                'vowel_ratio': vowel_ratio.item(),
                'word_length': len(word),
                'remaining_lives': 6 - wrong_guesses,
                'is_correct': next_letter in word_letters
            }
            eval_data.append(turn_data)
            
            # Update statistics
            total_guesses += 1
            if next_letter in word_letters:
                total_correct += 1
            
            # Update progress bar with detailed stats
            word_progress.set_postfix({
                'word': word, 
                'state': current_state, 
                'lives': 6 - wrong_guesses,
                'accuracy': f'{(total_correct/total_guesses):.2%}'
            })
            
            # Update game state
            guessed_letters.add(next_letter)
            if next_letter in word_letters:
                current_state = ''.join(c if c in guessed_letters else '_' for c in word)
            else:
                wrong_guesses += 1
    
    # Log final statistics
    accuracy = total_correct / total_guesses if total_guesses > 0 else 0
    logging.info(f"Detailed Evaluation Results:")
    logging.info(f"Total words evaluated: {num_words}")
    logging.info(f"Total guesses: {total_guesses}")
    logging.info(f"Correct guesses: {total_correct}")
    logging.info(f"Guess accuracy: {accuracy:.2%}")
    
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
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Add gradient clipping value
    max_grad_norm = 1.0
    
    # Add counters for monitoring
    nan_loss_count = 0
    max_nan_losses = 10  # Maximum number of NaN losses before stopping
    
    # Add counter for consecutive NaN validation losses
    consecutive_nan_val = 0
    max_consecutive_nan_val = 3  # Maximum allowed consecutive NaN validation losses
    
    # Curriculum learning parameters
    max_missing_letters = 6  # max number of missing letters
    current_max_missing = 1  # Start with 1 missing letter
    
    # Initial batch preparation
    logging.info("Preparing initial batches...")
    with Timer("Initial Batch Preparation"):
        # Get batches of states using curriculum with fixed batch size
        all_batches = prepare_curriculum_batches(
            all_states, 
            epoch=0, 
            max_missing=current_max_missing
        )
        logging.info(f"Number of batches: {len(all_batches)}")
    
    # Training loop
    for epoch in range(num_epochs):
        if epoch > 0 and current_max_missing < max_missing_letters:
            current_max_missing += 1
            logging.info(f"Advancing curriculum: max missing letters = {current_max_missing}")
            
            # Prepare new batches for current curriculum
            with Timer("Curriculum Batch Preparation"):
                all_batches = prepare_curriculum_batches(
                    all_states, 
                    epoch=epoch,
                    max_missing=current_max_missing if current_max_missing < max_missing_letters else None
                )
                logging.info(f"Number of batches: {len(all_batches)}")
        
        with Timer(f"Epoch {epoch + 1}"):
            model.train()
            total_loss = 0
            num_batches = 0
            
            # Training batches
            progress_bar = tqdm(all_batches, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch_idx, batch_states in enumerate(progress_bar):
                try:
                    optimizer.zero_grad()
                    
                    if batch_idx == 0:
                        prev_weights = {name: param.clone().detach() 
                                      for name, param in model.named_parameters()}
                    
                    # Get padded tensors and move to device in one go
                    padded_batch = prepare_padded_batch(batch_states)
                    char_indices, guessed, lengths, vowel_ratios, remaining_lives, targets = [
                        t.to(DEVICE) for t in padded_batch
                    ]
                    
                    predictions = model(
                        char_indices,
                        guessed,
                        lengths,
                        vowel_ratios,
                        remaining_lives
                    )
                    
                    # Add small epsilon to prevent log(0)
                    epsilon = 1e-7
                    predictions = torch.clamp(predictions, epsilon, 1.0 - epsilon)
                    targets = torch.clamp(targets, epsilon, 1.0 - epsilon)
                    
                    # Normalize predictions and targets
                    predictions = F.normalize(predictions, p=1, dim=1)
                    targets = F.normalize(targets, p=1, dim=1)
                    
                    # Calculate KL divergence loss
                    loss = F.kl_div(predictions.log(), targets, reduction='batchmean')
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        nan_loss_count += 1
                        logging.warning(f"NaN loss detected in epoch {epoch+1}, batch {batch_idx}")
                        logging.warning(f"Total NaN losses: {nan_loss_count}")
                        
                        # Rollback to previous weights if needed
                        if nan_loss_count > 5:  # Too many NaNs
                            logging.warning("Rolling back to start of epoch weights")
                            with torch.no_grad():
                                for name, param in model.named_parameters():
                                    param.copy_(prev_weights[name])
                            break  # Exit batch loop, try next epoch
                            
                        if nan_loss_count >= max_nan_losses:
                            logging.error(f"Too many NaN losses ({nan_loss_count}), stopping training")
                            return  # Exit training
                            
                        continue  # Skip this batch
                    
                    # Backward pass with gradient clipping
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    # Check for NaN gradients
                    has_nan_grad = False
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            has_nan_grad = True
                            logging.warning(f"NaN gradient detected in {name}")
                            break
                    
                    if has_nan_grad:
                        optimizer.zero_grad()  # Clear bad gradients
                        continue  # Skip this batch
                    
                    optimizer.step()
                    
                    # Update metrics
                    total_loss += loss.item()
                    num_batches += 1
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                    })
                    
                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}: {str(e)}")
                    logging.error(traceback.format_exc())
                    continue
            
            if num_batches == 0:
                logging.error("No valid batches in epoch")
                break
                
            avg_loss = total_loss/num_batches
            logging.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            # Validation and model saving
            val_loss = evaluate_validation_states(model, data['val_states'][:VALIDATION_STATES], epoch)
            logging.info(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}")
            
            # Run detailed evaluation every 2 epochs
            if (epoch + 1) % 2 == 0:
                detailed_stats = run_detailed_evaluation(model, data['val_words'], epoch=epoch)
                
                # Calculate completion rate
                successful_words = len([
                    word for word, group in detailed_stats.groupby('word')
                    if group.iloc[-1]['current_state'].count('_') == 0
                ])
                completion_rate = successful_words / len(detailed_stats['word'].unique())
                logging.info(f"Epoch {epoch+1} - Completion Rate: {completion_rate:.4f}")
            
            # Handle NaN validation loss
            if val_loss == float('inf') or torch.isnan(torch.tensor(val_loss)):
                consecutive_nan_val += 1
                logging.warning(f"NaN validation loss detected ({consecutive_nan_val}/{max_consecutive_nan_val})")
                if consecutive_nan_val >= max_consecutive_nan_val:
                    logging.error("Too many consecutive NaN validation losses - stopping training")
                    break
                continue  # Skip to next epoch
            else:
                consecutive_nan_val = 0  # Reset counter on valid loss
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_path = f'best_model_nv2_enhanced_loss{val_loss:.3f}.pth'
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