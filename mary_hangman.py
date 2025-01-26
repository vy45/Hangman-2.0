import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
import random
import pickle
import os
from tqdm import tqdm
from datetime import datetime
import sys
from pathlib import Path
import logging

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
BATCH_SIZE = 64
DATA_DIR = 'hangman_data'
VALIDATION_EPISODES = 10000
COMPLETION_EVAL_WORDS = 1000
SIMULATION_CORRECT_GUESS_PROB = 0.5

class MaryLSTMModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=27,  # One-hot encoded letters (26) + blank marker
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.combine = nn.Linear(hidden_dim + 26, hidden_dim)
        self.output = nn.Linear(hidden_dim, 26)
        
    def forward(self, word_state, guessed_letters):
        lstm_out, _ = self.lstm(word_state)
        final_lstm = lstm_out[:, -1]
        combined = torch.cat([final_lstm, guessed_letters], dim=1)
        hidden = F.relu(self.combine(combined))
        return F.softmax(self.output(hidden), dim=-1)

class EarlyStopping:
    def __init__(self, patience=4, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop

def load_and_preprocess_words(filename='words_250000_train.txt'):
    """Load words and split into train/validation sets"""
    with open(filename, 'r') as f:
        words = [word.strip().lower() for word in f.readlines()]
    
    # Filter words
    words = [w for w in words if all(c in ALPHABET for c in w)]
    word_lengths = [len(w) for w in words]
    
    # Count frequency of each length
    length_counts = defaultdict(int)
    for length in word_lengths:
        length_counts[length] += 1
    
    # Filter out words with lengths that appear only once
    min_samples_per_length = 5
    valid_lengths = {length for length, count in length_counts.items() 
                    if count >= min_samples_per_length}
    
    filtered_words = [w for w in words if len(w) in valid_lengths]
    filtered_lengths = [len(w) for w in filtered_words]
    
    try:
        train_words, val_words = train_test_split(
            filtered_words, test_size=0.2, stratify=filtered_lengths, random_state=42
        )
    except ValueError:
        print("Stratified split failed, falling back to random split")
        train_words, val_words = train_test_split(
            words, test_size=0.2, random_state=42
        )
    
    print(f"Train set: {len(train_words)} words")
    print(f"Validation set: {len(val_words)} words")
    
    return train_words, val_words

def simulate_game(word):
    """Simulate a single game and generate states"""
    states = []
    guessed_letters = set()
    word_letters = set(word)
    
    while len(guessed_letters) < 26 and len(word_letters - guessed_letters) > 0:
        # Current word state
        current_state = ''.join([c if c in guessed_letters else '_' for c in word])
        
        # Calculate target distribution (only unguessed correct letters)
        target_dist = {l: 0.0 for l in ALPHABET}
        remaining_letters = word_letters - guessed_letters
        for letter in remaining_letters:
            target_dist[letter] = 1.0
        
        # Normalize distribution
        total = sum(target_dist.values())
        if total > 0:
            target_dist = {k: v/total for k, v in target_dist.items()}
        
        # Store state
        states.append({
            'current_state': current_state,
            'guessed_letters': sorted(list(guessed_letters)),
            'target_distribution': target_dist
        })
        
        # Simulate next guess
        if random.random() < SIMULATION_CORRECT_GUESS_PROB and remaining_letters:
            next_letter = random.choice(list(remaining_letters))
        else:
            available_letters = set(ALPHABET) - guessed_letters
            next_letter = random.choice(list(available_letters))
        
        guessed_letters.add(next_letter)
    
    return states

def prepare_input(state):
    """Convert game state to model input format"""
    # Get actual word length
    word_len = len(state['current_state'])
    
    # Word state: one-hot encoded with 27 dims (26 letters + blank)
    word_state = torch.zeros(word_len, 27)
    for i, char in enumerate(state['current_state']):
        if char == '_':
            word_state[i, 26] = 1  # blank marker
        else:
            word_state[i, ord(char) - ord('a')] = 1
    
    # Guessed letters: 26 dimensions
    guessed_letters = torch.tensor([
        1 if chr(i + ord('a')) in state['guessed_letters'] else 0 
        for i in range(26)
    ], dtype=torch.float32)
    
    return word_state, guessed_letters

def save_data(train_states, val_states, val_words):
    """Save generated data with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f'mary_hangman_{timestamp}'
    
    os.makedirs(DATA_DIR, exist_ok=True)
    logging.info(f"Saving data to {DATA_DIR}")
    
    # Save train states
    train_path = f'{DATA_DIR}/{base_filename}_train.pkl'
    logging.info(f"Saving {len(train_states)} training states to {train_path}")
    with open(train_path, 'wb') as f:
        pickle.dump(train_states, f)
        
    # Save validation states
    val_path = f'{DATA_DIR}/{base_filename}_val.pkl'
    logging.info(f"Saving {len(val_states)} validation states to {val_path}")
    with open(val_path, 'wb') as f:
        pickle.dump(val_states, f)
        
    # Save validation words
    words_path = f'{DATA_DIR}/{base_filename}_val_words.pkl'
    logging.info(f"Saving {len(val_words)} validation words to {words_path}")
    with open(words_path, 'wb') as f:
        pickle.dump(val_words, f)
    
    # Save reference to latest data
    with open(f'{DATA_DIR}/latest_mary.txt', 'w') as f:
        f.write(base_filename)
    
    logging.info(f"Saved all data with base filename: {base_filename}")

def load_data():
    """Load latest saved data"""
    logging.info("Attempting to load saved data...")
    try:
        # Check if data directory exists
        if not os.path.exists(DATA_DIR):
            logging.info(f"Data directory {DATA_DIR} not found")
            return None, None, None
            
        # Check for latest data file
        latest_file = os.path.join(DATA_DIR, 'latest_mary.txt')
        if not os.path.exists(latest_file):
            logging.info(f"No latest data reference found at {latest_file}")
            return None, None, None
            
        with open(latest_file, 'r') as f:
            base_filename = f.read().strip()
        logging.info(f"Found latest data reference: {base_filename}")
            
        # Load train states
        train_path = f'{DATA_DIR}/{base_filename}_train.pkl'
        logging.info(f"Loading training states from {train_path}")
        with open(train_path, 'rb') as f:
            train_states = pickle.load(f)
        logging.info(f"Loaded {len(train_states)} training states")
            
        # Load validation states
        val_path = f'{DATA_DIR}/{base_filename}_val.pkl'
        logging.info(f"Loading validation states from {val_path}")
        with open(val_path, 'rb') as f:
            val_states = pickle.load(f)
        logging.info(f"Loaded {len(val_states)} validation states")
            
        # Load validation words
        words_path = f'{DATA_DIR}/{base_filename}_val_words.pkl'
        logging.info(f"Loading validation words from {words_path}")
        with open(words_path, 'rb') as f:
            val_words = pickle.load(f)
        logging.info(f"Loaded {len(val_words)} validation words")
            
        return train_states, val_states, val_words
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}", exc_info=True)
        return None, None, None

def setup_logging():
    """Setup logging to both file and console"""
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'mary_hangman_{timestamp}.log'
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Setup logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info(f"Logging setup complete. Log file: {log_file}")
    return log_file

def train_model(model, train_states, val_states, val_words, epochs=100):
    """Train model with progress tracking and gradient monitoring"""
    logging.info(f"Starting training for {epochs} epochs")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.3,
        patience=3,
        verbose=True,
        min_lr=1e-6
    )
    early_stopping = EarlyStopping(patience=4, min_delta=0.001)
    
    # Save best model
    best_val_loss = float('inf')
    best_model_path = None
    
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'completion_rate': [],
        'total_time': 0,
        'stopped_epoch': epochs,
        'gradient_norms': [],  # Track gradient norms
        'weight_norms': []     # Track weight norms
    }
    
    start_time = datetime.now()
    
    def save_checkpoint(epoch, val_loss):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f'{DATA_DIR}/mary_model_epoch{epoch}_{timestamp}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics
        }, path)
        return path
    
    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            
            # Training
            batch_iterator = tqdm(
                range(0, len(train_states), BATCH_SIZE),
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=False
            )
            
            for i in batch_iterator:
                batch_states = train_states[i:i + BATCH_SIZE]
                
                # Prepare batch
                word_states = []
                guessed_letters = []
                targets = []
                
                max_len = max(len(state['current_state']) for state in batch_states)
                
                for state in batch_states:
                    word_state, guessed = prepare_input(state)
                    padding = torch.zeros(max_len - len(word_state), 27)
                    padding[:, 26] = 1
                    word_state = torch.cat([word_state, padding])
                    
                    word_states.append(word_state)
                    guessed_letters.append(guessed)
                    targets.append(torch.tensor(
                        list(state['target_distribution'].values()), 
                        dtype=torch.float32
                    ))
                
                word_states = torch.stack(word_states).to(DEVICE)
                guessed_letters = torch.stack(guessed_letters).to(DEVICE)
                targets = torch.stack(targets).to(DEVICE)
                
                # Check for NaN in inputs
                if torch.isnan(word_states).any() or torch.isnan(guessed_letters).any():
                    logging.error("NaN detected in inputs!")
                    continue
                
                # Forward pass
                optimizer.zero_grad()  # Reset gradients
                predictions = model(word_states, guessed_letters)
                
                # Check predictions for NaN
                if torch.isnan(predictions).any():
                    logging.error("NaN detected in predictions!")
                    continue
                
                loss = F.kl_div(predictions.log(), targets, reduction='batchmean')
                
                # Check for NaN loss
                if torch.isnan(loss):
                    logging.error(f"NaN loss detected at epoch {epoch + 1}, batch {num_batches}")
                    continue
                
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Monitor gradients and weights
                grad_norm = 0
                weight_norm = 0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item()
                    weight_norm += param.data.norm().item()
                
                metrics['gradient_norms'].append(grad_norm)
                metrics['weight_norms'].append(weight_norm)
                
                if grad_norm > 10:  # Alert if gradients are too large
                    logging.warning(f"Large gradient norm detected: {grad_norm}")
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                batch_iterator.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}',
                    'grad_norm': f'{grad_norm:.2f}'
                })
            
            # Validation
            val_loss = evaluate_model(model, val_states)
            completion_rate = calculate_completion_rate(model, val_words[:COMPLETION_EVAL_WORDS])
            
            # Store metrics
            train_loss = total_loss / num_batches
            metrics['train_loss'].append(train_loss)
            metrics['val_loss'].append(val_loss)
            metrics['completion_rate'].append(completion_rate)
            
            logging.info(f"\nEpoch {epoch + 1} Results:")
            logging.info(f"Train Loss: {train_loss:.4f}")
            logging.info(f"Val Loss: {val_loss:.4f}")
            logging.info(f"Completion Rate: {completion_rate:.2%}")
            logging.info(f"Average Gradient Norm: {np.mean(metrics['gradient_norms'][-num_batches:]):.2f}")
            logging.info(f"Average Weight Norm: {np.mean(metrics['weight_norms'][-num_batches:]):.2f}")
            
            # Save model if it's the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = save_checkpoint(epoch + 1, val_loss)
                logging.info(f"New best model saved to: {best_model_path}")
            
            # Early stopping check
            if early_stopping(val_loss):
                logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                metrics['stopped_epoch'] = epoch + 1
                break
            
            scheduler.step(val_loss)
            
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}", exc_info=True)
        return metrics
    
    finally:
        total_time = (datetime.now() - start_time).total_seconds()
        metrics['total_time'] = total_time
        logging.info(f"\nTraining completed in {total_time:.2f} seconds")
        
        # Save final model regardless of performance
        final_path = save_checkpoint('final', val_loss)
        logging.info(f"Final model saved to: {final_path}")
    
    return metrics

def evaluate_model(model, val_states):
    """Evaluate model on validation states"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(val_states), BATCH_SIZE):
            batch_states = val_states[i:i + BATCH_SIZE]
            
            # Prepare batch
            word_states = []
            guessed_letters = []
            targets = []
            
            max_len = max(len(state['current_state']) for state in batch_states)
            
            for state in batch_states:
                word_state, guessed = prepare_input(state)
                padding = torch.zeros(max_len - len(word_state), 27)
                padding[:, 26] = 1
                word_state = torch.cat([word_state, padding])
                
                word_states.append(word_state)
                guessed_letters.append(guessed)
                targets.append(torch.tensor(
                    list(state['target_distribution'].values()), 
                    dtype=torch.float32
                ))
            
            word_states = torch.stack(word_states).to(DEVICE)
            guessed_letters = torch.stack(guessed_letters).to(DEVICE)
            targets = torch.stack(targets).to(DEVICE)
            
            predictions = model(word_states, guessed_letters)
            loss = F.kl_div(predictions.log(), targets, reduction='batchmean')
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def calculate_completion_rate(model, words):
    """Calculate word completion rate"""
    model.eval()
    completed = 0
    total = 0
    
    with torch.no_grad():
        for word in tqdm(words, desc="Calculating completion rate", leave=False):
            guessed_letters = set()
            current_state = '_' * len(word)
            word_completed = False
            wrong_guesses = 0
            word_letters = set(word)
            
            while len(guessed_letters) < 26 and wrong_guesses < 6:  # MAX_WRONG_GUESSES = 6
                # Prepare input
                state = {
                    'current_state': current_state,
                    'guessed_letters': sorted(list(guessed_letters))
                }
                
                word_state, guessed = prepare_input(state)
                word_state = word_state.unsqueeze(0).to(DEVICE)
                guessed = guessed.unsqueeze(0).to(DEVICE)
                
                # Get prediction
                predictions = model(word_state, guessed)
                valid_preds = [(i, p.item()) for i, p in enumerate(predictions[0]) 
                              if chr(i + ord('a')) not in guessed_letters]
                next_letter = chr(max(valid_preds, key=lambda x: x[1])[0] + ord('a'))
                
                # Update game state
                guessed_letters.add(next_letter)
                if next_letter not in word_letters:  # Wrong guess
                    wrong_guesses += 1
                
                current_state = ''.join([c if c in guessed_letters else '_' for c in word])
                
                if '_' not in current_state:
                    word_completed = True
                    break
            
            if word_completed and wrong_guesses < 6:  # Only count as completed if we didn't exceed wrong guesses
                completed += 1
            total += 1
            
            # if total % 10 == 0:  # Log progress every 10 words
            #     logging.info(f"Current completion rate: {completed/total:.2%} ({completed}/{total})")
    
    final_rate = completed / total
    # logging.info(f"Final completion rate: {final_rate:.2%} ({completed}/{total})")
    return final_rate

def run_detailed_validation(model, val_words):
    """Run detailed validation statistics on a model"""
    logging.info("\nRunning detailed validation statistics...")
    word_length_stats = defaultdict(lambda: {'total': 0, 'completed': 0, 'total_guesses': 0})
    
    with torch.no_grad():
        for word in tqdm(val_words, desc="Analyzing validation words"):
            guessed_letters = set()
            current_state = '_' * len(word)
            word_completed = False
            wrong_guesses = 0
            word_letters = set(word)
            num_guesses = 0
            
            while len(guessed_letters) < 26 and wrong_guesses < 6:
                state = {
                    'current_state': current_state,
                    'guessed_letters': sorted(list(guessed_letters))
                }
                
                word_state, guessed = prepare_input(state)
                word_state = word_state.unsqueeze(0).to(DEVICE)
                guessed = guessed.unsqueeze(0).to(DEVICE)
                
                predictions = model(word_state, guessed)
                valid_preds = [(i, p.item()) for i, p in enumerate(predictions[0]) 
                             if chr(i + ord('a')) not in guessed_letters]
                next_letter = chr(max(valid_preds, key=lambda x: x[1])[0] + ord('a'))
                
                guessed_letters.add(next_letter)
                num_guesses += 1
                if next_letter not in word_letters:
                    wrong_guesses += 1
                
                current_state = ''.join([c if c in guessed_letters else '_' for c in word])
                
                if '_' not in current_state:
                    word_completed = True
                    break
            
            # Record statistics
            length = len(word)
            word_length_stats[length]['total'] += 1
            word_length_stats[length]['total_guesses'] += num_guesses
            if word_completed and wrong_guesses < 6:
                word_length_stats[length]['completed'] += 1
    
    # Print statistics by word length
    logging.info("\nCompletion rates and average guesses by word length:")
    total_completed = 0
    total_words = 0
    total_guesses = 0
    
    for length in sorted(word_length_stats.keys()):
        stats = word_length_stats[length]
        completion_rate = stats['completed'] / stats['total']
        avg_guesses = stats['total_guesses'] / stats['total']
        total_completed += stats['completed']
        total_words += stats['total']
        total_guesses += stats['total_guesses']
        
        logging.info(
            f"Length {length:2d}: {completion_rate:6.2%} ({stats['completed']:4d}/{stats['total']:<4d}) "
            f"Avg guesses: {avg_guesses:.1f}"
        )
    
    overall_completion = total_completed / total_words
    overall_avg_guesses = total_guesses / total_words
    logging.info(f"\nOverall stats:")
    logging.info(f"Completion rate: {overall_completion:.2%} ({total_completed}/{total_words})")
    logging.info(f"Average guesses: {overall_avg_guesses:.1f}")
    
    return word_length_stats

def evaluate_saved_model(model_path):
    """Load and evaluate a saved model"""
    # Setup logging
    log_file = setup_logging()
    logging.info(f"Evaluating saved model: {model_path}")
    logging.info(f"Using device: {DEVICE}")
    
    try:
        # Load the model
        model = MaryLSTMModel().to(DEVICE)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logging.info("Model loaded successfully")
        
        # Load validation words from latest dataset
        _, _, val_words = load_data()
        if val_words is None:
            logging.error("Could not load validation words from saved dataset")
            return
        
        # Run detailed validation
        run_detailed_validation(model, val_words)
        
    except Exception as e:
        logging.error(f"Evaluation failed with error: {str(e)}", exc_info=True)
        raise
    finally:
        logging.info(f"Log file saved to: {log_file}")

def main():
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting Mary's Hangman training")
    logging.info(f"Using device: {DEVICE}")
    
    try:
        # Load or generate data
        train_states, val_states, val_words = load_data()
        
        if train_states is None:
            logging.info("Generating new dataset...")
            train_words, val_words = load_and_preprocess_words()
            
            logging.info("Simulating training games...")
            train_states = []
            for word in tqdm(train_words):
                train_states.extend(simulate_game(word))
                
            logging.info("Simulating validation games...")
            val_states = []
            for word in tqdm(val_words[:VALIDATION_EPISODES]):
                val_states.extend(simulate_game(word))
                
            save_data(train_states, val_states, val_words)
        
        # Initialize model
        model = MaryLSTMModel().to(DEVICE)
        num_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Model initialized with {num_params:,} parameters")
        
        # Train model
        logging.info("Starting model training...")
        metrics = train_model(model, train_states, val_states, val_words)
        
        # Run detailed validation
        stats = run_detailed_validation(model, val_words)
        metrics['word_length_stats'] = dict(stats)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'{DATA_DIR}/mary_model_{timestamp}.pt'
        metrics_path = f'{DATA_DIR}/mary_metrics_{timestamp}.pkl'
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }, model_path)
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        
        logging.info("\nTraining complete!")
        logging.info(f"Model saved to: {model_path}")
        logging.info(f"Metrics saved to: {metrics_path}")
        logging.info(f"Training completion rate: {metrics['completion_rate'][-1]:.2%}")
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise
    finally:
        logging.info(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # If model path provided, evaluate that model
        evaluate_saved_model(sys.argv[1])
    else:
        # Otherwise run normal training
        main() 