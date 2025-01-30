import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from datetime import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import random
from generate_hangman_data import (
    load_and_split_data, 
    build_ngram_dictionary, 
    generate_dataset,
    simulate_game_states
)
import argparse
from collections import defaultdict
import json
import sys
import traceback

# Constants
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
LETTER_TO_IDX = {letter: idx for idx, letter in enumerate(ALPHABET)}
HIDDEN_SIZE = 128
EMBEDDING_DIM = 32
NUM_LAYERS = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DATA_DIR = 'hangman_data'

# Additional Constants from mary_hangman
VALIDATION_EPISODES = 1000
MAX_GUESSES = 6
SAVE_METRICS = True
METRICS_DIR = 'metrics'

class HangmanModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim, num_layers=2):
        super(HangmanModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer for characters (add padding_idx)
        self.char_embedding = nn.Embedding(
            input_size, 
            embedding_dim,
            padding_idx=input_size-1  # Use last index for padding
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Additional features size (guessed letters + vowel ratio)
        additional_features = len(ALPHABET) + 1
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + additional_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, word_state, guessed_letters, vowel_ratio):
        # Create padding mask
        padding_mask = (word_state != len(ALPHABET))
        
        # Embed characters
        embedded = self.char_embedding(word_state)
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths=padding_mask.sum(1).cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # LSTM
        packed_output, _ = self.lstm(packed)
        
        # Unpack sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True
        )
        
        # Get last non-padded output for each sequence
        last_indices = padding_mask.sum(1) - 1
        batch_indices = torch.arange(lstm_out.size(0), device=lstm_out.device)
        lstm_last = lstm_out[batch_indices, last_indices]
        
        # Concatenate with additional features
        combined = torch.cat([lstm_last, guessed_letters, vowel_ratio.unsqueeze(1)], dim=1)
        
        # Final layers
        output = self.fc(combined)
        return output

class HangmanDataset(Dataset):
    def __init__(self, states):
        self.states = states
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = self.states[idx]
        
        # Convert current state to indices
        word_state = torch.tensor([
            LETTER_TO_IDX[c] if c in ALPHABET else len(ALPHABET) 
            for c in state['current_state']
        ], dtype=torch.long)
        
        # Convert guessed letters to one-hot
        guessed = torch.zeros(len(ALPHABET))
        for letter in state['guessed_letters']:
            if letter in LETTER_TO_IDX:
                guessed[LETTER_TO_IDX[letter]] = 1
        
        # Get vowel ratio
        vowel_ratio = torch.tensor(state['vowel_ratio'], dtype=torch.float32)
        
        # Get target distribution
        target = torch.tensor([
            state['target_distribution'][letter] 
            for letter in ALPHABET
        ], dtype=torch.float32)
        
        return word_state, guessed, vowel_ratio, target

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def load_or_generate_data(force_new=False):
    """Load existing dataset or generate new one"""
    # Check for existing datasets
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(exist_ok=True)
    
    dataset_files = list(data_dir.glob('hangman_states_*.pkl'))
    if dataset_files and not force_new:  # Only load if exists and not forcing new
        # Load most recent dataset
        latest_file = max(dataset_files, key=lambda p: p.stat().st_mtime)
        logging.info(f"Loading existing dataset: {latest_file}")
        with open(latest_file, 'rb') as f:
            data = pickle.load(f)
        return data['train_states'], data['val_states'], data['val_words']
    
    # Generate new dataset
    logging.info(f"{'Forcing new dataset generation...' if force_new else 'No existing dataset found. Generating new dataset...'}")
    
    # Load and split words
    training_words, validation_words = load_and_split_data()
    
    # Build n-gram dictionary
    with open('words_250000_train.txt', 'r') as f:
        words = [w.strip().lower() for w in f.readlines()]
    ngram_dict = build_ngram_dictionary(words)
    
    # Generate datasets
    train_states = generate_dataset(training_words, ngram_dict, is_validation=False, train_words=training_words)
    val_states = generate_dataset(validation_words, ngram_dict, is_validation=True, train_words=training_words)
    
    # Save dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = data_dir / f'hangman_states_{timestamp}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump({
            'train_states': train_states,
            'val_states': val_states,
            'val_words': validation_words
        }, f)
    
    logging.info(f"Dataset saved to {save_path}")
    return train_states, val_states, validation_words

def calculate_metrics(model, val_loader, device):
    """Calculate detailed validation metrics"""
    model.eval()
    metrics = defaultdict(float)
    total_batches = 0
    
    with torch.no_grad():
        for word_states, guessed, vowel_ratios, targets in val_loader:
            word_states = word_states.to(device)
            guessed = guessed.to(device)
            vowel_ratios = vowel_ratios.to(device)
            targets = targets.to(device)
            
            outputs = model(word_states, guessed, vowel_ratios)
            
            # Calculate various metrics
            pred_letters = outputs.argmax(dim=1)
            target_letters = targets.argmax(dim=1)
            
            # Accuracy
            correct = (pred_letters == target_letters).float()
            metrics['accuracy'] += correct.mean().item()
            
            # Top-k accuracy
            for k in [3, 5]:
                _, top_k = outputs.topk(k, dim=1)
                correct_k = torch.zeros_like(pred_letters, dtype=torch.float32)
                for i, target in enumerate(target_letters):
                    correct_k[i] = target in top_k[i]
                metrics[f'top_{k}_accuracy'] += correct_k.mean().item()
            
            total_batches += 1
    
    # Average metrics
    for key in metrics:
        metrics[key] /= total_batches
    
    return dict(metrics)

def save_metrics(metrics, model_path):
    """Save metrics to file"""
    metrics_dir = Path(METRICS_DIR)
    metrics_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_path = metrics_dir / f'metrics_{timestamp}.json'
    
    metrics_data = {
        'model_path': str(model_path),
        'timestamp': timestamp,
        'metrics': metrics
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    logging.info(f"Metrics saved to {metrics_path}")

def evaluate_model(model_path, val_loader, device):
    """Evaluate a saved model"""
    try:
        model = HangmanModel(
            input_size=len(ALPHABET) + 1,
            hidden_size=HIDDEN_SIZE,
            output_size=len(ALPHABET),
            embedding_dim=EMBEDDING_DIM,
            num_layers=NUM_LAYERS
        ).to(device)
        
        model.load_state_dict(torch.load(model_path))
        logging.info(f"Loaded model from {model_path}")
        
        metrics = calculate_metrics(model, val_loader, device)
        
        logging.info("\nValidation Metrics:")
        for key, value in metrics.items():
            logging.info(f"{key}: {value:.4f}")
        
        if SAVE_METRICS:
            save_metrics(metrics, model_path)
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error evaluating model: {str(e)}")
        logging.error(traceback.format_exc())
        return None

def train_model(model, train_loader, val_loader, num_epochs, device):
    """Train the model with improved validation and error handling"""
    try:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        best_val_loss = float('inf')
        best_model_path = None
        patience = 5
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0
            train_batches = 0
            
            try:
                for word_states, guessed, vowel_ratios, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                    word_states = word_states.to(device)
                    guessed = guessed.to(device)
                    vowel_ratios = vowel_ratios.to(device)
                    targets = targets.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(word_states, guessed, vowel_ratios)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                
                avg_train_loss = train_loss / train_batches
                
                # Validation
                model.eval()
                val_loss = 0
                val_batches = 0
                
                with torch.no_grad():
                    for word_states, guessed, vowel_ratios, targets in val_loader:
                        word_states = word_states.to(device)
                        guessed = guessed.to(device)
                        vowel_ratios = vowel_ratios.to(device)
                        targets = targets.to(device)
                        
                        outputs = model(word_states, guessed, vowel_ratios)
                        loss = criterion(outputs, targets)
                        
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                
                logging.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
                
                # Save best model with patience
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    if best_model_path:
                        os.remove(best_model_path)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    best_model_path = f'models/hangman_model_{timestamp}.pt'
                    torch.save(model.state_dict(), best_model_path)
                    logging.info(f"New best model saved to {best_model_path}")
                    
                    # Calculate and save metrics for best model
                    metrics = calculate_metrics(model, val_loader, device)
                    if SAVE_METRICS:
                        save_metrics(metrics, best_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(f"Early stopping after {patience} epochs without improvement")
                        break
                
            except Exception as e:
                logging.error(f"Error in epoch {epoch+1}: {str(e)}")
                logging.error(traceback.format_exc())
                continue
        
        return best_model_path
        
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        logging.error(traceback.format_exc())
        return None

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    # Find max length in the batch
    max_len = max(len(item[0]) for item in batch)
    
    # Pad sequences to max length
    word_states = [
        torch.nn.functional.pad(item[0], (0, max_len - len(item[0])), value=len(ALPHABET))
        for item in batch
    ]
    
    # Stack all tensors
    word_states = torch.stack(word_states)
    guessed = torch.stack([item[1] for item in batch])
    vowel_ratios = torch.stack([item[2] for item in batch])
    targets = torch.stack([item[3] for item in batch])
    
    return word_states, guessed, vowel_ratios, targets

def main():
    try:
        # Add argument parsing
        parser = argparse.ArgumentParser(description='Train Hangman Model')
        parser.add_argument('--force-new-data', action='store_true', 
                           help='Force generation of new dataset even if one exists')
        parser.add_argument('--evaluate', type=str,
                           help='Path to model for evaluation')
        args = parser.parse_args()
        
        setup_logging()
        
        # Create necessary directories
        for directory in ['models', METRICS_DIR]:
            Path(directory).mkdir(exist_ok=True)
        
        # Load or generate data with force flag
        train_states, val_states, val_words = load_or_generate_data(force_new=args.force_new_data)
        
        # Create datasets and dataloaders
        train_dataset = HangmanDataset(train_states)
        val_dataset = HangmanDataset(val_states)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if args.evaluate:
            # Evaluate existing model
            evaluate_model(args.evaluate, val_loader, device)
        else:
            # Train new model
            model = HangmanModel(
                input_size=len(ALPHABET) + 1,
                hidden_size=HIDDEN_SIZE,
                output_size=len(ALPHABET),
                embedding_dim=EMBEDDING_DIM,
                num_layers=NUM_LAYERS
            ).to(device)
            
            best_model_path = train_model(model, train_loader, val_loader, NUM_EPOCHS, device)
            
            if best_model_path:
                logging.info("\nEvaluating best model:")
                evaluate_model(best_model_path, val_loader, device)
    
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 