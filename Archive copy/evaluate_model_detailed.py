import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import pickle
import argparse
from tqdm import tqdm

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
DATA_DIR = 'hangman_data'

# Model definition (copy from mary_hangman.py to make standalone)
class MaryLSTMModel(nn.Module):
    def __init__(self, hidden_dim=128, embedding_dim=8, dropout_rate=0.2, use_batch_norm=False):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.char_embedding = nn.Embedding(27, embedding_dim)
        self.length_encoding = {
            i: torch.tensor([int(x) for x in format(i, '05b')], dtype=torch.float32)
            for i in range(1, 33)
        }
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim + 26 + 5)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.combine = nn.Linear(hidden_dim + 26 + 5, hidden_dim)
        self.output = nn.Linear(hidden_dim, 26)
        
    def forward(self, word_state, guessed_letters, word_length):
        embedded = self.char_embedding(word_state)
        lstm_out, _ = self.lstm(embedded)
        final_lstm = lstm_out[:, -1]
        length_encoding = torch.stack([self.length_encoding[l.item()] for l in word_length])
        combined_features = torch.cat([
            final_lstm,
            guessed_letters,
            length_encoding.to(final_lstm.device)
        ], dim=1)
        if self.use_batch_norm:
            combined_features = self.bn1(combined_features)
        hidden = self.dropout(F.relu(self.combine(combined_features)))
        if self.use_batch_norm:
            hidden = self.bn2(hidden)
        return F.softmax(self.output(hidden), dim=-1)

# Setup argument parsing
parser = argparse.ArgumentParser(description='Evaluate Hangman Model in Detail')
parser.add_argument('--model-path', type=str, help='Path to model for evaluation')
args = parser.parse_args()

# Find latest model if not specified
if args.model_path:
    model_path = args.model_path
else:
    model_files = list(Path(DATA_DIR).glob('mary_model_*.pt'))
    if not model_files:
        raise ValueError("No model files found!")
    model_path = max(model_files, key=lambda p: p.stat().st_mtime)

print(f"Using model: {model_path}")

# Load model
model = MaryLSTMModel().to(DEVICE)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Find and load latest dataset
dataset_files = list(Path(DATA_DIR).glob('hangman_states_*.pkl'))
if not dataset_files:
    raise ValueError("No dataset files found!")
latest_dataset = max(dataset_files, key=lambda p: p.stat().st_mtime)

with open(latest_dataset, 'rb') as f:
    data = pickle.load(f)
    val_states = data['val_states']
    val_words = data['val_words']

print("\n=== Testing on Validation States ===")
print(f"Number of validation states: {len(val_states)}")

# Test on first 5 validation states
for i, state in enumerate(val_states[:5]):
    print(f"\nValidation State {i+1}:")
    print(f"Current State: {state['current_state']}")
    print(f"Original Word: {state['original_word']}")
    print(f"Guessed Letters: {sorted(state['guessed_letters'])}")
    
    # Prepare input
    char_indices = torch.tensor([
        ALPHABET.index(c) if c in ALPHABET else 26 
        for c in state['current_state']
    ], dtype=torch.long)
    
    guessed = torch.zeros(26)
    for letter in state['guessed_letters']:
        if letter in ALPHABET:
            guessed[ALPHABET.index(letter)] = 1
    
    # Get model prediction
    with torch.no_grad():
        char_indices = char_indices.unsqueeze(0).to(DEVICE)
        guessed = guessed.unsqueeze(0).to(DEVICE)
        predictions = model(char_indices, guessed, torch.tensor([len(state['current_state'])]))
    
    # Print target distribution and model predictions
    print("\nTarget Distribution vs Model Q-Values:")
    print(f"{'Letter':^8} {'Target':^10} {'Q-Value':^10}")
    print("-" * 30)
    
    for letter in ALPHABET:
        if letter not in state['guessed_letters']:
            target = state['target_distribution'][letter]
            q_value = predictions[0][ALPHABET.index(letter)].item()
            print(f"{letter:^8} {target:^10.4f} {q_value:^10.4f}")
    
    # Get model's chosen letter
    valid_preds = [(i, p.item()) for i, p in enumerate(predictions[0]) 
                   if ALPHABET[i] not in state['guessed_letters']]
    next_letter = ALPHABET[max(valid_preds, key=lambda x: x[1])[0]]
    print(f"\nModel's chosen letter: {next_letter}")
    print("-" * 50)

print("\n=== Testing on Validation Words ===")
print(f"Number of validation words: {len(val_words)}")

# Test on first 5 validation words
for i, word in enumerate(val_words[:5]):
    print(f"\nValidation Word {i+1}: {word}")
    guessed_letters = set()
    current_state = '_' * len(word)
    num_guesses = 0
    
    while len(guessed_letters) < 26 and '_' in current_state and num_guesses < 6:
        print(f"\nGuess {num_guesses + 1}:")
        print(f"Current State: {current_state}")
        print(f"Guessed Letters: {sorted(guessed_letters)}")
        
        # Prepare input
        char_indices = torch.tensor([
            ALPHABET.index(c) if c in ALPHABET else 26 
            for c in current_state
        ], dtype=torch.long)
        
        guessed = torch.zeros(26)
        for letter in guessed_letters:
            if letter in ALPHABET:
                guessed[ALPHABET.index(letter)] = 1
        
        # Get model prediction
        with torch.no_grad():
            char_indices = char_indices.unsqueeze(0).to(DEVICE)
            guessed = guessed.unsqueeze(0).to(DEVICE)
            predictions = model(char_indices, guessed, torch.tensor([len(word)]))
        
        # Print model's Q-values for unguessed letters
        print("\nModel Q-Values for Unguessed Letters:")
        print(f"{'Letter':^8} {'Q-Value':^10}")
        print("-" * 20)
        
        valid_preds = [(ALPHABET[i], p.item()) for i, p in enumerate(predictions[0]) 
                       if ALPHABET[i] not in guessed_letters]
        for letter, q_value in sorted(valid_preds, key=lambda x: x[1], reverse=True):
            print(f"{letter:^8} {q_value:^10.4f}")
        
        # Make guess
        next_letter = max(valid_preds, key=lambda x: x[1])[0]
        print(f"\nModel's guess: {next_letter}")
        
        # Update game state
        guessed_letters.add(next_letter)
        current_state = ''.join([c if c in guessed_letters else '_' for c in word])
        num_guesses += 1
        
        if next_letter in word:
            print("Correct guess!")
        else:
            print("Wrong guess!")
    
    # Print final state
    print(f"\nFinal state: {current_state}")
    if '_' not in current_state:
        print(f"Word completed in {num_guesses} guesses!")
    else:
        print("Failed to complete word")
    print("-" * 50) 