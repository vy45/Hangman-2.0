import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
import json
import os
import pickle
import time  # Add for timing
from tqdm import tqdm, trange
from datetime import datetime
import sys
import psutil
from torchinfo import summary
import matplotlib.pyplot as plt

# Constants and configurations
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_WORD_LENGTH = 30  # Maximum word length to consider
MAX_WRONG_GUESSES = 6
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
VALIDATION_EPISODES = 1000
BATCH_SIZE = 64
DATA_DIR = 'hangman_data'
os.makedirs(DATA_DIR, exist_ok=True)
SIMULATION_CORRECT_GUESS_PROB = 0.5  # probability of picking a known correct letter
COMPLETION_EVAL_WORDS = 100  # number of words to use for completion rate calculation (Validation words are used for this)
INPUT_DIM = (MAX_WORD_LENGTH * 28) + 26 + 1  # 28 for one-hot (26 letters + blank + padding marker), 26 for guessed, 1 for lives
MAX_GRAD_NORM = 1.0  # For gradient clipping
LEARNING_RATE = 0.001  # Initial learning rate

# Helper functions for data generation and loading  
def load_and_preprocess_words(filename='words_250000_train.txt'):
    """Load words from file and perform stratified train/validation split"""
    with open(filename, 'r') as f:
        words = [word.strip().lower() for word in f.readlines()]
    
    # Filter words
    words = [w for w in words if len(w) <= MAX_WORD_LENGTH and all(c in ALPHABET for c in w)]
    word_lengths = [len(w) for w in words]
    
    # Count frequency of each length
    length_counts = defaultdict(int)
    for length in word_lengths:
        length_counts[length] += 1
    
    # Filter out words with lengths that appear only once
    min_samples_per_length = 5  # require at least 5 samples per length for better splitting
    valid_lengths = {length for length, count in length_counts.items() if count >= min_samples_per_length}
    
    filtered_words = [w for w in words if len(w) in valid_lengths]
    filtered_lengths = [len(w) for w in filtered_words]
    
    try:
        # Try stratified split
        train_words, val_words = train_test_split(
            filtered_words, test_size=0.2, stratify=filtered_lengths, random_state=42
        )
        print(f"Performed stratified split with {len(valid_lengths)} different word lengths")
    except ValueError as e:
        print("Stratified split failed, falling back to random split")
        train_words, val_words = train_test_split(
            words, test_size=0.2, random_state=42
        )
    
    print(f"Train set: {len(train_words)} words")
    print(f"Validation set: {len(val_words)} words")
    
    return train_words, val_words

def simulate_game(word):
    """Simulate a single game of hangman and return game states"""
    word = word.lower()
    states = []
    guessed_letters = set()
    lives = MAX_WRONG_GUESSES
    
    # Always add initial state (no guesses)
    initial_state = {
        'current_state': '_' * len(word),
        'guessed_letters': [],
        'lives': lives,
        'word_length': len(word)
    }
    
    # Calculate initial target distribution (uniform over all letters in word)
    target_dist = {l: 0.0 for l in ALPHABET}
    for letter in word:
        target_dist[letter] = 1.0
    total = sum(target_dist.values())
    if total > 0:
        target_dist = {k: v/total for k, v in target_dist.items()}
    
    # Add padded state for initial state
    padded_state = []
    for _ in range(len(word)):
        one_hot = [0] * 28
        one_hot[26] = 1  # blank marker
        padded_state.append(one_hot)
    
    padding_needed = MAX_WORD_LENGTH - len(word)
    padding_vector = [0] * 28
    padding_vector[27] = 1
    padded_state.extend([padding_vector] * padding_needed)
    
    initial_state['padded_state'] = padded_state
    initial_state['target_distribution'] = target_dist
    states.append(initial_state)
    
    # Rest of the game simulation...
    while lives > 0:
        # Create current masked state
        current_state = ''.join([c if c in guessed_letters else '_' for c in word])
        
        # No need to store the game state if all letters are guessed
        unguessed_word_letters = set(word) - guessed_letters
        if not unguessed_word_letters:
            break

        # Calculate target distribution for next guess
        remaining_letters = set(ALPHABET) - guessed_letters
        target_dist = {l: 0.0 for l in ALPHABET}
        for letter in remaining_letters:
            if letter in word:
                target_dist[letter] = 1.0
        
        # Normalize distribution
        total = sum(target_dist.values())
        if total > 0:
            target_dist = {k: v/total for k, v in target_dist.items()}
        
        # Store game state with padded one-hot encodings
        padded_state = []
        for c in current_state:
            if c == '_':
                one_hot = [0] * 28  # 26 letters + blank + padding marker
                one_hot[26] = 1  # blank marker
            else:
                one_hot = [0] * 28
                one_hot[ord(c) - ord('a')] = 1
            padded_state.append(one_hot)
        
        # Add padding with padding marker
        padding_needed = MAX_WORD_LENGTH - len(current_state)
        padding_vector = [0] * 28
        padding_vector[27] = 1  # padding marker
        padded_state.extend([padding_vector] * padding_needed)
        
        states.append({
            'current_state': current_state,
            'padded_state': padded_state,
            'guessed_letters': sorted(list(guessed_letters)),
            'target_distribution': target_dist,
            'lives': lives,
            'word_length': len(word)
        })
        
        # Simulate next guess with probability-based selection
        if random.random() < SIMULATION_CORRECT_GUESS_PROB and unguessed_word_letters:
            next_letter = random.choice(list(unguessed_word_letters))
        else:
            next_letter = random.choice(list(remaining_letters))
            
        guessed_letters.add(next_letter)
        if next_letter not in word:
            lives -= 1
    
    return states

def generate_training_data(train_words):
    """Generate training data from simulated games for all training words"""
    all_states = []
    for word in train_words:
        game_states = simulate_game(word)
        all_states.extend(game_states)
    return all_states

def save_simulation_data(train_states, val_states, val_words, filename_prefix='hangman'):
    """Save simulated training and validation data to disk with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f'{filename_prefix}_{timestamp}'
    
    with open(f'{DATA_DIR}/{base_filename}_train_states.pkl', 'wb') as f:
        pickle.dump(train_states, f)
    with open(f'{DATA_DIR}/{base_filename}_val_states.pkl', 'wb') as f:
        pickle.dump(val_states, f)
    with open(f'{DATA_DIR}/{base_filename}_val_words.pkl', 'wb') as f:
        pickle.dump(val_words, f)
    
    # Save filename reference for latest data
    with open(f'{DATA_DIR}/latest_simulation.txt', 'w') as f:
        f.write(base_filename)
    
    print(f"Saved simulation data to {DATA_DIR}/{base_filename}_*.pkl")

def load_simulation_data(filename_prefix=None):
    """Load simulated training and validation data from disk"""
    if filename_prefix is None:
        # Try to load latest simulation data
        try:
            with open(f'{DATA_DIR}/latest_simulation.txt', 'r') as f:
                filename_prefix = f.read().strip()
        except FileNotFoundError:
            print("No latest simulation reference found.")
            return None, None, None
    
    try:
        with open(f'{DATA_DIR}/{filename_prefix}_train_states.pkl', 'rb') as f:
            train_states = pickle.load(f)
        with open(f'{DATA_DIR}/{filename_prefix}_val_states.pkl', 'rb') as f:
            val_states = pickle.load(f)
        with open(f'{DATA_DIR}/{filename_prefix}_val_words.pkl', 'rb') as f:
            val_words = pickle.load(f)
        print(f"Loaded simulation data from {DATA_DIR}/{filename_prefix}_*.pkl")
        return train_states, val_states, val_words
    except FileNotFoundError:
        print(f"No simulation data found for prefix: {filename_prefix}")
        return None, None, None
    
# Model definitions
class TransformerModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        # Similar reshape as LSTM
        self.seq_len = MAX_WORD_LENGTH
        self.feature_dim = input_dim // MAX_WORD_LENGTH
        
        self.input_proj = nn.Linear(self.feature_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, MAX_WORD_LENGTH, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 26)
        
    def forward(self, x):
        batch_size = x.size(0)
        # Reshape input: [batch, features] -> [batch, seq_len, feature_dim]
        x = x.view(batch_size, self.seq_len, self.feature_dim)
        
        # Project to hidden_dim
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Transformer processing
        x = self.transformer(x)
        
        # Global pooling
        x = x.mean(dim=1)
        
        # Output with residual
        residual = self.fc1(x)
        x = F.relu(self.bn1(residual))
        x = x + residual
        
        return F.softmax(self.fc2(x), dim=-1)

class CANModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.attention = nn.MultiheadAttention(hidden_dim, 4)
        self.output = nn.Linear(hidden_dim, 26)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = attn_out.squeeze(1)
        return F.softmax(self.output(x), dim=-1)

class GNNModel(nn.Module):
    def __init__(self, node_features=54, hidden_dim=128):  # 54 = 28 one-hot + 26 guessed letters
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 26)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Global pooling
        x = global_mean_pool(x, data.batch)
        
        return F.softmax(self.output(x), dim=-1)

class MLPModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=128):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Just 2 residual blocks instead of 3
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # Reduced dropout
            ) for _ in range(2)
        ])
        
        self.output = nn.Linear(hidden_dim, 26)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.input_layer(x)))
        
        # Residual connections
        for layer in self.hidden_layers:
            residual = x
            x = layer(x)
            x = x + residual  # Skip connection
            
        return F.softmax(self.output(x), dim=-1)

class LSTMModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=128, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Reshape input to (batch, seq_len, features)
        self.seq_len = MAX_WORD_LENGTH
        self.feature_dim = input_dim // MAX_WORD_LENGTH
        
        # First reduce input dimension
        self.input_proj = nn.Linear(self.feature_dim, hidden_dim)
        
        # LSTM with multiple layers
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )
        
        # Simple attention
        self.attention = nn.MultiheadAttention(hidden_dim, 2)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 26)
        
    def forward(self, x):
        batch_size = x.size(0)
        # Reshape input: [batch, features] -> [batch, seq_len, feature_dim]
        x = x.view(batch_size, self.seq_len, self.feature_dim)
        
        # Project each position to hidden_dim
        x = self.input_proj(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global pooling
        x = attn_out.mean(dim=1)
        
        # Output layers with residual
        residual = self.fc1(x)
        x = F.relu(self.bn1(residual))
        x = x + residual
        
        return F.softmax(self.fc2(x), dim=-1)

class MaryLSTMModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # LSTM processing the obscured word
        self.lstm = nn.LSTM(
            input_size=27,  # One-hot encoded letters (26) + blank marker
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Final dense layers
        self.combine = nn.Linear(hidden_dim + 26, hidden_dim)
        self.output = nn.Linear(hidden_dim, 26)
        
    def forward(self, word_state, guessed_letters):
        # Process word through LSTM - no padding needed
        lstm_out, _ = self.lstm(word_state)
        
        # Take final LSTM state
        final_lstm = lstm_out[:, -1]
        
        # Combine with guessed letters
        combined = torch.cat([final_lstm, guessed_letters], dim=1)
        hidden = F.relu(self.combine(combined))
        
        return F.softmax(self.output(hidden), dim=-1)

def prepare_mary_input(state):
    """Convert game state to Mary's format"""
    # Get actual word length
    word_len = len(state['current_state'])
    
    # 1. Word state: one-hot encoded with 27 dims (26 letters + blank)
    word_state = torch.zeros(word_len, 27)
    for i, char in enumerate(state['current_state']):
        if char == '_':
            word_state[i, 26] = 1  # blank marker
        else:
            word_state[i, ord(char) - ord('a')] = 1
    
    # 2. Guessed letters: 26 dimensions
    guessed_letters = torch.tensor([
        1 if chr(i + ord('a')) in state['guessed_letters'] else 0 
        for i in range(26)
    ], dtype=torch.float32)
    
    return word_state, guessed_letters

def simulate_game_mary_style(word):
    """Simulate game following Mary's approach"""
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

def train_mary_model(model, train_states, val_states, epochs=10):
    """Training loop following Mary's approach"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Training
        for i in range(0, len(train_states), BATCH_SIZE):
            batch_states = train_states[i:i + BATCH_SIZE]
            
            # Prepare inputs
            word_states = []
            guessed_letters = []
            targets = []
            
            max_len = max(len(state['current_state']) for state in batch_states)
            
            for state in batch_states:
                word_state, guessed = prepare_mary_input(state)
                
                # Pad word state to max length in batch
                padding = torch.zeros(max_len - len(word_state), 27)
                padding[:, 26] = 1  # Use blank marker for padding
                word_state = torch.cat([word_state, padding])
                
                word_states.append(word_state)
                guessed_letters.append(guessed)
                targets.append(torch.tensor(
                    list(state['target_distribution'].values()), 
                    dtype=torch.float32
                ))
            
            # Stack batch
            word_states = torch.stack(word_states).to(DEVICE)
            guessed_letters = torch.stack(guessed_letters).to(DEVICE)
            targets = torch.stack(targets).to(DEVICE)
            
            # Forward pass
            predictions = model(word_states, guessed_letters)
            loss = F.kl_div(predictions.log(), targets, reduction='batchmean')
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = evaluate_model(model, val_states)
            completion_rate = calculate_completion_rate(model, val_words[:COMPLETION_EVAL_WORDS])
        
        print(f"Epoch {epoch + 1}")
        print(f"Train Loss: {total_loss / num_batches:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Completion Rate: {completion_rate:.2%}")

# Training and evaluation functions
def prepare_input(state):
    """Convert game state to model input using pre-computed padded one-hot encodings"""
    # 1. Padded state: MAX_WORD_LENGTH positions × 28 dimensions per position
    state_encoding = [val for vec in state['padded_state'] for val in vec]
    # Length so far: MAX_WORD_LENGTH * 28
    
    # 2. Guessed letters: 26 dimensions (one per letter)
    guessed_encoding = [1 if chr(i + ord('a')) in state['guessed_letters'] else 0 for i in range(26)]
    state_encoding.extend(guessed_encoding)
    # Length so far: (MAX_WORD_LENGTH * 28) + 26
    
    # 3. Lives: 1 dimension
    state_encoding.append(state['lives'] / MAX_WRONG_GUESSES)
    # Final length: (MAX_WORD_LENGTH * 28) + 26 + 1 = INPUT_DIM
    
    result = torch.tensor(state_encoding, dtype=torch.float32)
    assert result.shape[0] == INPUT_DIM, f"Input dimension mismatch in prepare_input! Expected {INPUT_DIM}, got {result.shape[0]}"
    return result

def calculate_completion_rate(model, words):
    """Calculate word completion rate for a set of words"""
    model.eval() # set model to evaluation mode (no gradient updates and dropout for consistency during inference)
    completed = 0
    total = 0
    
    with torch.no_grad():
        for word in tqdm(words, desc="Completion Rate", leave=False):
            state = {
                'current_state': '_' * len(word),
                'guessed_letters': [],
                'lives': MAX_WRONG_GUESSES,
                'word_length': len(word)
            }
            guessed = set()
            
            while len(guessed) < 26 and state['lives'] > 0:
                # Generate padded state for the current state
                padded_state = []
                for c in state['current_state']:
                    if c == '_':
                        one_hot = [0] * 28
                        one_hot[26] = 1  # blank marker
                    else:
                        one_hot = [0] * 28
                        one_hot[ord(c) - ord('a')] = 1
                    padded_state.append(one_hot)
                
                # Add padding with padding marker
                padding_needed = MAX_WORD_LENGTH - len(state['current_state'])
                padding_vector = [0] * 28
                padding_vector[27] = 1  # padding marker
                padded_state.extend([padding_vector] * padding_needed)
                
                # Add padded state to the state dictionary
                state['padded_state'] = padded_state
                
                inputs = prepare_input(state).unsqueeze(0)
                predictions = model(inputs)
                
                # Get next guess
                valid_preds = [(i, p.item()) for i, p in enumerate(predictions[0]) 
                              if chr(i + ord('a')) not in guessed]
                next_letter = chr(max(valid_preds, key=lambda x: x[1])[0] + ord('a'))
                guessed.add(next_letter)
                
                # Update state
                if next_letter not in word:
                    state['lives'] -= 1
                state['guessed_letters'] = sorted(list(guessed))
                state['current_state'] = ''.join(
                    [c if c in guessed else '_' for c in word]
                )
                
                # Check if word is completed
                if '_' not in state['current_state']:
                    break
            
            completed_word = '_' not in state['current_state'] and state['lives'] > 0
            if completed_word:
                completed += 1
            total += 1
    
    return completed / total

def display_predictions(model, state, model_type):
    """Display model's letter predictions for a given state"""
    model.eval()
    with torch.no_grad():
        if model_type == 'gnn':
            batch_data = prepare_gnn_batch([state])
            predictions = model(batch_data)
        else:
            inputs = prepare_input(state).unsqueeze(0)
            predictions = model(inputs)
        
        predictions = predictions[0].cpu().numpy()
        guessed = set(state['guessed_letters'])
        target_dist = state['target_distribution']
        word = state['current_state']  # Get current word state
        
        # Prepare data for display
        letter_data = []
        for i, pred in enumerate(predictions):
            letter = chr(i + ord('a'))
            letter_data.append({
                'letter': letter,
                'prediction': pred,
                'target': target_dist[letter],
                'guessed': letter in guessed,
                'in_word': target_dist[letter] > 0  # Check if letter is in word
            })
        
        # Sort by prediction value for unguessed letters
        unguessed_preds = [d for d in letter_data if not d['guessed']]
        unguessed_preds.sort(key=lambda x: x['prediction'], reverse=True)
        
        # Display predictions
        print("\nPrediction Analysis:")
        print("-" * 80)
        print(f"{'Letter':^8} | {'Prediction':^12} | {'Target':^12} | {'Status':^12} | {'In Word':^8}")
        print("-" * 80)
        
        # Show top 5 unguessed predictions
        print("Top 5 Predicted Letters:")
        for data in unguessed_preds[:5]:
            status = "Available"
            in_word = "Yes" if data['in_word'] else "No"
            print(f"{data['letter']:^8} | {data['prediction']:^12.4f} | {data['target']:^12.4f} | {status:^12} | {in_word:^8}")
        
        print("\nAlready Guessed Letters:")
        guessed_preds = [d for d in letter_data if d['guessed']]
        for data in guessed_preds:
            print(f"{data['letter']:^8} | {data['prediction']:^12.4f} | {data['target']:^12.4f} | {'Guessed':^12}")
        
        print("-" * 80)
        print(f"Current state: {state['current_state']}")
        print(f"Word length: {len(state['current_state'])}")
        print(f"Lives remaining: {state['lives']}")
        print(f"Guessed letters: {', '.join(sorted(guessed))}")
        
        # Show confidence distribution
        print("\nConfidence Distribution:")
        print("High confidence (>0.1):", [d['letter'] for d in letter_data if d['prediction'] > 0.1])
        print("Medium confidence (0.05-0.1):", [d['letter'] for d in letter_data if 0.05 <= d['prediction'] <= 0.1])
        print("Low confidence (<0.05):", [d['letter'] for d in letter_data if d['prediction'] < 0.05])

def get_memory_usage():
    """Get current memory usage statistics"""
    memory = psutil.Process().memory_info()
    stats = {
        'rss': f"{memory.rss / 1024 / 1024:.1f} MB",  # Resident Set Size
        'vms': f"{memory.vms / 1024 / 1024:.1f} MB",  # Virtual Memory Size
    }
    
    if torch.cuda.is_available():
        stats.update({
            'gpu_allocated': f"{torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB",
            'gpu_cached': f"{torch.cuda.memory_reserved() / 1024 / 1024:.1f} MB",
        })
    
    return stats

def log_model_architecture(model, model_name, input_dim):
    """Log model architecture details"""
    print(f"\n{model_name.upper()} Model Architecture:")
    print("-" * 80)
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Print model structure
    print("\nModel Structure:")
    print(model)
    
    # For non-GNN models, show detailed layer summary
    if model_name != 'gnn':
        try:
            print("\nLayer Summary:")
            summary(model, input_size=(1, input_dim))
        except Exception as e:
            print(f"\nCouldn't generate detailed layer summary: {str(e)}")
    else:
        print("\nGNN Model - Layer summary skipped")
        print("Input: Graph structure with node features of dimension", input_dim)
        print("Output: 26-dimensional probability distribution over letters")
    
    print("-" * 80)

def setup_logging():
    """Setup logging to both file and console"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'{DATA_DIR}/training_log_{timestamp}.txt'
    
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    
    # Log system information
    print(f"Logging to: {log_file}")
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nSystem Information:")
    print("-" * 80)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Total RAM: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    print("-" * 80)
    
    return log_file

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

def train_model(model, train_states, val_states, val_words, model_type, epochs=10):
    """Train a model and evaluate on validation data"""
    start_time = time.time()
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'completion_rate': [],
        'eval_times': [],
        'total_time': 0,
        'stopped_epoch': epochs
    }
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_valid_batches = 0
            random.shuffle(train_states)
            
            batch_iterator = tqdm(
                range(0, len(train_states), BATCH_SIZE),
                desc=f"Epoch {epoch + 1}/{epochs}",
                position=1,
                leave=False
            )
            
            for i in batch_iterator:
                batch_states = train_states[i:i + BATCH_SIZE]
                
                if model_type == 'gnn':
                    batch_data = prepare_gnn_batch(batch_states)
                    predictions = model(batch_data)
                else:
                    inputs = torch.stack([prepare_input(state) for state in batch_states])
                    predictions = model(inputs)
                
                targets = torch.tensor([
                    list(state['target_distribution'].values()) for state in batch_states
                ], dtype=torch.float32)
                
                # Add small epsilon to prevent log(0)
                epsilon = 1e-8
                predictions = torch.clamp(predictions, epsilon, 1.0 - epsilon)
                
                loss = F.kl_div(predictions.log(), targets, reduction='batchmean')
                
                # Check for invalid loss
                if not torch.isfinite(loss):
                    print(f"\nWarning: Invalid loss detected in batch {i//BATCH_SIZE}")
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                
                optimizer.step()
                
                batch_loss = loss.item()
                total_loss += batch_loss
                num_valid_batches += 1
                
                batch_iterator.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'avg_loss': f'{total_loss/num_valid_batches:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
            
            # Validation phase
            val_pbar = tqdm(desc="Validation", position=1, leave=False)
            eval_start_time = time.time()
            val_loss = evaluate_model(model, val_states)
            eval_time = time.time() - eval_start_time
            val_pbar.close()
            
            # Early stopping check
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                metrics['stopped_epoch'] = epoch + 1
                break
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
            # Completion rate calculation
            comp_pbar = tqdm(desc="Completion Rate", position=1, leave=False)
            completion_rate = calculate_completion_rate(model, val_words[:COMPLETION_EVAL_WORDS])
            comp_pbar.close()
            
            # Store metrics
            metrics['train_loss'].append(total_loss / num_valid_batches)
            metrics['val_loss'].append(val_loss)
            metrics['completion_rate'].append(completion_rate)
            metrics['eval_times'].append(eval_time)
            
            # Print epoch results
            print(f"\nEpoch {epoch + 1}/{epochs} Results:")
            print(f"Train Loss: {total_loss / num_valid_batches:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Completion Rate: {completion_rate:.2%}")
            print(f"Evaluation Time: {eval_time:.2f}s")
            
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        print("Skipping to next model...")
        return metrics
        
    finally:
        total_time = time.time() - start_time
        metrics['total_time'] = total_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        if metrics['stopped_epoch'] < epochs:
            print(f"Stopped early at epoch {metrics['stopped_epoch']}")
    
    return metrics

def evaluate_model(model, val_states):
    """Evaluate model on validation states"""
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(val_states), BATCH_SIZE):
            batch_states = val_states[i:i + BATCH_SIZE]
            
            if model_type == 'gnn':
                batch_data = prepare_gnn_batch(batch_states)
                predictions = model(batch_data)
            else:
                inputs = torch.stack([prepare_input(state) for state in batch_states])
                predictions = model(inputs)
            
            targets = torch.tensor([
                list(state['target_distribution'].values()) for state in batch_states
            ], dtype=torch.float32)
            
            loss = F.kl_div(predictions.log(), targets, reduction='batchmean')
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def prepare_gnn_batch(states):
    """Prepare batch data for GNN model"""
    batch_data = []
    for state in states:
        try:
            # Create node features - one node per character position
            node_features = []
            word_length = len(state['current_state'])
            
            # Create fixed-size node features for all positions up to MAX_WORD_LENGTH
            for i in range(MAX_WORD_LENGTH):
                if i < word_length:
                    c = state['current_state'][i]
                    if c == '_':
                        one_hot = [0] * 27
                        one_hot[26] = 1  # blank marker
                    else:
                        one_hot = [0] * 27
                        one_hot[ord(c) - ord('a')] = 1
                else:
                    one_hot = [0] * 27  # padding position
                node_features.append(one_hot)
            
            # Add guessed letters and lives information to each node
            guessed_encoding = [1 if chr(i + ord('a')) in state['guessed_letters'] else 0 for i in range(26)]
            lives_encoding = [state['lives'] / MAX_WRONG_GUESSES]
            
            # Extend each node's features
            for i in range(MAX_WORD_LENGTH):
                node_features[i].extend(guessed_encoding + lives_encoding)
            
            # Create edges (fully connected graph)
            edge_index = []
            for i in range(MAX_WORD_LENGTH):
                for j in range(MAX_WORD_LENGTH):
                    if i != j:
                        edge_index.append([i, j])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            x = torch.tensor(node_features, dtype=torch.float32)
            
            data = Data(x=x, edge_index=edge_index)
            batch_data.append(data)
            
        except Exception as e:
            print(f"Error preparing GNN batch data: {str(e)}")
            continue
    
    if not batch_data:
        raise ValueError("Failed to prepare any valid data in the batch")
        
    return Batch.from_data_list(batch_data)

def plot_training_metrics(results):
    """Plot training metrics for all models"""
    models = list(results.keys())
    metrics = ['train_loss', 'val_loss', 'completion_rate']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for model_name in models:
        model_metrics = results[model_name]['metrics']
        
        # Training Loss
        line, = axes[0].plot(model_metrics['train_loss'], label=model_name.upper())
        if model_metrics.get('stopped_epoch', len(model_metrics['train_loss'])) < len(model_metrics['train_loss']):
            axes[0].axvline(x=model_metrics.get('stopped_epoch', len(model_metrics['train_loss'])), color=line.get_color(), linestyle='--', alpha=0.5)
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        axes[0].legend()
        
        # Validation Loss
        axes[1].plot(model_metrics['val_loss'], label=model_name.upper())
        axes[1].set_title('Validation Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True)
        axes[1].legend()
        
        # Completion Rate
        axes[2].plot(model_metrics['completion_rate'], label=model_name.upper())
        axes[2].set_title('Completion Rate')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Rate')
        axes[2].grid(True)
        axes[2].legend()
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f'{DATA_DIR}/training_metrics_{timestamp}.png'
    plt.savefig(plot_path)
    print(f"\nTraining metrics plot saved to: {plot_path}")
    
    # Print final metrics
    print("\nFinal Metrics:")
    print("-" * 100)
    print(f"{'Model':^15} | {'Train Loss':^12} | {'Val Loss':^12} | {'Completion':^12} | {'Epochs':^12}")
    print("-" * 100)
    
    for model_name in models:
        metrics = results[model_name]['metrics']
        final_train = metrics['train_loss'][-1] if metrics['train_loss'] else float('nan')
        final_val = metrics['val_loss'][-1] if metrics['val_loss'] else float('nan')
        final_comp = metrics['completion_rate'][-1] if metrics['completion_rate'] else 0.0
        stopped_epoch = metrics.get('stopped_epoch', len(metrics['train_loss']))
        
        print(f"{model_name.upper():^15} | {final_train:^12.4f} | {final_val:^12.4f} | "
              f"{final_comp:^12.2%} | {stopped_epoch:^12d}")
    print("-" * 100)

def run_benchmarks(model_ids, load_saved_data=True):
    """Train and evaluate specified models"""
    # Try to load saved data first if requested
    train_states = val_states = val_words = None
    if load_saved_data:
        train_states, val_states, val_words = load_simulation_data()
    
    # Generate new data if needed
    if train_states is None or val_states is None or val_words is None:
        train_words, val_words = load_and_preprocess_words()
        train_states = generate_training_data(train_words)
        # Generate validation states once
        val_states = []
        for word in val_words[:VALIDATION_EPISODES]:
            val_states.extend(simulate_game(word))
        # Save all data
        save_simulation_data(train_states, val_states, val_words)
    
    # Verify input dimensions
    sample_input = prepare_input(train_states[0])
    actual_dim = sample_input.shape[0]
    assert actual_dim == INPUT_DIM, f"Input dimension mismatch! Expected {INPUT_DIM}, got {actual_dim}"
    
    models = {
        1: ('transformer', TransformerModel(hidden_dim=128, num_heads=4, num_layers=2)),  # Kept smaller but enhanced
        2: ('lstm', LSTMModel(hidden_dim=128, num_layers=2)),  # New LSTM model
        3: ('can', CANModel()),  # Existing CAN model
        4: ('gnn', GNNModel()),  # Existing GNN model
        5: ('mlp', MLPModel(hidden_dim=128)),  # Enhanced MLP with residuals but same size
        6: ('mary_lstm', MaryLSTMModel(hidden_dim=128))  # New MaryLSTM model
    }
    
    results = {}
    training_times = []
    
    for model_id in model_ids:
        if model_id not in models:
            print(f"Invalid model ID: {model_id}")
            continue
            
        model_name, model = models[model_id]
        print(f"\nTraining {model_name.upper()} model...")
        
        try:
            # Validate model input dimensions
            if model_name == 'gnn':
                # Test GNN batch preparation with a single sample
                test_batch = prepare_gnn_batch([train_states[0]])
                print(f"GNN input validation successful. Node features: {test_batch.x.shape}")
            else:
                sample_input = prepare_input(train_states[0])
                print(f"Model input validation successful. Input shape: {sample_input.shape}")
            
            model = model.to(DEVICE)
            log_model_architecture(model, model_name, INPUT_DIM)
            
            metrics = train_model(model, train_states, val_states, val_words, model_name)
            training_times.append((model_name, metrics['total_time']))
            
            results[model_name] = {
                'final_val_loss': metrics['val_loss'][-1] if metrics['val_loss'] else float('nan'),
                'final_completion_rate': metrics['completion_rate'][-1] if metrics['completion_rate'] else 0.0,
                'metrics': metrics,
                'training_time': metrics['total_time']
            }
            
        except Exception as e:
            print(f"\nError with {model_name} model:")
            print(f"Type: {type(e).__name__}")
            print(f"Error: {str(e)}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
            print("\nSkipping to next model...")
            training_times.append((model_name, float('nan')))
            continue
    
    # Print training time summary
    print("\nTraining Time Summary:")
    print("-" * 50)
    print(f"{'Model':^15} | {'Time (seconds)':^15} | {'Time (minutes)':^15}")
    print("-" * 50)
    for model_name, train_time in training_times:
        if np.isfinite(train_time):
            print(f"{model_name:^15} | {train_time:^15.2f} | {train_time/60:^15.2f}")
        else:
            print(f"{model_name:^15} | {'Failed':^15} | {'Failed':^15}")
    print("-" * 50)
    
    # Save results
    with open(f'{DATA_DIR}/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot training metrics
    plot_training_metrics(results)

def simulate_and_display_game(model, model_type, word, max_wrong=3):
    """Simulate and display a game up to specified wrong guesses"""
    print(f"\nSimulating game for word: {word}")
    print("=" * 80)
    
    # Initialize state
    state = {
        'current_state': '_' * len(word),
        'guessed_letters': [],
        'lives': MAX_WRONG_GUESSES,
        'word_length': len(word)
    }
    
    # Calculate target distribution for the word
    target_dist = {l: 0.0 for l in ALPHABET}
    for letter in word:
        target_dist[letter] = 1.0
    total = sum(target_dist.values())
    if total > 0:
        target_dist = {k: v/total for k, v in target_dist.items()}
    
    state['target_distribution'] = target_dist
    
    # Track Q-values history
    history = []
    wrong_guesses = 0
    
    while wrong_guesses < max_wrong and '_' in state['current_state']:
        # Prepare state
        padded_state = []
        for c in state['current_state']:
            if c == '_':
                one_hot = [0] * 28
                one_hot[26] = 1
            else:
                one_hot = [0] * 28
                one_hot[ord(c) - ord('a')] = 1
            padded_state.append(one_hot)
        
        padding_needed = MAX_WORD_LENGTH - len(word)
        padding_vector = [0] * 28
        padding_vector[27] = 1
        padded_state.extend([padding_vector] * padding_needed)
        
        state['padded_state'] = padded_state
        
        # Get model predictions
        model.eval()
        with torch.no_grad():
            if model_type == 'gnn':
                batch_data = prepare_gnn_batch([state])
                predictions = model(batch_data)
            else:
                inputs = prepare_input(state).unsqueeze(0)
                predictions = model(inputs)
            
            predictions = predictions[0].cpu().numpy()
        
        # Get next guess (highest Q-value for unguessed letter)
        guessed = set(state['guessed_letters'])
        valid_preds = [(chr(i + ord('a')), p) for i, p in enumerate(predictions) 
                      if chr(i + ord('a')) not in guessed]
        valid_preds.sort(key=lambda x: x[1], reverse=True)
        next_letter = valid_preds[0][0]
        
        # Store state information before making the guess
        history.append({
            'state': state['current_state'],
            'guessed': sorted(list(guessed)),
            'predictions': valid_preds,
            'choice': next_letter,
            'lives': state['lives'],
            'wrong_guesses': wrong_guesses
        })
        
        # Update state
        guessed.add(next_letter)
        state['guessed_letters'] = sorted(list(guessed))
        
        # Check if guess was wrong (letter not in word)
        if next_letter not in word:
            state['lives'] -= 1
            wrong_guesses += 1
        
        # Update current state
        new_state = ''
        for c in word:
            if c in guessed:
                new_state += c
            else:
                new_state += '_'
        state['current_state'] = new_state
    
    # Display game history
    print("\nGame Progress:")
    print("-" * 80)
    print(f"{'Turn':^5} | {'State':^{len(word)}} | {'Guess':^6} | {'Result':^8} | {'Top 3 Q-values':^40} | {'Wrong':^5}")
    print("-" * 80)
    
    for i, turn in enumerate(history):
        # Get top 3 predictions for this turn
        top_3 = [f"{p[0]}:{p[1]:.3f}" for p in turn['predictions'][:3]]
        top_3_str = ', '.join(top_3)
        
        # Determine if guess was correct
        next_state = history[i+1]['state'] if i+1 < len(history) else state['current_state']
        result = "✓" if next_state.count('_') < turn['state'].count('_') else "✗"
        
        print(f"{i+1:^5} | {turn['state']:^{len(word)}} | {turn['choice']:^6} | {result:^8} | "
              f"{top_3_str:<40} | {turn['wrong_guesses']:^5}")
    
    print("-" * 80)
    print(f"Final state: {state['current_state']}")
    print(f"Lives remaining: {state['lives']} ({wrong_guesses} wrong guesses)")
    print(f"Letters guessed: {', '.join(state['guessed_letters'])}")
    print(f"Target word: {word}")
    print("=" * 80)

def test_predictions_on_word(model, model_type, word="congratulations"):
    """Test model predictions on a specific word"""
    print(f"\nTesting predictions on word: {word}")
    
    # Create initial state
    state = {
        'current_state': '_' * len(word),
        'guessed_letters': [],
        'lives': MAX_WRONG_GUESSES,
        'word_length': len(word)
    }
    
    # Add padded state
    padded_state = []
    for _ in range(len(word)):
        one_hot = [0] * 28
        one_hot[26] = 1  # blank marker
        padded_state.append(one_hot)
    
    padding_needed = MAX_WORD_LENGTH - len(word)
    padding_vector = [0] * 28
    padding_vector[27] = 1
    padded_state.extend([padding_vector] * padding_needed)
    
    # Calculate target distribution
    target_dist = {l: 0.0 for l in ALPHABET}
    for letter in word:
        target_dist[letter] = 1.0
    total = sum(target_dist.values())
    if total > 0:
        target_dist = {k: v/total for k, v in target_dist.items()}
    
    state['padded_state'] = padded_state
    state['target_distribution'] = target_dist
    
    display_predictions(model, state, model_type)
    
    # # Add game simulation
    # print("\nSimulating partial game:")
    # simulate_and_display_game(model, model_type, word)

if __name__ == "__main__":
    # Setup logging before anything else
    log_file = setup_logging()
    
    try:
        # Only train Transformer (1), MLP (4), and MaryLSTM (6)
        models_to_train = [5, 2, 3, 1, 4, 6]
        run_benchmarks(models_to_train, load_saved_data=True)
    finally:
        # Restore original stdout and close log file
        if hasattr(sys.stdout, 'log'):
            sys.stdout.log.close()
            sys.stdout = sys.stdout.terminal
        print(f"\nLog file saved to: {log_file}") 