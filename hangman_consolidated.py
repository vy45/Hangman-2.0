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

# Constants and configurations
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_WORD_LENGTH = 20  # Maximum word length to consider
MAX_WRONG_GUESSES = 6
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
TRAIN_EPISODES = 10000
VALIDATION_EPISODES = 1000
BATCH_SIZE = 64
DATA_DIR = 'hangman_data'
os.makedirs(DATA_DIR, exist_ok=True)

# Helper functions for data generation
def load_and_preprocess_words(filename='words_250000_train.txt'):
    """Load words from file and perform stratified train/validation split"""
    with open(filename, 'r') as f:
        words = [word.strip().lower() for word in f.readlines()]
    
    # Filter words and create length-based stratification
    words = [w for w in words if len(w) <= MAX_WORD_LENGTH and all(c in ALPHABET for c in w)]
    word_lengths = [len(w) for w in words]
    
    # Perform stratified split
    train_words, val_words = train_test_split(
        words, test_size=0.2, stratify=word_lengths, random_state=42
    )
    return train_words, val_words

def simulate_game(word):
    """Simulate a single game of hangman and return game states"""
    word = word.lower()
    states = []
    guessed_letters = set()
    lives = MAX_WRONG_GUESSES
    
    while lives > 0:
        # Create current masked state
        current_state = ''.join([c if c in guessed_letters else '_' for c in word])
        
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
        
        # Store game state
        states.append({
            'current_state': current_state,
            'guessed_letters': sorted(list(guessed_letters)),
            'target_distribution': target_dist,
            'lives': lives,
            'word_length': len(word)
        })
        
        # Simulate next guess
        unguessed_word_letters = set(word) - guessed_letters
        if not unguessed_word_letters:
            break
            
        next_letter = random.choice(list(remaining_letters))
        guessed_letters.add(next_letter)
        if next_letter not in word:
            lives -= 1
    
    return states

def generate_training_data(train_words, num_episodes=TRAIN_EPISODES):
    """Generate training data from simulated games"""
    all_states = []
    for _ in range(num_episodes):
        word = random.choice(train_words)
        game_states = simulate_game(word)
        all_states.extend(game_states)
    return all_states

# Model definitions
class TransformerModel(nn.Module):
    def __init__(self, input_dim=53, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim*4),
            num_layers
        )
        self.output = nn.Linear(hidden_dim, 26)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)
        return F.softmax(self.output(x), dim=-1)

class CANModel(nn.Module):
    def __init__(self, input_dim=53, hidden_dim=128):
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
    def __init__(self, input_dim=53, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 26)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, data.batch)
        return F.softmax(self.output(x), dim=-1)

class MLPModel(nn.Module):
    def __init__(self, input_dim=53, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 26)
        )
        
    def forward(self, x):
        return F.softmax(self.network(x), dim=-1)

# Training and evaluation functions
def prepare_input(state):
    """Convert game state to model input"""
    # Create one-hot encoding for current state
    state_encoding = []
    for c in state['current_state']:
        if c == '_':
            state_encoding.extend([0] * 27)  # 26 letters + blank
        else:
            one_hot = [0] * 27
            one_hot[ord(c) - ord('a')] = 1
            state_encoding.extend(one_hot)
            
    # Pad to maximum length
    padding_needed = MAX_WORD_LENGTH - len(state['current_state'])
    state_encoding.extend([0] * (27 * padding_needed))
    
    # Add guessed letters and lives information
    guessed_encoding = [1 if chr(i + ord('a')) in state['guessed_letters'] else 0 for i in range(26)]
    state_encoding.extend(guessed_encoding)
    state_encoding.append(state['lives'] / MAX_WRONG_GUESSES)
    
    return torch.tensor(state_encoding, dtype=torch.float32)

def save_simulation_data(train_states, val_words, filename_prefix='hangman'):
    """Save simulated training data and validation words to disk"""
    with open(f'{DATA_DIR}/{filename_prefix}_train_states.pkl', 'wb') as f:
        pickle.dump(train_states, f)
    with open(f'{DATA_DIR}/{filename_prefix}_val_words.pkl', 'wb') as f:
        pickle.dump(val_words, f)
    print(f"Saved simulation data to {DATA_DIR}/")

def load_simulation_data(filename_prefix='hangman'):
    """Load simulated training data and validation words from disk"""
    try:
        with open(f'{DATA_DIR}/{filename_prefix}_train_states.pkl', 'rb') as f:
            train_states = pickle.load(f)
        with open(f'{DATA_DIR}/{filename_prefix}_val_words.pkl', 'rb') as f:
            val_words = pickle.load(f)
        print(f"Loaded simulation data from {DATA_DIR}/")
        return train_states, val_words
    except FileNotFoundError:
        print("No saved simulation data found.")
        return None, None

def calculate_completion_rate(model, words, model_type):
    """Calculate word completion rate for a set of words"""
    model.eval()
    completed = 0
    total = 0
    
    with torch.no_grad():
        for word in words:
            completed_word = True
            state = {
                'current_state': '_' * len(word),
                'guessed_letters': [],
                'lives': MAX_WRONG_GUESSES,
                'word_length': len(word)
            }
            guessed = set()
            
            while len(guessed) < 26 and state['lives'] > 0:
                if model_type == 'gnn':
                    batch_data = prepare_gnn_batch([state])
                    predictions = model(batch_data)
                else:
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

def train_model(model, train_states, val_words, model_type, epochs=10):
    """Train a model and evaluate on validation data"""
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    
    # Track metrics
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'completion_rate': []
    }
    
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(train_states)
        
        for i in range(0, len(train_states), BATCH_SIZE):
            batch_states = train_states[i:i + BATCH_SIZE]
            
            if model_type == 'gnn':
                # Special handling for GNN input
                batch_data = prepare_gnn_batch(batch_states)
                predictions = model(batch_data)
            else:
                inputs = torch.stack([prepare_input(state) for state in batch_states])
                predictions = model(inputs)
            
            targets = torch.tensor([
                list(state['target_distribution'].values()) for state in batch_states
            ], dtype=torch.float32)
            
            loss = F.kl_div(predictions.log(), targets, reduction='batchmean')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation and metrics
        val_loss = evaluate_model(model, val_words, model_type)
        completion_rate = calculate_completion_rate(model, val_words[:100], model_type)
        
        metrics['train_loss'].append(total_loss/len(train_states))
        metrics['val_loss'].append(val_loss)
        metrics['completion_rate'].append(completion_rate)
        
        print(f'Epoch {epoch+1}:')
        print(f'  Train Loss: {total_loss/len(train_states):.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Completion Rate: {completion_rate:.2%}')
    
    return metrics

def evaluate_model(model, val_words, model_type):
    """Evaluate model on validation words"""
    model.eval()
    total_loss = 0
    val_states = []
    
    # Generate validation states
    for word in val_words[:VALIDATION_EPISODES]:
        val_states.extend(simulate_game(word))
    
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
    
    return total_loss / len(val_states)

def prepare_gnn_batch(states):
    """Prepare batch data for GNN model"""
    batch_data = []
    for state in states:
        # Create node features
        node_features = []
        for c in state['current_state']:
            if c == '_':
                node_features.append([0] * 27)
            else:
                one_hot = [0] * 27
                one_hot[ord(c) - ord('a')] = 1
                node_features.append(one_hot)
        
        # Add guessed letters and lives information to each node
        guessed_encoding = [1 if chr(i + ord('a')) in state['guessed_letters'] else 0 for i in range(26)]
        lives_encoding = [state['lives'] / MAX_WRONG_GUESSES]
        
        for i in range(len(node_features)):
            node_features[i].extend(guessed_encoding + lives_encoding)
        
        # Create edges (fully connected graph)
        num_nodes = len(node_features)
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        x = torch.tensor(node_features, dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index)
        batch_data.append(data)
    
    return Batch.from_data_list(batch_data)

def run_benchmarks(model_ids, load_saved_data=True):
    """Train and evaluate specified models"""
    # Try to load saved data first if requested
    train_states = val_words = None
    if load_saved_data:
        train_states, val_words = load_simulation_data()
    
    # Generate new data if needed
    if train_states is None or val_words is None:
        train_words, val_words = load_and_preprocess_words()
        train_states = generate_training_data(train_words)
        # Save the newly generated data
        save_simulation_data(train_states, val_words)
    
    models = {
        1: ('transformer', TransformerModel()),
        2: ('can', CANModel()),
        3: ('gnn', GNNModel()),
        4: ('mlp', MLPModel())
    }
    
    results = {}
    for model_id in model_ids:
        if model_id not in models:
            print(f"Invalid model ID: {model_id}")
            continue
            
        model_name, model = models[model_id]
        print(f"\nTraining {model_name.upper()} model...")
        model = model.to(DEVICE)
        metrics = train_model(model, train_states, val_words, model_name)
        
        # Save model and metrics
        torch.save({
            'state_dict': model.state_dict(),
            'metrics': metrics
        }, f'{DATA_DIR}/hangman_{model_name}_model.pt')
        
        results[model_name] = {
            'final_val_loss': metrics['val_loss'][-1],
            'final_completion_rate': metrics['completion_rate'][-1],
            'metrics': metrics
        }
    
    # Save results
    with open(f'{DATA_DIR}/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    # Specify which models to train (1: Transformer, 2: CAN, 3: GNN, 4: MLP)
    models_to_train = [1, 2, 3, 4]  # Change this list to train specific models
    run_benchmarks(models_to_train, load_saved_data=True) 