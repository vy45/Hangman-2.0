import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle
import logging
import traceback
from pathlib import Path
from datetime import datetime
import argparse
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.optim import Adam

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
BATCH_SIZE = 64
DATA_DIR = 'hangman_data'
MAX_WORD_LENGTH = 32  # Maximum word length to handle
LEARNING_RATE = 0.001
SIMULATION_CORRECT_GUESS_PROB = 0.5
HIDDEN_SIZES = [512, 256, 128]  # Deep network architecture
PAD_TOKEN = '*'  # Different from '_' used for unknown letters
LEARNING_RATE = 0.001
COMPLETION_EVAL_WORDS = 10000
EPOCHS = 100

class MLPHangmanModel(nn.Module):
    def __init__(self, max_word_length=MAX_WORD_LENGTH, dropout_rate=0.4, use_batch_norm=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        
        # Initialize with proper weight scaling
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        # Embedding for each position (28 tokens: 0-25=a-z, 26=unknown, 27=padding)
        self.char_embedding = nn.Embedding(28, 8, padding_idx=27)
        
        # Calculate total input size
        char_features = max_word_length * 8  # embedded characters
        additional_features = 26 + 5 + 1  # guessed + length_encoding + vowel_ratio
        input_size = char_features + additional_features
        
        # Build deep network
        layers = []
        prev_size = input_size
        
        for hidden_size in HIDDEN_SIZES:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Final output layer
        layers.append(nn.Linear(prev_size, 26))
        
        self.layers = nn.Sequential(*layers)
        
        # Apply weight initialization
        self.apply(init_weights)
        
    def forward(self, word_states, guessed_letters, lengths, vowel_ratios):
        """Forward pass"""
        # Get batch size
        batch_size = word_states.size(0)
        
        # Embed characters (word_states should be [batch_size, max_word_length])
        embedded = self.char_embedding(word_states)  # [batch_size, max_word_length, embedding_dim]
        
        # Flatten the embeddings
        embedded = embedded.view(batch_size, -1)  # [batch_size, max_word_length * embedding_dim]
        
        # Get length encoding (5-bit)
        length_encoding = torch.stack([
            torch.tensor([int(x) for x in format(int(l.item()), '05b')], dtype=torch.float32)
            for l in lengths
        ]).to(embedded.device)
        
        # Ensure all inputs have correct dimensions
        guessed_letters = guessed_letters.float()  # [batch_size, 26]
        vowel_ratios = vowel_ratios.float().unsqueeze(1)  # [batch_size, 1]
        
        # Concatenate all features
        combined = torch.cat([
            embedded,  # [batch_size, max_word_length * embedding_dim]
            guessed_letters,  # [batch_size, 26]
            length_encoding,  # [batch_size, 5]
            vowel_ratios  # [batch_size, 1]
        ], dim=1)
        
        # Pass through layers
        x = combined
        for layer in self.layers:
            x = layer(x)
        
        # Output distribution
        return F.softmax(x, dim=-1)

def simulate_game_states(word):
    """Simulate two games with proper vowel-aware strategy"""
    all_states = []
    for _ in range(2):  # Simulate each word twice
        states = []
        guessed_letters = set()
        word_letters = set(word)
        VOWELS = set('aeiou')
        unguessed_vowels = VOWELS - guessed_letters
        wrong_guesses = 0
        
        while len(guessed_letters) < 26 and len(word_letters - guessed_letters) > 0 and wrong_guesses < 6:
            # Calculate current state and vowel ratio
            current_state = ''.join([c if c in guessed_letters else '_' for c in word])
            known_vowels = sum(1 for c in word if c in VOWELS and c in guessed_letters)
            vowel_ratio = known_vowels / len(word)
            
            # Create state
            state = {
                'current_state': current_state,
                'guessed_letters': sorted(list(guessed_letters)),
                'original_word': word,
                'vowel_ratio': vowel_ratio,
                'remaining_lives': 6 - wrong_guesses
            }
            
            # Calculate target distribution (only for known missing letters)
            remaining_letters = word_letters - guessed_letters
            if not remaining_letters:
                return all_states
            
            target_dist = {l: 0.0 for l in ALPHABET}
            # Set target distribution based on remaining letters
            for letter in remaining_letters:
                target_dist[letter] = 1.0
            
            # Normalize target distribution
            total = sum(target_dist.values())
            target_dist = {k: v/total for k, v in target_dist.items()}
            
            state['target_distribution'] = target_dist
            states.append(state)
            
            # Choose next letter based on strategy
            if random.random() < SIMULATION_CORRECT_GUESS_PROB:
                # Guess from known letters
                consonant_choices = remaining_letters - VOWELS
                vowel_choices = remaining_letters & VOWELS

                if vowel_ratio <= 0.2 and vowel_choices:
                    next_letter = random.choice(list(vowel_choices))
                elif vowel_ratio >= 0.4 and consonant_choices:
                    next_letter = random.choice(list(consonant_choices))
                else:
                    next_letter = random.choice(list(remaining_letters))
            else:
                # Random guess from unguessed letters
                available_letters = set(ALPHABET) - guessed_letters
                consonant_choices = available_letters - VOWELS
                vowel_choices = available_letters & VOWELS
                
                if vowel_ratio <= 0.2 and vowel_choices:
                    next_letter = random.choice(list(vowel_choices))
                elif vowel_ratio >= 0.4 and consonant_choices:
                    next_letter = random.choice(list(consonant_choices))
                else:
                    next_letter = random.choice(list(available_letters))
            
            guessed_letters.add(next_letter)
            if next_letter not in word:
                wrong_guesses += 1
        
        all_states.extend(states)
    
    return all_states

def prepare_input(state):
    """Prepare input tensors with proper initialization"""
    char_indices = []
    current_state = state['current_state']
    
    # Handle word characters
    for c in current_state:
        if c in ALPHABET:
            char_indices.append(ALPHABET.index(c))
        else:  # Unknown letter '_'
            char_indices.append(26)
    
    # Add padding
    padding_needed = MAX_WORD_LENGTH - len(current_state)
    char_indices.extend([27] * padding_needed)
    
    # Create tensors properly
    char_indices = torch.LongTensor(char_indices)
    guessed = torch.zeros(26, dtype=torch.float32)
    
    for letter in state['guessed_letters']:
        if letter in ALPHABET:
            guessed[ALPHABET.index(letter)] = 1
    
    vowel_ratio = torch.tensor(state['vowel_ratio'], dtype=torch.float32)
    
    return char_indices, guessed, vowel_ratio

def save_data(train_states, val_states, val_words):
    """Save MLP-specific dataset"""
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = data_dir / f'mlp_hangman_states_{timestamp}.pkl'
    
    try:
        with open(save_path, 'wb') as f:
            pickle.dump({
                'train_states': train_states,
                'val_states': val_states,
                'val_words': val_words
            }, f)
        logging.info(f"Dataset saved to {save_path}")
        
    except Exception as e:
        logging.error(f"Error saving dataset: {str(e)}")
        logging.error(traceback.format_exc())
        raise 


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


def load_data(force_new=False):
    """Load existing MLP dataset or return None if not found"""
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(exist_ok=True)
    
    dataset_files = list(data_dir.glob('mlp_hangman_states_*.pkl'))
    if dataset_files and not force_new:
        latest_file = max(dataset_files, key=lambda p: p.stat().st_mtime)
        logging.info(f"Loading existing dataset: {latest_file}")
        try:
            with open(latest_file, 'rb') as f:
                data = pickle.load(f)
            return data['train_states'], data['val_states'], data['val_words']
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            return None, None, None
    return None, None, None

def setup_logging():
    """Setup logging to both file and console"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'mlp_hangman_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def train_model(model, train_states, val_states, val_words, epochs=EPOCHS):
    """Train the model with optimized logging but maintaining training functionality"""
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'completion_rate': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Prepare and shuffle batches
        batches = prepare_length_batches(train_states)
        random.shuffle(batches)
        
        # Progress bar with minimal output
        batch_iterator = tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch in batch_iterator:
            # Prepare batch tensors
            word_states = torch.stack([prepare_input(state)[0] for state in batch]).to(DEVICE)
            guessed_letters = torch.stack([prepare_input(state)[1] for state in batch]).to(DEVICE)
            lengths = torch.tensor([len(state['current_state']) for state in batch]).to(DEVICE)
            vowel_ratios = torch.stack([prepare_input(state)[2] for state in batch]).to(DEVICE)
            targets = torch.stack([torch.tensor(
                list(state['target_distribution'].values()), 
                dtype=torch.float32
            ) for state in batch]).to(DEVICE)
            
            # Zero gradients (essential - do not remove)
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(word_states, guessed_letters, lengths, vowel_ratios)
            loss = F.kl_div(predictions.log(), targets, reduction='batchmean')
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            batch_iterator.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation phase
        val_loss = evaluate_model(model, val_states)
        completion_rate = calculate_completion_rate(model, val_words[:COMPLETION_EVAL_WORDS])
        
        # Store metrics
        avg_train_loss = total_loss / num_batches
        metrics['train_loss'].append(avg_train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['completion_rate'].append(completion_rate)
        
        # Minimal epoch logging (using logging instead of print)
        logging.info(f"\nEpoch {epoch+1}:")
        logging.info(f"Train Loss: {avg_train_loss:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}")
        logging.info(f"Completion Rate: {completion_rate:.2%}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'hangman_data/mlp_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
        
        scheduler.step(val_loss)
        
        # Early stopping check
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print("Learning rate too small, stopping training")
            break
    
    return model, metrics

def evaluate_model(model, val_states):
    """Evaluate model with minimal logging"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for state in val_states:
            char_indices = prepare_input(state['current_state'])[0].to(DEVICE)
            guessed = prepare_input(state)[1].to(DEVICE)
            target = torch.tensor(list(state['target_distribution'].values()), dtype=torch.float32).to(DEVICE)
            
            prediction = model(char_indices, guessed, torch.tensor([len(state['current_state'])]).to(DEVICE), torch.tensor([state['vowel_ratio']]).to(DEVICE))
            loss = F.kl_div(prediction.log(), target, reduction='batchmean')
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def calculate_completion_rate(model, words):
    """Calculate completion rate with minimal output"""
    model.eval()
    completed = 0
    
    with torch.no_grad():
        for word in tqdm(words, desc="Calculating completion rate", leave=False):
            if play_game(model, word):
                completed += 1
    
    return completed / len(words)

def prepare_length_batches(states, batch_size=BATCH_SIZE):
    """Group states by length and create batches"""
    length_groups = defaultdict(list)
    for state in states:
        length_groups[len(state['current_state'])].append(state)
    
    batches = []
    for states in length_groups.values():
        for i in range(0, len(states), batch_size):
            batches.append(states[i:i + batch_size])
    return batches

def play_game(model, word):
    """Play a single game of hangman"""
    guessed_letters = set()
    wrong_guesses = 0
    word_letters = set(word)
    
    while len(guessed_letters) < 26 and wrong_guesses < 6:
        current_state = ''.join([c if c in guessed_letters else '_' for c in word])
        known_vowels = sum(1 for c in word if c in 'aeiou' and c in guessed_letters)
        vowel_ratio = known_vowels / len(word)
        
        # Prepare input
        char_indices = prepare_input({'current_state': current_state})[0].unsqueeze(0).to(DEVICE)
        guessed = prepare_input({'guessed_letters': sorted(list(guessed_letters))})[1].unsqueeze(0).to(DEVICE)
        length = torch.tensor([len(word)], dtype=torch.float32).to(DEVICE)
        vowel_ratio = torch.tensor([vowel_ratio], dtype=torch.float32).to(DEVICE)
        
        # Get prediction
        prediction = model(char_indices, guessed, length, vowel_ratio)
        valid_preds = [(i, p.item()) for i, p in enumerate(prediction[0]) 
                      if chr(i + ord('a')) not in guessed_letters]
        next_letter = chr(max(valid_preds, key=lambda x: x[1])[0] + ord('a'))
        
        # Update game state
        guessed_letters.add(next_letter)
        if next_letter not in word_letters:
            wrong_guesses += 1
            
        if all(c in guessed_letters for c in word):
            return True
            
    return False

def main():
    parser = argparse.ArgumentParser(description='Train MLP Hangman Model')
    parser.add_argument('--force-new-data', action='store_true',
                       help='Force generation of new dataset')
    args = parser.parse_args()
    
    log_file = setup_logging()
    logging.info("Starting MLP Hangman training")
    
    try:
        # Load or generate data
        train_states, val_states, val_words = load_data(force_new=args.force_new_data)
        
        if train_states is None or args.force_new_data:
            # Split into train/val
            train_words, val_words = load_and_preprocess_words()
            
            # Generate datasets
            logging.info("Generating training states...")
            train_states = []
            for word in tqdm(train_words):
                train_states.extend(simulate_game_states(word))
            
            logging.info("Generating validation states...")
            val_states = []
            for word in tqdm(val_words):  
                val_states.extend(simulate_game_states(word))
            
            save_data(train_states, val_states, val_words)
        
        # Initialize model
        model = MLPHangmanModel().to(DEVICE)
        
        # Train model
        model, metrics = train_model(model, train_states, val_states, val_words, epochs=EPOCHS)
        
        logging.info("Training complete!")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        logging.error(traceback.format_exc())
    finally:
        logging.info(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    main() 