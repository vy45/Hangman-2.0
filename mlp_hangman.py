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

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
BATCH_SIZE = 64
DATA_DIR = 'hangman_data'
MAX_WORD_LENGTH = 32  # Maximum word length to handle
LEARNING_RATE = 0.001
HIDDEN_SIZES = [512, 256, 128]  # Deep network architecture
PAD_TOKEN = '*'  # Different from '_' used for unknown letters

class MLPHangmanModel(nn.Module):
    def __init__(self, max_word_length=MAX_WORD_LENGTH, dropout_rate=0.4, use_batch_norm=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        
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
            torch.tensor([int(x) for x in format(l.item(), '05b')], dtype=torch.float32)
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
    """Simulate two games for each word with vowel-ratio-based strategy"""
    all_states = []
    for _ in range(2):  # Simulate each word twice
        states = []
        guessed_letters = set()
        word_letters = set(word)
        VOWELS = set('aeiou')
        unguessed_vowels = VOWELS - guessed_letters
        wrong_guesses = 0
        
        while (len(guessed_letters) < 26 and 
               len(word_letters - guessed_letters) > 0 and 
               wrong_guesses < 6):  # Stop at 6 wrong guesses
            
            current_state = ''.join([c if c in guessed_letters else '_' for c in word])
            known_vowels = sum(1 for c in word if c in VOWELS and c in guessed_letters)
            vowel_ratio = known_vowels / len(word)
            
            # Create state
            state = {
                'current_state': current_state,
                'guessed_letters': sorted(list(guessed_letters)),
                'original_word': word,
                'vowel_ratio': vowel_ratio,
                'remaining_lives': 6 - wrong_guesses  # Add remaining lives
            }
            
            # Calculate target distribution (only known missing letters)
            remaining_letters = set(word) - guessed_letters
            target_dist = {l: 0.0 for l in ALPHABET}
            for letter in remaining_letters:
                target_dist[letter] = 1.0
            
            # Normalize target distribution
            total = sum(target_dist.values())
            if total > 0:
                target_dist = {k: v/total for k, v in target_dist.items()}
            
            state['target_distribution'] = target_dist
            states.append(state)
            
            # Choose next letter based on vowel ratio strategy
            if vowel_ratio < 0.2 and unguessed_vowels:
                next_letter = random.choice(list(unguessed_vowels))
                unguessed_vowels.remove(next_letter)
            elif vowel_ratio > 0.4 and len(guessed_letters) < 25:
                available = set(ALPHABET) - guessed_letters - VOWELS
                if not available:
                    available = set(ALPHABET) - guessed_letters
                next_letter = random.choice(list(available))
            else:
                if random.random() < 0.5 and (word_letters - guessed_letters):
                    next_letter = random.choice(list(word_letters - guessed_letters))
                else:
                    available = set(ALPHABET) - guessed_letters
                    next_letter = random.choice(list(available))
            
            guessed_letters.add(next_letter)
            if next_letter not in word:
                wrong_guesses += 1
        
        all_states.extend(states)  # Keep all states regardless of game outcome
    
    return all_states

def prepare_input(state):
    """Prepare input tensors for MLP model with distinct padding"""
    # Use 27 for padding token, 26 for unknown letters ('_'), 0-25 for a-z
    char_indices = []
    current_state = state['current_state']
    
    # First handle the actual word characters
    for c in current_state:
        if c in ALPHABET:
            char_indices.append(ALPHABET.index(c))
        else:  # Unknown letter '_'
            char_indices.append(26)
    
    # Add padding tokens
    padding_needed = MAX_WORD_LENGTH - len(current_state)
    char_indices.extend([27] * padding_needed)  # Use 27 for padding
    
    char_indices = torch.tensor(char_indices, dtype=torch.long)
    
    # Prepare other features
    guessed = torch.zeros(26)
    for letter in state['guessed_letters']:
        if letter in ALPHABET:
            guessed[ALPHABET.index(letter)] = 1
            
    vowel_ratio = torch.tensor([state['vowel_ratio']], dtype=torch.float32)
    
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

def train_model(model, train_states, val_states, val_words, epochs=100):
    """Train the MLP model"""
    logging.info(f"Starting training for {epochs} epochs")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    best_val_loss = float('inf')
    best_model_path = None
    
    # Prepare batches
    train_batches = [train_states[i:i + BATCH_SIZE] 
                    for i in range(0, len(train_states), BATCH_SIZE)]
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Shuffle batches
        random.shuffle(train_batches)
        
        for batch in tqdm(train_batches, desc=f"Epoch {epoch + 1}/{epochs}"):
            # Prepare batch
            word_states = []
            guessed_letters = []
            lengths = []
            vowel_ratios = []
            targets = []
            
            for state in batch:
                char_indices, guessed, vowel_ratio = prepare_input(state)
                word_states.append(char_indices)
                guessed_letters.append(guessed)
                lengths.append(len(state['current_state']))
                vowel_ratios.append(vowel_ratio)
                targets.append(torch.tensor(
                    list(state['target_distribution'].values()), 
                    dtype=torch.float32
                ))
            
            # Stack tensors with correct shapes
            word_states = torch.stack(word_states).to(DEVICE)  # [batch_size, max_word_length]
            guessed_letters = torch.stack(guessed_letters).to(DEVICE)  # [batch_size, 26]
            lengths = torch.tensor(lengths, dtype=torch.float32).to(DEVICE)  # [batch_size]
            vowel_ratios = torch.tensor(vowel_ratios, dtype=torch.float32).to(DEVICE)  # [batch_size]
            targets = torch.stack(targets).to(DEVICE)  # [batch_size, vocab_size]
            
            # Forward pass
            predictions = model(word_states, guessed_letters, lengths, vowel_ratios)
            loss = criterion(predictions.log(), targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        val_loss = evaluate_model(model, val_states)
        completion_rate = calculate_completion_rate(model, val_words[:1000])
        
        logging.info(f"Epoch {epoch + 1}:")
        logging.info(f"Train Loss: {total_loss/len(train_batches):.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}")
        logging.info(f"Completion Rate: {completion_rate:.2%}")
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            best_model_path = f'{DATA_DIR}/mlp_model_{timestamp}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_model_path)
            logging.info(f"New best model saved to: {best_model_path}")
    
    return model, best_model_path

def evaluate_model(model, val_states):
    """Evaluate model on validation states"""
    model.eval()
    total_loss = 0
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    val_batches = [val_states[i:i + BATCH_SIZE] 
                   for i in range(0, len(val_states), BATCH_SIZE)]
    
    with torch.no_grad():
        for batch in val_batches:
            # Prepare batch data (same as training)
            word_states = []
            guessed_letters = []
            lengths = []
            vowel_ratios = []
            targets = []
            
            for state in batch:
                char_indices, guessed, vowel_ratio = prepare_input(state)
                word_states.append(char_indices)
                guessed_letters.append(guessed)
                lengths.append(len(state['current_state']))
                vowel_ratios.append(vowel_ratio)
                targets.append(torch.tensor(
                    list(state['target_distribution'].values()),
                    dtype=torch.float32
                ))
            
            # Stack tensors
            word_states = torch.stack(word_states).to(DEVICE)
            guessed_letters = torch.stack(guessed_letters).to(DEVICE)
            lengths = torch.tensor(lengths).to(DEVICE)
            vowel_ratios = torch.stack(vowel_ratios).to(DEVICE)
            targets = torch.stack(targets).to(DEVICE)
            
            predictions = model(word_states, guessed_letters, lengths, vowel_ratios)
            loss = criterion(predictions.log(), targets)
            total_loss += loss.item()
    
    return total_loss / len(val_batches)

def calculate_completion_rate(model, words):
    """Calculate word completion rate"""
    model.eval()
    completed = 0
    total = len(words)
    
    with torch.no_grad():
        for word in tqdm(words, desc="Calculating completion rate"):
            guessed_letters = set()
            current_state = '_' * len(word)
            wrong_guesses = 0
            
            while wrong_guesses < 6 and '_' in current_state:
                # Prepare input
                state = {
                    'current_state': current_state,
                    'guessed_letters': sorted(list(guessed_letters)),
                    'vowel_ratio': sum(1 for c in current_state if c in 'aeiou') / len(current_state)
                }
                
                char_indices, guessed, vowel_ratio = prepare_input(state)
                char_indices = char_indices.unsqueeze(0).to(DEVICE)
                guessed = guessed.unsqueeze(0).to(DEVICE)
                length = torch.tensor([len(current_state)]).to(DEVICE)
                vowel_ratio = vowel_ratio.unsqueeze(0).to(DEVICE)
                
                # Get model prediction
                predictions = model(char_indices, guessed, length, vowel_ratio)
                
                # Choose next letter
                valid_preds = [(i, p.item()) for i, p in enumerate(predictions[0])
                              if chr(i + ord('a')) not in guessed_letters]
                next_letter = chr(max(valid_preds, key=lambda x: x[1])[0] + ord('a'))
                
                # Update game state
                guessed_letters.add(next_letter)
                if next_letter in word:
                    current_state = ''.join(c if c in guessed_letters else '_' for c in word)
                else:
                    wrong_guesses += 1
            
            if '_' not in current_state:
                completed += 1
    
    return completed / total

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
            # Load words
            with open('words_250000_train.txt', 'r') as f:
                words = [w.strip().lower() for w in f.readlines()]
            
            # Split into train/val
            train_words = words[:int(0.8 * len(words))]
            val_words = words[int(0.8 * len(words)):]
            
            # Generate datasets
            logging.info("Generating training states...")
            train_states = []
            for word in tqdm(train_words):
                train_states.extend(simulate_game_states(word))
            
            logging.info("Generating validation states...")
            val_states = []
            for word in tqdm(val_words[:1000]):  # Limit validation set
                val_states.extend(simulate_game_states(word))
            
            save_data(train_states, val_states, val_words)
        
        # Initialize model
        model = MLPHangmanModel().to(DEVICE)
        
        # Train model
        model, best_model_path = train_model(model, train_states, val_states, val_words)
        
        logging.info("Training complete!")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        logging.error(traceback.format_exc())
    finally:
        logging.info(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    main() 