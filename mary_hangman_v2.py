# Including Remaining Lives as a feature. 
# no fancy target distribution

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
import argparse
import traceback
import torch.optim as optim
import re
from collections import Counter
from multiprocessing import Pool
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import string

def setup_logging():
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'mary_hangman_v2_{timestamp}.log'
    
    # Remove any existing handlers to avoid duplicate logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Changed from DEBUG to INFO to reduce noise
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("Logging setup complete")
    logging.info(f"Log file: {log_file}")
    return log_file

# Constants
QUICK_TEST = False
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
FALLBACK_DEVICE = torch.device("cpu")  # For operations not supported by MPS
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
BATCH_SIZE = 64
DATA_DIR = 'hangman_data'
VALIDATION_EPISODES = 10000
FREQ_CUTOFF = 10
SIMULATION_CORRECT_GUESS_PROB = 0.5
MIN_NGRAM_LENGTH = 3
MAX_NGRAM_LENGTH = 5
EPOCHS = 3 if QUICK_TEST else 100
COMPLETION_EVAL_WORDS = 1000 if QUICK_TEST else 10000

class MaryLSTMModel(nn.Module):
    def __init__(self, hidden_dim=128, embedding_dim=8, dropout_rate=0.4, use_batch_norm=False):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        
        # Initialize components on CPU first
        self.char_embedding = nn.Embedding(27, embedding_dim).to(FALLBACK_DEVICE)
        
        # Length encoding dictionary (5-bit representation)
        self.length_encoding = {
            i: torch.tensor([int(x) for x in format(i, '05b')], dtype=torch.float32)
            for i in range(1, 33)  # Support lengths 1-32
        }
        
        # LSTM processing the embedded sequence
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        ).to(FALLBACK_DEVICE)
        
        # Batch normalization layers (optional)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim + 26 + 5 + 2).to(FALLBACK_DEVICE)  # +2 for vowel ratio and remaining lives
            self.bn2 = nn.BatchNorm1d(hidden_dim).to(FALLBACK_DEVICE)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate).to(FALLBACK_DEVICE)
        
        # Final dense layers
        self.combine = nn.Linear(hidden_dim + 26 + 5 + 2, hidden_dim).to(FALLBACK_DEVICE)  # +2 for vowel ratio and remaining lives
        self.output = nn.Linear(hidden_dim, 26).to(FALLBACK_DEVICE)
        
        # Move entire model to target device after initialization
        self.to(DEVICE)
        
    def forward(self, word_state, guessed_letters, word_length, vowel_ratio, remaining_lives):
        # Get batch size
        batch_size = word_state.size(0)
        
        # LSTM processing
        embedded = self.char_embedding(word_state)  # [batch_size, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_size*2]
        
        # Get final LSTM output
        final_lstm = lstm_out[:, -1, :]  # Take last sequence output [batch_size, hidden_size*2]
        
        # Create length encoding
        length_encoding = torch.stack([
            torch.tensor([int(x) for x in format(int(l.item()), '05b')], dtype=torch.float32)
            for l in word_length
        ]).to(final_lstm.device)

        # Fix vowel_ratio and remaining_lives dimensions
        if vowel_ratio.dim() == 1:
            vowel_ratio = vowel_ratio.unsqueeze(-1)
        elif vowel_ratio.dim() == 2 and vowel_ratio.size(1) > 1:
            vowel_ratio = vowel_ratio.squeeze(-1).unsqueeze(-1)
            
        if remaining_lives.dim() == 1:
            remaining_lives = remaining_lives.unsqueeze(-1)
        elif remaining_lives.dim() == 2 and remaining_lives.size(1) > 1:
            remaining_lives = remaining_lives.squeeze(-1).unsqueeze(-1)
        
        # Concatenate features
        combined_features = torch.cat([
            final_lstm,  # [batch_size, hidden_size*2]
            guessed_letters,  # [batch_size, 26]
            length_encoding,  # [batch_size, 5]
            vowel_ratio,  # [batch_size, 1]
            remaining_lives  # [batch_size, 1]
        ], dim=1)
        
        # Apply batch norm if enabled
        if self.use_batch_norm:
            combined_features = self.bn1(combined_features)
        
        # First dense layer with dropout
        hidden = self.dropout(F.relu(self.combine(combined_features)))
        
        # Second batch norm if enabled
        if self.use_batch_norm:
            hidden = self.bn2(hidden)
        
        # Output layer
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
    
    if QUICK_TEST:
        words = words[:10000]

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

def get_masked_ngrams(current_state, n):
    """Get all n-grams from current state with at most one mask"""
    masked_ngrams = []
    state_len = len(current_state)
    
    if state_len < n:
        return masked_ngrams
    
    # Get all n-grams
    for i in range(state_len - n + 1):
        ngram = current_state[i:i+n]
        # Only keep n-grams with 0 or 1 mask
        if ngram.count('_') <= 1:
            masked_ngrams.append(ngram)
    
    # logging.info(f"Found {n}-grams: {masked_ngrams}")
    return masked_ngrams

def calculate_ngram_weights(current_state, ngram_dict, guessed_letters):
    """Calculate letter weights based on n-gram frequencies"""
    # Separate dictionary for each n-gram length
    length_weights = {}
    unguessed_letters = set(ALPHABET) - set(guessed_letters)
    
    # For each possible n-gram length
    for n in range(MIN_NGRAM_LENGTH, min(MAX_NGRAM_LENGTH + 1, len(current_state) + 1)):
        letter_weights = defaultdict(int)
        
        # Get masked n-grams
        masked_ngrams = get_masked_ngrams(current_state, n)
        
        # For each masked n-gram
        for ngram in masked_ngrams:
            if '_' in ngram:
                mask_pos = ngram.index('_')
                # Try each unguessed letter
                for letter in unguessed_letters:
                    filled = ngram[:mask_pos] + letter + ngram[mask_pos+1:]
                    if filled in ngram_dict:
                        letter_weights[letter] += ngram_dict[filled]
                        # logging.info(f"Found match for {filled} with count {ngram_dict[filled]}")
        
        length_weights[n] = dict(letter_weights)
    
    return length_weights

def get_matching_words(current_state, guessed_wrong, word_dict):
    """Find all words that match current state pattern and don't contain wrong guesses"""
    # Pre-compile pattern and create set of wrong guesses once
    pattern_re = re.compile(current_state.replace('_', '.'))
    wrong_set = set(guessed_wrong)
    
    # Use list comprehension for better performance
    return [word for word in word_dict 
            if not (wrong_set & set(word)) and pattern_re.match(word)]

def calculate_letter_frequencies(matching_words):
    """Calculate letter frequencies from matching words"""
    # Use Counter for better performance
    letter_counts = Counter(''.join(matching_words))
    total = sum(letter_counts.values())
    
    if total > 0:
        return {k: v/total for k, v in letter_counts.items()}
    return defaultdict(float)

def simulate_game_states(word):
    """Simulate a game with proper vowel-aware strategy"""
    states = []
    guessed_letters = set()
    word_letters = set(word)
    VOWELS = set('aeiou')
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
        remaining_letters = word_letters - guessed_letters # will never be empty

        target_dist = {l: 0.0 for l in ALPHABET}
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
    
    return states

def adjust_vowel_weights(target_dist, current_state, vowel_ratio):
    """Adjust vowel weights based on current state vowel ratio"""
    VOWELS = set('aeiou')
    
    # Get max weight of known letters (if any) # this is not max weight of known letters, it should be from
    known_max = max(target_dist.values())
    
    # Adjust vowel weights based on current ratio
    if vowel_ratio > 0.48:  # Too many vowels already, reduce vowel weights
        for vowel in VOWELS:
            # if vowel not in current_state:  # Only adjust if not a known letter
            target_dist[vowel] *= 0.8
    elif vowel_ratio <= 0.38:  # Too few vowels, boost vowel weights
        for vowel in VOWELS:
            # if vowel not in current_state:  # Only adjust if not a known letter
            target_dist[vowel] *= 1.2
            # Cap at known_max to prevent overpowering known letters
            target_dist[vowel] = min(target_dist[vowel], known_max*1.1)
    
    return target_dist

def calculate_target_distribution(current_state, guessed_letters, original_word, train_words=None, ngram_dict=None, is_validation=False, vowel_ratio=None):
    """Calculate target distribution using three strategies"""
    word_length = len(current_state)
    target_dist = {l: 0.0 for l in ALPHABET}
    
    # Strategy 1: Known missing letters (always used)
    known_dist = {l: 0.0 for l in ALPHABET}
    remaining_letters = set(original_word) - set(guessed_letters)
    for letter in remaining_letters:
        known_dist[letter] = 1.0
    
    # Get max value from known distribution for scaling
    known_max = max(known_dist.values()) if known_dist.values() else 1.0
    
    # Initialize frequency distribution
    freq_dist = {l: 0.0 for l in ALPHABET}
    
    # Always initialize with known distribution and small weights
    for letter in ALPHABET:
        target_dist[letter] = known_dist[letter] if letter in remaining_letters else 0.01
    
    # Then optionally add frequency information for short words
    if train_words is not None and word_length < FREQ_CUTOFF:
        matching = get_matching_words(current_state, set(guessed_letters) - set(current_state), train_words)
        freqs = calculate_letter_frequencies(matching)
        freq_dist.update(freqs)
        
        # Combine known (0.5) and frequency (0.5)
        for letter in ALPHABET:
            target_dist[letter] = 0.5 * known_dist[letter] + 0.5 * freq_dist[letter]
    
    # Add n-gram predictions for training data with longer words
    if not is_validation and word_length >= 10 and ngram_dict is not None:
        ngram_dist = {l: 0.0 for l in ALPHABET}
        weights = calculate_ngram_weights(current_state, ngram_dict, guessed_letters)
        
        # Combine n-gram predictions with increasing weights
        total_weight = 0
        for n, letter_weights in weights.items():
            weight = n * n  # Square the n-gram length for weight
            total = sum(letter_weights.values())
            if total > 0:
                normalized = {k: v/total for k, v in letter_weights.items()}
                total_weight += weight
                for letter, prob in normalized.items():
                    ngram_dist[letter] += prob * weight
        
        # Scale n-gram distribution
        if total_weight > 0:
            ngram_dist = {k: v/total_weight for k, v in ngram_dist.items()}
            max_weight = max(target_dist.values()) if target_dist.values() else 1.0
            if max(ngram_dist.values(), default=0) > 0:
                ngram_dist = {k: v * max_weight / max(ngram_dist.values()) for k, v in ngram_dist.items()}
            
            # Combine with dynamic weight (more weight for longer words)
            ngram_weight = min(0.5, 0.3 + 0.02 * (word_length - 10))
            for letter in ALPHABET:
                target_dist[letter] = (1 - ngram_weight) * target_dist[letter] + ngram_weight * ngram_dist[letter]
    
    # Zero out guessed letters
    for letter in guessed_letters:
        target_dist[letter] = 0.0
    
    # Adjust vowel weights based on current ratio
    target_dist = adjust_vowel_weights(target_dist, current_state, vowel_ratio)
    
    # Normalize
    total = sum(target_dist.values())
    if total > 0:
        target_dist = {k: v/total for k, v in target_dist.items()}
    
    return target_dist

def calculate_upsampling_factors(length_counts):
    """Calculate upsampling factors for each length"""
    factors = {}
    
    # Short word upsampling
    factors.update({
        1: 10.0,
        2: 5.4,
        3: 2.1,
        4: 1.4
    })
    
    # Long word upsampling
    factors.update({
        16: 1.2,
        17: 1.35,
        18: 1.7,
        19: 2.0,
        20: 2.0,
        21: 2.0,
        22: 2.0,
        23: 2.0,
        24: 2.0,
        25: 2.0,
        26: 2.0,
        27: 2.0,
        28: 2.0,
        29: 2.0,
        30: 2.0
    })
    
    # Ensure minimum states per length
    min_states = BATCH_SIZE  # At least one batch
    
    return {length: max(
        factors.get(length, 1.0),  # Get factor or default to 1.0
        min_states / count if count < min_states else 1.0  # Ensure minimum states
    ) for length, count in length_counts.items()}

def time_block(description):
    """Context manager for timing code blocks"""
    class Timer:
        def __init__(self, description):
            self.description = description
        
        def __enter__(self):
            self.start = time.time()
            return self
        
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            logging.info(f"{self.description}: {elapsed:.2f} seconds")
    
    return Timer(description)

def generate_dataset(words, ngram_dict, num_games_per_word=1, is_validation=False, train_words=None):
    """Generate dataset with balanced upsampling"""
    with time_block("Initial dataset generation"):
        logging.info("Generating initial dataset...")
        all_states = []
        length_counts = defaultdict(int)
        word_length_map = defaultdict(list)
        
        # Generate initial states
        for word in tqdm(words, desc="Generating game states"):
            game_states = []
            for _ in range(num_games_per_word):
                current_state = ['_'] * len(word)
                guessed_letters = []
                remaining_lives = 6
                
                while remaining_lives > 0 and '_' in current_state:
                    # Calculate current state info
                    vowel_count = sum(1 for c in current_state if c in 'aeiou')
                    vowel_ratio = vowel_count / len(current_state)
                    
                    # Calculate target distribution (only known missing letters)
                    target_dist = {l: 0.0 for l in ALPHABET}
                    remaining_letters = set(word) - set(guessed_letters)
                    for letter in remaining_letters:
                        target_dist[letter] = 1.0
                    # Normalize
                    total = sum(target_dist.values())
                    if total > 0:
                        target_dist = {k: v/total for k, v in target_dist.items()}
                    
                    # Record state
                    state_info = {
                        'current_state': ''.join(current_state),
                        'guessed_letters': guessed_letters.copy(),
                        'remaining_lives': remaining_lives,
                        'vowel_ratio': vowel_ratio,
                        'original_word': word,
                        'target_distribution': target_dist
                    }
                    game_states.append(state_info)
                    
                    # Simulate guess
                    available_letters = [c for c in string.ascii_lowercase if c not in guessed_letters]
                    if not available_letters:
                        break
                    
                    guess = random.choice(available_letters)
                    guessed_letters.append(guess)
                    
                    if guess in word:
                        for i, letter in enumerate(word):
                            if letter == guess:
                                current_state[i] = guess
                    else:
                        remaining_lives -= 1
                
                # Add final state with final target distribution
                vowel_count = sum(1 for c in current_state if c in 'aeiou')
                vowel_ratio = vowel_count / len(current_state)
                
                # Final target distribution
                final_target_dist = {l: 0.0 for l in ALPHABET}
                remaining_letters = set(word) - set(guessed_letters)
                for letter in remaining_letters:
                    final_target_dist[letter] = 1.0
                # Normalize
                total = sum(final_target_dist.values())
                if total > 0:
                    final_target_dist = {k: v/total for k, v in final_target_dist.items()}
                
                final_state = {
                    'current_state': ''.join(current_state),
                    'guessed_letters': guessed_letters.copy(),
                    'remaining_lives': remaining_lives,
                    'vowel_ratio': vowel_ratio,
                    'original_word': word,
                    'target_distribution': final_target_dist
                }
                game_states.append(final_state)
            
            # Add all states from this word
            all_states.extend(game_states)
            length = len(word)
            length_counts[length] += len(game_states)
            word_length_map[length].append(word)
    
    # Handle upsampling if needed
    if not is_validation:
        with time_block("Upsampling process"):
            # Print initial statistics
            total_states = len(all_states)
            logging.info("\nInitial state distribution:")
            for length in sorted(length_counts.keys()):
                count = length_counts[length]
                percentage = count / total_states
                factor = calculate_upsampling_factors(length_counts)[length]
                if factor > 1.0:
                    logging.info(f"Length {length:2d}: {count:6d} states ({percentage:.2%}) - Upsampling factor: {factor:.2f}")
                else:
                    logging.info(f"Length {length:2d}: {count:6d} states ({percentage:.2%})")
            
            # Upsample underrepresented lengths
            upsampling_factors = calculate_upsampling_factors(length_counts)
            logging.info("\nUpsampling underrepresented lengths...")
            
            additional_states = []
            for length, factor in tqdm(upsampling_factors.items(), desc="Upsampling"):
                if factor <= 1.0:
                    continue
                
                current_count = length_counts[length]
                target_count = int(current_count * factor)
                additional_needed = target_count - current_count
                
                if additional_needed <= 0:
                    continue
                
                words_of_length = word_length_map[length]
                while len(additional_states) < additional_needed:
                    word = random.choice(words_of_length)
                    # Generate one more game for this word
                    game_states = simulate_game_states(word)
                    additional_states.extend(game_states)
                
                additional_states = additional_states[:additional_needed]
            
            # Add upsampled states
            all_states.extend(additional_states)
            
            # Print final statistics
            total_states = len(all_states)
            logging.info("\nFinal state distribution:")
            for length in sorted(length_counts.keys()):
                count = length_counts[length]
                percentage = count / total_states
                logging.info(f"Length {length:2d}: {count:6d} states ({percentage:.2%})")
    
    return all_states

def build_ngram_dictionary(words):
    """Build dictionary of n-grams from words with length-dependent n-gram sizes"""
    logging.info("Building n-gram dictionary...")
    ngram_dict = defaultdict(int)
    
    for word in tqdm(words, desc="Generating n-grams"):
        word_len = len(word)
        
        # For words up to length 7, use all possible n-grams
        if word_len <= 7:
            max_len = word_len
        else:
            # For longer words, use n-grams of length 3 to 7
            max_len = 7
            
        # Generate n-grams for each length
        for n in range(3, max_len + 1):
            for i in range(word_len - n + 1):
                ngram = word[i:i+n]
                ngram_dict[ngram] += 1
    
    # logging.info(f"Generated {len(ngram_dict)} unique n-grams")
    return ngram_dict

def prepare_input(state):
    """Prepare input tensors with proper device placement"""
    char_indices = torch.tensor([
        ALPHABET.index(c) if c in ALPHABET else 26 
        for c in state['current_state']
    ], dtype=torch.long, device=DEVICE)
    
    guessed = torch.zeros(26, device=DEVICE)
    for letter in state['guessed_letters']:
        if letter in ALPHABET:
            guessed[ALPHABET.index(letter)] = 1
            
    vowel_ratio = torch.tensor([state['vowel_ratio']], dtype=torch.float32, device=DEVICE)
    remaining_lives = torch.tensor([state['remaining_lives']], dtype=torch.float32, device=DEVICE) / 6.0
    
    return char_indices, guessed, vowel_ratio, remaining_lives

def save_data(train_states, val_states, val_words):
    """Save datasets with timestamp"""
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = data_dir / f'hangman_states_v2_{timestamp}.pkl'
    
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
    """Load existing dataset or return None if not found"""
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(exist_ok=True)
    
    dataset_files = list(data_dir.glob('hangman_states_v2_*.pkl'))
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

def prepare_length_batches(train_states):
    """Organize training states into length-specific batches"""
    # Group states by word length
    length_groups = defaultdict(list)
    for state in train_states:
        length = len(state['current_state'])
        length_groups[length].append(state)
    
    # Ensure each group has multiple of BATCH_SIZE states
    batches = []
    for length, states in length_groups.items():
        num_states = len(states)
        num_full_batches = num_states // BATCH_SIZE
        if num_full_batches == 0:
            continue
            
        # Trim to multiple of batch size
        states = states[:num_full_batches * BATCH_SIZE]
        
        # Split into batches
        for i in range(0, len(states), BATCH_SIZE):
            batches.append(states[i:i + BATCH_SIZE])
    
    return batches

def train_model(model, train_states, val_states, val_words, epochs=EPOCHS):
    """Modified training loop with curriculum learning and MPS optimizations"""
    logging.info(f"Starting training for {epochs} epochs")
    
    # Clear MPS memory before training
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
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
        'gradient_norms': [],
        'weight_norms': []
    }
    
    start_time = datetime.now()
    
    def save_checkpoint(epoch, val_loss):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f'{DATA_DIR}/mary_model_v2_epoch{epoch}_{timestamp}.pt'
        
        # Convert any defaultdicts in metrics to regular dicts before saving
        metrics_copy = metrics.copy()
        for key in metrics_copy:
            if isinstance(metrics_copy[key], defaultdict):
                metrics_copy[key] = dict(metrics_copy[key])
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics_copy
        }, path)
        return path
    
    try:
        # Optimize batch size for MPS
        effective_batch_size = BATCH_SIZE * 2 if DEVICE.type == "mps" else BATCH_SIZE
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            
            # Prepare and shuffle batches for this epoch
            batches = prepare_length_batches(train_states)
            random.shuffle(batches)
            
            batch_iterator = tqdm(
                batches,
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=False
            )
            
            for batch in batch_iterator:
                # Prepare batch (all states have same length)
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
                
                # Stack tensors and move to device
                word_states = torch.stack(word_states).to(DEVICE)
                guessed_letters = torch.stack(guessed_letters).to(DEVICE)
                lengths = torch.tensor(lengths).to(DEVICE)
                vowel_ratios = torch.stack(vowel_ratios).to(DEVICE)
                remaining_lives = torch.stack(remaining_lives).to(DEVICE)
                targets = torch.stack(targets).to(DEVICE)
                
                # Forward pass
                optimizer.zero_grad()
                predictions = model(word_states, guessed_letters, lengths, vowel_ratios, remaining_lives)
                
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
                
                if grad_norm > 10:
                    logging.warning(f"Large gradient norm detected: {grad_norm}")
                
                optimizer.step()
                
                # Periodically clear MPS cache
                if DEVICE.type == "mps" and num_batches % 100 == 0:
                    torch.mps.empty_cache()
                
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
            
            # Run detailed validation every 4 epochs
            if (epoch + 1) % 4 == 0:
                logging.info("\nRunning detailed validation...")
                detailed_stats = run_detailed_validation(model, val_words[:COMPLETION_EVAL_WORDS])
                metrics[f'detailed_stats_epoch_{epoch+1}'] = dict(detailed_stats)
            
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
        logging.error(f"Training failed with error: {str(e)}")
        logging.error(traceback.format_exc())
        raise
    
    finally:
        # Clear MPS memory after training
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        total_time = (datetime.now() - start_time).total_seconds()
        metrics['total_time'] = total_time
        logging.info(f"\nTraining completed in {total_time:.2f} seconds")
        
        # Save final model regardless of performance
        final_path = save_checkpoint('final', val_loss)
        logging.info(f"Final model saved to: {final_path}")
    
    return model, optimizer, metrics

def evaluate_model(model, val_states):
    """Evaluate model on validation states using length-specific batches"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Prepare validation batches
    val_batches = prepare_length_batches(val_states)
    
    with torch.no_grad():
        for batch in val_batches:
            # Prepare batch (all states have same length)
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
            loss = F.kl_div(predictions.log(), targets, reduction='batchmean')
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else float('inf')

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
                # Calculate vowel ratio
                known_vowels = sum(1 for c in word if c in 'aeiou' and c in guessed_letters)
                vowel_ratio = known_vowels / len(word)
                
                # Prepare input
                state = {
                    'current_state': current_state,
                    'guessed_letters': sorted(list(guessed_letters)),
                    'vowel_ratio': vowel_ratio,  # Add vowel ratio to state
                    'remaining_lives': 6 - wrong_guesses
                }
                
                char_indices, guessed, vowel_ratio, lives = prepare_input(state)
                
                # Add batch dimension and convert to tensors
                char_indices = char_indices.unsqueeze(0).to(DEVICE)  # [1, max_word_length]
                guessed = guessed.unsqueeze(0).to(DEVICE)  # [1, 26]
                length = torch.tensor([len(current_state)], dtype=torch.float32).to(DEVICE)  # [1]
                vowel_ratio = vowel_ratio.unsqueeze(0).to(DEVICE)  # [1]
                lives = torch.tensor([lives], dtype=torch.float32).to(DEVICE)  # [1]
                
                # Get model prediction
                predictions = model(char_indices, guessed, length, vowel_ratio, lives)
                
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

def run_detailed_validation(model, val_words):
    """Run detailed validation statistics with detailed prediction logging"""
    logging.info("\nRunning detailed validation statistics...")
    word_length_stats = defaultdict(lambda: {'total': 0, 'completed': 0, 'total_guesses': 0})
    
    # Create CSV filename with model type and timestamp
    model_type = model.__class__.__name__.replace('HangmanModel', '').lower()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = f'{DATA_DIR}/predictions_{model_type}_v2_{timestamp}.csv'
    
    # Open CSV file with context manager for automatic closing
    with open(csv_file, 'w') as f:
        # Write header
        header = ['word', 'current_state', 'guessed_letters', 'q_values', 'chosen_letter', 'correct']
        f.write(','.join(header) + '\n')
        
        with torch.no_grad():
            for word in tqdm(val_words, desc="Analyzing validation words"):
                length = len(word)
                guessed_letters = set()
                current_state = '_' * length
                word_completed = False
                wrong_guesses = 0
                num_guesses = 0
                word_letters = set(word)
                
                while len(guessed_letters) < 26 and wrong_guesses < 6:
                    # Calculate vowel ratio
                    known_vowels = sum(1 for c in word if c in 'aeiou' and c in guessed_letters)
                    vowel_ratio = known_vowels / length
                    
                    state = {
                        'current_state': current_state,
                        'guessed_letters': sorted(list(guessed_letters)),
                        'vowel_ratio': vowel_ratio,
                        'remaining_lives': 6 - wrong_guesses
                    }
                    
                    # Get input tensors
                    char_indices, guessed, vowel_ratio, lives = prepare_input(state)
                    char_indices = char_indices.unsqueeze(0).to(DEVICE)
                    guessed = guessed.unsqueeze(0).to(DEVICE)
                    length_tensor = torch.tensor([length], dtype=torch.float32).to(DEVICE)
                    vowel_ratio = torch.tensor([vowel_ratio], dtype=torch.float32).to(DEVICE)
                    lives = torch.tensor([lives], dtype=torch.float32).to(DEVICE)
                    
                    # Get prediction
                    predictions = model(char_indices, guessed, length_tensor, vowel_ratio, lives)
                    
                    # Get next letter (excluding already guessed letters)
                    valid_preds = [(i, p.item()) for i, p in enumerate(predictions[0]) 
                                  if chr(i + ord('a')) not in guessed_letters]
                    next_letter = chr(max(valid_preds, key=lambda x: x[1])[0] + ord('a'))
                    
                    # Format Q-values for CSV
                    q_values = {chr(i + ord('a')): f"{p.item():.4f}" 
                              for i, p in enumerate(predictions[0])}
                    q_values_str = repr(q_values).replace(',', ';')  # Avoid CSV confusion
                    
                    # Write prediction data
                    row = [
                        word,
                        current_state,
                        ''.join(sorted(guessed_letters)),
                        q_values_str,
                        next_letter,
                        str(next_letter in word_letters)
                    ]
                    f.write(','.join(row) + '\n')
                    
                    # Update game state
                    guessed_letters.add(next_letter)
                    num_guesses += 1
                    if next_letter not in word_letters:
                        wrong_guesses += 1
                    
                    current_state = ''.join([c if c in guessed_letters else '_' for c in word])
                    
                    if '_' not in current_state:
                        word_completed = True
                        break
                
                # Update statistics
                length = len(word)
                word_length_stats[length]['total'] += 1
                word_length_stats[length]['total_guesses'] += num_guesses
                if word_completed and wrong_guesses < 6:
                    word_length_stats[length]['completed'] += 1
    
    logging.info(f"Detailed predictions saved to: {csv_file}")
    
    # Print statistics
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
    
    return dict(word_length_stats)  # Convert defaultdict to regular dict before returning

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

def validate_states(states, num_samples=5):
    """Validate state data structure and tensor dimensions"""
    logging.info(f"\nValidating {num_samples} random states...")
    
    if not states:
        raise ValueError("States list is empty")
        
    logging.info(f"Total states to validate: {len(states)}")
    
    # Print first state for debugging
    logging.info(f"Example state keys: {states[0].keys()}")
    logging.info(f"Example state values: {states[0]}")
    
    # Sample random states
    sample_states = random.sample(states, min(num_samples, len(states)))
    
    for i, state in enumerate(sample_states):
        logging.info(f"\nChecking state {i+1}:")
        
        # Check required keys
        required_keys = {'current_state', 'guessed_letters', 'original_word', 
                        'vowel_ratio', 'target_distribution', 'remaining_lives'}
        missing_keys = required_keys - state.keys()
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")
        
        # Check if word is complete
        if '_' not in state['current_state']:
            raise ValueError(f"Found completed word state: {state['current_state']}")
        
        try:
            # Validate dimensions by preparing tensors
            char_indices, guessed, vowel_ratio, lives = prepare_input(state)
            length = torch.tensor([len(state['current_state'])], dtype=torch.float32, device=DEVICE)
            target = torch.tensor(list(state['target_distribution'].values()), 
                                dtype=torch.float32, device=DEVICE)
            
            # Add batch dimension for testing
            char_indices = char_indices.unsqueeze(0)  # [1, max_word_length]
            guessed = guessed.unsqueeze(0)  # [1, 26]
            length_encoding = torch.tensor([[int(x) for x in format(int(length.item()), '05b')]], 
                                        dtype=torch.float32, device=DEVICE)  # [1, 5]
            vowel_ratio = vowel_ratio.unsqueeze(0)  # [1, 1]
            lives = lives.unsqueeze(0)  # [1, 1]
            
            # Test concatenation
            combined = torch.cat([
                char_indices.float(),  # [1, max_word_length]
                guessed,  # [1, 26]
                length_encoding,  # [1, 5]
                vowel_ratio,  # [1, 1]
                lives  # [1, 1]
            ], dim=1)
            
            logging.info(f"Combined shape: {combined.shape}")
            
            # Validate target distribution
            target_sum = sum(state['target_distribution'].values())
            if not 0.99 <= target_sum <= 1.01:  # Allow small floating point errors
                raise ValueError(f"Target distribution sum = {target_sum}, should be 1.0")
            
        except Exception as e:
            raise ValueError(f"State validation failed: {str(e)}")
    
    logging.info("\nAll validation checks passed!")
    return True

def main():
    # Setup logging first thing in main()
    log_file = setup_logging()
    
    # Now log device info after logging is set up
    logging.info(f"Using device: {DEVICE}")
    if DEVICE.type == "mps":
        logging.info("Running on Apple Silicon")
        if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "enable_mem_efficient_sdp"):
            torch.backends.mps.enable_mem_efficient_sdp(True)

    parser = argparse.ArgumentParser(description='Train Mary Hangman Model')
    parser.add_argument('--force-new-data', action='store_true', 
                       help='Force generation of new dataset even if one exists')
    parser.add_argument('--evaluate', type=str,
                       help='Path to model for evaluation')
    args = parser.parse_args()

    logging.info("Starting Mary's Hangman training")
    
    try:
        # Load or generate data
        train_states, val_states, val_words = load_data(force_new=args.force_new_data)
        
        if train_states is None or args.force_new_data:
            logging.info(f"{'Forcing new dataset generation...' if args.force_new_data else 'No existing dataset found. Generating new dataset...'}")
            training_words, val_words = load_and_preprocess_words()
            
            ngram_dict = None
            
            # Add logging to track dataset generation
            logging.info(f"Generating training states for {len(training_words)} words...")
            train_states = generate_dataset(training_words, ngram_dict, num_games_per_word=1, is_validation=False)
            logging.info(f"Generated {len(train_states)} training states")
            
            logging.info(f"Generating validation states for {len(val_words)} words...")
            val_states = generate_dataset(val_words, ngram_dict, num_games_per_word=1, is_validation=True)
            logging.info(f"Generated {len(val_states)} validation states")
            
            # Verify states before saving
            if not train_states or not val_states:
                raise ValueError("Generated states are empty!")
                
            logging.info("Saving generated datasets...")
            save_data(train_states, val_states, val_words)
        
        # Add more validation logging
        logging.info(f"Validating {len(train_states)} training states...")
        validate_states(train_states)
        logging.info(f"Validating {len(val_states)} validation states...")
        validate_states(val_states)
        logging.info("State validation complete")

        if args.evaluate:
            evaluate_saved_model(args.evaluate)
            return

        # Initialize model
        model = MaryLSTMModel().to(DEVICE)
        num_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Model initialized with {num_params:,} parameters")
        
        # Train model
        logging.info("Starting model training...")
        model, optimizer, metrics = train_model(model, train_states, val_states, val_words)
        
        # Run detailed validation
        stats = run_detailed_validation(model, val_words)
        metrics['word_length_stats'] = dict(stats)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'{DATA_DIR}/mary_model_v2_{timestamp}.pt'
        metrics_path = f'{DATA_DIR}/mary_metrics_v2_{timestamp}.pkl'
        
        # Now we have access to both model and optimizer
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
        logging.error(f"Training failed with error: {str(e)}")
        logging.error(traceback.format_exc())
        raise
    finally:
        logging.info(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    main() 