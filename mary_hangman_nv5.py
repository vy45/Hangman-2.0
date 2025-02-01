#!/usr/bin/env python3

import sys
import subprocess
import pkg_resources
import logging
from pathlib import Path
import platform

def check_and_install_packages():
    """Check if required packages are installed and install if missing"""
    required = {
        'torch': '2.0.0',
        'numpy': '1.21.0',
        'scikit-learn': '1.0.0',
        'tqdm': '4.65.0',
        'pandas': '1.3.0',
        'matplotlib': '3.4.0',
        'seaborn': '0.11.0',
        'psutil': '5.9.0',
        'torchinfo': '1.7.0'
    }
    
    optional = {
        'torch-geometric': '2.0.0',  # For GPU support
        'pytest': '6.0.0',           # For testing
        'black': '22.0.0',           # For code formatting
        'pylint': '2.12.0'           # For code quality
    }
    
    missing = []
    missing_optional = []
    
    # Check installed packages
    installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    # Check required packages
    for package, min_version in required.items():
        if package not in installed:
            missing.append(f"{package}>={min_version}")
        else:
            current = pkg_resources.parse_version(installed[package])
            required_ver = pkg_resources.parse_version(min_version)
            if current < required_ver:
                missing.append(f"{package}>={min_version}")
    
    # Check optional packages
    for package, min_version in optional.items():
        if package not in installed:
            missing_optional.append(f"{package}>={min_version}")
    
    if missing:
        print("Installing required packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("Required package installation complete!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install required packages: {str(e)}")
            sys.exit(1)
    
    if missing_optional:
        print("\nOptional packages are not installed. These might be useful:")
        for pkg in missing_optional:
            print(f"  {pkg}")
        print("\nYou can install them with:")
        print(f"pip install {' '.join(missing_optional)}")
    
    # Check for MPS support on Apple Silicon
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            import torch
            if torch.backends.mps.is_available():
                print("MPS (Metal Performance Shaders) is available - will use Apple Silicon GPU")
            else:
                print("MPS is not available. Please ensure you have:")
                print("1. macOS 12.3 or later")
                print("2. PyTorch 2.0 or later")
                print("3. Proper PyTorch installation for Apple Silicon")
                print("\nYou can install PyTorch for Apple Silicon with:")
                print("pip3 install --pre torch torchvision torchaudio")
        except Exception as e:
            print(f"Error checking MPS availability: {str(e)}")

# Run package check before other imports
if __name__ == "__main__":
    check_and_install_packages()

# Including Remaining Lives as a feature. 
# more data, bigger model, transformer

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
import traceback
import torch.optim as optim
import re
from collections import Counter
from multiprocessing import Pool
import time
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Constants
QUICK_TEST = False
DEVICE = (
    torch.device("mps") 
    if torch.backends.mps.is_available() 
    else torch.device("cuda") 
    if torch.cuda.is_available() 
    else torch.device("cpu")
)
logging.info(f"Using device: {DEVICE}")
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
BATCH_SIZE = 64
DATA_DIR = 'hangman_data'
VALIDATION_EPISODES = 10000
FREQ_CUTOFF = 10
SIMULATION_CORRECT_GUESS_PROB = 0.5
MIN_NGRAM_LENGTH = 3
MAX_NGRAM_LENGTH = 5
EPOCHS = 10 if QUICK_TEST else 100
COMPLETION_EVAL_WORDS = 1000 if QUICK_TEST else 10000
 
class MaryTransformerModel(nn.Module):
    def __init__(self, hidden_dim=512, embedding_dim=64, nhead=8, num_layers=6, dropout_rate=0.4, use_batch_norm=False):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        
        # Embedding layer for characters (28 inputs: 0-25=a-z, 26=mask, 27=padding)
        self.char_embedding = nn.Embedding(28, embedding_dim, padding_idx=27)
        
        # Length encoding dictionary (5-bit representation)
        self.length_encoding = {
            i: torch.tensor([int(x) for x in format(i, '05b')], dtype=torch.float32)
            for i in range(1, 33)  # Support lengths 1-32
        }
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout_rate)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Batch normalization layers (optional)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(embedding_dim + 26 + 5 + 2)  # +2 for vowel ratio and remaining lives
            self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final dense layers
        self.combine = nn.Linear(embedding_dim + 26 + 5 + 2, hidden_dim)
        self.output = nn.Linear(hidden_dim, 26)
        
    def forward(self, word_state, guessed_letters, word_length, vowel_ratio, remaining_lives):
        # Get batch size
        batch_size = word_state.size(0)
        
        # Embedding and positional encoding
        embedded = self.char_embedding(word_state)  # [batch_size, seq_len, embedding_dim]
        embedded = self.pos_encoder(embedded)
        
        # Create attention mask for padding
        padding_mask = (word_state == 27)  # True for padding tokens
        
        # Transformer processing
        transformer_out = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        # Get final output (use mean pooling over sequence length)
        final_out = transformer_out.mean(dim=1)  # [batch_size, embedding_dim]
        
        # Create length encoding
        length_encoding = torch.stack([
            self.length_encoding[int(l.item())]
            for l in word_length
        ]).to(final_out.device)
        
        # Fix dimensions
        if vowel_ratio.dim() == 1:
            vowel_ratio = vowel_ratio.unsqueeze(-1)
        if remaining_lives.dim() == 1:
            remaining_lives = remaining_lives.unsqueeze(-1)
        
        # Concatenate features
        combined_features = torch.cat([
            final_out,  # [batch_size, embedding_dim]
            guessed_letters,  # [batch_size, 26]
            length_encoding,  # [batch_size, 5]
            vowel_ratio,  # [batch_size, 1]
            remaining_lives  # [batch_size, 1]
        ], dim=1)
        
        # Apply batch norm if enabled
        if self.use_batch_norm:
            combined_features = self.bn1(combined_features)
        
        # Dense layers
        hidden = self.dropout(F.relu(self.combine(combined_features)))
        if self.use_batch_norm:
            hidden = self.bn2(hidden)
            
        return F.softmax(self.output(hidden), dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=32):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x) 
    

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.last_loss = None
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.last_loss = val_loss
            return False
            
        # If loss improved significantly from last epoch, reset counter
        if val_loss < self.last_loss - self.min_delta:
            self.counter = 0
            if val_loss < self.best_loss:
                self.best_loss = val_loss
        # If loss got worse or stayed roughly the same
        else:
            self.counter += 1
            
        self.last_loss = val_loss
        
        # Only stop if we've had patience number of epochs without improvement
        if self.counter >= self.patience:
            return True
            
        return False

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

def simulate_game_states(args):
    """Simulate a game with proper vowel-aware strategy"""
    word = args[0]  # Unpack word from args tuple
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

def build_letter_frequencies(words):
    """Build dictionary of letter frequencies by word length"""
    length_freqs = {}
    
    # Group words by length
    words_by_length = defaultdict(list)
    for word in words:
        words_by_length[len(word)].append(word)
    
    # Calculate frequencies for each length
    for length, word_list in words_by_length.items():
        freqs = calculate_letter_frequencies(word_list)
        length_freqs[length] = freqs
    
    return length_freqs

def calculate_target_distribution(current_state, guessed_letters, original_word, train_words=None, 
                                ngram_dict=None, is_validation=False, vowel_ratio=None, 
                                remaining_lives=None, length_freqs=None):
    """Calculate target distribution using modified strategy"""
    word_length = len(current_state)
    target_dist = {l: 0.0 for l in ALPHABET}
    
    # Strategy 1: Known missing letters (always used)
    known_dist = {l: 0.0 for l in ALPHABET}
    remaining_letters = set(original_word) - set(guessed_letters)
    for letter in remaining_letters:
        known_dist[letter] = 1.0
    
    # For validation states or single remaining letter, just use known missing letters
    if is_validation or len(remaining_letters)==1:
        # Normalize known distribution
        total = sum(known_dist.values())
        if total > 0:
            return {k: v/total for k, v in known_dist.items()}
        return known_dist
    
    # Adjust vowel weights at the beginning
    known_dist = adjust_vowel_weights(known_dist, current_state, vowel_ratio)
    
    # For training states:
    # 1. Check if we have any valid n-grams with one mask
    valid_ngrams = []
    for n in range(3, 7):  # n-grams from length 3 to 6
        ngrams = get_masked_ngrams(current_state, n)
        # Only add ngrams that have exactly one mask
        valid_ngrams.extend([ng for ng in ngrams if ng.count('_') == 1])
    
    # If no valid n-grams found, use letter frequencies
    if not valid_ngrams and length_freqs is not None:
        # Get pre-calculated frequencies for this word length
        freqs = length_freqs.get(word_length, {})
        
        # Multiply known distribution with frequencies
        for letter in remaining_letters:
            target_dist[letter] = known_dist[letter] * freqs.get(letter, 0.0)
    
    # If we have valid n-grams, use n-gram based weights
    else:
        ngram_weights = {l: 0.0 for l in remaining_letters}
        total_weight = 0
        
        for ngram in valid_ngrams:
            # We know each ngram has exactly one mask from the filtering above
            mask_pos = ngram.index('_')
            ngram_len = len(ngram)
            # Scale weight based on n-gram length
            length_weight = (ngram_len) ** 2  # 36.0 for length 6, 9 for length 3
            
            # Only try known missing letters
            for letter in remaining_letters:
                filled = ngram[:mask_pos] + letter + ngram[mask_pos+1:]
                if filled in ngram_dict:
                    ngram_weights[letter] += ngram_dict[filled] * length_weight
                    total_weight += ngram_dict[filled] * length_weight
        
        # Normalize n-gram weights
        if total_weight > 0:
            ngram_weights = {k: v/total_weight for k, v in ngram_weights.items()}
            
            # Multiply known distribution with n-gram weights
            for letter in remaining_letters:
                target_dist[letter] = known_dist[letter] * ngram_weights.get(letter, 0.0)
    
    # Zero out guessed letters
    for letter in guessed_letters:
        target_dist[letter] = 0.0
    
    # Normalize final distribution
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

def calculate_target_distribution_wrapper(args):
    """Wrapper function to unpack arguments for multiprocessing"""
    current_state, guessed_letters, original_word, train_words, ngram_dict, is_validation, vowel_ratio, remaining_lives, length_freqs = args
    return calculate_target_distribution(
        current_state, 
        guessed_letters, 
        original_word, 
        train_words, 
        ngram_dict, 
        is_validation, 
        vowel_ratio,
        remaining_lives,
        length_freqs
    )

def generate_dataset(words, ngram_dict, is_validation=False, train_words=None):
    """Generate dataset of game states"""
    all_states = []
    
    # Pre-calculate letter frequencies by word length
    length_freqs = build_letter_frequencies(train_words) if train_words is not None else None
    
    # Generate states for each word
    with Pool() as pool:
        # Simulate two games for each word
        word_pairs = [(word,) for word in words for _ in range(2)]  # Create tuples for each game
        all_states = list(tqdm(
            pool.imap(simulate_game_states, word_pairs, chunksize=10),
            total=len(word_pairs),  # This will be double the number of words
            desc="Generating game states"
        ))
    all_states = [s for states in all_states for s in states]  # Flatten
    
    with time_block("Target distribution calculation"):
        with Pool() as pool:
            calc_args = [
                (s['current_state'], 
                 set(s['guessed_letters']), 
                 s['original_word'],
                 train_words,
                 None if is_validation else ngram_dict,
                 is_validation,
                 s['vowel_ratio'],
                 s['remaining_lives'],
                 length_freqs) for s in all_states
            ]
            
            target_dists = list(tqdm(
                pool.imap(
                    calculate_target_distribution_wrapper,
                    calc_args,
                    chunksize=100
                ),
                total=len(all_states),
                desc="Calculating target distributions"
            ))
        
        for state, dist in zip(all_states, target_dists):
            state['target_distribution'] = dist
    
    return all_states

def build_ngram_dictionary(words):
    """Build dictionary of n-gram frequencies for lengths 3-6"""
    ngram_dict = defaultdict(int)
    
    for word in tqdm(words, desc="Building n-gram dictionary"):
        word_len = len(word)
        for n in range(3, 7):  # Only n-grams of length 3-6
            if word_len >= n:
                for i in range(word_len - n + 1):
                    ngram = word[i:i+n]
                    ngram_dict[ngram] += 1
    
    return dict(ngram_dict)  # Convert to regular dict

# def simulate_game(word):
#     """Simulate a single game and generate states"""
#     states = []
#     guessed_letters = set()
#     word_letters = set(word)
    
#     while len(guessed_letters) < 26 and len(word_letters - guessed_letters) > 0:
#         # Current word state
#         current_state = ''.join([c if c in guessed_letters else '_' for c in word])
        
#         # Calculate target distribution (only unguessed correct letters)
#         target_dist = {l: 0.0 for l in ALPHABET}
#         remaining_letters = word_letters - guessed_letters
#         for letter in remaining_letters:
#             target_dist[letter] = 1.0
        
#         # Normalize distribution
#         total = sum(target_dist.values())
#         if total > 0:
#             target_dist = {k: v/total for k, v in target_dist.items()}
        
#         # Store state
#         states.append({
#             'current_state': current_state,
#             'guessed_letters': sorted(list(guessed_letters)),
#             'target_distribution': target_dist
#         })
        
#         # Simulate next guess
#         if random.random() < SIMULATION_CORRECT_GUESS_PROB and remaining_letters:
#             next_letter = random.choice(list(remaining_letters))
#         else:
#             available_letters = set(ALPHABET) - guessed_letters
#             next_letter = random.choice(list(available_letters))
        
#         guessed_letters.add(next_letter)
    
#     return states

def prepare_input(state):
    """Prepare input tensors for model"""
    char_indices = torch.tensor([
        ALPHABET.index(c) if c in ALPHABET else 26 
        for c in state['current_state']
    ], dtype=torch.long)
    
    guessed = torch.zeros(26)
    for letter in state['guessed_letters']:
        if letter in ALPHABET:
            guessed[ALPHABET.index(letter)] = 1
            
    vowel_ratio = torch.tensor([state['vowel_ratio']], dtype=torch.float32)
    remaining_lives = torch.tensor([state['remaining_lives']], dtype=torch.float32) / 6.0  # Scale to [0,1]
    
    return char_indices, guessed, vowel_ratio, remaining_lives

def save_data(train_states, val_states, val_words):
    """Save datasets with timestamp"""
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = data_dir / f'hangman_states_nv3_{timestamp}.pkl'
    
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
    
    dataset_files = list(data_dir.glob('hangman_states_nv3_*.pkl'))
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
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'mary_hangman_nv5_{timestamp}.log'
    
    # Configure logging to show DEBUG messages
    logging.basicConfig(
        level=logging.DEBUG,  # Changed from INFO to DEBUG
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("Logging setup complete. Log file: " + str(log_file))
    return log_file



def count_missing_letters(state):
    """Count number of missing letter positions in a state"""
    return state['current_state'].count('_')

def prepare_curriculum_batches(train_states, epoch, max_missing=None):
    """Prepare batches with curriculum learning based on number of missing letters"""
    # Filter states based on epoch (curriculum)
    if max_missing is not None:
        filtered_states = [
            state for state in train_states 
            if count_missing_letters(state) <= max_missing
        ]
    else:
        filtered_states = train_states
    
    # Shuffle all states first
    random.shuffle(filtered_states)
    
    # Sort by length
    sorted_states = sorted(filtered_states, key=lambda x: len(x['current_state']))
    
    # Create batches (not necessarily same length)
    batches = []
    for i in range(0, len(sorted_states), BATCH_SIZE):
        batch = sorted_states[i:i + BATCH_SIZE]
        batches.append(batch)
    
    # Shuffle batches
    random.shuffle(batches)
    return batches

def prepare_padded_batch(batch):
    """Prepare a batch with padding to max length in batch"""
    # Find max length in batch
    max_len = max(len(state['current_state']) for state in batch)
    
    # Prepare tensors
    word_states = []
    guessed_letters = []
    lengths = []
    vowel_ratios = []
    remaining_lives = []
    targets = []
    
    for state in batch:
        # Get basic tensors
        char_indices, guessed, vowel_ratio, lives = prepare_input(state)
        current_len = len(char_indices)
        
        # Pad char_indices if needed (use 27 for padding)
        if current_len < max_len:
            padding = torch.full((max_len - current_len,), 27, dtype=torch.long)
            char_indices = torch.cat([char_indices, padding])
        
        word_states.append(char_indices)
        guessed_letters.append(guessed)
        lengths.append(len(state['current_state']))  # Original length before padding
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
    
    return word_states, guessed_letters, lengths, vowel_ratios, remaining_lives, targets

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
    """Modified training loop with curriculum learning"""
    logging.info(f"Starting training for {epochs} epochs")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=4, min_delta=0.001)
    
    # Save best model
    best_val_loss = float('inf')
    best_model_path = None
    
    # Track metrics
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
        path = f'{DATA_DIR}/mary_model_nv5_epoch{epoch}_{timestamp}.pt'
        
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
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            
            # Curriculum: increase max missing letters with epoch
            # First 5 epochs: increase from 1 to 5 missing letters
            # After that: use full dataset
            max_missing = epoch + 1 if epoch < 5 else None
            
            # Prepare batches for this epoch
            batches = prepare_curriculum_batches(train_states, epoch, max_missing)
            
            batch_iterator = tqdm(
                batches,
                desc=f"Epoch {epoch + 1}/{epochs} (max missing: {max_missing if max_missing else 'all'})",
                leave=False
            )
            
            for batch in batch_iterator:
                # Prepare padded batch
                word_states, guessed_letters, lengths, vowel_ratios, remaining_lives, targets = prepare_padded_batch(batch)
                
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
            
            # Validation phase using length-specific batches
            model.eval()
            val_loss = evaluate_model(model, val_states)  # This uses prepare_length_batches
            completion_rate = calculate_completion_rate(model, val_words[:COMPLETION_EVAL_WORDS])
            
            # Run detailed validation every 4 epochs
            if (epoch + 1) % 4 == 0:
                logging.info("\nRunning detailed validation...")
                detailed_stats = run_detailed_validation(model, val_words[:COMPLETION_EVAL_WORDS])
                metrics[f'detailed_stats_epoch_{epoch+1}'] = dict(detailed_stats)
            
            # Store metrics
            metrics['val_loss'].append(val_loss)
            metrics['completion_rate'].append(completion_rate)
            metrics['train_loss'].append(total_loss / num_batches)
            
            # Log progress
            logging.info(f"\nEpoch {epoch + 1} Results:")
            logging.info(f"Train Loss: {total_loss / num_batches:.4f}")
            logging.info(f"Val Loss: {val_loss:.4f}")
            logging.info(f"Completion Rate: {completion_rate:.2%}")
            logging.info(f"Average Gradient Norm: {np.mean(metrics['gradient_norms'][-num_batches:]):.2f}")
            logging.info(f"Average Weight Norm: {np.mean(metrics['weight_norms'][-num_batches:]):.2f}")
            
            # Save model if it's the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = save_checkpoint(epoch + 1, val_loss)
                logging.info(f"New best model saved to: {best_model_path}")
            
            # Early stopping check (only after epoch 7)
            if epoch >= 7 and early_stopping(val_loss):
                logging.info(f"\nEarly stopping triggered at epoch {epoch + 1}")
                metrics['stopped_epoch'] = epoch + 1
                break
            
            scheduler.step(val_loss)
            
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        logging.error(traceback.format_exc())
        raise
    
    finally:
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
    csv_file = f'{DATA_DIR}/predictions_{model_type}_{timestamp}.csv'
    
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
        model = MaryTransformerModel().to(DEVICE)
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
        
        # Validate dimensions by preparing tensors
        char_indices, guessed, vowel_ratio, lives = prepare_input(state)
        length = torch.tensor([len(state['current_state'])], dtype=torch.float32)
        
        # Try tensor concatenation
        try:
            # Add batch dimension for testing
            char_indices = char_indices.unsqueeze(0)  # [1, max_word_length]
            guessed = guessed.unsqueeze(0)  # [1, 26]
            length_encoding = torch.tensor([[int(x) for x in format(int(length.item()), '05b')]], 
                                        dtype=torch.float32)  # [1, 5]
            vowel_ratio = torch.tensor([[vowel_ratio]], dtype=torch.float32)  # [1, 1]
            lives = torch.tensor([[lives]], dtype=torch.float32)  # [1, 1]
            
            # Test concatenation with transformer output
            mock_transformer_out = torch.zeros((1, 64))  # [1, embedding_dim]
            combined = torch.cat([
                mock_transformer_out,  # [1, embedding_dim]
                guessed,  # [1, 26]
                length_encoding,  # [1, 5]
                vowel_ratio,  # [1, 1]
                lives  # [1, 1]
            ], dim=1)
            
            logging.info(f"Combined shape: {combined.shape}")
            
        except Exception as e:
            raise ValueError(f"Tensor concatenation failed: {str(e)}")
    
    logging.info("\nAll validation checks passed!")
    return True

def main():
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Train Mary Hangman Model')
    parser.add_argument('--force-new-data', action='store_true', 
                       help='Force generation of new dataset even if one exists')
    parser.add_argument('--evaluate', type=str,
                       help='Path to model for evaluation')
    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging()
    logging.info("Starting Mary's Hangman training")
    logging.info(f"Using device: {DEVICE}")
    
    try:
        # Load or generate data
        train_states, val_states, val_words = load_data(force_new=args.force_new_data)
        
        if train_states is None or args.force_new_data:
            logging.info(f"{'Forcing new dataset generation...' if args.force_new_data else 'No existing dataset found. Generating new dataset...'}")
            training_words, val_words = load_and_preprocess_words()
            
            # Build n-gram dictionary
            with open('words_250000_train.txt', 'r') as f:
                words = [w.strip().lower() for w in f.readlines()]
            ngram_dict = build_ngram_dictionary(words)
            
            # Generate datasets
            train_states = generate_dataset(training_words, ngram_dict, is_validation=False, train_words=training_words)
            val_states = generate_dataset(val_words, ngram_dict, is_validation=True, train_words=training_words)
            
            save_data(train_states, val_states, val_words)
        
        # Validate states before training
        validate_states(train_states)
        validate_states(val_states)

        if args.evaluate:
            evaluate_saved_model(args.evaluate)
            return

        # Initialize model
        model = MaryTransformerModel().to(DEVICE)
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
        model_path = f'{DATA_DIR}/mary_model_nv5_{timestamp}.pt'
        metrics_path = f'{DATA_DIR}/mary_metrics_nv5_{timestamp}.pkl'
        
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