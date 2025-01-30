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
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
BATCH_SIZE = 64
DATA_DIR = 'hangman_data'
VALIDATION_EPISODES = 10000
COMPLETION_EVAL_WORDS = 1000
FREQ_CUTOFF = 10
SIMULATION_CORRECT_GUESS_PROB = 0.5
MIN_NGRAM_LENGTH = 3
MAX_NGRAM_LENGTH = 5
LEARNING_RATE = 0.0003  # Lower learning rate for transformer

class TransformerHangmanModel(nn.Module):
    def __init__(self, hidden_dim=128, embedding_dim=8, num_heads=4, num_layers=3, dropout_rate=0.4, use_batch_norm=False):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        
        # Embedding layer for characters
        self.char_embedding = nn.Embedding(27, embedding_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            self.init_positional_encoding(32, embedding_dim),
            requires_grad=False  # Usually kept fixed #why sinusoidal initialisation?
        )
        
        # Length encoding
        self.length_encoding = {
            i: torch.tensor([int(x) for x in format(i, '05b')], dtype=torch.float32)
            for i in range(1, 33)
        }
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Batch normalization
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(embedding_dim + 26 + 5 + 1)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Dense layers
        self.combine = nn.Linear(embedding_dim + 26 + 5 + 1, hidden_dim)
        self.output = nn.Linear(hidden_dim, 26)
        
    def init_positional_encoding(self, max_len, d_model):
        """Initialize positional encoding with sinusoidal values"""
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        return pe
        
    def create_padding_mask(self, word_state):
        # Create mask for padding tokens
        return word_state == 26  # 26 is the padding token
        
    def forward(self, word_state, guessed_letters, word_length, vowel_ratio):
        # Get sequence length for each batch
        batch_size = word_state.size(0)
        seq_len = word_state.size(1)
        
        # Embed characters and add positional encoding
        embedded = self.char_embedding(word_state)
        embedded = embedded + self.pos_encoding[:, :seq_len, :]
        
        # Create padding mask
        padding_mask = self.create_padding_mask(word_state)
        
        # Pass through transformer
        transformer_out = self.transformer(
            embedded,
            src_key_padding_mask=padding_mask
        )
        
        # Global average pooling over sequence length
        pooled = torch.mean(transformer_out, dim=1)
        
        # Get length encoding
        length_encoding = torch.stack([self.length_encoding[l.item()] for l in word_length])
        
        # Combine features
        combined_features = torch.cat([
            pooled,
            guessed_letters,
            length_encoding.to(pooled.device),
            vowel_ratio.unsqueeze(1)
        ], dim=1)
        
        # Apply batch norm if enabled
        if self.use_batch_norm:
            combined_features = self.bn1(combined_features)
        
        # Dense layers with dropout
        hidden = self.dropout(F.relu(self.combine(combined_features)))
        
        if self.use_batch_norm:
            hidden = self.bn2(hidden)
        
        # Output probabilities
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
    """Simulate a game and generate states"""
    states = []
    guessed_letters = set()
    word_letters = set(word)
    VOWELS = set('aeiou')
    unguessed_vowels = VOWELS - guessed_letters
    
    while len(guessed_letters) < 26 and len(word_letters - guessed_letters) > 0 and wrong_guesses < 6:
        # Calculate current state and vowel ratio
        current_state = ''.join([c if c in guessed_letters else '_' for c in word])
        # Calculate vowel ratio based on total word length instead of known letters
        known_vowels = sum(1 for c in word if c in VOWELS and c in guessed_letters)
        vowel_ratio = known_vowels / len(current_state)
        
        states.append({
            'current_state': current_state,
            'guessed_letters': sorted(list(guessed_letters)),
            'original_word': word,
            'vowel_ratio': vowel_ratio
        })
        
        # Choose next letter
        if vowel_ratio < 0.2 and unguessed_vowels:  # Prioritize vowels early
            next_letter = random.choice(list(unguessed_vowels))
            unguessed_vowels.remove(next_letter)
        else:
            # Regular guessing logic
            if random.random() < SIMULATION_CORRECT_GUESS_PROB and (word_letters - guessed_letters):
                next_letter = random.choice(list(word_letters - guessed_letters))
            else:
                available_letters = set(ALPHABET) - guessed_letters
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
        matching = get_matching_words(current_state, guessed_letters - set(current_state), train_words)
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

def calculate_target_distribution_wrapper(args):
    """Wrapper function to unpack arguments for multiprocessing"""
    current_state, guessed_letters, original_word, train_words, ngram_dict, is_validation, vowel_ratio = args
    return calculate_target_distribution(
        current_state, 
        guessed_letters, 
        original_word, 
        train_words, 
        ngram_dict, 
        is_validation, 
        vowel_ratio
    )

def generate_dataset(words, ngram_dict, is_validation=False, train_words=None):
    """Generate dataset with balanced upsampling and target distributions"""
    with time_block("Organizing words by length"):
        # Pre-organize words by length for faster matching
        words_by_length = defaultdict(list)
        for word in words:
            words_by_length[len(word)].append(word)
        
        # Pre-organize training words by length if needed for validation
        train_words_by_length = None
        if train_words:
            train_words_by_length = defaultdict(list)
            for word in train_words:
                train_words_by_length[len(word)].append(word)
        else:
            # For training dataset, use words_by_length
            train_words_by_length = words_by_length
        
        # Debug logging
        # for length, words_list in train_words_by_length.items():
            # logging.debug(f"Length {length}: {len(words_list)} training words available")
    
    with time_block("Initial dataset generation"):
        logging.info("Generating initial dataset...")
        all_states = []
        length_counts = defaultdict(int)
        word_length_map = defaultdict(list)
        
        # First pass: generate initial states
        for word in tqdm(words, desc="Initial simulation"):
            states = simulate_game_states(word)
            all_states.extend(states)
            length = len(word)
            length_counts[length] += len(states)
            word_length_map[length].append(word)
    
    with time_block("Upsampling process"):
        if not is_validation:
            # Print initial statistics and calculate upsampling
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
            
            # Upsample based on calculated factors
            upsampling_factors = calculate_upsampling_factors(length_counts)
            logging.info("\nUpsampling underrepresented lengths...")
            
            for length, factor in upsampling_factors.items():
                if factor > 1.0:
                    current_count = length_counts[length]
                    target_count = int(current_count * factor)
                    additional_needed = target_count - current_count
                    
                    if additional_needed > 0:
                        logging.info(f"\nLength {length} needs {additional_needed} more states")
                        
                        additional_states = []
                        words = word_length_map[length]
                        while len(additional_states) < additional_needed:
                            word = random.choice(words)
                            states = simulate_game_states(word)
                            additional_states.extend(states)
                        
                        # Trim to exact number needed
                        all_states.extend(additional_states[:additional_needed])
                        length_counts[length] += additional_needed
            
            # Print final statistics
            total_states = len(all_states)
            logging.info("\nFinal state distribution:")
            for length in sorted(length_counts.keys()):
                count = length_counts[length]
                percentage = count / total_states
                logging.info(f"Length {length:2d}: {count:6d} states ({percentage:.2%})")
                
                if count < BATCH_SIZE:
                    logging.warning(f"Length {length} has fewer than {BATCH_SIZE} states!")
    
    with time_block("Target distribution calculation"):
        # Calculate target distributions in parallel
        with Pool() as pool:
            # Prepare arguments as tuples
            calc_args = [
                (s['current_state'], 
                 set(s['guessed_letters']), 
                 s['original_word'],
                 # Only pass train_words for short words
                 train_words_by_length[len(s['current_state'])] if len(s['current_state']) < FREQ_CUTOFF else None,
                 None if is_validation else ngram_dict,
                 is_validation,
                 s['vowel_ratio']) for s in all_states
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
        
        # Update states with calculated distributions
        for state, dist in zip(all_states, target_dists):
            state['target_distribution'] = dist
    
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
    
    return char_indices, guessed, vowel_ratio

def save_data(train_states, val_states, val_words):
    """Save datasets with timestamp"""
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = data_dir / f'hangman_states_{timestamp}.pkl'
    
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
    
    dataset_files = list(data_dir.glob('hangman_states_*.pkl'))
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
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'transformer_hangman_{timestamp}.log'
    
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

def train_model(model, train_states, val_states, val_words, epochs=100):
    """Modified training loop for length-specific batches"""
    logging.info(f"Starting training for {epochs} epochs")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
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
        'gradient_norms': [],  # Track gradient norms
        'weight_norms': []     # Track weight norms
    }
    
    start_time = datetime.now()
    
    def save_checkpoint(epoch, val_loss):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f'{DATA_DIR}/transformer_model_epoch{epoch}_{timestamp}.pt'
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
            
            # Prepare and shuffle batches for this epoch
            batches = prepare_length_batches(train_states)
            random.shuffle(batches)  # Shuffle batch order
            
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
                
                # Forward pass
                optimizer.zero_grad()
                predictions = model(word_states, guessed_letters, lengths, vowel_ratios)
                
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
            loss = F.kl_div(predictions.log(), targets, reduction='batchmean')
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else float('inf')

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
                
                char_indices, guessed, vowel_ratio = prepare_input(state)
                char_indices = char_indices.unsqueeze(0).to(DEVICE)
                guessed = guessed.unsqueeze(0).to(DEVICE)
                
                # Get prediction
                predictions = model(char_indices, guessed, torch.tensor([len(word)]), torch.tensor([vowel_ratio]))
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
                
                char_indices, guessed, vowel_ratio = prepare_input(state)
                char_indices = char_indices.unsqueeze(0).to(DEVICE)
                guessed = guessed.unsqueeze(0).to(DEVICE)
                
                predictions = model(char_indices, guessed, torch.tensor([len(word)]), torch.tensor([vowel_ratio]))
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
        model = TransformerHangmanModel(
            hidden_dim=128,
            embedding_dim=32,
            num_heads=4,
            num_layers=3,
            dropout_rate=0.4,
            use_batch_norm=True
        ).to(DEVICE)
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
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Train transformer Hangman Model')
    parser.add_argument('--force-new-data', action='store_true', 
                       help='Force generation of new dataset even if one exists')
    parser.add_argument('--evaluate', type=str,
                       help='Path to model for evaluation')
    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging()
    logging.info("Starting transformer's Hangman training")
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
        
        if args.evaluate:
            evaluate_saved_model(args.evaluate)
            return

        # Initialize model
        model = TransformerHangmanModel(
            hidden_dim=128,
            embedding_dim=32,
            num_heads=4,
            num_layers=3,
            dropout_rate=0.4,
            use_batch_norm=True
        ).to(DEVICE)
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
        model_path = f'{DATA_DIR}/transformer_model_{timestamp}.pt'
        metrics_path = f'{DATA_DIR}/transformer_metrics_{timestamp}.pkl'
        
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