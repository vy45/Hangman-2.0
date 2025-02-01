import torch
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
import random
import pickle
import os
from tqdm import tqdm
from datetime import datetime
import logging
from pathlib import Path
import re
from collections import Counter
from multiprocessing import Pool
from functools import partial
import time

# Constants
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
BATCH_SIZE = 64
DATA_DIR = 'hangman_data'
MIN_NGRAM_LENGTH = 3
MAX_NGRAM_LENGTH = 5
FREQ_CUTOFF = 12

def setup_logging():
    """Setup logging similar to mary_hangman.py"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'generate_data_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

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
    
    while len(guessed_letters) < 26 and len(word_letters - guessed_letters) > 0:
        # Current word state
        current_state = ''.join([c if c in guessed_letters else '_' for c in word])
        
        # Calculate vowel ratio from current state
        VOWELS = set('aeiou')
        known_letters = set(c for c in current_state if c != '_')
        known_vowels = sum(1 for c in known_letters if c in VOWELS)
        known_consonants = len(known_letters) - known_vowels
        
        # If we have known letters, calculate ratio from them
        # Otherwise use a default ratio (0.4 - typical English ratio)
        if known_letters:
            vowel_ratio = known_vowels / len(known_letters)
        else:
            vowel_ratio = 0
        
        states.append({
            'current_state': current_state,
            'guessed_letters': sorted(list(guessed_letters)),
            'original_word': word,
            'vowel_ratio': vowel_ratio
        })
        
        # Simulate next guess
        if random.random() < 0.5 and (word_letters - guessed_letters):
            next_letter = random.choice(list(word_letters - guessed_letters))
        else:
            available_letters = set(ALPHABET) - guessed_letters
            next_letter = random.choice(list(available_letters))
        
        guessed_letters.add(next_letter)
    
    return states

def adjust_vowel_weights(target_dist, current_state, vowel_ratio):
    """Adjust vowel weights based on current state vowel ratio"""
    VOWELS = set('aeiou')
    
    # Get max weight of known letters (if any) # this is not max weight of known letters, it should be from
    known_max = max(target_dist.values())
    
    # Adjust vowel weights based on current ratio
    if vowel_ratio >= 0.4:  # Too many vowels already, reduce vowel weights
        for vowel in VOWELS:
            # if vowel not in current_state:  # Only adjust if not a known letter
            target_dist[vowel] *= 0.7
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
        23: 2.0
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

def load_and_split_data(filename='words_250000_train.txt'):
    """Load words and split into train/validation sets with stratification"""
    with open(filename, 'r') as f:
        words = [word.strip().lower() for word in f.readlines()]
    
    # words = words[:10000]

    # Filter words
    words = [w for w in words if all(c in ALPHABET for c in w)]
    word_lengths = [len(w) for w in words]
    
    # Count frequency of each length
    length_counts = defaultdict(int)
    for length in word_lengths:
        length_counts[length] += 1
    
    # Filter out words with lengths that appear too infrequently
    min_samples_per_length = 10  # Ensure enough samples for validation
    valid_lengths = {length for length, count in length_counts.items() 
                    if count >= min_samples_per_length}
    
    filtered_words = [w for w in words if len(w) in valid_lengths]
    filtered_lengths = [len(w) for w in filtered_words]
    
    # Perform stratified split
    train_words, val_words = train_test_split(
        filtered_words, 
        test_size=0.2, 
        stratify=filtered_lengths,
        random_state=42
    )
    
    return train_words, val_words

def debug_target_distribution(current_state, original_word, guessed_letters, vowel_ratio, train_words, is_validation=False):
    """Debug helper to see step-by-step target distribution calculation"""
    print("\nDEBUG TARGET DISTRIBUTION CALCULATION")
    print("=====================================")
    print(f"Input State: {current_state}")
    print(f"Original Word: {original_word}")
    print(f"Guessed Letters: {sorted(list(guessed_letters))}")
    print(f"Vowel Ratio: {vowel_ratio:.2f}")
    print(f"Mode: {'Validation' if is_validation else 'Training'}")
    
    print("\nStep 1: Known Missing Letters")
    print("---------------------------")
    known_dist = {l: 0.0 for l in ALPHABET}
    remaining_letters = set(original_word) - set(guessed_letters)
    for letter in remaining_letters:
        known_dist[letter] = 1.0
    print(f"Remaining letters: {sorted(list(remaining_letters))}")
    print("Known distribution:")
    for l, v in sorted(known_dist.items()):
        if v > 0:
            print(f"{l}: {v:.2f}")
    
    print("\nStep 2: Letter Frequencies")
    print("------------------------")
    word_length = len(current_state)
    target_dist = {l: 0.0 for l in ALPHABET}
    
    if train_words is not None and word_length < FREQ_CUTOFF:
        print(f"Word length {word_length} < {FREQ_CUTOFF}, calculating frequencies")
        print(f"Available training words: {len(train_words)}")
        matching = get_matching_words(current_state, guessed_letters - set(current_state), train_words)
        print(f"Matching words: {matching[:10]}{'...' if len(matching) > 10 else ''}")
        print(f"Total matching words: {len(matching)}")
        
        freqs = calculate_letter_frequencies(matching)
        print("\nCalculated frequencies:")
        for l, v in sorted(freqs.items(), key=lambda x: x[1], reverse=True):
            if v > 0:
                print(f"{l}: {v:.4f}")
        
        print("\nCombining known (0.5) and frequencies (0.5):")
        for letter in ALPHABET:
            target_dist[letter] = 0.5 * known_dist[letter] + 0.5 * freqs.get(letter, 0.0)
    
    if not is_validation and word_length >= 10:
        print("\nStep 3: N-gram Predictions")
        print("------------------------")
        print(f"Word length {word_length} >= 10, calculating n-gram weights")
        
        ngram_dist = {l: 0.0 for l in ALPHABET}
        weights = calculate_ngram_weights(current_state, ngram_dict, guessed_letters)
        
        print("\nN-gram weights by length:")
        total_weight = 0
        for n, letter_weights in weights.items():
            weight = n * n
            total = sum(letter_weights.values())
            if total > 0:
                print(f"\n{n}-gram predictions:")
                normalized = {k: v/total for k, v in letter_weights.items()}
                total_weight += weight
                for letter, prob in sorted(normalized.items(), key=lambda x: x[1], reverse=True):
                    if prob > 0:
                        print(f"{letter}: {prob:.4f} (weight: {weight})")
                    ngram_dist[letter] += prob * weight
        
        if total_weight > 0:
            print("\nScaled n-gram distribution:")
            ngram_dist = {k: v/total_weight for k, v in ngram_dist.items()}
            max_weight = max(target_dist.values()) if target_dist.values() else 1.0
            if max(ngram_dist.values(), default=0) > 0:
                ngram_dist = {k: v * max_weight / max(ngram_dist.values()) for k, v in ngram_dist.items()}
            
            ngram_weight = min(0.5, 0.3 + 0.02 * (word_length - 10))
            print(f"N-gram weight: {ngram_weight:.2f}")
            
            for letter in ALPHABET:
                target_dist[letter] = (1 - ngram_weight) * target_dist[letter] + ngram_weight * ngram_dist[letter]
    
    print("\nStep 4: Vowel Adjustment")
    print("----------------------")
    print(f"Current vowel ratio: {vowel_ratio:.2f}")
    if vowel_ratio > 0.48:
        print("Too many vowels (>0.48), reducing vowel weights by 0.7")
    elif vowel_ratio <= 0.38:
        print("Too few vowels (<=0.38), boosting vowel weights by 1.3")
    target_dist = adjust_vowel_weights(target_dist, current_state, vowel_ratio)
    
    print("\nAfter vowel adjustment:")
    for l, v in sorted(target_dist.items(), key=lambda x: x[1], reverse=True):
        if v > 0:
            print(f"{l}: {v:.4f}")
    
    print("\nStep 5: Normalization")
    print("-------------------")
    total = sum(target_dist.values())
    if total > 0:
        target_dist = {k: v/total for k, v in target_dist.items()}
    
    print("Final normalized distribution:")
    for l, v in sorted(target_dist.items(), key=lambda x: x[1], reverse=True):
        if v > 0:
            print(f"{l}: {v:.4f}")

if __name__ == "__main__":
    setup_logging()
    
    with time_block("Building n-gram dictionary"):
        with open('words_250000_train.txt', 'r') as f:
            words = [w.strip().lower() for w in f.readlines()]
        ngram_dict = build_ngram_dictionary(words)
    
    with time_block("Loading and splitting data"):
        training_words, validation_words = load_and_split_data()
    
    with time_block("Generating training dataset"):
        logging.info("Generating training dataset...")
        train_states = generate_dataset(training_words, ngram_dict, is_validation=False, train_words=training_words)
    
    with time_block("Generating validation dataset"):
        logging.info("Generating validation dataset...")
        val_states = generate_dataset(validation_words, ngram_dict, is_validation=True, train_words=training_words)
    
    # Print sample states with clearer formatting
    def print_sample_states(states, mode="Training"):
        logging.info(f"\nSample {mode} States:")
        for i, state in enumerate(random.sample(states, 5)):
            logging.info(f"\n{mode} Sample {i+1}:")
            logging.info(f"Current State: {state['current_state']}")
            logging.info(f"Original Word: {state['original_word']}")
            logging.info(f"Guessed Letters: {state['guessed_letters']}")
            logging.info(f"Vowel Ratio: {state['vowel_ratio']:.2f}")
            logging.info("Target Distribution:")
            # Only show probabilities > 0.01 for clarity
            for letter, prob in sorted(state['target_distribution'].items(), key=lambda x: x[1], reverse=True):
                if prob > 0.01:
                    logging.info(f"{letter}: {prob:.4f}")
            logging.info("-" * 40)
    
    print_sample_states(train_states, "Training")
    print_sample_states(val_states, "Validation")
    
    # Save datasets
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'{DATA_DIR}/hangman_states_{timestamp}.pkl'
    
    with open(save_path, 'wb') as f:
        pickle.dump({
            'train_states': train_states,
            'val_states': val_states,
            'val_words': validation_words
        }, f)
    
    logging.info(f"\nDataset saved to {save_path}")
    
    # # Debug example
    # print("\n=== DEBUG EXAMPLE ===")
    # sample_state = "h_ll_"
    # sample_word = "hello"
    # sample_guessed = {'h', 'p', 'r'}
    # sample_ratio = 0.0  # no vowels known yet
    
    # # Get some training words for testing
    # with open('words_250000_train.txt', 'r') as f:
    #     debug_train_words = [w.strip().lower() for w in f.readlines() if len(w.strip()) == 5]
    
    # # Test both training and validation modes
    # print("\nTRAINING MODE:")
    # debug_target_distribution(sample_state, sample_word, sample_guessed, sample_ratio, debug_train_words, is_validation=False)
    
    # print("\nVALIDATION MODE:")
    # debug_target_distribution(sample_state, sample_word, sample_guessed, sample_ratio, debug_train_words, is_validation=True) 