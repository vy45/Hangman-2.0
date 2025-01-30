import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
import torch
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import time
from tqdm import tqdm
from pathlib import Path
import psutil

class WordProcessor:
    """Organizes words by length for the Hangman game."""
    
    def __init__(self, word_list: List[str]):
        self.word_list = [w.lower() for w in word_list]
        self.words_by_length = self._organize_by_length()
        
    def _organize_by_length(self) -> Dict[int, Set[str]]:
        by_length = {}
        for word in self.word_list:
            length = len(word)
            if length not in by_length:
                by_length[length] = set()
            by_length[length].add(word)
        return by_length

def split_words_stratified(words: List[str], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Split words into train and val sets, maintaining length distribution"""
    np.random.seed(seed)
    
    # Group words by length
    by_length = defaultdict(list)
    for word in words:
        by_length[len(word)].append(word)
    
    train_words, val_words = [], []
    
    # Split each length group with the same ratio
    for length, word_group in by_length.items():
        n_train = int(len(word_group) * train_ratio)
        indices = np.random.permutation(len(word_group))
        
        train_words.extend([word_group[i] for i in indices[:n_train]])
        val_words.extend([word_group[i] for i in indices[n_train:]])
    
    return train_words, val_words

class HangmanSimulator:
    """Generates training data through biased sampling"""
    
    def __init__(self, word_processor: WordProcessor, optimal_prob: float = 0.7):
        self.word_processor = word_processor
        self.optimal_prob = optimal_prob
        self.stats = {
            'total_guesses': 0,
            'total_games': 0,
            'wins': 0,
            'losses': 0,
            'by_length': {},
            'game_lengths': []
        }
        
    def generate_game_state(self, word: str, guessed: Set[str]) -> Dict:
        """Create a game state representation"""
        pattern = ['_'] * len(word)
        for i, letter in enumerate(word):
            if letter in guessed:
                pattern[i] = letter
                
        # Create feature vectors
        word_state = np.zeros((len(word), 28))  # 26 letters + mask + padding
        for i, char in enumerate(pattern):
            if char == '_':
                word_state[i, 26] = 1  # mask token
            else:
                word_state[i, ord(char) - ord('a')] = 1
                
        alphabet_state = np.zeros(26)
        for letter in guessed:
            alphabet_state[ord(letter) - ord('a')] = 1
            
        # Create target distribution from actual remaining letters
        remaining_letters = set(word) - guessed
        target_dist = np.zeros(26)
        if remaining_letters:  # If there are letters left to guess
            prob = 1.0 / len(remaining_letters)
            for letter in remaining_letters:
                target_dist[ord(letter) - ord('a')] = prob
            
        return {
            'word_state': word_state,
            'alphabet_state': alphabet_state,
            'word_length': len(word),
            'target_distribution': target_dist
        }
    
    def simulate_game(self, word: str) -> List[Dict]:
        states = []
        guessed = set()
        wrong_guesses = 0
        n_guesses = 0
        word_len = len(word)
        
        # Get unique letters in word for biased sampling
        word_letters = set(word)
        
        # Initialize per-length stats if not exists
        if word_len not in self.stats['by_length']:
            self.stats['by_length'][word_len] = {
                'total_games': 0,
                'wins': 0,
                'total_guesses': 0,
                'guesses_list': []
            }
            
        while wrong_guesses < 6 and len(guessed) < 26:
            state = self.generate_game_state(word, guessed)
            states.append(state)
            
            # Get next guess
            if np.random.random() < self.optimal_prob:
                # Choose randomly from letters in word that haven't been guessed
                available_word_letters = word_letters - guessed
                if available_word_letters:
                    next_guess = np.random.choice(list(available_word_letters))
                else:
                    # If all word letters guessed, choose random unguessed letter
                    valid_indices = np.where(state['alphabet_state'] == 0)[0]
                    if len(valid_indices) == 0:
                        break
                    next_guess = chr(np.random.choice(valid_indices) + ord('a'))
            else:
                # Choose random from valid (unguessed) letters
                valid_indices = np.where(state['alphabet_state'] == 0)[0]
                if len(valid_indices) == 0:
                    break
                next_guess = chr(np.random.choice(valid_indices) + ord('a'))
            
            guessed.add(next_guess)
            n_guesses += 1
            
            if next_guess not in word:
                wrong_guesses += 1
                
            if all(letter in guessed for letter in word):
                self.stats['wins'] += 1
                self.stats['by_length'][word_len]['wins'] += 1
                break
                
        # Update statistics
        self.stats['total_games'] += 1
        self.stats['total_guesses'] += n_guesses
        self.stats['game_lengths'].append(n_guesses)
        
        self.stats['by_length'][word_len]['total_games'] += 1
        self.stats['by_length'][word_len]['total_guesses'] += n_guesses
        self.stats['by_length'][word_len]['guesses_list'].append(n_guesses)
        
        if wrong_guesses >= 6:
            self.stats['losses'] += 1
            
        return states

    def generate_training_data(self, words: List[str], save_dir: str, batch_size: int = 10000):
        """Generate and save training data in batches to manage memory"""
        print(f"Generating training data from {len(words):,} words...")
        start_time = time.time()
        
        # Create directory for states
        states_dir = Path(save_dir) / 'states'
        states_dir.mkdir(exist_ok=True)
        
        total_states = 0
        batch_states = []
        batch_num = 0
        
        # Create progress bars
        pbar_words = tqdm(words, desc="Processing words")
        pbar_states = tqdm(desc="Total states", unit=" states")
        
        for word in pbar_words:
            states = self.simulate_game(word)
            batch_states.extend(states)
            total_states += len(states)
            
            # Update states progress bar
            pbar_states.update(len(states))
            # Update word progress bar description with stats
            pbar_words.set_postfix({
                'states': f"{total_states:,}",
                'avg_states/word': f"{total_states/(pbar_words.n+1):.1f}"
            })
            
            # Save batch if it reaches batch_size
            if len(batch_states) >= batch_size:
                self._save_batch(batch_states, states_dir, batch_num)
                batch_num += 1
                batch_states = []
        
        # Save any remaining states
        if batch_states:
            self._save_batch(batch_states, states_dir, batch_num)
        
        pbar_words.close()
        pbar_states.close()
        
        # Save statistics
        stats_file = Path(save_dir) / 'stats.json'
        with open(stats_file, 'w') as f:
            json.dump({
                'total_states': total_states,
                'total_games': self.stats['total_games'],
                'wins': self.stats['wins'],
                'losses': self.stats['losses'],
                'by_length': self.stats['by_length'],
                'game_lengths': self.stats['game_lengths'],
                'memory_usage_mb': get_process_memory_mb()
            }, f, indent=4)
        
        print(f"\nGenerated {total_states:,} states from {len(words):,} words")
        print(f"Average states per word: {total_states/len(words):.2f}")
        print(f"Time taken: {time.time() - start_time:.2f}s")
        print(f"Data saved to: {save_dir}")
        print(f"Memory usage: {get_process_memory_mb():.1f} MB")
        
        return total_states
    
    def _save_batch(self, states: List[Dict], save_dir: Path, batch_num: int):
        """Save a batch of states to compressed npz file
        
        Args:
            states: List of state dictionaries
            save_dir: Directory to save batch files
            batch_num: Batch number for filename
        """
        # Save each batch as a single compressed npz file
        batch_file = save_dir / f'batch_{batch_num}.npz'
        
        # Convert states to arrays, handling variable lengths
        word_states = np.array([s['word_state'].tobytes() for s in states], dtype=object)
        word_shapes = np.array([(s['word_state'].shape) for s in states])
        alphabet_states = np.array([s['alphabet_state'] for s in states])
        targets = np.array([s['target_distribution'] for s in states])
        word_lengths = np.array([s['word_length'] for s in states])
        
        # Save all components in a single compressed file
        np.savez_compressed(
            batch_file,
            word_states=word_states,
            word_shapes=word_shapes,
            alphabet_states=alphabet_states,
            targets=targets,
            word_lengths=word_lengths
        )

def analyze_dataset_stats(stats):
    """Analyze and display dataset statistics"""
    print("\nDataset Statistics:")
    print(f"Total Games: {stats['total_games']}")
    print(f"Win Rate: {stats['wins']/stats['total_games']*100:.2f}%")
    print(f"Average Guesses: {stats['total_guesses']/stats['total_games']:.2f}")
    
    lengths = sorted(stats['by_length'].keys())
    length_stats = []
    for length in lengths:
        data = stats['by_length'][length]
        length_stats.append({
            'length': length,
            'games': data['total_games'],
            'win_rate': data['wins']/data['total_games']*100,
            'avg_guesses': data['total_guesses']/data['total_games']
        })
    
    df = pd.DataFrame(length_stats)
    print("\nStats by word length:")
    print(df.to_string())
    return df

def plot_comparative_stats(all_stats, p_values, save_dir):
    """Create comparative plots for different p values"""
    # Use a valid style
    plt.style.use('seaborn-v0_8')
    colors = sns.color_palette("husl", len(p_values))
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("Available keys in stats:", all_stats[p_values[0]].keys())
    
    # Calculate total_guesses for each p value
    for stats in all_stats.values():
        stats['total_guesses'] = sum(length_data['total_guesses'] 
                                   for length_data in stats['by_length'].values())
    
    # 1. Win Rates
    plt.figure(figsize=(10, 6))
    win_rates = [stats['wins']/stats['total_games']*100 for stats in all_stats.values()]
    sns.barplot(x=[f"p={p}" for p in p_values], y=win_rates)
    plt.title('Win Rates by p Value')
    plt.ylabel('Win Rate (%)')
    plt.savefig(os.path.join(save_dir, 'win_rates.png'))
    plt.close()
    
    # 2. Average Guesses
    plt.figure(figsize=(10, 6))
    avg_guesses = [stats['total_guesses']/stats['total_games'] for stats in all_stats.values()]
    sns.barplot(x=[f"p={p}" for p in p_values], y=avg_guesses)
    plt.title('Average Guesses by p Value')
    plt.ylabel('Average Guesses')
    plt.savefig(os.path.join(save_dir, 'avg_guesses.png'))
    plt.close()
    
    # 3. Game Length Distributions
    plt.figure(figsize=(12, 6))
    for p, stats in all_stats.items():
        if 'game_lengths' in stats:
            sns.kdeplot(stats['game_lengths'], label=f"p={p}")
    plt.title('Distribution of Game Lengths')
    plt.xlabel('Number of Guesses')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'game_lengths_dist.png'))
    plt.close()
    
    # Get all lengths across all datasets and sort numerically
    all_lengths = set()
    for stats in all_stats.values():
        if 'by_length' in stats:
            all_lengths.update(int(l) for l in stats['by_length'].keys())
    all_lengths = sorted(list(all_lengths))
    
    # 4. Success Rate by Word Length
    plt.figure(figsize=(15, 6))
    for p, stats in all_stats.items():
        if 'by_length' in stats:
            success_rates = [
                stats['by_length'][str(l)]['wins'] / stats['by_length'][str(l)]['total_games'] * 100 
                if str(l) in stats['by_length'] else 0
                for l in all_lengths
            ]
            plt.plot(all_lengths, success_rates, marker='o', label=f"p={p}")
    plt.title('Success Rate by Word Length')
    plt.xlabel('Word Length')
    plt.ylabel('Success Rate (%)')
    plt.legend()
    plt.grid(True)
    plt.xticks(all_lengths)  # Force x-axis to show all lengths
    plt.savefig(os.path.join(save_dir, 'success_rate_by_length.png'))
    plt.close()

    # 5. Average Guesses by Word Length
    plt.figure(figsize=(15, 6))
    for p, stats in all_stats.items():
        if 'by_length' in stats:
            avg_guesses = [
                stats['by_length'][str(l)]['total_guesses'] / stats['by_length'][str(l)]['total_games']
                if str(l) in stats['by_length'] else 0
                for l in all_lengths
            ]
            plt.plot(all_lengths, avg_guesses, marker='o', label=f"p={p}")
    plt.title('Average Guesses by Word Length')
    plt.xlabel('Word Length')
    plt.ylabel('Average Guesses')
    plt.legend()
    plt.grid(True)
    plt.xticks(all_lengths)  # Force x-axis to show all lengths
    plt.savefig(os.path.join(save_dir, 'avg_guesses_by_length.png'))
    plt.close()

def get_process_memory_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def load_existing_stats(save_dir: str, p_values: List[float]) -> Dict:
    """Load statistics from existing generated datasets"""
    all_stats = {}
    for p in p_values:
        p_dir = os.path.join(save_dir, f"p_{int(p*100)}")
        stats_file = os.path.join(p_dir, "stats.json")
        
        if not os.path.exists(stats_file):
            raise FileNotFoundError(f"Stats file not found: {stats_file}")
            
        with open(stats_file, 'r') as f:
            all_stats[p] = json.load(f)
            
    return all_stats

def main():
    # Configuration
    generate_p_values = [0.4, 0.6]  # Only generate these new datasets
    word_file = "words_250000_train.txt"
    save_dir = "p_analysis"
    train_ratio = 0.8
    generate_data = True  # Set to True to generate new p-value datasets
    
    print(f"Using save directory: {save_dir}")
    
    if generate_data:
        # Check which p-values need to be generated
        existing_datasets = set()
        for d in os.listdir(save_dir):
            if d.startswith('p_'):
                try:
                    p_value = float(d.split('_')[1]) / 100
                    existing_datasets.add(p_value)
                except (IndexError, ValueError):
                    continue
        
        new_p_values = [p for p in generate_p_values if p not in existing_datasets]
        
        if not new_p_values:
            print("All requested datasets already exist. Skipping generation.")
        else:
            print(f"Generating datasets for p values: {new_p_values}")
            print(f"Using word file: {word_file}")
            print(f"Saving results to: {save_dir}")
            os.makedirs(save_dir, exist_ok=True)
            
            # Load and split words
            with open(word_file, 'r') as f:
                all_words = [word.strip().lower() for word in f.readlines()]
            
            train_words, val_words = split_words_stratified(all_words, train_ratio)
            print(f"\nSplit {len(all_words)} words into:")
            print(f"Training: {len(train_words)} words")
            print(f"Validation: {len(val_words)} words")
            
            # Save validation words if not already saved
            val_file = os.path.join(save_dir, "validation_words.txt")
            if not os.path.exists(val_file):
                with open(val_file, 'w') as f:
                    f.write('\n'.join(val_words))
            
            # Generate only new datasets
            for p in new_p_values:
                print(f"\nGenerating dataset with p = {p}")
                p_dir = os.path.join(save_dir, f"p_{int(p*100)}")
                os.makedirs(p_dir, exist_ok=True)
                
                word_processor = WordProcessor(train_words)
                simulator = HangmanSimulator(word_processor, optimal_prob=p)
                
                simulator.generate_training_data(train_words, p_dir)
                analyze_dataset_stats(simulator.stats)
    
    # Find all available p-value datasets
    available_p_values = []
    for d in os.listdir(save_dir):
        if d.startswith('p_'):
            try:
                p_value = float(d.split('_')[1]) / 100
                available_p_values.append(p_value)
            except (IndexError, ValueError):
                continue
    
    available_p_values.sort()
    print(f"\nFound datasets for p values: {available_p_values}")
    
    # Load all available stats
    print("Loading all available datasets...")
    try:
        all_stats = load_existing_stats(save_dir, available_p_values)
        print("Successfully loaded all datasets")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Create comparative plots and save summary
    print("\nGenerating comparative plots...")
    try:
        plot_comparative_stats(all_stats, available_p_values, save_dir)
        print("Plots saved successfully")
    except Exception as e:
        print(f"Error generating plots: {e}")
        print("Make sure seaborn is installed: pip install seaborn")
        return
    
    # Generate summary statistics
    summary_df = pd.DataFrame([{
        'p_value': p,
        'win_rate': stats['wins']/stats['total_games']*100,
        'avg_guesses': stats['total_guesses']/stats['total_games'],
        'total_games': stats['total_games'],
        'total_states': stats['total_states'],
        'avg_states_per_word': stats['total_states']/stats['total_games']
    } for p, stats in all_stats.items()])
    
    print("\nSummary Statistics:")
    print(summary_df.to_string(float_format=lambda x: '{:.2f}'.format(x)))
    summary_df.to_csv(os.path.join(save_dir, "summary_stats.csv"), index=False)

if __name__ == "__main__":
    main() 