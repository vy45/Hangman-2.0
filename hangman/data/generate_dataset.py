import torch
import numpy as np
from typing import List, Dict, Set, Optional
import os
import argparse
from word_processor import WordProcessor
from simulator import HangmanSimulator
import time
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    batch_size: int = 1000
    p_values: List[float] = [0.5]
    max_word_length: int = 30
    feature_dim: int = 28
    validation_split: float = 0.1

class DatasetGenerator:
    def __init__(
        self,
        word_file: Path,
        output_dir: Path,
        config: DatasetConfig
    ):
        """Initialize dataset generator
        
        Args:
            word_file: Path to word list file
            output_dir: Output directory for generated data
            config: Generation configuration
        """
        # Initialize paths
        self.word_file = Path(word_file)
        self.output_dir = Path(output_dir)
        self.config = config
        
        # Initialize state
        self.words: List[str] = []
        self.train_words: List[str] = []
        self.val_words: List[str] = []
        self.processor: Optional[WordProcessor] = None
        self.simulator: Optional[HangmanSimulator] = None
        
        # Initialize statistics
        self.stats: Dict[str, Any] = {
            'total_words': 0,
            'total_states': 0,
            'start_time': time.time(),
            'batch_counts': {}
        }
        
        # Validate setup
        self._validate_setup()

def generate_dataset(
    word_file: str,
    save_dir: str = 'data',
    word_percentage: float = 100.0,  # Default to using all words
    optimal_prob: float = 0.7  # Default to 70% optimal, 30% random
) -> None:
    """Generate and save dataset with stratified split
    
    Args:
        word_file: Path to word list file
        save_dir: Directory to save dataset files
        word_percentage: Percentage of words to use (1-100)
        optimal_prob: Probability of optimal strategy (0-1)
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load words
    with open(word_file, 'r') as f:
        all_words = [word.strip().lower() for word in f.readlines()]
    
    # Sample words if percentage < 100
    if word_percentage < 100:
        n_words = int(len(all_words) * word_percentage / 100)
        print(f"Using {word_percentage}% of words ({n_words} words)")
        np.random.shuffle(all_words)
        words = all_words[:n_words]
    else:
        words = all_words
    
    train_path = f'{save_dir}/full_training_states.pt'
    val_path = f'{save_dir}/full_validation_states.pt'
    
    print(f"Generating dataset from {len(words)} words...")
    start_time = time.time()
    
    # Stratified split by word length
    word_lengths = [len(word) for word in words]
    unique_lengths = sorted(set(word_lengths))
    
    train_words = []
    val_words = []
    
    print("Performing stratified split by word length...")
    for length in unique_lengths:
        length_words = [w for w, l in zip(words, word_lengths) if l == length]
        n_words = len(length_words)
        n_train = int(0.8 * n_words)
        
        # Shuffle words of this length
        np.random.shuffle(length_words)
        train_words.extend(length_words[:n_train])
        val_words.extend(length_words[n_train:])
    
    print(f"\nGenerating training states ({len(train_words)} words)...")
    word_processor = WordProcessor(train_words)
    simulator = HangmanSimulator(word_processor)
    train_states = simulator.generate_training_data(train_words, train_path)
    
    print(f"\nGenerating validation states ({len(val_words)} words)...")
    simulator = HangmanSimulator(WordProcessor(val_words))
    val_states = simulator.generate_training_data(val_words, val_path)
    
    total_time = time.time() - start_time
    print(f"\nDataset generation complete!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Training states: {len(train_states)}")
    print(f"Validation states: {len(val_states)}")
    print(f"Files saved to: {save_dir}")

    # Create filename with parameters
    base_name = f"hangman_states_w{len(words)}_p{int(optimal_prob*100)}"
    version = 1
    while os.path.exists(f'{save_dir}/{base_name}_v{version}.pt'):
        version += 1
    save_path = f'{save_dir}/{base_name}_v{version}.pt'

    print("\nGenerating statistics plots...")
    simulator.plot_statistics(save_dir)
    
    print("\nOverall Statistics:")
    print(f"Total Games: {simulator.stats['total_games']}")
    print(f"Win Rate: {simulator.stats['wins']/simulator.stats['total_games']*100:.2f}%")
    print(f"Average Guesses: {simulator.stats['total_guesses']/simulator.stats['total_games']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Hangman dataset')
    parser.add_argument('--word_file', type=str, required=True,
                       help='Path to word list file')
    parser.add_argument('--save_dir', type=str, default='data',
                       help='Directory to save dataset files')
    parser.add_argument('--word_percentage', type=float, default=100.0,
                       help='Percentage of words to use (1-100)')
    parser.add_argument('--optimal_prob', type=float, default=0.7,
                       help='Probability of optimal strategy (0-1)')
    
    args = parser.parse_args()
    generate_dataset(args.word_file, args.save_dir, args.word_percentage, args.optimal_prob) 