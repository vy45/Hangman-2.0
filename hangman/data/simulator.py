import numpy as np
from typing import Dict, List, Set, Tuple, Any
from .word_processor import WordProcessor
import torch
import time
import os
import matplotlib.pyplot as plt
import json

class HangmanSimulator:
    """Simulates Hangman game states."""
    
    def __init__(self, word_processor, optimal_prob: float = 0.5):
        """Initialize simulator.
        
        Args:
            word_processor: WordProcessor instance
            optimal_prob: Probability for target distribution (default: 0.5)
        """
        self.word_processor = word_processor
        self.optimal_prob = optimal_prob
        self.stats = {
            'total_guesses': 0,
            'total_games': 0,
            'wins': 0,
            'losses': 0,
            'by_length': {},  # Will store stats per word length
            'game_lengths': []  # Store number of guesses for each game
        }
        
    def generate_game_state(self, word: str, guessed: Set[str]) -> Dict[str, Any]:
        """Generate a game state for a word with guessed letters.
        
        Args:
            word: The target word
            guessed: Set of already guessed letters
            
        Returns:
            Dictionary containing:
                - word_state: One-hot encoded pattern
                - alphabet_state: Available letters (1 for available, 0 for guessed)
                - word_length: Length of word
                - target_distribution: Target probabilities for next guess
        """
        # Create word state (one-hot encoding for each position)
        pattern = ['_' if letter not in guessed else letter for letter in word]
        word_state = np.zeros((len(word), 28))  # 26 letters + '_' + padding
        
        for i, char in enumerate(pattern):
            if char == '_':
                word_state[i, 26] = 1  # '_' position
            else:
                word_state[i, ord(char) - ord('a')] = 1
                
        # Create alphabet state (1 for available letters)
        alphabet_state = np.zeros(26)
        for i in range(26):
            letter = chr(ord('a') + i)
            if letter not in guessed:
                alphabet_state[i] = 1
                
        # Create target distribution based on actual letters in word
        target = np.zeros(26)
        remaining_letters = set(word) - guessed
        if remaining_letters:
            # Equal probability for all remaining letters in word
            for letter in remaining_letters:
                target[ord(letter) - ord('a')] = 1.0 / len(remaining_letters)
        
        return {
            'word_state': word_state,
            'alphabet_state': alphabet_state,
            'word_length': len(word),
            'target_distribution': target
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
                'guesses_list': []  # For box plot
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
            
            # Track wrong guesses
            if next_guess not in word:
                wrong_guesses += 1
                
            # Check if word is complete
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

    def plot_statistics(self, save_dir: str):
        """Generate and save statistical plots"""
        # Create plots directory
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Box plot of guesses by word length
        plt.figure(figsize=(12, 6))
        lengths = sorted(self.stats['by_length'].keys())
        box_data = [self.stats['by_length'][l]['guesses_list'] for l in lengths]
        plt.boxplot(box_data, labels=lengths)
        plt.title('Distribution of Guesses by Word Length')
        plt.xlabel('Word Length')
        plt.ylabel('Number of Guesses')
        plt.savefig(f'{plots_dir}/guesses_by_length_box.png')
        plt.close()
        
        # 2. Success rate by word length
        plt.figure(figsize=(12, 6))
        success_rates = [
            self.stats['by_length'][l]['wins'] / self.stats['by_length'][l]['total_games'] * 100 
            for l in lengths
        ]
        plt.bar(lengths, success_rates)
        plt.title('Success Rate by Word Length')
        plt.xlabel('Word Length')
        plt.ylabel('Success Rate (%)')
        plt.savefig(f'{plots_dir}/success_rate_by_length.png')
        plt.close()
        
        # 3. Game length distribution
        plt.figure(figsize=(12, 6))
        plt.hist(self.stats['game_lengths'], bins=20)
        plt.title('Distribution of Game Lengths')
        plt.xlabel('Number of Guesses')
        plt.ylabel('Frequency')
        plt.savefig(f'{plots_dir}/game_length_distribution.png')
        plt.close()
        
        # Save numerical statistics
        stats_summary = {
            'overall': {
                'total_games': self.stats['total_games'],
                'win_rate': self.stats['wins'] / self.stats['total_games'] * 100,
                'avg_guesses': self.stats['total_guesses'] / self.stats['total_games'],
            },
            'by_length': {
                length: {
                    'total_games': data['total_games'],
                    'win_rate': (data['wins'] / data['total_games'] * 100),
                    'avg_guesses': data['total_guesses'] / data['total_games']
                }
                for length, data in self.stats['by_length'].items()
            }
        }
        
        with open(f'{plots_dir}/statistics_summary.json', 'w') as f:
            json.dump(stats_summary, f, indent=4)

    @property
    def average_guesses_per_word(self):
        return self.stats['total_guesses'] / max(1, self.stats['total_games'])

    def generate_training_data(self, words: List[str], save_path: str):
        """Generate and save training states for all words"""
        print(f"Generating training data from {len(words)} words...")
        start_time = time.time()
        
        all_states = []
        for i, word in enumerate(words):
            if i % 1000 == 0:  # Progress update
                print(f"Processed {i}/{len(words)} words...")
            
            states = self.simulate_game(word)
            all_states.extend(states)
            
        print(f"Generated {len(all_states)} states from {len(words)} words")
        print(f"Average states per word: {len(all_states)/len(words):.2f}")
        print(f"Time taken: {time.time() - start_time:.2f}s")
        
        # Save to file
        torch.save({
            'states': all_states,
            'stats': self.stats
        }, save_path)
        
        return len(all_states) 