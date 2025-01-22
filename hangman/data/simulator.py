import numpy as np
from typing import Dict, List, Set, Tuple
from .word_processor import WordProcessor

class HangmanSimulator:
    """Generates training data through biased sampling"""
    
    def __init__(self, word_processor: WordProcessor, optimal_prob: float = 0.7):
        self.word_processor = word_processor
        self.optimal_prob = optimal_prob
        
    def generate_game_state(self, word: str, guessed: Set[str]) -> Dict:
        """Create a game state representation"""
        pattern = ['_'] * len(word)
        for i, letter in enumerate(word):
            if letter in guessed:
                pattern[i] = letter
                
        # Get next guess distribution
        freq = self.word_processor.get_pattern_frequencies(''.join(pattern), guessed)
        
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
            
        # Create target distribution
        target_dist = np.zeros(26)
        for letter, prob in freq.items():
            target_dist[ord(letter) - ord('a')] = prob
            
        return {
            'word_state': word_state,
            'alphabet_state': alphabet_state,
            'word_length': len(word),
            'target_distribution': target_dist
        }
    
    def simulate_game(self, word: str) -> List[Dict]:
        """Simulate a complete game and return all states"""
        states = []
        guessed = set()
        
        while len(guessed) < 26:
            state = self.generate_game_state(word, guessed)
            states.append(state)
            
            # Get next guess
            if np.random.random() < self.optimal_prob:
                # Choose optimal guess
                next_guess = chr(np.argmax(state['target_distribution']) + ord('a'))
            else:
                # Choose random valid guess
                valid_indices = np.where(state['alphabet_state'] == 0)[0]
                if len(valid_indices) == 0:
                    break
                next_guess = chr(np.random.choice(valid_indices) + ord('a'))
            
            guessed.add(next_guess)
            
            # Check if word is complete
            if all(letter in guessed for letter in word):
                break
                
        return states 