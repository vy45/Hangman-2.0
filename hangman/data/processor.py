from typing import List, Dict, Set, Optional
import numpy as np
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ProcessorConfig:
    """Configuration for word processing"""
    max_word_length: int = 30
    min_word_length: int = 3
    feature_dim: int = 28
    alphabet: str = 'abcdefghijklmnopqrstuvwxyz'
    
class WordProcessor:
    """Process words and game states for Hangman"""
    
    def __init__(self, config: ProcessorConfig):
        """Initialize word processor
        
        Args:
            config: Processor configuration
        """
        self.config = config
        
        # Initialize character mappings
        self.char_to_idx: Dict[str, int] = {
            char: idx for idx, char in enumerate(config.alphabet)
        }
        self.idx_to_char: Dict[int, str] = {
            idx: char for char, idx in self.char_to_idx.items()
        }
        
        # Initialize state tracking
        self.vocab: Set[str] = set()
        self.word_lengths: Dict[int, int] = {}
        
    def encode_word_state(
        self,
        word: str,
        guessed_letters: Set[str]
    ) -> np.ndarray:
        """Encode word state as feature matrix
        
        Args:
            word: Target word
            guessed_letters: Set of guessed letters
            
        Returns:
            Feature matrix [word_length, feature_dim]
        """
        word_length = len(word)
        features = np.zeros((word_length, self.config.feature_dim))
        
        for i, char in enumerate(word):
            # One-hot encode character if guessed
            if char in guessed_letters:
                features[i, self.char_to_idx[char]] = 1
            else:
                features[i, -1] = 1  # Mask token
                
        return features
    
    def encode_alphabet_state(
        self,
        guessed_letters: Set[str]
    ) -> np.ndarray:
        """Encode alphabet state
        
        Args:
            guessed_letters: Set of guessed letters
            
        Returns:
            Binary vector [26] indicating guessed letters
        """
        state = np.zeros(26)
        for char in guessed_letters:
            if char in self.char_to_idx:
                state[self.char_to_idx[char]] = 1
        return state 