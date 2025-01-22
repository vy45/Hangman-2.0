import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple

class WordProcessor:
    """Handles word analysis and letter frequency calculations"""
    
    def __init__(self, word_list: List[str]):
        self.word_list = [w.lower() for w in word_list]
        self.letter_frequencies = self._calculate_global_frequencies()
        self.word_patterns = self._precompute_word_patterns()
        
    def _calculate_global_frequencies(self) -> Dict[str, float]:
        """Calculate global letter frequencies across all words"""
        freq = defaultdict(int)
        total = 0
        for word in self.word_list:
            for letter in set(word):  # Count each letter once per word
                freq[letter] += 1
            total += 1
        return {k: v/total for k, v in freq.items()}
    
    def _precompute_word_patterns(self) -> Dict[str, Set[str]]:
        """Precompute word patterns for faster lookup during simulation"""
        patterns = defaultdict(set)
        for word in self.word_list:
            patterns[len(word)].add(word)
        return patterns
    
    def get_pattern_frequencies(self, pattern: str, guessed: Set[str]) -> Dict[str, float]:
        """Calculate letter frequencies for words matching the current pattern"""
        matching_words = self._get_matching_words(pattern, guessed)
        if not matching_words:
            return self.letter_frequencies  # Fallback to global frequencies
            
        freq = defaultdict(int)
        for word in matching_words:
            remaining_letters = set(word) - guessed
            for letter in remaining_letters:
                freq[letter] += 1
                
        total = sum(freq.values())
        return {k: v/total for k, v in freq.items() if k not in guessed}
    
    def _get_matching_words(self, pattern: str, guessed: Set[str]) -> Set[str]:
        """Get all words that match the current pattern and guessed letters"""
        matching = set()
        pattern_len = len(pattern)
        
        for word in self.word_patterns[pattern_len]:
            if self._word_matches_pattern(word, pattern, guessed):
                matching.add(word)
                
        return matching
    
    @staticmethod
    def _word_matches_pattern(word: str, pattern: str, guessed: Set[str]) -> bool:
        """Check if a word matches the given pattern and guessed letters"""
        if len(word) != len(pattern):
            return False
            
        for w_char, p_char in zip(word, pattern):
            if p_char != '_' and p_char != w_char:
                return False
            if p_char == '_' and w_char in guessed:
                return False
        return True 