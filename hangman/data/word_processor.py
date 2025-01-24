from typing import Dict, List, Set

class WordProcessor:
    """Simple word processor for Hangman game."""
    
    def __init__(self, word_list: List[str]):
        """Initialize with a list of words.
        
        Args:
            word_list: List of words to process
        """
        self.word_list = [w.lower().strip() for w in word_list]
        self.words_by_length = self._organize_by_length()
        
    def _organize_by_length(self) -> Dict[int, Set[str]]:
        """Organize words by their length."""
        by_length = {}
        for word in self.word_list:
            length = len(word)
            if length not in by_length:
                by_length[length] = set()
            by_length[length].add(word)
        return by_length 

    def get_pattern_frequencies(self, pattern: str, guessed: Set[str]) -> Dict[str, float]:
        """Get letter frequencies for a given pattern and guessed letters.
        
        Args:
            pattern: Current word pattern (e.g. "h_ll_")
            guessed: Set of already guessed letters
            
        Returns:
            Dictionary mapping unguessed letters to their frequencies
        """
        # Get words matching pattern length
        word_length = len(pattern)
        possible_words = self.words_by_length.get(word_length, set())
        
        # Filter words matching pattern
        matching_words = set()
        for word in possible_words:
            matches = True
            for i, (p, w) in enumerate(zip(pattern, word)):
                if p != '_' and p != w:
                    matches = False
                    break
            if matches:
                matching_words.add(word)
        
        # Count letter frequencies
        letter_counts = {chr(i): 0 for i in range(ord('a'), ord('z')+1)}
        total_words = len(matching_words) or 1  # Avoid division by zero
        
        for word in matching_words:
            for letter in set(word):  # Count each letter once per word
                if letter not in guessed:
                    letter_counts[letter] += 1
        
        # Convert to frequencies
        frequencies = {letter: count/total_words 
                     for letter, count in letter_counts.items() 
                     if letter not in guessed}
        
        return frequencies 