from typing import Dict, List, Set

class WordProcessor:
    """Organizes words by length for the Hangman game.
    
    This class maintains a collection of words organized by their length,
    which can be useful for validation and testing purposes.
    """
    
    def __init__(self, word_list: List[str]):
        """Initialize WordProcessor with a list of words.
        
        Args:
            word_list: List of words to process
        """
        self.word_list = [w.lower() for w in word_list]
        self.words_by_length = self._organize_by_length()
        
    def _organize_by_length(self) -> Dict[int, Set[str]]:
        """Organize words by their length for quick access.
        
        Returns:
            Dictionary mapping word lengths to sets of words of that length
        """
        by_length = {}
        for word in self.word_list:
            length = len(word)
            if length not in by_length:
                by_length[length] = set()
            by_length[length].add(word)
        return by_length 