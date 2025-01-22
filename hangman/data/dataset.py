import torch
from torch.utils.data import Dataset
from typing import List
from .simulator import HangmanSimulator
from .word_processor import WordProcessor

class HangmanDataset(Dataset):
    def __init__(self, words: List[str], word_processor: WordProcessor = None, max_word_length: int = 30):
        self.words = words
        self.word_processor = word_processor if word_processor else WordProcessor(words)
        self.simulator = HangmanSimulator(self.word_processor)
        self.max_word_length = max_word_length
        
    def __len__(self):
        return len(self.words)
        
    def pad_word_state(self, word_state, length):
        """Pad word state to max length"""
        if length < self.max_word_length:
            padding = torch.zeros(self.max_word_length - length, 28)
            padding[:, -1] = 1  # Set padding token
            return torch.cat([torch.FloatTensor(word_state), padding], dim=0)
        return torch.FloatTensor(word_state)
        
    def __getitem__(self, idx):
        word = self.words[idx]
        states = self.simulator.simulate_game(word)
        if not states:  # If simulation failed, try another word
            return self.__getitem__((idx + 1) % len(self))
            
        # Randomly select one state from the game
        state = states[torch.randint(len(states), (1,)).item()]
        
        # Pad word state to max length
        word_state = self.pad_word_state(state['word_state'], len(word))
        
        return {
            'word_state': word_state,
            'alphabet_state': torch.FloatTensor(state['alphabet_state']),
            'word_length': torch.LongTensor([state['word_length']]),
            'target_distribution': torch.FloatTensor(state['target_distribution'])
        } 